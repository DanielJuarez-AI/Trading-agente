"""
Runtime de inferencia rápida en producción CPU para Candle Predictor v2 (`NextCandleGRU`).
Verifica hashes SHA-256 al cargar y mantiene inferencia estrictamente causal.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from backend.app.ml.candle_predictor.config import (
    CLASSES,
    CandleV2Config,
    canonical_symbol,
    sha256_file,
)
from backend.app.ml.candle_predictor.features import (
    CANDLE_V2_FEATURES,
    CausalRobustScaler,
    compute_candle_v2_features,
)
from backend.app.ml.candle_predictor.network import NextCandleGRU, calibrated_proba


@dataclass
class CandleV2Output:
    pred_logret: float
    pred_return: float
    pred_tr: float
    probs: Dict[str, float]
    edge: float
    direction_confidence: float
    abstain: bool
    latent: List[float]
    production_enabled: bool
    ensemble_weight: float
    latency_ms: float


class CandleV2Runtime:
    """Cargador e inferencia de baja latencia (< 2 ms) por temporalidad (`1d` / `1h`)."""

    def __init__(
        self,
        interval: str,
        model: NextCandleGRU,
        scaler: CausalRobustScaler,
        cfg: CandleV2Config,
        temperature: float,
        conf_threshold: float,
        ensemble_weight: float,
        production_enabled: bool,
    ) -> None:
        self.interval = str(interval).strip().lower()
        self.model = model.eval()
        self.scaler = scaler
        self.cfg = cfg
        self.temperature = float(temperature)
        self.conf_threshold = float(conf_threshold)
        self.ensemble_weight = float(ensemble_weight)
        self.production_enabled = bool(production_enabled)

    @classmethod
    def from_manifest(
        cls,
        artifacts_dir: Path,
        interval: str,
        verify_hashes: bool = True,
    ) -> "CandleV2Runtime":
        manifest_path = Path(artifacts_dir) / "candle_v2_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Falta {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models_map = manifest.get("models", {})
        tf = str(interval).strip().lower()
        if tf not in models_map:
            raise KeyError(f"Temporalidad '{tf}' ausente en {manifest_path}")

        entry = models_map[tf]
        feat_cols = list(entry.get("feature_columns", []))
        if feat_cols and feat_cols != CANDLE_V2_FEATURES:
            raise ValueError(
                f"Contrato de features Candle v2 inconsistente para {tf}: {len(feat_cols)} vs {len(CANDLE_V2_FEATURES)}"
            )

        weights_path = Path(artifacts_dir) / str(entry["weights_file"])
        scaler_path = Path(artifacts_dir) / str(entry["scaler_file"])
        if not weights_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Artefactos v2 incompletos en {artifacts_dir} para '{tf}'"
            )

        if verify_hashes:
            hashes = entry.get("sha256", {})
            if "weights" in hashes and sha256_file(weights_path) != hashes["weights"]:
                raise RuntimeError(f"Hash SHA-256 inválido para {weights_path.name}")
            if "scaler" in hashes and sha256_file(scaler_path) != hashes["scaler"]:
                raise RuntimeError(f"Hash SHA-256 inválido para {scaler_path.name}")

        cfg = CandleV2Config.from_dict(entry["config"])
        scaler = CausalRobustScaler.from_dict(
            json.loads(scaler_path.read_text(encoding="utf-8"))
        )
        model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=cfg)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            interval=tf,
            model=model,
            scaler=scaler,
            cfg=cfg,
            temperature=float(entry.get("temperature", 1.0)),
            conf_threshold=float(entry.get("conf_threshold", 0.50)),
            ensemble_weight=float(entry.get("ensemble_weight", cfg.ensemble_weight)),
            production_enabled=bool(entry.get("production_enabled", False)),
        )

    def predict(self, candles_df: pd.DataFrame, symbol: str) -> Optional[CandleV2Output]:
        """
        Predice la siguiente vela `(t+1)` usando las últimas `seq_len` filas válidas
        estrictamente causales. Devuelve `None` si no hay historial suficiente tras warmup.
        """
        canon = canonical_symbol(symbol)
        feat_df = compute_candle_v2_features(
            df=candles_df,
            symbol=canon,
            interval=self.interval,
            threshold_bps=self.cfg.threshold_bps,
        )
        seq_len = int(self.cfg.seq_len)
        if len(feat_df) < seq_len:
            return None

        tail_valid = feat_df["valid_mask"].iloc[-seq_len:].to_numpy(dtype=bool)
        if not bool(tail_valid.all()):
            return None

        raw_win = feat_df[CANDLE_V2_FEATURES].iloc[-seq_len:].to_numpy(dtype=np.float32)
        scaled_win = self.scaler.transform(raw_win)
        x_t = torch.from_numpy(np.ascontiguousarray(scaled_win, dtype=np.float32)).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x_t)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_logret = float(out["ret"][0].item()) * self.scaler.ret_scale
        pred_return = float(math.expm1(max(-5.0, min(5.0, pred_logret))))
        pred_tr = float(max(0.0, float(out["tr"][0].item()) * self.scaler.tr_scale))

        probs_np = calibrated_proba(
            out["logits"].numpy(), temperature=self.temperature
        )[0]
        probs_dict = {
            CLASSES[0]: round(float(probs_np[0]), 6),
            CLASSES[1]: round(float(probs_np[1]), 6),
            CLASSES[2]: round(float(probs_np[2]), 6),
        }
        pred_cls = int(np.argmax(probs_np))
        max_dir_prob = float(max(probs_np[0], probs_np[2]))
        dir_conf = float(np.max(probs_np))
        abstain = bool(pred_cls == 1 or max_dir_prob < self.conf_threshold)
        edge = float(probs_np[2] - probs_np[0])
        latent = [round(float(v), 6) for v in out["latent"][0].tolist()]

        return CandleV2Output(
            pred_logret=pred_logret,
            pred_return=pred_return,
            pred_tr=pred_tr,
            probs=probs_dict,
            edge=edge,
            direction_confidence=dir_conf,
            abstain=abstain,
            latent=latent,
            production_enabled=self.production_enabled,
            ensemble_weight=self.ensemble_weight,
            latency_ms=latency_ms,
        )

    def predict_batch(self, candles_df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """
        Inferencia causal fila a fila sobre un DataFrame histórico de un mismo ticker.
        Las filas sin ventana válida completa permanecen en `NaN` (y `abstain=True`).
        """
        canon = canonical_symbol(symbol)
        feat_df = compute_candle_v2_features(
            df=candles_df,
            symbol=canon,
            interval=self.interval,
            threshold_bps=self.cfg.threshold_bps,
        )
        n = len(feat_df)
        seq_len = int(self.cfg.seq_len)
        latent_dim = int(self.cfg.latent_dim)

        pred_logret = np.full(n, np.nan, dtype=np.float64)
        pred_return = np.full(n, np.nan, dtype=np.float64)
        pred_tr = np.full(n, np.nan, dtype=np.float64)
        probs = np.full((n, 3), np.nan, dtype=np.float64)
        abstain = np.ones(n, dtype=bool)
        latent = np.full((n, latent_dim), np.nan, dtype=np.float32)

        valid_mask = feat_df["valid_mask"].to_numpy(dtype=bool)
        raw_x = feat_df[CANDLE_V2_FEATURES].to_numpy(dtype=np.float32)
        scaled_x = self.scaler.transform(raw_x)

        end_indices: List[int] = []
        windows: List[np.ndarray] = []
        for end_idx in range(seq_len - 1, n):
            start_idx = end_idx - seq_len + 1
            if bool(valid_mask[start_idx : end_idx + 1].all()):
                end_indices.append(end_idx)
                windows.append(scaled_x[start_idx : end_idx + 1])

        if windows:
            x_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(windows), dtype=np.float32))
            with torch.no_grad():
                out = self.model(x_tensor)

            r_log = out["ret"].numpy() * self.scaler.ret_scale
            t_r = np.maximum(0.0, out["tr"].numpy() * self.scaler.tr_scale)
            pr_mat = calibrated_proba(out["logits"].numpy(), temperature=self.temperature)
            lat_mat = out["latent"].numpy().astype(np.float32)

            idx_arr = np.asarray(end_indices, dtype=np.int64)
            pred_logret[idx_arr] = r_log
            pred_return[idx_arr] = np.expm1(np.clip(r_log, -5.0, 5.0))
            pred_tr[idx_arr] = t_r
            probs[idx_arr] = pr_mat
            latent[idx_arr] = lat_mat

            cls_pred = np.argmax(pr_mat, axis=-1)
            max_dir = np.maximum(pr_mat[:, 0], pr_mat[:, 2])
            abstain[idx_arr] = (cls_pred == 1) | (max_dir < self.conf_threshold)

        return {
            "pred_logret": pred_logret,
            "pred_return": pred_return,
            "pred_tr": pred_tr,
            "probs": probs,
            "abstain": abstain,
            "latent": latent,
        }
