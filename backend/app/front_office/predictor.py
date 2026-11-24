"""
Front Office - Predictor Cuantitativo Multi-Temporalidad (LSTM T4 + XGBoost Walk-Forward 1h/1d).

Carga los artefactos congelados (model_manifest.json, scaler_v1.joblib, lstm_v1.pt,
y quant_v2_manifest.json + quant_xgb_1d.joblib / quant_xgb_1h.joblib),
verifica su integridad SHA-256 y el contrato exacto de columnas antes de emitir
el rendimiento estimado r_hat_{t+1} en escala original y series históricas sin look-ahead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.app.core.config import ARTIFACTS_DIR
from backend.app.front_office.features import (
    DEFAULT_SEQ_LEN,
    ENHANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_sequences,
    compute_technical_features,
)
from backend.app.ml.candle_predictor.inference import CandleV2Runtime


class LSTMRegressor(nn.Module):
    """Red LSTM para predicción cuantitativa de rendimiento siguiente r_{t+1}."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class MLPRegressor(nn.Module):
    """Perceptrón Multicapa para ventana temporal aplanada."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        seq_len: int = DEFAULT_SEQ_LEN,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        flat_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        return self.net(flat).squeeze(-1)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PredictionResult:
    symbol: str
    candle_close_utc: str
    pred_return_t1: float
    direction: str
    ema_ratio: float
    vol_20: float
    rsi_14: float
    momentum_5: float
    ret_lag_1: float
    model_version: str
    features_verified: bool
    atr_14: float = 0.0
    atr_pct: float = 0.0
    interval: str = "1d"
    model_detail: str = "lstm_v1"
    pred_volatility_t1: float = 0.0
    candle_v2_return: float = 0.0
    direction_probs: Dict[str, float] = field(default_factory=dict)
    direction_confidence: float = 0.0
    abstain: bool = False
    candle_v2_latent: List[float] = field(default_factory=list)
    candle_v2_enabled: bool = False


class QuantPredictor:
    """Cargador e inferencia de producción para LSTM T4 y Quant v2 XGBoost (1h/1d)."""

    def __init__(self, artifacts_dir: Path | None = None, verify_hashes: bool = True) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.manifest_path = self.artifacts_dir / "model_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No se encontró model_manifest.json en {self.manifest_path}. "
                "Ejecute primero notebooks/generate_phase1_artifacts.py."
            )

        self.manifest: Dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.feature_columns: List[str] = list(
            self.manifest.get("feature_columns", FEATURE_COLUMNS)
        )
        self.seq_len: int = int(self.manifest.get("seq_len", DEFAULT_SEQ_LEN))
        self.model_version: str = str(self.manifest.get("model_version", "v1"))

        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Contrato de features inconsistente: manifest={self.feature_columns} "
                f"vs runtime={FEATURE_COLUMNS}"
            )

        scaler_path = self.artifacts_dir / self.manifest.get("scaler_file", "scaler_v1.joblib")
        model_path = self.artifacts_dir / self.manifest.get("model_file", "lstm_v1.pt")
        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Faltan artefactos de modelo en {self.artifacts_dir}: "
                f"scaler={scaler_path.exists()}, model={model_path.exists()}"
            )

        if verify_hashes:
            expected_hashes = self.manifest.get("sha256", {})
            if "scaler" in expected_hashes:
                actual_scaler_hash = compute_sha256(scaler_path)
                if actual_scaler_hash != expected_hashes["scaler"]:
                    raise RuntimeError("Hash SHA-256 inválido para scaler_v1.joblib")
            if "model" in expected_hashes:
                actual_model_hash = compute_sha256(model_path)
                if actual_model_hash != expected_hashes["model"]:
                    raise RuntimeError("Hash SHA-256 inválido para lstm_v1.pt")

        self.scaler = joblib.load(scaler_path)
        arch = self.manifest.get("architecture", {})
        hidden_dim = int(arch.get("hidden_dim", 32))
        num_layers = int(arch.get("num_layers", 1))
        dropout = float(arch.get("dropout", 0.1))

        self.model = LSTMRegressor(
            input_dim=len(self.feature_columns),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Cargar modelos Quant v2 (Walk-Forward XGBoost 1d & 1h) con verificación SHA-256
        self.xgb_1d: Optional[Any] = None
        self.xgb_1h: Optional[Any] = None
        self.quant_v2_manifest: Dict[str, Any] = {}
        self._load_quant_v2(verify_hashes=verify_hashes)

        # Cargar Candle Predictor v2 (NextCandleGRU multi-cabezal por intervalo)
        self.candle_v2: Dict[str, CandleV2Runtime] = {}
        self._load_candle_v2(verify_hashes=verify_hashes)

    def _load_candle_v2(self, verify_hashes: bool = True) -> None:
        c2_man_path = self.artifacts_dir / "candle_v2_manifest.json"
        if not c2_man_path.exists():
            return
        try:
            man = json.loads(c2_man_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for tf in man.get("models", {}).keys():
            try:
                rt = CandleV2Runtime.from_manifest(
                    self.artifacts_dir, interval=str(tf), verify_hashes=verify_hashes
                )
                self.candle_v2[str(tf).strip().lower()] = rt
            except RuntimeError:
                raise
            except Exception:
                continue

    def _load_quant_v2(self, verify_hashes: bool = True) -> None:
        v2_man_path = self.artifacts_dir / "quant_v2_manifest.json"
        if not v2_man_path.exists():
            return
        try:
            man = json.loads(v2_man_path.read_text(encoding="utf-8"))
            self.quant_v2_manifest = man
            models_meta = man.get("models", {})
            for tf, attr_name in (("1d", "xgb_1d"), ("1h", "xgb_1h")):
                m_info = models_meta.get(tf)
                if not m_info:
                    continue
                m_path = self.artifacts_dir / str(m_info.get("model_file", f"quant_xgb_{tf}.joblib"))
                if not m_path.exists():
                    continue
                if verify_hashes and "sha256" in m_info:
                    if compute_sha256(m_path) != m_info["sha256"]:
                        raise RuntimeError(f"Hash SHA-256 inválido para {m_path.name}")
                setattr(self, attr_name, joblib.load(m_path))
        except RuntimeError:
            raise
        except Exception:
            self.xgb_1d = None
            self.xgb_1h = None

    def _component_returns(
        self, feat_df: pd.DataFrame, interval: str = "1d"
    ) -> Dict[str, np.ndarray]:
        """Calcula los retornos de los componentes legacy (`lstm_v1` y `xgb` si aplica)."""
        x_raw = feat_df[self.feature_columns].to_numpy(dtype=np.float32)
        x_scaled = self.scaler.transform(x_raw)
        seqs, _ = build_sequences(x_scaled, targets=None, seq_len=self.seq_len)
        with torch.no_grad():
            lstm_preds = self.model(torch.tensor(seqs, dtype=torch.float32)).numpy()

        out: Dict[str, np.ndarray] = {"lstm_v1": lstm_preds.astype(float)}
        aligned = feat_df.iloc[self.seq_len - 1 :].reset_index(drop=True)
        atr_pcts = np.maximum(aligned["atr_pct"].to_numpy(dtype=np.float32), 1e-4)

        if interval == "1h" and self.xgb_1h is not None:
            x_enh = aligned[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            out["xgb"] = (self.xgb_1h.predict(x_enh) * atr_pcts).astype(float)
        elif interval == "1d" and self.xgb_1d is not None:
            x_enh = aligned[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            out["xgb"] = (self.xgb_1d.predict(x_enh) * atr_pcts).astype(float)
        return out

    def _predict_batch_returns(
        self, feat_df: pd.DataFrame, interval: str = "1d"
    ) -> tuple[np.ndarray, str]:
        comps = self._component_returns(feat_df, interval=interval)
        if interval == "1h" and "xgb" in comps:
            return comps["xgb"], "quant_xgb_1h"
        if interval == "1d" and "xgb" in comps:
            blended = (0.65 * comps["xgb"] + 0.35 * comps["lstm_v1"]).astype(float)
            return blended, "ensemble_xgb1d_lstm"
        return comps["lstm_v1"], self.model_version

    def predict_from_candles(
        self, candles_df: pd.DataFrame, symbol: str, interval: str = "1d"
    ) -> PredictionResult:
        """
        Recibe un DataFrame histórico de velas cerradas ordenado ascendentemente,
        calcula features sin leakage y predice r_hat_{t+1} ajustado a `interval`.
        """
        if len(candles_df) < 35:
            raise ValueError(
                f"Se requieren al menos 35 velas cerradas para inferencia segura; recibidas={len(candles_df)}"
            )

        feat_df = compute_technical_features(candles_df, drop_na=True)
        if len(feat_df) < self.seq_len:
            raise ValueError(
                f"Filas válidas tras warmup ({len(feat_df)}) < seq_len ({self.seq_len})."
            )

        comps = self._component_returns(feat_df, interval=interval)
        if interval == "1h" and "xgb" in comps:
            pred_val = float(comps["xgb"][-1])
            model_detail = "quant_xgb_1h"
        elif interval == "1d" and "xgb" in comps:
            pred_val = float(0.65 * comps["xgb"][-1] + 0.35 * comps["lstm_v1"][-1])
            model_detail = "ensemble_xgb1d_lstm"
        else:
            pred_val = float(comps["lstm_v1"][-1])
            model_detail = self.model_version

        direction = "UP" if pred_val > 0 else ("DOWN" if pred_val < 0 else "FLAT")

        pred_volatility_t1 = 0.0
        candle_v2_return = 0.0
        direction_probs: Dict[str, float] = {}
        direction_confidence = 0.0
        abstain = False
        candle_v2_latent: List[float] = []
        candle_v2_enabled = False

        runtime = self.candle_v2.get(str(interval).strip().lower())
        if runtime is not None:
            try:
                c2_out = runtime.predict(candles_df, symbol=symbol)
            except Exception:
                c2_out = None
            if c2_out is not None:
                pred_volatility_t1 = float(c2_out.pred_tr)
                candle_v2_return = float(c2_out.pred_return)
                direction_probs = dict(c2_out.probs)
                direction_confidence = float(c2_out.direction_confidence)
                abstain = bool(c2_out.abstain)
                candle_v2_latent = list(c2_out.latent)
                candle_v2_enabled = bool(c2_out.production_enabled)

                if c2_out.production_enabled:
                    w = float(c2_out.ensemble_weight)
                    if "xgb" in comps:
                        base_xgb = float(comps["xgb"][-1])
                        pred_val = float((1.0 - w) * base_xgb + w * c2_out.pred_return)
                        model_detail = f"ensemble_xgb{interval}_candlev2"
                    else:
                        pred_val = float(c2_out.pred_return)
                        model_detail = f"candle_v2_{interval}"

                    if c2_out.abstain:
                        direction = "FLAT"
                    else:
                        direction = "UP" if pred_val > 0 else ("DOWN" if pred_val < 0 else "FLAT")

        last_row = feat_df.iloc[-1]
        close_time = str(last_row.get("close_time", last_row.get("timestamp", "")))

        return PredictionResult(
            symbol=symbol,
            candle_close_utc=close_time,
            pred_return_t1=pred_val,
            direction=direction,
            ema_ratio=float(last_row["ema_ratio"]),
            vol_20=float(last_row["vol_20"]),
            rsi_14=float(last_row["rsi_14"]),
            momentum_5=float(last_row["momentum_5"]),
            ret_lag_1=float(last_row["ret_lag_1"]),
            model_version=self.model_version,
            features_verified=True,
            atr_14=float(last_row.get("atr_14", 0.0)),
            atr_pct=float(last_row.get("atr_pct", 0.015)),
            interval=interval,
            model_detail=model_detail,
            pred_volatility_t1=pred_volatility_t1,
            candle_v2_return=candle_v2_return,
            direction_probs=direction_probs,
            direction_confidence=direction_confidence,
            abstain=abstain,
            candle_v2_latent=candle_v2_latent,
            candle_v2_enabled=candle_v2_enabled,
        )

    def predict_historical_series(
        self,
        candles_df: pd.DataFrame,
        symbol: str,
        interval: str = "1d",
        last_n: int = 35,
    ) -> List[Dict[str, Any]]:
        """
        Calcula los pronósticos pasados t -> t+1 sobre las últimas `last_n` velas
        para graficar precisión visual histórica en el dashboard sin leakage.
        """
        if len(candles_df) < 35:
            return []
        feat_df = compute_technical_features(candles_df, drop_na=True)
        if len(feat_df) <= self.seq_len:
            return []

        preds, _ = self._predict_batch_returns(feat_df, interval=interval)
        aligned = feat_df.iloc[self.seq_len - 1 :].reset_index(drop=True)
        n_pairs = len(aligned) - 1
        if n_pairs <= 0:
            return []

        start_idx = max(0, n_pairs - int(last_n))
        series: List[Dict[str, Any]] = []
        for idx in range(start_idx, n_pairs):
            row_t = aligned.iloc[idx]
            row_next = aligned.iloc[idx + 1]
            c_t = float(row_t["close"])
            c_next = float(row_next["close"])
            r_hat = float(preds[idx])
            r_actual = ((c_next - c_t) / c_t) if c_t > 0 else 0.0
            hit = ((r_hat > 0 and r_actual > 0) or (r_hat < 0 and r_actual < 0))
            series.append(
                {
                    "open_time": str(row_next.get("open_time", "")),
                    "close_time": str(row_next.get("close_time", row_next.get("timestamp", ""))),
                    "predicted_close": round(c_t * (1.0 + r_hat), 4),
                    "actual_next_close": round(c_next, 4),
                    "pred_return_t1": round(r_hat, 6),
                    "actual_return_t1": round(r_actual, 6),
                    "hit": bool(hit),
                }
            )
        return series
