"""
Entrenador Walk-Forward Cronológico (Piloto y Universo 30 Tickers + BTCUSDT)
para Candle Predictor v2 (`NextCandleGRU`).

Genera artefactos versionados y verificados con SHA-256:
- `lstm_v2_{interval}.pt`
- `candle_v2_scaler_{interval}.json`
- `candle_v2_manifest.json`
- `reports/candle_v2_train_{interval}.json`
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.app.core.config import ARTIFACTS_DIR
from backend.app.ml.candle_predictor.config import (
    CLASSES,
    DEFAULT_PILOT_TICKERS,
    CandleV2Config,
    default_universe_tickers,
    load_config,
    sha256_file,
)
from backend.app.ml.candle_predictor.data import load_candles
from backend.app.ml.candle_predictor.evaluator import (
    evaluate_predictions,
    select_conf_threshold,
)
from backend.app.ml.candle_predictor.features import (
    CANDLE_V2_FEATURES,
    CausalRobustScaler,
    MultiTickerSequenceDataset,
    TickerFrame,
    build_ticker_frame,
)
from backend.app.ml.candle_predictor.network import (
    NextCandleGRU,
    calibrated_proba,
    fit_temperature,
)
from backend.app.ml.candle_predictor.splits import make_walk_forward_splits

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_causal_scaler_from_split(
    frames: Sequence[TickerFrame],
    train_index: Sequence[Tuple[int, int]],
    max_end_by_ticker: Dict[int, int],
) -> CausalRobustScaler:
    """
    Ajusta `CausalRobustScaler` usando únicamente filas cronológicamente pasadas
    del split de entrenamiento actual (nunca filas de validación, cal o TE).
    """
    x_parts: List[np.ndarray] = []
    for t_idx, max_end in sorted(max_end_by_ticker.items()):
        frame = frames[t_idx]
        sub_x = frame.X[: max_end + 1]
        sub_valid = frame.valid[: max_end + 1]
        if bool(sub_valid.any()):
            x_parts.append(sub_x[sub_valid])

    if not x_parts:
        raise RuntimeError("No hay filas válidas de entrenamiento para ajustar CausalRobustScaler.")

    y_ret_tr = np.asarray([float(frames[ti].y_ret[ei]) for ti, ei in train_index], dtype=np.float64)
    y_tr_tr = np.asarray([float(frames[ti].y_tr[ei]) for ti, ei in train_index], dtype=np.float64)

    scaler = CausalRobustScaler(feature_names=CANDLE_V2_FEATURES)
    scaler.fit(np.vstack(x_parts), y_ret=y_ret_tr, y_tr=y_tr_tr)
    return scaler


def _compute_balanced_class_weights(
    frames: Sequence[TickerFrame],
    train_index: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    y_dir = np.asarray([int(frames[ti].y_dir[ei]) for ti, ei in train_index], dtype=np.int64)
    counts = np.bincount(y_dir, minlength=len(CLASSES)).astype(np.float64)
    total = float(len(y_dir))
    weights = np.where(
        counts > 0.0,
        total / (float(len(CLASSES)) * np.maximum(counts, 1.0)),
        1.0,
    )
    weights = np.clip(weights, 0.5, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def _train_single_split(
    frames: Sequence[TickerFrame],
    train_index: Sequence[Tuple[int, int]],
    val_index: Sequence[Tuple[int, int]],
    max_end_by_ticker: Dict[int, int],
    cfg: CandleV2Config,
    seed: int,
    slippage_bps: float,
    fee_pct: float,
) -> Dict[str, Any]:
    """Entrena un split (fold o final) con early stopping y calibración sobre validación."""
    _set_seeds(seed)
    scaler = _fit_causal_scaler_from_split(frames, train_index, max_end_by_ticker)

    train_ds = MultiTickerSequenceDataset(
        frames=frames,
        scaler=scaler,
        seq_len=cfg.seq_len,
        index=train_index,
        require_target=True,
    )
    val_ds = MultiTickerSequenceDataset(
        frames=frames,
        scaler=scaler,
        seq_len=cfg.seq_len,
        index=val_index,
        require_target=True,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Dataset vacío en split: train={len(train_ds)}, val={len(val_ds)}")

    # PROHIBIDO shuffle=True para garantizar estricto orden cronológico
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    class_weights_t = _compute_balanced_class_weights(frames, train_ds.index)
    model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, int(cfg.max_epochs) + 1):
        model.train()
        for xb, yb_r, yb_t, yb_d in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            l_ret = F.smooth_l1_loss(out["ret"], yb_r / scaler.ret_scale)
            l_tr = F.smooth_l1_loss(out["tr"], yb_t / scaler.tr_scale)
            l_dir = F.cross_entropy(out["logits"], yb_d, weight=class_weights_t)
            loss = cfg.loss_w_ret * l_ret + cfg.loss_w_tr * l_tr + cfg.loss_w_dir * l_dir
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb_r, yb_t, yb_d in val_loader:
                out = model(xb)
                l_ret = F.smooth_l1_loss(out["ret"], yb_r / scaler.ret_scale)
                l_tr = F.smooth_l1_loss(out["tr"], yb_t / scaler.tr_scale)
                l_dir = F.cross_entropy(out["logits"], yb_d, weight=class_weights_t)
                vloss = cfg.loss_w_ret * l_ret + cfg.loss_w_tr * l_tr + cfg.loss_w_dir * l_dir
                bs = int(xb.size(0))
                val_loss_sum += float(vloss.item()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)
        if np.isfinite(val_loss) and val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(cfg.patience):
                break

    model.load_state_dict(best_state)
    model.eval()

    val_outputs = _predict_dataset(model, scaler, val_ds, batch_size=cfg.batch_size)
    temperature = fit_temperature(val_outputs["logits"], val_outputs["y_dir"])
    val_probs = calibrated_proba(val_outputs["logits"], temperature=temperature)
    val_meta = list(zip(val_ds.symbols, val_ds.end_ts.tolist()))

    conf_thr = select_conf_threshold(
        meta_rows=val_meta,
        probs=val_probs,
        y_ret_log=val_outputs["y_ret"],
        y_dir=val_outputs["y_dir"],
        conf_grid=cfg.conf_grid,
        min_coverage=cfg.min_coverage,
        slippage_bps=slippage_bps,
        fee_pct=fee_pct,
    )
    val_eval = evaluate_predictions(
        meta_rows=val_meta,
        pred_ret_log=val_outputs["pred_ret"],
        pred_tr=val_outputs["pred_tr"],
        probs=val_probs,
        y_ret_log=val_outputs["y_ret"],
        y_tr=val_outputs["y_tr"],
        y_dir=val_outputs["y_dir"],
        conf_threshold=conf_thr,
        slippage_bps=slippage_bps,
        fee_pct=fee_pct,
    )

    return {
        "model": model,
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "temperature": temperature,
        "conf_threshold": conf_thr,
        "val_outputs": val_outputs,
        "val_probs": val_probs,
        "val_meta": val_meta,
        "val_metrics": val_eval,
    }


def _predict_dataset(
    model: NextCandleGRU,
    scaler: CausalRobustScaler,
    ds: MultiTickerSequenceDataset,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pr, pt, lg, yr, yt, yd = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb_r, yb_t, yb_d in loader:
            out = model(xb)
            pr.append(out["ret"].numpy() * scaler.ret_scale)
            pt.append(out["tr"].numpy() * scaler.tr_scale)
            lg.append(out["logits"].numpy())
            yr.append(yb_r.numpy())
            yt.append(yb_t.numpy())
            yd.append(yb_d.numpy())

    return {
        "pred_ret": np.concatenate(pr) if pr else np.zeros(0, dtype=np.float64),
        "pred_tr": np.concatenate(pt) if pt else np.zeros(0, dtype=np.float64),
        "logits": np.vstack(lg) if lg else np.zeros((0, 3), dtype=np.float64),
        "y_ret": np.concatenate(yr) if yr else np.zeros(0, dtype=np.float64),
        "y_tr": np.concatenate(yt) if yt else np.zeros(0, dtype=np.float64),
        "y_dir": np.concatenate(yd) if yd else np.zeros(0, dtype=np.int64),
    }


def measure_inference_latency_ms(
    model: NextCandleGRU, seq_len: int, n_features: int, runs: int = 200
) -> float:
    """Mide latencia p50 (ms) de un forward batch=1 en CPU con 1 hilo."""
    model.eval()
    dummy = torch.zeros((1, int(seq_len), int(n_features)), dtype=torch.float32)
    prev_threads = torch.get_num_threads()
    latencies: List[float] = []
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            # Warmup corto
            for _ in range(10):
                model(dummy)
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000.0)
    finally:
        torch.set_num_threads(prev_threads)
    return round(float(np.median(latencies)), 4)


def _read_previous_baselines(interval: str, artifacts_dir: Path) -> Tuple[float, float]:
    fallback_q = -0.1133 if interval == "1d" else -0.2273
    fallback_l = -0.1063

    # Leer desde ARTIFACTS_DIR oficial como referencia estable aunque se guarde en pilot/
    base_dir = ARTIFACTS_DIR if (ARTIFACTS_DIR / "quant_v2_manifest.json").exists() else artifacts_dir
    q_path = base_dir / "quant_v2_manifest.json"
    if q_path.exists():
        try:
            q_man = json.loads(q_path.read_text(encoding="utf-8"))
            q_val = q_man.get("models", {}).get(interval, {}).get("holdout_te_metrics", {}).get("sharpe_step")
            if q_val is not None:
                fallback_q = float(q_val)
        except Exception:
            pass

    l_path = base_dir / "laya1_drl_manifest.json"
    if l_path.exists():
        try:
            l_man = json.loads(l_path.read_text(encoding="utf-8"))
            l_val = l_man.get("metrics", {}).get("TE", {}).get("sharpe_step")
            if l_val is not None:
                fallback_l = float(l_val)
        except Exception:
            pass

    return fallback_q, fallback_l


def save_candle_v2_artifacts(
    model: NextCandleGRU,
    scaler: CausalRobustScaler,
    cfg: CandleV2Config,
    out_dir: Path,
    interval: str,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Guarda pesos `.pt`, escalador `.json`, computa SHA-256 y fusiona la entrada de
    `interval` en `candle_v2_manifest.json` preservando el resto de temporalidades.
    """
    tf = str(interval).strip().lower()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / f"lstm_v2_{tf}.pt"
    scaler_path = out_dir / f"candle_v2_scaler_{tf}.json"
    manifest_path = out_dir / "candle_v2_manifest.json"

    torch.save(model.state_dict(), weights_path)
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")

    weights_sha = sha256_file(weights_path)
    scaler_sha = sha256_file(scaler_path)

    existing_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}

    models_section: Dict[str, Any] = dict(existing_manifest.get("models", {}))

    entry: Dict[str, Any] = {
        "interval": tf,
        "cell": cfg.cell,
        "weights_file": weights_path.name,
        "scaler_file": scaler_path.name,
        "sha256": {
            "weights": weights_sha,
            "scaler": scaler_sha,
        },
        "config": cfg.to_dict(),
        "feature_columns": list(CANDLE_V2_FEATURES),
        "classes": list(CLASSES),
        "temperature": 1.0,
        "conf_threshold": 0.50,
        "ensemble_weight": float(cfg.ensemble_weight),
        "latent_dim": int(cfg.latent_dim),
        "laya_state_dim_v2": 17 + int(cfg.latent_dim),
        "production_enabled": False,
        "gate_reason": "manual_artifact_save",
        "torch_version": torch.__version__,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra_manifest:
        entry.update(extra_manifest)

    models_section[tf] = entry
    manifest = {
        "manifest_version": "candle_v2",
        "models": models_section,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_interval(
    interval: str,
    tickers: Sequence[str],
    artifacts_dir: Path,
    source: str = "auto",
    period: Optional[str] = None,
    refresh: bool = False,
    fee_pct: float = 0.001,
    slippage_bps: float = 5.0,
    force_enable: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ejecuta el entrenamiento Walk-Forward completo + calibración + TE holdout para un intervalo."""
    cfg = load_config(interval=interval, overrides=cfg_overrides)
    embargo = cfg.resolved_embargo_bars

    candles_map, data_meta = load_candles(
        tickers=tickers,
        interval=interval,
        source=source,
        period=period,
        refresh=refresh,
        seq_len=cfg.seq_len,
    )
    if len(candles_map) < 3:
        raise RuntimeError(
            f"Se requieren al menos 3 tickers con histórico suficiente para {interval}; "
            f"cargados={len(candles_map)} ({data_meta['tickers_skipped']})."
        )

    frames: List[TickerFrame] = [
        build_ticker_frame(
            df=df,
            symbol=sym,
            interval=interval,
            threshold_bps=cfg.threshold_bps,
        )
        for sym, df in candles_map.items()
    ]

    plan = make_walk_forward_splits(
        frames=frames,
        seq_len=cfg.seq_len,
        n_folds=cfg.n_folds,
        te_frac=cfg.te_frac,
        embargo=embargo,
    )

    fold_reports: List[Dict[str, Any]] = []
    oos_meta: List[Tuple[str, int]] = []
    oos_ret, oos_tr, oos_probs = [], [], []
    oos_yr, oos_yt, oos_yd = [], [], []

    for fold in plan.folds:
        res = _train_single_split(
            frames=frames,
            train_index=fold.train_index,
            val_index=fold.val_index,
            max_end_by_ticker=fold.train_max_end_by_ticker,
            cfg=cfg,
            seed=cfg.seed + fold.fold,
            slippage_bps=slippage_bps,
            fee_pct=fee_pct,
        )
        compact_val = {k: v for k, v in res["val_metrics"].items() if k != "per_ticker"}
        fold_reports.append(
            {
                "fold": fold.fold,
                "best_epoch": res["best_epoch"],
                "best_val_loss": res["best_val_loss"],
                "temperature": res["temperature"],
                "conf_threshold": res["conf_threshold"],
                "train_samples": len(fold.train_index),
                "val_samples": len(fold.val_index),
                **compact_val,
            }
        )
        oos_meta.extend(res["val_meta"])
        oos_ret.append(res["val_outputs"]["pred_ret"])
        oos_tr.append(res["val_outputs"]["pred_tr"])
        oos_probs.append(res["val_probs"])
        oos_yr.append(res["val_outputs"]["y_ret"])
        oos_yt.append(res["val_outputs"]["y_tr"])
        oos_yd.append(res["val_outputs"]["y_dir"])

    wf_agg = evaluate_predictions(
        meta_rows=oos_meta,
        pred_ret_log=np.concatenate(oos_ret),
        pred_tr=np.concatenate(oos_tr),
        probs=np.vstack(oos_probs),
        y_ret_log=np.concatenate(oos_yr),
        y_tr=np.concatenate(oos_yt),
        y_dir=np.concatenate(oos_yd),
        conf_threshold=float(np.median([r["conf_threshold"] for r in fold_reports])),
        slippage_bps=slippage_bps,
        fee_pct=fee_pct,
    )
    wf_agg_compact = {k: v for k, v in wf_agg.items() if k != "per_ticker"}

    # Modelo final sobre Dev - Calibración con Early Stopping y Temperatura sobre Cal
    final_res = _train_single_split(
        frames=frames,
        train_index=plan.final_train_index,
        val_index=plan.final_cal_index,
        max_end_by_ticker=plan.final_train_max_end_by_ticker,
        cfg=cfg,
        seed=cfg.seed,
        slippage_bps=slippage_bps,
        fee_pct=fee_pct,
    )
    final_model: NextCandleGRU = final_res["model"]
    final_scaler: CausalRobustScaler = final_res["scaler"]
    final_temp = float(final_res["temperature"])
    final_conf_thr = float(final_res["conf_threshold"])

    # Evaluación ÚNICA estricta en Holdout TE
    te_ds = MultiTickerSequenceDataset(
        frames=frames,
        scaler=final_scaler,
        seq_len=cfg.seq_len,
        index=plan.final_te_index,
        require_target=True,
    )
    te_out = _predict_dataset(final_model, final_scaler, te_ds, batch_size=cfg.batch_size)
    te_probs = calibrated_proba(te_out["logits"], temperature=final_temp)
    te_meta = list(zip(te_ds.symbols, te_ds.end_ts.tolist()))

    te_primary_full = evaluate_predictions(
        meta_rows=te_meta,
        pred_ret_log=te_out["pred_ret"],
        pred_tr=te_out["pred_tr"],
        probs=te_probs,
        y_ret_log=te_out["y_ret"],
        y_tr=te_out["y_tr"],
        y_dir=te_out["y_dir"],
        conf_threshold=final_conf_thr,
        slippage_bps=slippage_bps,
        fee_pct=fee_pct,
    )
    te_zero_fee_full = evaluate_predictions(
        meta_rows=te_meta,
        pred_ret_log=te_out["pred_ret"],
        pred_tr=te_out["pred_tr"],
        probs=te_probs,
        y_ret_log=te_out["y_ret"],
        y_tr=te_out["y_tr"],
        y_dir=te_out["y_dir"],
        conf_threshold=final_conf_thr,
        slippage_bps=slippage_bps,
        fee_pct=0.0,
    )

    te_primary_compact = {k: v for k, v in te_primary_full.items() if k != "per_ticker"}
    te_zero_fee_compact = {k: v for k, v in te_zero_fee_full.items() if k != "per_ticker"}

    prev_quant_sharpe, prev_laya1_sharpe = _read_previous_baselines(interval, artifacts_dir)
    improved_vs_prev = bool(float(te_primary_compact["sharpe_step"]) > prev_quant_sharpe)

    checks = {
        "sharpe_beats_prev_te": bool(float(te_primary_compact["sharpe_step"]) > prev_quant_sharpe),
        "ev_non_negative": bool(float(te_primary_compact["ev_net_bps_per_active_step"]) >= 0.0),
        "coverage_ge_min": bool(float(te_primary_compact["coverage"]) >= float(cfg.min_coverage)),
        "not_cache_only": bool(not data_meta["cache_only"]),
    }
    passed_all = all(checks.values())
    production_enabled = bool(force_enable or passed_all)
    if force_enable and not passed_all:
        gate_reason = f"force_enabled_override (failed={ [k for k, v in checks.items() if not v] })"
    elif passed_all:
        gate_reason = "passed_all_oos_gates"
    else:
        failed = [k for k, v in checks.items() if not v]
        gate_reason = f"blocked_by_gate: {', '.join(failed)}"

    latency_p50 = measure_inference_latency_ms(
        final_model, seq_len=cfg.seq_len, n_features=len(CANDLE_V2_FEATURES)
    )

    compact_data_meta = {
        "requested_source": data_meta["requested_source"],
        "period": data_meta["period"],
        "tickers_count": len(data_meta["tickers_loaded"]),
        "tickers_loaded": data_meta["tickers_loaded"],
        "tickers_skipped": data_meta["tickers_skipped"],
        "start_utc": data_meta["start_utc"],
        "end_utc": data_meta["end_utc"],
        "dataset_sha256": data_meta["dataset_sha256"],
        "cache_only": data_meta["cache_only"],
    }

    extra_manifest = {
        "temperature": final_temp,
        "conf_threshold": final_conf_thr,
        "final_best_epoch": final_res["best_epoch"],
        "data": compact_data_meta,
        "splits": plan.summary_dict(),
        "walk_forward": fold_reports,
        "walk_forward_aggregate": wf_agg_compact,
        "holdout_te": {
            "primary_fee_001": te_primary_compact,
            "paper_fee_000": te_zero_fee_compact,
        },
        "baseline_comparison": {
            "prev_quant_v2_te_sharpe": round(prev_quant_sharpe, 4),
            "prev_laya1_te_sharpe": round(prev_laya1_sharpe, 4),
            "buy_hold_te_sharpe": te_primary_compact["buy_hold_sharpe_step"],
            "improved_vs_prev": improved_vs_prev,
        },
        "production_enabled": production_enabled,
        "gate_reason": gate_reason,
        "inference_latency_ms_p50": latency_p50,
    }

    manifest = save_candle_v2_artifacts(
        model=final_model,
        scaler=final_scaler,
        cfg=cfg,
        out_dir=artifacts_dir,
        interval=interval,
        extra_manifest=extra_manifest,
    )

    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    train_report = {
        "interval": interval,
        "manifest_entry": manifest["models"][interval],
        "data_source_per_ticker": data_meta["source_per_ticker"],
        "data_rows_per_ticker": data_meta["rows_per_ticker"],
        "holdout_te_per_ticker_fee_001": te_primary_full["per_ticker"],
        "holdout_te_per_ticker_fee_000": te_zero_fee_full["per_ticker"],
    }
    report_path = reports_dir / f"candle_v2_train_{interval}.json"
    report_path.write_text(json.dumps(train_report, indent=2), encoding="utf-8")

    return manifest["models"][interval]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Entrenamiento Walk-Forward Candle Predictor v2")
    parser.add_argument("--interval", choices=["1d", "1h", "both"], required=True)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--universe", choices=["all"], default=None)
    parser.add_argument("--source", choices=["auto", "yfinance", "cache"], default="auto")
    parser.add_argument("--period", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=None)
    parser.add_argument("--te-frac", type=float, default=None)
    parser.add_argument("--threshold-bps", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--cell", choices=["gru", "lstm"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--force-enable", action="store_true")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.pilot:
        tickers = list(DEFAULT_PILOT_TICKERS)
    elif args.universe == "all":
        tickers = default_universe_tickers()
    else:
        tickers = default_universe_tickers()

    if args.artifacts_dir:
        out_dir = Path(args.artifacts_dir)
    elif args.pilot:
        out_dir = ARTIFACTS_DIR / "pilot"
    else:
        out_dir = ARTIFACTS_DIR

    overrides = {
        "n_folds": args.folds,
        "embargo_bars": args.embargo,
        "te_frac": args.te_frac,
        "threshold_bps": args.threshold_bps,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "cell": args.cell,
        "max_epochs": args.epochs,
        "seed": args.seed,
    }

    intervals = ["1d", "1h"] if args.interval == "both" else [args.interval]
    results: Dict[str, Any] = {}
    for tf in intervals:
        print(f"\n=== Entrenando Candle Predictor v2 [{tf}] en {out_dir} ({len(tickers)} tickers solicitados) ===")
        entry = train_interval(
            interval=tf,
            tickers=tickers,
            artifacts_dir=out_dir,
            source=args.source,
            period=args.period,
            refresh=args.refresh,
            fee_pct=args.fee_pct,
            slippage_bps=args.slippage_bps,
            force_enable=args.force_enable,
            cfg_overrides=overrides,
        )
        results[tf] = {
            "production_enabled": entry["production_enabled"],
            "gate_reason": entry["gate_reason"],
            "temperature": entry["temperature"],
            "conf_threshold": entry["conf_threshold"],
            "inference_latency_ms_p50": entry["inference_latency_ms_p50"],
            "walk_forward_aggregate": entry["walk_forward_aggregate"],
            "holdout_te_primary": entry["holdout_te"]["primary_fee_001"],
            "baseline_comparison": entry["baseline_comparison"],
        }
        print(json.dumps(results[tf], indent=2))


if __name__ == "__main__":
    main()
