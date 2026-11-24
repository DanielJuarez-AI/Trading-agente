"""
Evaluación estricta fuera de muestra (OOS / TE) para Candle Predictor v2:
- Errores de retorno logarítmico y rango esperado (MAE / RMSE)
- Precisión direccional cruda y filtrada por confianza con abstención explícita (HOLD)
- Matriz de confusión 3x3 y Expected Calibration Error (ECE)
- Métricas económicas por paso y por ticker descontando slippage y comisiones sin cruzar símbolos
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from backend.app.core.config import ARTIFACTS_DIR
from backend.app.ml.candle_predictor.config import (
    DEFAULT_PILOT_TICKERS,
    CandleV2Config,
    default_universe_tickers,
)
from backend.app.ml.candle_predictor.data import load_candles
from backend.app.ml.candle_predictor.features import (
    CANDLE_V2_FEATURES,
    CausalRobustScaler,
    MultiTickerSequenceDataset,
    build_index,
    build_ticker_frame,
)
from backend.app.ml.candle_predictor.network import (
    NextCandleGRU,
    calibrated_proba,
    expected_calibration_error,
)


def _trading_stats_from_series(
    symbols: Sequence[str],
    ts_arr: np.ndarray,
    filtered_cls: np.ndarray,
    y_ret_log: np.ndarray,
    y_dir: np.ndarray,
    active_mask: np.ndarray,
    cost_per_side: float,
) -> Dict[str, Any]:
    """Calcula Sharpe, EV neto y Buy-and-Hold sin cruzar fronteras entre tickers."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, sym in enumerate(symbols):
        groups[str(sym)].append(i)

    all_net_lo: List[np.ndarray] = []
    all_net_ls: List[np.ndarray] = []
    all_net_bh: List[np.ndarray] = []
    all_pos_lo: List[np.ndarray] = []
    total_entries_lo = 0
    total_entries_ls = 0
    per_ticker: Dict[str, Dict[str, Any]] = {}

    for sym in sorted(groups.keys()):
        idxs = groups[sym]
        idxs.sort(key=lambda j: int(ts_arr[j]))
        idx_np = np.asarray(idxs, dtype=np.int64)

        simple_r = np.exp(y_ret_log[idx_np]) - 1.0
        cls_sym = filtered_cls[idx_np]

        pos_lo = (cls_sym == 2).astype(np.float64)
        diff_lo = np.diff(pos_lo, prepend=0.0)
        net_lo = pos_lo * simple_r - np.abs(diff_lo) * cost_per_side
        entries_lo = int(np.sum(diff_lo > 0.5))

        pos_ls = np.where(cls_sym == 2, 1.0, np.where(cls_sym == 0, -1.0, 0.0))
        diff_ls = np.diff(pos_ls, prepend=0.0)
        net_ls = pos_ls * simple_r - np.abs(diff_ls) * cost_per_side
        entries_ls = int(np.sum((diff_ls != 0.0) & (pos_ls != 0.0)))

        pos_bh = np.ones_like(simple_r, dtype=np.float64)
        diff_bh = np.diff(pos_bh, prepend=0.0)
        net_bh = pos_bh * simple_r - np.abs(diff_bh) * cost_per_side

        all_net_lo.append(net_lo)
        all_net_ls.append(net_ls)
        all_net_bh.append(net_bh)
        all_pos_lo.append(pos_lo)
        total_entries_lo += entries_lo
        total_entries_ls += entries_ls

        std_lo = float(np.std(net_lo))
        std_bh = float(np.std(net_bh))
        n_act_sym = int(np.sum(pos_lo > 0.0))
        act_sym_mask = active_mask[idx_np]
        if int(np.sum(act_sym_mask)) > 0:
            sym_acc = float(np.mean(cls_sym[act_sym_mask] == y_dir[idx_np][act_sym_mask]))
        else:
            sym_acc = None

        per_ticker[sym] = {
            "samples": int(len(idx_np)),
            "coverage": round(float(np.mean(act_sym_mask)), 4),
            "active_long_steps": n_act_sym,
            "n_trades": entries_lo,
            "filtered_3class_accuracy": round(sym_acc, 4) if sym_acc is not None else None,
            "sharpe_step": round(float(np.mean(net_lo) / std_lo) if std_lo > 1e-8 else 0.0, 4),
            "ev_net_bps_per_active_step": round(
                float(np.sum(net_lo) / n_act_sym * 10000.0) if n_act_sym > 0 else 0.0, 4
            ),
            "buy_hold_sharpe_step": round(
                float(np.mean(net_bh) / std_bh) if std_bh > 1e-8 else 0.0, 4
            ),
            "total_net_return_pct": round(float((np.prod(1.0 + net_lo) - 1.0) * 100.0), 4),
        }

    net_lo_all = np.concatenate(all_net_lo) if all_net_lo else np.zeros(0, dtype=np.float64)
    net_ls_all = np.concatenate(all_net_ls) if all_net_ls else np.zeros(0, dtype=np.float64)
    net_bh_all = np.concatenate(all_net_bh) if all_net_bh else np.zeros(0, dtype=np.float64)
    pos_lo_all = np.concatenate(all_pos_lo) if all_pos_lo else np.zeros(0, dtype=np.float64)

    std_lo = float(np.std(net_lo_all)) if len(net_lo_all) else 0.0
    std_ls = float(np.std(net_ls_all)) if len(net_ls_all) else 0.0
    std_bh = float(np.std(net_bh_all)) if len(net_bh_all) else 0.0

    sharpe_lo = float(np.mean(net_lo_all) / std_lo) if std_lo > 1e-8 else 0.0
    sharpe_ls = float(np.mean(net_ls_all) / std_ls) if std_ls > 1e-8 else 0.0
    sharpe_bh = float(np.mean(net_bh_all) / std_bh) if std_bh > 1e-8 else 0.0

    active_long_steps = int(np.sum(pos_lo_all > 0.0))
    sum_net_lo = float(np.sum(net_lo_all))
    ev_active_bps = (
        float(sum_net_lo / active_long_steps * 10000.0) if active_long_steps > 0 else 0.0
    )
    ev_trade_bps = (
        float(sum_net_lo / total_entries_lo * 10000.0) if total_entries_lo > 0 else 0.0
    )

    return {
        "sharpe_step": round(sharpe_lo, 4),
        "long_short_sharpe_step": round(sharpe_ls, 4),
        "buy_hold_sharpe_step": round(sharpe_bh, 4),
        "ev_net_bps_per_active_step": round(ev_active_bps, 4),
        "ev_net_bps_per_trade": round(ev_trade_bps, 4),
        "n_trades": int(total_entries_lo),
        "n_trades_long_short": int(total_entries_ls),
        "active_long_steps": active_long_steps,
        "mean_step_net_bps": round(float(np.mean(net_lo_all) * 10000.0) if len(net_lo_all) else 0.0, 4),
        "per_ticker": per_ticker,
    }


def evaluate_predictions(
    meta_rows: Sequence[Tuple[str, int]],
    pred_ret_log: np.ndarray,
    pred_tr: np.ndarray,
    probs: np.ndarray,
    y_ret_log: np.ndarray,
    y_tr: np.ndarray,
    y_dir: np.ndarray,
    conf_threshold: float = 0.50,
    slippage_bps: float = 5.0,
    fee_pct: float = 0.001,
) -> Dict[str, Any]:
    """
    Calcula métricas estadísticas y económicas completas para un split.
    `meta_rows` contiene `(symbol, ts_ns)` alineado 1 a 1 con las predicciones.
    """
    n = len(y_ret_log)
    if n == 0:
        raise ValueError("No hay muestras para evaluar.")

    pred_r = np.asarray(pred_ret_log, dtype=np.float64)
    true_r = np.asarray(y_ret_log, dtype=np.float64)
    pred_vol = np.asarray(pred_tr, dtype=np.float64)
    true_vol = np.asarray(y_tr, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_dir_arr = np.asarray(y_dir, dtype=np.int64)

    mae_ret = float(np.mean(np.abs(pred_r - true_r)))
    rmse_ret = float(np.sqrt(np.mean((pred_r - true_r) ** 2)))
    mae_tr = float(np.mean(np.abs(pred_vol - true_vol)))
    rmse_tr = float(np.sqrt(np.mean((pred_vol - true_vol) ** 2)))

    nz_mask = true_r != 0.0
    raw_dir_acc = (
        float(np.mean(np.sign(pred_r[nz_mask]) == np.sign(true_r[nz_mask])))
        if int(np.sum(nz_mask)) > 0
        else 0.0
    )

    pred_cls = np.argmax(probs_arr, axis=-1)
    max_prob = np.max(probs_arr, axis=-1)
    active = (pred_cls != 1) & (max_prob >= float(conf_threshold))
    filtered_cls = np.where(active, pred_cls, 1)

    coverage = float(np.mean(active))
    if int(np.sum(active)) > 0:
        filt_3class_acc = float(np.mean(filtered_cls[active] == y_dir_arr[active]))
        act_nz = active & nz_mask
        if int(np.sum(act_nz)) > 0:
            pred_sign = np.where(filtered_cls[act_nz] == 2, 1.0, -1.0)
            filt_sign_acc = float(np.mean(pred_sign == np.sign(true_r[act_nz])))
        else:
            filt_sign_acc = 0.0
    else:
        filt_3class_acc = 0.0
        filt_sign_acc = 0.0

    cm = np.zeros((3, 3), dtype=np.int64)
    for t_c, p_c in zip(y_dir_arr, filtered_cls):
        if 0 <= t_c <= 2 and 0 <= p_c <= 2:
            cm[t_c, p_c] += 1

    ece = expected_calibration_error(probs_arr, y_dir_arr, bins=10)

    symbols = [r[0] for r in meta_rows]
    ts_arr = np.asarray([int(r[1]) for r in meta_rows], dtype=np.int64)
    cost_per_side = float(slippage_bps) / 10000.0 + float(fee_pct)
    trading = _trading_stats_from_series(
        symbols=symbols,
        ts_arr=ts_arr,
        filtered_cls=filtered_cls,
        y_ret_log=true_r,
        y_dir=y_dir_arr,
        active_mask=active,
        cost_per_side=cost_per_side,
    )

    return {
        "samples": int(n),
        "conf_threshold": float(conf_threshold),
        "slippage_bps": float(slippage_bps),
        "fee_pct": float(fee_pct),
        "mae_ret_log": round(mae_ret, 6),
        "rmse_ret_log": round(rmse_ret, 6),
        "mae_tr": round(mae_tr, 6),
        "rmse_tr": round(rmse_tr, 6),
        "raw_directional_accuracy": round(raw_dir_acc, 4),
        "filtered_directional_accuracy": round(filt_sign_acc, 4),
        "filtered_3class_accuracy": round(filt_3class_acc, 4),
        "coverage": round(coverage, 4),
        "ece": round(ece, 4),
        "confusion_matrix_3x3": cm.tolist(),
        **trading,
    }


def select_conf_threshold(
    meta_rows: Sequence[Tuple[str, int]],
    probs: np.ndarray,
    y_ret_log: np.ndarray,
    y_dir: np.ndarray,
    conf_grid: Sequence[float],
    min_coverage: float = 0.02,
    slippage_bps: float = 5.0,
    fee_pct: float = 0.001,
) -> float:
    """
    Selecciona el umbral de confianza exclusivamente sobre el conjunto de validación/calibración,
    maximizando el Sharpe neto OOS sujeto a una cobertura mínima `min_coverage`.
    """
    dummy_ret = np.zeros_like(y_ret_log, dtype=np.float64)
    dummy_tr = np.zeros_like(y_ret_log, dtype=np.float64)

    candidates: List[Tuple[float, float, float, float]] = []
    fallback_cands: List[Tuple[float, float, float]] = []
    for thr in conf_grid:
        rep = evaluate_predictions(
            meta_rows=meta_rows,
            pred_ret_log=dummy_ret,
            pred_tr=dummy_tr,
            probs=probs,
            y_ret_log=y_ret_log,
            y_tr=dummy_tr,
            y_dir=y_dir,
            conf_threshold=float(thr),
            slippage_bps=slippage_bps,
            fee_pct=fee_pct,
        )
        cov = float(rep["coverage"])
        sharpe = float(rep["sharpe_step"])
        ev = float(rep["ev_net_bps_per_active_step"])
        fallback_cands.append((cov, sharpe, float(thr)))
        if cov >= float(min_coverage) and int(rep["active_long_steps"]) > 0:
            candidates.append((sharpe, ev, -abs(float(thr) - 0.50), float(thr)))

    if candidates:
        candidates.sort(reverse=True)
        return float(candidates[0][3])

    fallback_cands.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    return float(fallback_cands[0][2]) if fallback_cands else float(conf_grid[0])


def run_cli_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = artifacts_dir / "candle_v2_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró candle_v2_manifest.json en {artifacts_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models_map = manifest.get("models", {})
    if args.interval not in models_map:
        raise KeyError(f"Intervalo '{args.interval}' no encontrado en {manifest_path}")

    entry = models_map[args.interval]
    from backend.app.ml.candle_predictor.config import sha256_file

    weights_path = artifacts_dir / entry["weights_file"]
    scaler_path = artifacts_dir / entry["scaler_file"]
    expected_hashes = entry.get("sha256", {})
    if sha256_file(weights_path) != expected_hashes.get("weights"):
        raise RuntimeError(f"SHA-256 inválido para {weights_path.name}")
    if sha256_file(scaler_path) != expected_hashes.get("scaler"):
        raise RuntimeError(f"SHA-256 inválido para {scaler_path.name}")

    cfg = CandleV2Config.from_dict(entry["config"])
    scaler = CausalRobustScaler.from_dict(json.loads(scaler_path.read_text(encoding="utf-8")))

    model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if args.universe == "all" or not args.tickers:
        tickers = default_universe_tickers()
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    candles_map, data_meta = load_candles(
        tickers=tickers,
        interval=args.interval,
        source=args.source,
        seq_len=cfg.seq_len,
    )
    frames = [
        build_ticker_frame(
            df=df,
            symbol=sym,
            interval=args.interval,
            threshold_bps=cfg.threshold_bps,
        )
        for sym, df in candles_map.items()
    ]

    te_start_str = str(entry.get("splits", {}).get("te_start_utc", ""))
    if not te_start_str:
        raise RuntimeError("El manifiesto no contiene splits.te_start_utc.")
    te_start_ns = int(pd.Timestamp(te_start_str).tz_convert("UTC").value)

    te_index = build_index(
        frames=frames,
        seq_len=cfg.seq_len,
        ts_lo=te_start_ns,
        ts_hi=None,
        require_target=True,
    )
    te_index.sort(key=lambda pair: (int(frames[pair[0]].ts[pair[1]]), pair[0], pair[1]))
    ds = MultiTickerSequenceDataset(
        frames=frames,
        scaler=scaler,
        seq_len=cfg.seq_len,
        index=te_index,
        require_target=True,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    preds_ret, preds_tr, all_logits, ys_ret, ys_tr, ys_dir = [], [], [], [], [], []
    with torch.no_grad():
        for xb, yb_r, yb_t, yb_d in loader:
            out = model(xb)
            preds_ret.append(out["ret"].numpy() * scaler.ret_scale)
            preds_tr.append(out["tr"].numpy() * scaler.tr_scale)
            all_logits.append(out["logits"].numpy())
            ys_ret.append(yb_r.numpy())
            ys_tr.append(yb_t.numpy())
            ys_dir.append(yb_d.numpy())

    pred_ret_arr = np.concatenate(preds_ret)
    pred_tr_arr = np.concatenate(preds_tr)
    logits_arr = np.vstack(all_logits)
    y_ret_arr = np.concatenate(ys_ret)
    y_tr_arr = np.concatenate(ys_tr)
    y_dir_arr = np.concatenate(ys_dir)

    probs = calibrated_proba(logits_arr, temperature=float(entry.get("temperature", 1.0)))
    meta_rows = list(zip(ds.symbols, ds.end_ts.tolist()))
    conf_thr = float(entry.get("conf_threshold", 0.50))

    primary_eval = evaluate_predictions(
        meta_rows=meta_rows,
        pred_ret_log=pred_ret_arr,
        pred_tr=pred_tr_arr,
        probs=probs,
        y_ret_log=y_ret_arr,
        y_tr=y_tr_arr,
        y_dir=y_dir_arr,
        conf_threshold=conf_thr,
        slippage_bps=args.slippage_bps,
        fee_pct=args.fee_pct,
    )
    zero_fee_eval = evaluate_predictions(
        meta_rows=meta_rows,
        pred_ret_log=pred_ret_arr,
        pred_tr=pred_tr_arr,
        probs=probs,
        y_ret_log=y_ret_arr,
        y_tr=y_tr_arr,
        y_dir=y_dir_arr,
        conf_threshold=conf_thr,
        slippage_bps=args.slippage_bps,
        fee_pct=0.0,
    )

    report = {
        "interval": args.interval,
        "te_start_utc": te_start_str,
        "conf_threshold": conf_thr,
        "temperature": float(entry.get("temperature", 1.0)),
        "dataset_meta": {
            "tickers_loaded": len(data_meta["tickers_loaded"]),
            "cache_only": data_meta["cache_only"],
            "dataset_sha256": data_meta["dataset_sha256"],
        },
        "te_primary_fee_001": primary_eval,
        "te_paper_fee_000": {
            k: v for k, v in zero_fee_eval.items() if k != "per_ticker"
        },
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / f"candle_v2_eval_{args.interval}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {k: v for k, v in primary_eval.items() if k != "per_ticker"}
    print(f"[Evaluator {args.interval}] TE guardado en {out_file}")
    print(json.dumps({"primary_fee_001": summary, "paper_fee_000": report["te_paper_fee_000"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluador OOS/TE para Candle Predictor v2")
    parser.add_argument("--interval", choices=["1d", "1h"], required=True)
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--tickers", default="")
    parser.add_argument("--universe", choices=["all", "custom"], default="all")
    parser.add_argument("--source", choices=["auto", "yfinance", "cache"], default="auto")
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()
    if args.tickers:
        args.universe = "custom"
    elif args.universe == "all" and not args.tickers:
        args.tickers = ",".join(default_universe_tickers())
    elif not args.tickers:
        args.tickers = ",".join(DEFAULT_PILOT_TICKERS)
    run_cli_evaluation(args)


if __name__ == "__main__":
    main()
