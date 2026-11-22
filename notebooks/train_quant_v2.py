"""
Quant v2 - Entrenamiento Multi-Temporalidad (1h y 1d) con Validación Walk-Forward Purgada.

Características:
- Features causales normalizadas por ATR (`ENHANCED_FEATURE_COLUMNS`).
- Target adimensional normalizado por volatilidad: `clip(target_ret_t1 / atr_pct, -4.0, 4.0)`,
  permitiendo compartir señal consistente entre activos de distinta volatilidad (KO vs BTC-USD).
- Validación Cruzada Walk-Forward Expansiva de 4 pliegues con purge gap.
- Exporta:
  - `notebooks/artifacts/quant_xgb_1d.joblib`
  - `notebooks/artifacts/quant_xgb_1h.joblib`
  - `notebooks/artifacts/quant_v2_manifest.json`
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import ARTIFACTS_DIR, DATA_DIR, PRIMARY_TICKERS
from backend.app.front_office.features import (
    ENHANCED_FEATURE_COLUMNS,
    compute_technical_features,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_panel(cache_path: Path) -> Dict[str, pd.DataFrame]:
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    panel: Dict[str, pd.DataFrame] = {}
    for ticker in PRIMARY_TICKERS:
        if ticker not in raw:
            continue
        df = pd.DataFrame(raw[ticker])
        if df.empty or len(df) < 55:
            continue
        feat = compute_technical_features(df, drop_na=True)
        sup = feat.dropna(subset=["target_ret_t1"]).reset_index(drop=True)
        if len(sup) < 30:
            continue
        atr_pct = np.maximum(sup["atr_pct"].to_numpy(dtype=np.float32), 1e-4)
        sup["target_atr_z"] = np.clip(
            sup["target_ret_t1"].to_numpy(dtype=np.float32) / atr_pct, -4.0, 4.0
        )
        panel[ticker] = sup
    return panel


def _make_xgb(seed: int = 42, n_estimators: int = 160) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=25,
        random_state=seed,
        n_jobs=4,
    )


def _eval_predictions(
    y_true_ret: np.ndarray, y_pred_ret: np.ndarray
) -> Dict[str, Any]:
    mae = float(np.mean(np.abs(y_pred_ret - y_true_ret)))
    rmse = float(np.sqrt(np.mean((y_pred_ret - y_true_ret) ** 2)))
    mask = y_true_ret != 0.0
    if int(np.sum(mask)) > 0:
        dir_acc = float(np.mean(np.sign(y_pred_ret[mask]) == np.sign(y_true_ret[mask])))
    else:
        dir_acc = 0.0

    # Retorno largo-efectivo simple tomando señal positiva con 5 bps de fricción
    pos = (y_pred_ret > 0.0).astype(float)
    strat_ret = pos * y_true_ret - np.abs(np.diff(pos, prepend=0.0)) * 0.0005
    std_s = float(np.std(strat_ret))
    sharpe = float(np.mean(strat_ret) / std_s) if std_s > 1e-8 else 0.0
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(dir_acc, 4),
        "sharpe_step": round(sharpe, 4),
    }


def train_timeframe(interval: str, cache_path: Path, output_path: Path) -> Dict[str, Any]:
    panel = _prepare_panel(cache_path)
    n_folds = 4
    purge_gap = 2
    fold_reports: List[Dict[str, Any]] = []
    oos_true: List[np.ndarray] = []
    oos_pred: List[np.ndarray] = []

    for fold in range(1, n_folds + 1):
        tr_x_parts, tr_y_parts = [], []
        va_x_parts, va_y_parts, va_ret_parts, va_atr_parts = [], [], [], []

        tr_end_frac = 0.40 + (fold - 1) * 0.13
        va_end_frac = min(0.95, tr_end_frac + 0.13)

        for _, df in panel.items():
            n = len(df)
            tr_end = max(12, int(n * tr_end_frac))
            va_start = min(n - 3, tr_end + purge_gap)
            va_end = min(n, int(n * va_end_frac))
            if va_end <= va_start:
                continue

            tr_df = df.iloc[:tr_end]
            va_df = df.iloc[va_start:va_end]

            tr_x_parts.append(tr_df[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            tr_y_parts.append(tr_df["target_atr_z"].to_numpy(dtype=np.float32))
            va_x_parts.append(va_df[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            va_y_parts.append(va_df["target_atr_z"].to_numpy(dtype=np.float32))
            va_ret_parts.append(va_df["target_ret_t1"].to_numpy(dtype=np.float32))
            va_atr_parts.append(va_df["atr_pct"].to_numpy(dtype=np.float32))

        x_tr = np.vstack(tr_x_parts)
        y_tr = np.concatenate(tr_y_parts)
        x_va = np.vstack(va_x_parts)
        y_va_ret = np.concatenate(va_ret_parts)
        va_atr = np.concatenate(va_atr_parts)

        model = _make_xgb(seed=42 + fold, n_estimators=160)
        model.fit(x_tr, y_tr)
        pred_z = model.predict(x_va)
        pred_ret = pred_z * va_atr

        metrics = _eval_predictions(y_va_ret, pred_ret)
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(x_tr)),
                "val_rows": int(len(x_va)),
                **metrics,
            }
        )
        oos_true.append(y_va_ret)
        oos_pred.append(pred_ret)

    agg_metrics = _eval_predictions(np.concatenate(oos_true), np.concatenate(oos_pred))

    # Entrenamiento final hasta el 85% más reciente (dejando el 15% TE intacto o para calibración temporal)
    final_x, final_y = [], []
    te_x, te_ret, te_atr = [], [], []
    for _, df in panel.items():
        n = len(df)
        cut = max(20, int(n * 0.85))
        tr_df = df.iloc[:cut]
        te_df = df.iloc[cut:]
        final_x.append(tr_df[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        final_y.append(tr_df["target_atr_z"].to_numpy(dtype=np.float32))
        if len(te_df) > 0:
            te_x.append(te_df[ENHANCED_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            te_ret.append(te_df["target_ret_t1"].to_numpy(dtype=np.float32))
            te_atr.append(te_df["atr_pct"].to_numpy(dtype=np.float32))

    final_model = _make_xgb(seed=42, n_estimators=180)
    final_model.fit(np.vstack(final_x), np.concatenate(final_y))

    te_pred_ret = final_model.predict(np.vstack(te_x)) * np.concatenate(te_atr)
    te_metrics = _eval_predictions(np.concatenate(te_ret), te_pred_ret)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_path)
    return {
        "interval": interval,
        "model_file": output_path.name,
        "sha256": _sha256_file(output_path),
        "tickers_count": len(panel),
        "walk_forward_folds": fold_reports,
        "walk_forward_aggregate_oos": agg_metrics,
        "holdout_te_metrics": te_metrics,
    }


def main() -> None:
    report_1d = train_timeframe(
        interval="1d",
        cache_path=DATA_DIR / "replay_candles.json",
        output_path=ARTIFACTS_DIR / "quant_xgb_1d.joblib",
    )
    report_1h = train_timeframe(
        interval="1h",
        cache_path=DATA_DIR / "replay_candles_1h.json",
        output_path=ARTIFACTS_DIR / "quant_xgb_1h.joblib",
    )

    manifest = {
        "model_version": "quant_v2_xgb_walkforward",
        "feature_columns": ENHANCED_FEATURE_COLUMNS,
        "target_definition": "clip(target_ret_t1 / atr_pct, -4.0, 4.0)",
        "models": {
            "1d": report_1d,
            "1h": report_1h,
        },
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = ARTIFACTS_DIR / "quant_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Quant v2 Walk-Forward entrenamiento completo:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
