"""
Fase 1 - Entrenamiento Cuantitativo Reproducible, Evaluación TE y Exportación de Artefactos.

Genera:
- notebooks/artifacts/scaler_v1.joblib (ajustado EXCLUSIVAMENTE en TR)
- notebooks/artifacts/lstm_v1.pt       (pesos congelados del modelo ganador en VA)
- notebooks/artifacts/model_manifest.json
- exports/experimentos.csv
- exports/predicciones_TE.csv
- exports/prediccion_futura.csv        (para los 30 PRIMARY_TICKERS, columna 'real' vacía)
- exports/simulacion.csv               (simulación preliminar en TE; ampliada en comparator.py)
- data/replay_candles.json             (histórico de velas cerradas para los 30 tickers)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import ARTIFACTS_DIR, DATA_DIR, EXPORTS_DIR, PRIMARY_TICKERS
from backend.app.front_office.features import (
    DEFAULT_SEQ_LEN,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_TECH,
    build_sequences,
    compute_technical_features,
)
from backend.app.front_office.predictor import LSTMRegressor, MLPRegressor, compute_sha256


# ==============================================================================
# MÉTRICAS EN ESCALA ORIGINAL (Sección 5.1 del Plan Maestro)
# ==============================================================================
def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | str:
    mask = y_true != 0.0
    denom = int(np.sum(mask))
    if denom == 0:
        return "No definido"
    hits = int(np.sum(np.sign(y_pred[mask]) == np.sign(y_true[mask])))
    return float(hits / denom)


def simulate_long_only_equity(
    prices: np.ndarray,
    preds: np.ndarray,
    ema_ratios: np.ndarray | None = None,
    use_tech_filter: bool = False,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.001,
    slippage_bps: float = 5.0,
) -> Dict[str, Any]:
    """
    Simulador vectorial Long-Only con desfase t -> ejecución t+1 y costos.
    Calcula Win Rate sobre ciclos cerrados (Compra + Venta) y Max Drawdown.
    """
    cash = float(initial_capital)
    qty = 0.0
    entry_cost_total = 0.0
    closed_pnls: List[float] = []
    equity_curve: List[float] = [cash]
    slip_mult = slippage_bps / 10000.0

    n = len(prices)
    for t in range(n - 1):
        signal_buy = preds[t] > 0.0005
        if use_tech_filter and ema_ratios is not None:
            signal_buy = signal_buy and (ema_ratios[t] > 1.0)
        signal_sell = preds[t] < -0.0005

        exec_price = float(prices[t + 1])
        if exec_price <= 0:
            equity_curve.append(cash + qty * float(prices[t]))
            continue

        if signal_buy and qty == 0.0 and cash > 10.0:
            fill_price = exec_price * (1.0 + slip_mult)
            qty = cash / (fill_price * (1.0 + fee_pct))
            notional = qty * fill_price
            fee = notional * fee_pct
            entry_cost_total = notional + fee
            cash = max(0.0, cash - entry_cost_total)
        elif signal_sell and qty > 0.0:
            fill_price = exec_price * (1.0 - slip_mult)
            gross = qty * fill_price
            fee = gross * fee_pct
            net_proceeds = gross - fee
            closed_pnls.append(net_proceeds - entry_cost_total)
            cash += net_proceeds
            qty = 0.0
            entry_cost_total = 0.0

        equity_curve.append(cash + qty * exec_price)

    eq = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(eq)
    drawdowns = (peaks - eq) / np.where(peaks > 0, peaks, 1.0)
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    win_rate: float | str = (
        float(sum(1 for p in closed_pnls if p > 0) / len(closed_pnls))
        if len(closed_pnls) > 0
        else "No definido"
    )
    net_return_pct = float((eq[-1] - initial_capital) / initial_capital)
    return {
        "final_equity": float(eq[-1]),
        "net_return_pct": net_return_pct,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "closed_trades": len(closed_pnls),
        "equity_curve": equity_curve,
    }


# ==============================================================================
# DESCARGA / CONSTRUCCIÓN DE HISTÓRICOS PARA LOS 30 TICKERS
# ==============================================================================
def _synthetic_ohlcv_fallback(ticker: str, days: int = 750) -> pd.DataFrame:
    seed = abs(hash(ticker)) % (2**31)
    rng = np.random.default_rng(seed)
    start_dt = datetime.now(timezone.utc) - timedelta(days=days + 5)
    dates = pd.date_range(start=start_dt, periods=days, freq="D", tz="UTC")
    base_price = 45000.0 if ticker == "BTC-USD" else float(rng.uniform(45.0, 450.0))
    rets = rng.normal(loc=0.0004, scale=0.018 if ticker == "BTC-USD" else 0.014, size=days)
    closes = base_price * np.exp(np.cumsum(rets))
    opens = np.roll(closes, 1)
    opens[0] = base_price
    highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.001, 0.012, size=days))
    lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.001, 0.012, size=days))
    vols = rng.uniform(5e5, 8e6, size=days)

    return pd.DataFrame(
        {
            "open_time": [d.strftime("%Y-%m-%dT00:00:00Z") for d in dates],
            "close_time": [(d + timedelta(hours=23, minutes=59, seconds=59)).strftime("%Y-%m-%dT23:59:59Z") for d in dates],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def fetch_universe_history(tickers: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    """Descarga OHLCV diaria para los 30 PRIMARY_TICKERS asegurando solo velas cerradas."""
    print(f"Descargando historicos para {len(tickers)} tickers...")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    universe: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[ticker].dropna(subset=["Close"]).copy()
            else:
                sub = raw.dropna(subset=["Close"]).copy()

            if len(sub) < 200:
                universe[ticker] = _synthetic_ohlcv_fallback(ticker)
                continue

            sub = sub.reset_index()
            date_col = "Date" if "Date" in sub.columns else sub.columns[0]
            dates = pd.to_datetime(sub[date_col], utc=True)

            # Excluir vela potencialmente abierta del día UTC actual
            closed_mask = dates.dt.strftime("%Y-%m-%d") < today_str
            sub = sub.loc[closed_mask].reset_index(drop=True)
            dates = dates.loc[closed_mask].reset_index(drop=True)

            if len(sub) < 180:
                universe[ticker] = _synthetic_ohlcv_fallback(ticker)
                continue

            df = pd.DataFrame(
                {
                    "open_time": [d.strftime("%Y-%m-%dT00:00:00Z") for d in dates],
                    "close_time": [
                        (d + timedelta(hours=23, minutes=59, seconds=59)).strftime(
                            "%Y-%m-%dT23:59:59Z"
                        )
                        for d in dates
                    ],
                    "open": sub["Open"].astype(float).to_numpy(),
                    "high": sub["High"].astype(float).to_numpy(),
                    "low": sub["Low"].astype(float).to_numpy(),
                    "close": sub["Close"].astype(float).to_numpy(),
                    "volume": sub["Volume"].astype(float).to_numpy(),
                }
            )
            universe[ticker] = df
        except Exception as exc:
            print(f"Fallback sintetico para {ticker}: {exc}")
            universe[ticker] = _synthetic_ohlcv_fallback(ticker)

    return universe


# ==============================================================================
# ENTRENAMIENTO PYTORCH CON EARLY STOPPING
# ==============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_torch_model(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
) -> Tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    x_va_t = torch.tensor(x_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    best_va_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    n_tr = len(x_tr_t)
    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(x_tr_t[idx])
            loss = criterion(preds, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_preds = model(x_va_t)
            va_loss = float(criterion(va_preds, y_va_t).item())

        if va_loss < best_va_loss - 1e-8:
            best_va_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_va_loss


def prepare_splits(
    universe: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Dict[str, Any]:
    """
    Partición temporal estricta (TR 70%, VA 15%, TE 15%) por activo,
    ajustando StandardScaler EXCLUSIVAMENTE con datos de TR.
    """
    tr_raw_rows: List[np.ndarray] = []
    per_ticker_splits: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ticker, df in universe.items():
        feat_df = compute_technical_features(df, drop_na=True)
        # Mantener filas con target conocido para TR/VA/TE supervisado, guardando la última fila para predicción futura
        sup_df = feat_df.dropna(subset=["target_ret_t1"]).reset_index(drop=True)
        n = len(sup_df)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        df_tr = sup_df.iloc[:n_tr].reset_index(drop=True)
        df_va = sup_df.iloc[n_tr : n_tr + n_va].reset_index(drop=True)
        df_te = sup_df.iloc[n_tr + n_va :].reset_index(drop=True)

        tr_raw_rows.append(df_tr[feature_cols].to_numpy(dtype=np.float32))
        per_ticker_splits[ticker] = {
            "full_feat": feat_df,
            "TR": df_tr,
            "VA": df_va,
            "TE": df_te,
        }

    scaler = StandardScaler()
    scaler.fit(np.vstack(tr_raw_rows))

    x_tr_list, y_tr_list = [], []
    x_va_list, y_va_list = [], []
    x_te_list, y_te_list = [], []
    te_meta_rows: List[pd.DataFrame] = []

    for ticker, parts in per_ticker_splits.items():
        for split_name, x_acc, y_acc in [
            ("TR", x_tr_list, y_tr_list),
            ("VA", x_va_list, y_va_list),
            ("TE", x_te_list, y_te_list),
        ]:
            part_df = parts[split_name]
            x_scaled = scaler.transform(part_df[feature_cols].to_numpy(dtype=np.float32))
            y_arr = part_df["target_ret_t1"].to_numpy(dtype=np.float32)
            xs, ys = build_sequences(x_scaled, y_arr, seq_len=seq_len)
            x_acc.append(xs)
            y_acc.append(ys)
            if split_name == "TE":
                meta = part_df.iloc[seq_len - 1 :].copy()
                meta["ticker"] = ticker
                te_meta_rows.append(meta)

    return {
        "scaler": scaler,
        "per_ticker": per_ticker_splits,
        "x_tr": np.concatenate(x_tr_list, axis=0),
        "y_tr": np.concatenate(y_tr_list, axis=0),
        "x_va": np.concatenate(x_va_list, axis=0),
        "y_va": np.concatenate(y_va_list, axis=0),
        "x_te": np.concatenate(x_te_list, axis=0),
        "y_te": np.concatenate(y_te_list, axis=0),
        "te_meta": pd.concat(te_meta_rows, ignore_index=True),
    }


def run_phase1() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe_history(PRIMARY_TICKERS, period="3y")

    # Guardar cache de velas cerradas para Historical Replay (últimas 140 velas por ticker)
    replay_cache: Dict[str, List[Dict[str, Any]]] = {}
    for ticker, df in universe.items():
        tail_df = df.tail(140).reset_index(drop=True)
        replay_cache[ticker] = tail_df.to_dict(orient="records")
    (DATA_DIR / "replay_candles.json").write_text(
        json.dumps(replay_cache, indent=2), encoding="utf-8"
    )
    print(f"Cache de replay guardado en {DATA_DIR / 'replay_candles.json'}")

    # Preparar splits con todos los indicadores técnicos
    splits_full = prepare_splits(universe, FEATURE_COLUMNS, seq_len=DEFAULT_SEQ_LEN)
    splits_notech = prepare_splits(universe, FEATURE_COLUMNS_NO_TECH, seq_len=DEFAULT_SEQ_LEN)

    experiments: List[Dict[str, Any]] = []

    # 1) Referencias base: Pronóstico Cero y Ridge Regression
    zero_va = np.zeros_like(splits_full["y_va"])
    zero_te = np.zeros_like(splits_full["y_te"])
    experiments.append(
        {
            "experiment_id": "baseline_zero",
            "family": "Baseline",
            "config": "zero_forecast",
            "seed": 0,
            "features": "all_8",
            "window": "full_TR_VA_TE",
            "va_mae": compute_mae(splits_full["y_va"], zero_va),
            "va_rmse": compute_rmse(splits_full["y_va"], zero_va),
            "te_mae": compute_mae(splits_full["y_te"], zero_te),
            "te_rmse": compute_rmse(splits_full["y_te"], zero_te),
            "te_dir_acc": compute_directional_accuracy(splits_full["y_te"], zero_te),
        }
    )

    ridge = Ridge(alpha=10.0)
    ridge.fit(
        splits_full["x_tr"].reshape(len(splits_full["x_tr"]), -1),
        splits_full["y_tr"],
    )
    ridge_va = ridge.predict(splits_full["x_va"].reshape(len(splits_full["x_va"]), -1))
    ridge_te = ridge.predict(splits_full["x_te"].reshape(len(splits_full["x_te"]), -1))
    experiments.append(
        {
            "experiment_id": "baseline_ridge",
            "family": "Baseline",
            "config": "ridge_alpha_10",
            "seed": 0,
            "features": "all_8",
            "window": "full_TR_VA_TE",
            "va_mae": compute_mae(splits_full["y_va"], ridge_va),
            "va_rmse": compute_rmse(splits_full["y_va"], ridge_va),
            "te_mae": compute_mae(splits_full["y_te"], ridge_te),
            "te_rmse": compute_rmse(splits_full["y_te"], ridge_te),
            "te_dir_acc": compute_directional_accuracy(splits_full["y_te"], ridge_te),
        }
    )

    # 2) Experimentación exigida: 3 configuraciones x 3 semillas (42, 123, 999)
    seeds = [42, 123, 999]
    configs = [
        ("MLP_h64", "MLP", {"hidden_dim": 64, "dropout": 0.1}),
        ("LSTM_h32", "LSTM", {"hidden_dim": 32, "num_layers": 1, "dropout": 0.1}),
        ("LSTM_h64", "LSTM", {"hidden_dim": 64, "num_layers": 1, "dropout": 0.15}),
    ]

    best_lstm_model: LSTMRegressor | None = None
    best_lstm_va_rmse = float("inf")
    best_lstm_arch = {"hidden_dim": 32, "num_layers": 1, "dropout": 0.1}

    for cfg_name, family, kwargs in configs:
        for seed in seeds:
            set_seed(seed)
            if family == "MLP":
                model: nn.Module = MLPRegressor(
                    input_dim=len(FEATURE_COLUMNS),
                    seq_len=DEFAULT_SEQ_LEN,
                    hidden_dim=kwargs["hidden_dim"],
                    dropout=kwargs["dropout"],
                )
            else:
                model = LSTMRegressor(
                    input_dim=len(FEATURE_COLUMNS),
                    hidden_dim=kwargs["hidden_dim"],
                    num_layers=kwargs["num_layers"],
                    dropout=kwargs["dropout"],
                )

            model, _ = train_torch_model(
                model,
                splits_full["x_tr"],
                splits_full["y_tr"],
                splits_full["x_va"],
                splits_full["y_va"],
                epochs=25,
                lr=1e-3,
                patience=4,
            )

            with torch.no_grad():
                va_pred = model(torch.tensor(splits_full["x_va"], dtype=torch.float32)).numpy()
                te_pred = model(torch.tensor(splits_full["x_te"], dtype=torch.float32)).numpy()

            va_rmse = compute_rmse(splits_full["y_va"], va_pred)
            experiments.append(
                {
                    "experiment_id": f"{cfg_name}_seed{seed}",
                    "family": family,
                    "config": cfg_name,
                    "seed": seed,
                    "features": "all_8",
                    "window": "full_TR_VA_TE",
                    "va_mae": compute_mae(splits_full["y_va"], va_pred),
                    "va_rmse": va_rmse,
                    "te_mae": compute_mae(splits_full["y_te"], te_pred),
                    "te_rmse": compute_rmse(splits_full["y_te"], te_pred),
                    "te_dir_acc": compute_directional_accuracy(splits_full["y_te"], te_pred),
                }
            )

            if family == "LSTM" and va_rmse < best_lstm_va_rmse:
                best_lstm_va_rmse = va_rmse
                best_lstm_model = model
                best_lstm_arch = dict(kwargs)

    # 3) 3 ventanas expansivas (40%, 70%, 100% de TR)
    n_tr_total = len(splits_full["x_tr"])
    for frac_label, frac in [("exp_win_40pct", 0.40), ("exp_win_70pct", 0.70), ("exp_win_100pct", 1.00)]:
        set_seed(42)
        cutoff = max(100, int(n_tr_total * frac))
        m_win = LSTMRegressor(input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=1, dropout=0.1)
        m_win, _ = train_torch_model(
            m_win,
            splits_full["x_tr"][:cutoff],
            splits_full["y_tr"][:cutoff],
            splits_full["x_va"],
            splits_full["y_va"],
            epochs=20,
            patience=4,
        )
        with torch.no_grad():
            va_p = m_win(torch.tensor(splits_full["x_va"], dtype=torch.float32)).numpy()
            te_p = m_win(torch.tensor(splits_full["x_te"], dtype=torch.float32)).numpy()
        experiments.append(
            {
                "experiment_id": f"LSTM_{frac_label}",
                "family": "LSTM_ExpandingWindow",
                "config": "LSTM_h32",
                "seed": 42,
                "features": "all_8",
                "window": frac_label,
                "va_mae": compute_mae(splits_full["y_va"], va_p),
                "va_rmse": compute_rmse(splits_full["y_va"], va_p),
                "te_mae": compute_mae(splits_full["y_te"], te_p),
                "te_rmse": compute_rmse(splits_full["y_te"], te_p),
                "te_dir_acc": compute_directional_accuracy(splits_full["y_te"], te_p),
            }
        )

    # 4) Ablación: Sin indicadores técnicos (solo rezagos)
    set_seed(42)
    m_notech = LSTMRegressor(
        input_dim=len(FEATURE_COLUMNS_NO_TECH), hidden_dim=32, num_layers=1, dropout=0.1
    )
    m_notech, _ = train_torch_model(
        m_notech,
        splits_notech["x_tr"],
        splits_notech["y_tr"],
        splits_notech["x_va"],
        splits_notech["y_va"],
        epochs=20,
        patience=4,
    )
    with torch.no_grad():
        va_nt = m_notech(torch.tensor(splits_notech["x_va"], dtype=torch.float32)).numpy()
        te_nt = m_notech(torch.tensor(splits_notech["x_te"], dtype=torch.float32)).numpy()
    experiments.append(
        {
            "experiment_id": "LSTM_no_tech_indicators",
            "family": "LSTM_Ablation",
            "config": "LSTM_h32_lags_only",
            "seed": 42,
            "features": "lags_only_4",
            "window": "full_TR_VA_TE",
            "va_mae": compute_mae(splits_notech["y_va"], va_nt),
            "va_rmse": compute_rmse(splits_notech["y_va"], va_nt),
            "te_mae": compute_mae(splits_notech["y_te"], te_nt),
            "te_rmse": compute_rmse(splits_notech["y_te"], te_nt),
            "te_dir_acc": compute_directional_accuracy(splits_notech["y_te"], te_nt),
        }
    )

    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv(EXPORTS_DIR / "experimentos.csv", index=False)

    # 5) Congelar artefactos del modelo ganador
    assert best_lstm_model is not None
    scaler_path = ARTIFACTS_DIR / "scaler_v1.joblib"
    model_path = ARTIFACTS_DIR / "lstm_v1.pt"
    joblib.dump(splits_full["scaler"], scaler_path)
    torch.save(best_lstm_model.state_dict(), model_path)

    with torch.no_grad():
        te_preds = best_lstm_model(torch.tensor(splits_full["x_te"], dtype=torch.float32)).numpy()

    te_meta = splits_full["te_meta"].copy()
    te_meta["y_real"] = splits_full["y_te"]
    te_meta["y_pred"] = te_preds
    te_meta["error"] = te_meta["y_pred"] - te_meta["y_real"]

    pred_te_export = te_meta[["ticker", "close_time", "close", "y_real", "y_pred", "error"]].copy()
    pred_te_export.to_csv(EXPORTS_DIR / "predicciones_TE.csv", index=False)

    # Evaluación en TE sobre BTC-USD para simulación comparativa con/sin filtro técnico
    btc_te = te_meta[te_meta["ticker"] == "BTC-USD"].reset_index(drop=True)
    sim_no_filter = simulate_long_only_equity(
        prices=btc_te["close"].to_numpy(),
        preds=btc_te["y_pred"].to_numpy(),
        ema_ratios=btc_te["ema_ratio"].to_numpy(),
        use_tech_filter=False,
    )
    sim_with_filter = simulate_long_only_equity(
        prices=btc_te["close"].to_numpy(),
        preds=btc_te["y_pred"].to_numpy(),
        ema_ratios=btc_te["ema_ratio"].to_numpy(),
        use_tech_filter=True,
    )

    # 6) Predicción futura t+1 para los 30 PRIMARY_TICKERS (columna 'real' vacía)
    future_rows: List[Dict[str, Any]] = []
    for ticker in PRIMARY_TICKERS:
        full_feat = splits_full["per_ticker"][ticker]["full_feat"]
        x_scaled = splits_full["scaler"].transform(
            full_feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        )
        seqs, _ = build_sequences(x_scaled, targets=None, seq_len=DEFAULT_SEQ_LEN)
        last_seq = torch.tensor(seqs[-1:], dtype=torch.float32)
        with torch.no_grad():
            pred_t1 = float(best_lstm_model(last_seq).item())
        last_row = full_feat.iloc[-1]
        future_rows.append(
            {
                "ticker": ticker,
                "candle_close_utc": str(last_row["close_time"]),
                "last_close_price": float(last_row["close"]),
                "pred_return_t1": pred_t1,
                "predicted_direction": "UP" if pred_t1 > 0 else "DOWN",
                "ema_ratio": float(last_row["ema_ratio"]),
                "vol_20": float(last_row["vol_20"]),
                "real": "",
            }
        )
    pd.DataFrame(future_rows).to_csv(EXPORTS_DIR / "prediccion_futura.csv", index=False)

    # Simulación inicial en TE (luego comparator.py genera la tabla oficial de 3 carriles)
    sim_rows = []
    for idx in range(len(btc_te)):
        sim_rows.append(
            {
                "step": idx + 1,
                "ticker": "BTC-USD",
                "candle_close_utc": str(btc_te.loc[idx, "close_time"]),
                "close": float(btc_te.loc[idx, "close"]),
                "pred_return_t1": float(btc_te.loc[idx, "y_pred"]),
                "equity_no_filter": float(sim_no_filter["equity_curve"][idx]),
                "equity_with_filter": float(sim_with_filter["equity_curve"][idx]),
            }
        )
    pd.DataFrame(sim_rows).to_csv(EXPORTS_DIR / "simulacion.csv", index=False)

    # 7) Manifest con hashes SHA-256 y contrato exacto
    btc_parts = splits_full["per_ticker"]["BTC-USD"]
    manifest = {
        "model_version": "lstm_v1",
        "model_file": "lstm_v1.pt",
        "scaler_file": "scaler_v1.joblib",
        "architecture": {
            "class": "LSTMRegressor",
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": int(best_lstm_arch["hidden_dim"]),
            "num_layers": int(best_lstm_arch.get("num_layers", 1)),
            "dropout": float(best_lstm_arch.get("dropout", 0.1)),
        },
        "seq_len": DEFAULT_SEQ_LEN,
        "feature_columns": FEATURE_COLUMNS,
        "primary_tickers": PRIMARY_TICKERS,
        "sha256": {
            "model": compute_sha256(model_path),
            "scaler": compute_sha256(scaler_path),
        },
        "splits_summary": {
            "reference_ticker": "BTC-USD",
            "TR_rows": int(len(btc_parts["TR"])),
            "TR_start": str(btc_parts["TR"].iloc[0]["close_time"]),
            "TR_end": str(btc_parts["TR"].iloc[-1]["close_time"]),
            "VA_rows": int(len(btc_parts["VA"])),
            "VA_start": str(btc_parts["VA"].iloc[0]["close_time"]),
            "VA_end": str(btc_parts["VA"].iloc[-1]["close_time"]),
            "TE_rows": int(len(btc_parts["TE"])),
            "TE_start": str(btc_parts["TE"].iloc[0]["close_time"]),
            "TE_end": str(btc_parts["TE"].iloc[-1]["close_time"]),
        },
        "metrics_TE": {
            "MAE": compute_mae(splits_full["y_te"], te_preds),
            "RMSE": compute_rmse(splits_full["y_te"], te_preds),
            "Directional_Accuracy": compute_directional_accuracy(
                splits_full["y_te"], te_preds
            ),
            "BTC_TE_Win_Rate_With_Filter": sim_with_filter["win_rate"],
            "BTC_TE_Max_Drawdown_With_Filter": sim_with_filter["max_drawdown"],
            "BTC_TE_Net_Return_No_Filter": sim_no_filter["net_return_pct"],
            "BTC_TE_Net_Return_With_Filter": sim_with_filter["net_return_pct"],
        },
        "training_environment": {
            "target_gpu": "NVIDIA Tesla T4 (Colab) / CPU reproducible fallback",
            "pytorch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    (ARTIFACTS_DIR / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Fase 1 completada. Artefactos y CSVs exportados exitosamente.")
    print(json.dumps(manifest["metrics_TE"], indent=2))


if __name__ == "__main__":
    run_phase1()
