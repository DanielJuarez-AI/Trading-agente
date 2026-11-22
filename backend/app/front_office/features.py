"""
Front Office - Feature Engineering Módulo Determinista (Sin Data Leakage).

Calcula exactamente los mismos indicadores y rezagos utilizados en el entrenamiento
de Fase 1 (Colab T4 / generate_phase1_artifacts.py) más features causales normalizadas
por ATR para modelos multi-activo y multi-temporalidad (Quant v2 XGBoost 1h/1d).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_5",
    "ema_ratio",
    "vol_20",
    "rsi_14",
    "momentum_5",
]

FEATURE_COLUMNS_NO_TECH: List[str] = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_5",
]

ENHANCED_FEATURE_COLUMNS: List[str] = [
    "ret_atr_1",
    "ret_atr_3",
    "ret_atr_5",
    "ema_ratio",
    "ema_10_20_atr",
    "atr_pct",
    "vol_20",
    "vol_ratio",
    "rsi_norm",
    "macd_hist",
    "bb_pos",
    "volume_z",
]

DEFAULT_SEQ_LEN: int = 10


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI causal calculado con EWM (Wilder smoothing) sin fuga futura."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Cuando avg_loss es 0 y avg_gain > 0 -> RSI = 100; ambos 0 -> RSI = 50
    rsi = rsi.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return rsi


def compute_technical_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Calcula las features causales sobre un DataFrame ordenado cronológicamente
    que contiene al menos la columna 'close'.

    No aplica backfill (.bfill) ni promedios centrados para evitar cualquier
    data leakage entre filas futuras y pasadas.
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame de velas debe contener la columna 'close'.")

    out = df.copy()
    close = out["close"].astype(float)
    ret_1 = close.pct_change()

    # Rezagos de rendimiento conocidos al cierre de la vela t
    out["ret_lag_1"] = ret_1
    out["ret_lag_2"] = ret_1.shift(1)
    out["ret_lag_3"] = ret_1.shift(2)
    out["ret_lag_5"] = ret_1.shift(4)

    # Indicadores técnicos causales base (contrato Fase 1)
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    out["ema_ratio"] = ema_10 / ema_20.replace(0.0, np.nan)
    out["vol_20"] = ret_1.rolling(window=20, min_periods=20).std()
    out["rsi_14"] = _compute_rsi(close, period=14)
    out["momentum_5"] = close / close.shift(5) - 1.0

    # ATR-14 Wilder causal (con fallback a abs diff si faltan high/low)
    prev_close = close.shift(1)
    if "high" in out.columns and "low" in out.columns:
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
    else:
        tr = (close - prev_close).abs().fillna(0.0)

    atr_14 = tr.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
    min_atr = close * 1e-4
    atr_14 = np.maximum(atr_14, min_atr)
    out["atr_14"] = atr_14
    out["atr_pct"] = (atr_14 / close.replace(0.0, np.nan)).fillna(0.015)
    safe_atr_pct = out["atr_pct"].replace(0.0, 1e-4)

    # Features causales normalizadas por ATR (invariantes a escala y temporalidad)
    out["ret_atr_1"] = np.clip((ret_1 / safe_atr_pct).fillna(0.0), -5.0, 5.0)
    out["ret_atr_3"] = np.clip(((close / close.shift(3) - 1.0) / safe_atr_pct).fillna(0.0), -6.0, 6.0)
    out["ret_atr_5"] = np.clip((out["momentum_5"] / safe_atr_pct).fillna(0.0), -8.0, 8.0)
    out["ema_10_20_atr"] = np.clip(((out["ema_ratio"] - 1.0) / safe_atr_pct).fillna(0.0), -5.0, 5.0)
    out["rsi_norm"] = ((out["rsi_14"] - 50.0) / 50.0).fillna(0.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = ((macd_line - macd_sig) / close.replace(0.0, np.nan)).fillna(0.0)

    sma_20 = close.rolling(window=20, min_periods=5).mean()
    std_20 = close.rolling(window=20, min_periods=5).std().replace(0.0, np.nan)
    out["bb_pos"] = np.clip(((close - sma_20) / (2.0 * std_20)).fillna(0.0), -2.0, 2.0)

    vol_mean_20 = out["vol_20"].rolling(window=20, min_periods=5).mean().replace(0.0, np.nan)
    out["vol_ratio"] = np.clip((out["vol_20"] / vol_mean_20).fillna(1.0), 0.2, 4.0)

    if "volume" in out.columns:
        vol_s = out["volume"].astype(float)
        v_mean = vol_s.rolling(window=20, min_periods=5).mean()
        v_std = vol_s.rolling(window=20, min_periods=5).std().replace(0.0, np.nan)
        out["volume_z"] = np.clip(((vol_s - v_mean) / v_std).fillna(0.0), -4.0, 4.0)
    else:
        out["volume_z"] = 0.0

    # Target supervisado (solo usado en entrenamiento/evaluación; en vivo el último valor es NaN)
    out["target_ret_t1"] = ret_1.shift(-1)

    if drop_na:
        out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return out


def build_sequences(
    feature_matrix: np.ndarray,
    targets: np.ndarray | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Construye ventanas temporales causales de longitud `seq_len` para MLP/LSTM.
    Cada ventana i contiene filas [i - seq_len + 1 .. i] (solo pasado y presente t).
    """
    if len(feature_matrix) < seq_len:
        raise ValueError(
            f"Se requieren al menos {seq_len} filas válidas tras el warmup, "
            f"pero solo hay {len(feature_matrix)}."
        )

    xs = []
    ys = []
    for end_idx in range(seq_len - 1, len(feature_matrix)):
        start_idx = end_idx - seq_len + 1
        xs.append(feature_matrix[start_idx : end_idx + 1])
        if targets is not None:
            ys.append(targets[end_idx])

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32) if targets is not None else None
    return x_arr, y_arr
