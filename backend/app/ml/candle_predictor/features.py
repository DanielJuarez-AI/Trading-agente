"""
Pipeline causal de ingeniería de características adimensionales y Dataset multi-ticker
sin fugas entre símbolos ni look-ahead bias para Candle Predictor v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.app.front_office.features import _compute_rsi
from backend.app.ml.candle_predictor.config import canonical_symbol, is_crypto

EPS: float = 1e-12

CANDLE_V2_FEATURES: List[str] = [
    "body_rel",
    "upper_wick_rel",
    "lower_wick_rel",
    "logret_1",
    "logret_2",
    "logret_3",
    "logret_5",
    "logret_10",
    "logret_20",
    "rv_10",
    "rv_20",
    "rv_60",
    "atr_pct",
    "log_rel_volume",
    "has_volume",
    "dist_ema20",
    "dist_ema50",
    "ema20_slope",
    "bb_width",
    "rsi_01",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "session_flag",
    "is_crypto",
]

TARGET_COLUMNS: Tuple[str, ...] = ("target_logret", "target_tr", "target_dir")


def _extract_timestamps(df: pd.DataFrame) -> pd.Series:
    """Extrae timestamps UTC a partir de open_time, close_time o timestamp."""
    for col in ("open_time", "close_time", "timestamp"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def compute_candle_v2_features(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "1d",
    threshold_bps: float = 25.0,
) -> pd.DataFrame:
    """
    Calcula 26 características estrictamente causales, estacionarias y adimensionales
    (agnósticas al nivel de precio nominal) más targets (t+1) solo para entrenamiento/evaluación.

    No elimina filas por NaN de warmup: expone la columna booleana `valid_mask`.
    No utiliza bfill, ventanas centradas ni estadísticos globales sobre el dataset.
    """
    if "close" not in df.columns:
        raise ValueError("El DataFrame OHLCV debe contener al menos la columna 'close'.")

    out = df.copy()
    ts = _extract_timestamps(out)
    has_valid_ts = bool(ts.notna().any())
    if has_valid_ts and not ts.is_monotonic_increasing:
        out = out.assign(_ts_sort=ts).sort_values("_ts_sort", kind="mergesort").reset_index(drop=True)
        ts = out.pop("_ts_sort")
    else:
        out = out.reset_index(drop=True)
        ts = ts.reset_index(drop=True)

    close = out["close"].astype(float)
    close_safe = close.where(close > 0.0, np.nan)
    open_p = out["open"].astype(float) if "open" in out.columns else close.copy()
    high_p = out["high"].astype(float) if "high" in out.columns else close.copy()
    low_p = out["low"].astype(float) if "low" in out.columns else close.copy()

    # 1-3: Anatomía de vela relativa al rango (H - L + eps)
    hl_range = (high_p - low_p).clip(lower=0.0) + EPS
    body_top = np.maximum(open_p, close)
    body_bot = np.minimum(open_p, close)
    out["body_rel"] = (close - open_p) / hl_range
    out["upper_wick_rel"] = (high_p - body_top) / hl_range
    out["lower_wick_rel"] = (body_bot - low_p) / hl_range

    # 4-9: Dinámica de retornos logarítmicos acumulados a horizonte k
    for lag in (1, 2, 3, 5, 10, 20):
        prev = close_safe.shift(lag)
        ratio = close_safe / prev
        out[f"logret_{lag}"] = np.log(ratio.where(ratio > 0.0, np.nan))

    # 10-12: Volatilidad realizada en ventanas de 10, 20 y 60 velas
    r1 = out["logret_1"]
    out["rv_10"] = r1.rolling(window=10, min_periods=10).std()
    out["rv_20"] = r1.rolling(window=20, min_periods=20).std()
    out["rv_60"] = r1.rolling(window=60, min_periods=60).std()

    # 13: ATR-14 Wilder normalizado por Close_t
    prev_close = close_safe.shift(1)
    tr = pd.concat(
        [
            (high_p - low_p).abs(),
            (high_p - prev_close).abs(),
            (low_p - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    out["atr_pct"] = atr_14 / close_safe

    # 14-15: Volumen relativo seguro frente a activos sin volumen
    if "volume" in out.columns:
        vol = out["volume"].astype(float).clip(lower=0.0)
        sma_vol = vol.rolling(window=20, min_periods=20).mean()
        valid_vol = sma_vol.notna() & (sma_vol > 0.0)
        rel_vol = np.where(valid_vol, np.clip(vol / sma_vol.where(valid_vol, 1.0), 0.05, 20.0), 1.0)
        out["log_rel_volume"] = np.where(valid_vol, np.log(rel_vol), 0.0)
        out["has_volume"] = np.where(valid_vol, 1.0, 0.0)
    else:
        out["log_rel_volume"] = 0.0
        out["has_volume"] = 0.0

    # 16-20: Contexto de tendencia, ancho de Bollinger y RSI-14 Wilder [0, 1]
    ema_20 = close_safe.ewm(span=20, min_periods=20, adjust=False).mean()
    ema_50 = close_safe.ewm(span=50, min_periods=50, adjust=False).mean()
    out["dist_ema20"] = (close_safe / ema_20) - 1.0
    out["dist_ema50"] = (close_safe / ema_50) - 1.0

    ema20_prev5 = ema_20.shift(5)
    raw_slope = ((ema_20 / ema20_prev5) - 1.0) / (out["atr_pct"] + EPS)
    out["ema20_slope"] = np.where(raw_slope.notna(), np.clip(raw_slope, -10.0, 10.0), np.nan)

    sma_20 = close_safe.rolling(window=20, min_periods=20).mean()
    std_20 = close_safe.rolling(window=20, min_periods=20).std()
    out["bb_width"] = (4.0 * std_20) / sma_20

    out["rsi_01"] = _compute_rsi(close_safe, period=14) / 100.0

    # 21-26: Codificación temporal cíclica y régimen de sesión / clase de activo
    crypto_flag = 1.0 if is_crypto(symbol) else 0.0
    tf = str(interval).strip().lower()

    if has_valid_ts:
        valid_ts = ts.notna()
        hour = ts.dt.hour.fillna(0).astype(float)
        dow = ts.dt.day_of_week.fillna(0).astype(float)
        out["hour_sin"] = np.where(valid_ts, np.sin(2.0 * np.pi * hour / 24.0), 0.0)
        out["hour_cos"] = np.where(valid_ts, np.cos(2.0 * np.pi * hour / 24.0), 0.0)
        out["dow_sin"] = np.where(valid_ts, np.sin(2.0 * np.pi * dow / 7.0), 0.0)
        out["dow_cos"] = np.where(valid_ts, np.cos(2.0 * np.pi * dow / 7.0), 0.0)
        if crypto_flag > 0.5 or tf == "1d":
            out["session_flag"] = 1.0
        else:
            out["session_flag"] = np.where(valid_ts & (hour >= 13.0) & (hour <= 20.0), 1.0, 0.0)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 0.0
        out["dow_sin"] = 0.0
        out["dow_cos"] = 0.0
        out["session_flag"] = 1.0 if (crypto_flag > 0.5 or tf == "1d") else 0.0

    out["is_crypto"] = crypto_flag

    # Targets estacionarios y agnósticos al precio nominal (t+1; NaN en la última fila)
    next_close = close_safe.shift(-1)
    next_high = high_p.shift(-1)
    next_low = low_p.shift(-1)
    next_ratio = next_close / close_safe
    out["target_logret"] = np.log(next_ratio.where(next_ratio > 0.0, np.nan))
    out["target_tr"] = (next_high - next_low).clip(lower=0.0) / close_safe

    thr = float(threshold_bps) / 10000.0
    r_next = out["target_logret"]
    target_dir = pd.Series(np.nan, index=out.index, dtype=float)
    valid_target = r_next.notna()
    target_dir.loc[valid_target] = np.where(
        r_next.loc[valid_target] < -thr,
        0.0,
        np.where(r_next.loc[valid_target] > thr, 2.0, 1.0),
    )
    out["target_dir"] = target_dir

    # Sanitizar +-inf -> NaN
    for col in CANDLE_V2_FEATURES + list(TARGET_COLUMNS):
        vals = out[col].to_numpy(dtype=np.float64)
        vals = np.where(np.isinf(vals), np.nan, vals)
        out[col] = vals

    out["ts"] = ts
    feat_matrix = out[CANDLE_V2_FEATURES].to_numpy(dtype=np.float64)
    out["valid_mask"] = np.isfinite(feat_matrix).all(axis=1)
    return out


class CausalRobustScaler:
    """
    Escalador robusto (mediana + IQR) ajustado exclusivamente sobre el bloque
    de entrenamiento cronológico pasado, evitando todo look-ahead bias.
    Almacena además las escalas de los targets (`ret_scale`, `tr_scale`) del fold.
    """

    def __init__(
        self,
        center: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        ret_scale: float = 1e-2,
        tr_scale: float = 1e-2,
        clip_val: float = 6.0,
        feature_names: Sequence[str] = CANDLE_V2_FEATURES,
    ) -> None:
        n_feat = len(feature_names)
        self.feature_names: Tuple[str, ...] = tuple(feature_names)
        self.center: np.ndarray = (
            np.asarray(center, dtype=np.float64)
            if center is not None
            else np.zeros(n_feat, dtype=np.float64)
        )
        self.scale: np.ndarray = (
            np.asarray(scale, dtype=np.float64)
            if scale is not None
            else np.ones(n_feat, dtype=np.float64)
        )
        self.ret_scale: float = max(float(ret_scale), 1e-4)
        self.tr_scale: float = max(float(tr_scale), 1e-4)
        self.clip_val: float = float(clip_val)

    @classmethod
    def identity(
        cls,
        n_features: int = len(CANDLE_V2_FEATURES),
        clip_val: float = 1e9,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CausalRobustScaler":
        names = tuple(feature_names) if feature_names is not None else tuple(
            CANDLE_V2_FEATURES[:n_features]
        )
        return cls(
            center=np.zeros(len(names), dtype=np.float64),
            scale=np.ones(len(names), dtype=np.float64),
            ret_scale=1.0,
            tr_scale=1.0,
            clip_val=clip_val,
            feature_names=names,
        )

    def fit(
        self,
        X: np.ndarray,
        y_ret: Optional[np.ndarray] = None,
        y_tr: Optional[np.ndarray] = None,
    ) -> "CausalRobustScaler":
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("CausalRobustScaler.fit requiere una matriz 2D no vacía.")

        # Calculamos por columna ignorando NaNs de forma segura incluso si una columna es toda NaN
        n_cols = arr.shape[1]
        center = np.zeros(n_cols, dtype=np.float64)
        scale = np.ones(n_cols, dtype=np.float64)
        for j in range(n_cols):
            col = arr[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                continue
            med = float(np.median(finite))
            q25, q75 = np.percentile(finite, [25.0, 75.0])
            iqr = float(q75 - q25)
            center[j] = med
            scale[j] = iqr if (np.isfinite(iqr) and iqr >= 1e-8) else 1.0

        self.center = center
        self.scale = scale

        if y_ret is not None:
            vr = np.asarray(y_ret, dtype=np.float64)
            vr = vr[np.isfinite(vr)]
            if len(vr) > 0:
                self.ret_scale = max(float(np.std(vr)), 1e-4)
        if y_tr is not None:
            vt = np.asarray(y_tr, dtype=np.float64)
            vt = vt[np.isfinite(vt)]
            if len(vt) > 0:
                self.tr_scale = max(float(np.median(vt)), 1e-4)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        scaled = (arr - self.center) / self.scale
        clipped = np.clip(scaled, -self.clip_val, self.clip_val)
        return clipped.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scaler_type": "CausalRobustScaler",
            "feature_names": list(self.feature_names),
            "center": [float(v) for v in self.center],
            "scale": [float(v) for v in self.scale],
            "ret_scale": float(self.ret_scale),
            "tr_scale": float(self.tr_scale),
            "clip_val": float(self.clip_val),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalRobustScaler":
        return cls(
            center=np.asarray(data["center"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            ret_scale=float(data.get("ret_scale", 1e-2)),
            tr_scale=float(data.get("tr_scale", 1e-2)),
            clip_val=float(data.get("clip_val", 6.0)),
            feature_names=tuple(data.get("feature_names", CANDLE_V2_FEATURES)),
        )


@dataclass
class TickerFrame:
    """Contenedor en memoria de features y targets alineados por símbolo e intervalo."""

    symbol: str
    interval: str
    X: np.ndarray          # (N, F) float32 sin escalar
    y_ret: np.ndarray      # (N,) float32
    y_tr: np.ndarray       # (N,) float32
    y_dir: np.ndarray      # (N,) float32 (NaN si no hay target; {0,1,2} si válido)
    ts: np.ndarray         # (N,) int64 nanosegundos UTC
    valid: np.ndarray      # (N,) bool (features finitas en la fila)


def build_ticker_frame(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "1d",
    threshold_bps: float = 25.0,
) -> TickerFrame:
    """Construye un `TickerFrame` causal a partir del histórico OHLCV de un solo símbolo."""
    canon = canonical_symbol(symbol)
    feat_df = compute_candle_v2_features(
        df=df, symbol=canon, interval=interval, threshold_bps=threshold_bps
    )
    ts_series = feat_df["ts"]
    if ts_series.notna().all():
        ts_ns = pd.DatetimeIndex(ts_series).as_unit("ns").asi8.astype(np.int64)
    elif ts_series.notna().any():
        # Interpolar monotonicidad solo si hay pocos NaT sueltos no; mejor fallback sintético seguro
        ts_ns = np.arange(len(feat_df), dtype=np.int64) * 3_600_000_000_000
    else:
        ts_ns = np.arange(len(feat_df), dtype=np.int64) * 3_600_000_000_000

    return TickerFrame(
        symbol=canon,
        interval=str(interval).strip().lower(),
        X=feat_df[CANDLE_V2_FEATURES].to_numpy(dtype=np.float32),
        y_ret=feat_df["target_logret"].to_numpy(dtype=np.float32),
        y_tr=feat_df["target_tr"].to_numpy(dtype=np.float32),
        y_dir=feat_df["target_dir"].to_numpy(dtype=np.float32),
        ts=ts_ns,
        valid=feat_df["valid_mask"].to_numpy(dtype=bool),
    )


def _has_valid_target(frame: TickerFrame, end_idx: int) -> bool:
    return bool(
        np.isfinite(frame.y_ret[end_idx])
        and np.isfinite(frame.y_tr[end_idx])
        and np.isfinite(frame.y_dir[end_idx])
    )


def build_index(
    frames: Sequence[TickerFrame],
    seq_len: int,
    ts_lo: Optional[int] = None,
    ts_hi: Optional[int] = None,
    require_target: bool = True,
) -> List[Tuple[int, int]]:
    """
    Construye lista de pares `(ticker_idx, end_idx)` pertenecientes íntegramente a un mismo
    `TickerFrame` (jamás cruza fronteras entre tickers), filtrando por rango `[ts_lo, ts_hi)`
    en la fila final `end_idx` de cada ventana.
    """
    pairs: List[Tuple[int, int]] = []
    for t_idx, frame in enumerate(frames):
        n = len(frame.X)
        if n < seq_len:
            continue
        for end_idx in range(seq_len - 1, n):
            end_ts = int(frame.ts[end_idx])
            if ts_lo is not None and end_ts < ts_lo:
                continue
            if ts_hi is not None and end_ts >= ts_hi:
                continue
            start_idx = end_idx - seq_len + 1
            if not bool(frame.valid[start_idx : end_idx + 1].all()):
                continue
            if require_target and not _has_valid_target(frame, end_idx):
                continue
            pairs.append((t_idx, end_idx))
    return pairs


class MultiTickerSequenceDataset(Dataset):
    """
    Dataset PyTorch estricto por símbolo: cada secuencia `[end - seq_len + 1 .. end]`
    proviene de un único `TickerFrame` validado y ordenado cronológicamente.
    """

    def __init__(
        self,
        frames: Sequence[TickerFrame],
        scaler: CausalRobustScaler,
        seq_len: int,
        index: Sequence[Tuple[int, int]],
        require_target: bool = True,
    ) -> None:
        super().__init__()
        self.frames = list(frames)
        self.scaler = scaler
        self.seq_len = int(seq_len)
        self.require_target = bool(require_target)

        self._scaled_X: List[np.ndarray] = [
            scaler.transform(f.X) for f in self.frames
        ]

        valid_index: List[Tuple[int, int]] = []
        symbols: List[str] = []
        end_ts: List[int] = []

        for t_idx, end_idx in index:
            if t_idx < 0 or t_idx >= len(self.frames):
                raise IndexError(f"ticker_idx fuera de rango: {t_idx}")
            frame = self.frames[t_idx]
            start_idx = end_idx - self.seq_len + 1
            if start_idx < 0 or end_idx >= len(frame.X):
                continue
            if not bool(frame.valid[start_idx : end_idx + 1].all()):
                continue
            if self.require_target and not _has_valid_target(frame, end_idx):
                continue
            valid_index.append((int(t_idx), int(end_idx)))
            symbols.append(frame.symbol)
            end_ts.append(int(frame.ts[end_idx]))

        self.index: List[Tuple[int, int]] = valid_index
        self.symbols: List[str] = symbols
        self.end_ts: np.ndarray = np.asarray(end_ts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_idx, end_idx = self.index[idx]
        start_idx = end_idx - self.seq_len + 1
        x_win = self._scaled_X[t_idx][start_idx : end_idx + 1]
        frame = self.frames[t_idx]

        y_ret = float(frame.y_ret[end_idx]) if self.require_target else 0.0
        y_tr = float(frame.y_tr[end_idx]) if self.require_target else 0.0
        y_dir = int(frame.y_dir[end_idx]) if self.require_target else 1

        return (
            torch.from_numpy(np.ascontiguousarray(x_win, dtype=np.float32)),
            torch.tensor(y_ret, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            torch.tensor(y_dir, dtype=torch.long),
        )
