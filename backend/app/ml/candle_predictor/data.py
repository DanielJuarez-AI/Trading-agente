"""
Cargador unificado de velas OHLCV cerradas (`1h` / `1d`) con soporte para:
- Histórico extendido vía `yfinance` con caché CSV local en `data/history/{interval}/{SYMBOL}.csv`
- Fallback o lectura directa desde las cachés del repo (`data/replay_candles*.json`)
- Verificación determinista SHA-256 del dataset consolidado
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from backend.app.core.config import DATA_DIR
from backend.app.ml.candle_predictor.config import WARMUP_BARS, deduplicate_tickers

logger = logging.getLogger(__name__)

DEFAULT_PERIOD_BY_INTERVAL: Dict[str, str] = {
    "1d": "5y",
    "1h": "729d",
}

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _normalize_candle_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Estandariza columnas, orden ascendente y exclusión de velas no cerradas."""
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    out = df.copy()
    if "open_time" not in out.columns and "timestamp" in out.columns:
        out["open_time"] = out["timestamp"]
    if "close_time" not in out.columns:
        delta = timedelta(hours=23, minutes=59, seconds=59) if interval == "1d" else timedelta(minutes=59, seconds=59)
        o_dt = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
        out["close_time"] = (o_dt + delta).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    o_dt = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    c_dt = pd.to_datetime(out["close_time"], utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz=timezone.utc)

    valid = o_dt.notna() & c_dt.notna() & (c_dt <= now_utc)
    out = out.loc[valid].copy()
    out["_o_dt"] = o_dt.loc[valid]

    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col == "volume":
            out[col] = 0.0
        else:
            out[col] = out["close"]

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.loc[out["close"] > 0.0]
    out = out.sort_values("_o_dt", kind="mergesort")
    out = out.drop_duplicates(subset=["_o_dt"], keep="last").drop(columns=["_o_dt"]).reset_index(drop=True)
    return out[list(CANONICAL_COLUMNS)]


def _history_df_from_yf_sub(
    sub: pd.DataFrame, interval: str, now_utc: datetime
) -> pd.DataFrame:
    """Convierte la salida cruda de yfinance al formato canónico cerradas UTC."""
    if sub is None or sub.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    work = sub.dropna(subset=["Close"]).copy().reset_index()
    if work.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    date_col = "Datetime" if "Datetime" in work.columns else (
        "Date" if "Date" in work.columns else work.columns[0]
    )
    raw_dts = pd.to_datetime(work[date_col])

    rows: List[Dict[str, Any]] = []
    for idx, row in work.iterrows():
        raw_ts = raw_dts.iloc[idx]
        if interval == "1d":
            # Preservar la fecha bursátil original antes de proyectar a UTC 00:00..23:59:59Z
            bar_date = raw_ts.date()
            open_dt = datetime(bar_date.year, bar_date.month, bar_date.day, 0, 0, 0, tzinfo=timezone.utc)
            close_dt = open_dt + timedelta(hours=23, minutes=59, seconds=59)
        else:
            ts_utc = (
                raw_ts.tz_localize(timezone.utc)
                if raw_ts.tzinfo is None
                else raw_ts.tz_convert(timezone.utc)
            )
            open_dt = ts_utc.to_pydatetime()
            close_dt = open_dt + timedelta(minutes=59, seconds=59)

        if close_dt > now_utc:
            continue

        o_val = float(row.get("Open", row["Close"]))
        h_val = float(row.get("High", row["Close"]))
        l_val = float(row.get("Low", row["Close"]))
        c_val = float(row["Close"])
        v_val = float(row.get("Volume", 0.0) or 0.0)
        if c_val <= 0.0:
            continue

        rows.append(
            {
                "open_time": open_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "close_time": close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": round(o_val, 6),
                "high": round(h_val, 6),
                "low": round(l_val, 6),
                "close": round(c_val, 6),
                "volume": round(max(0.0, v_val), 4),
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    return _normalize_candle_df(pd.DataFrame(rows), interval=interval)


def _fetch_yfinance_symbol(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Descarga histórico de un símbolo vía yfinance de forma segura."""
    import yfinance as yf

    now_utc = datetime.now(timezone.utc)
    raw = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            sub = raw[symbol]
        elif symbol in raw.columns.get_level_values(1):
            sub = raw.xs(symbol, axis=1, level=1)
        else:
            sub = raw.droplevel(1, axis=1) if raw.columns.nlevels == 2 else raw
    else:
        sub = raw
    return _history_df_from_yf_sub(sub, interval=interval, now_utc=now_utc)


def _load_repo_cache_map(interval: str, data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    cache_name = "replay_candles_1h.json" if interval == "1h" else "replay_candles.json"
    cache_path = data_dir / cache_name
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _compute_dataset_sha256(candles_map: Dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for sym in sorted(candles_map.keys()):
        h.update(f"# {sym}\n".encode("utf-8"))
        csv_bytes = candles_map[sym][list(CANONICAL_COLUMNS)].to_csv(index=False).encode("utf-8")
        h.update(csv_bytes)
    return h.hexdigest()


def load_candles(
    tickers: Iterable[str],
    interval: str = "1d",
    source: str = "auto",
    period: Optional[str] = None,
    refresh: bool = False,
    seq_len: int = 24,
    data_dir: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Carga históricos OHLCV cerrados para una lista de tickers.

    Args:
        tickers: Símbolos crudos (se canonicalizan y deduplican; ej. BTCUSDT -> BTC-USD).
        interval: "1d" o "1h".
        source: "auto" (yfinance/CSV con fallback a caché), "yfinance", o "cache".
        period: Ventana de yfinance ("5y" en 1d, "729d" en 1h por defecto).
        refresh: Ignora caché CSV `data/history/{interval}/{SYMBOL}.csv` y re-descarga.
        seq_len: Longitud de ventana para exigir un mínimo de velas útiles por ticker.
        data_dir: Raíz de datos (por defecto `DATA_DIR`).
    """
    tf = str(interval).strip().lower()
    if tf not in DEFAULT_PERIOD_BY_INTERVAL:
        raise ValueError(f"Temporalidad inválida: {interval}")
    if source not in ("auto", "yfinance", "cache"):
        raise ValueError(f"source inválido: {source}. Use 'auto', 'yfinance' o 'cache'.")

    root_data = data_dir or DATA_DIR
    resolved_period = period or DEFAULT_PERIOD_BY_INTERVAL[tf]
    canon_tickers = deduplicate_tickers(tickers)
    min_rows = WARMUP_BARS + int(seq_len) + 20

    history_dir = root_data / "history" / tf
    if source in ("auto", "yfinance"):
        history_dir.mkdir(parents=True, exist_ok=True)

    repo_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if source in ("cache", "auto"):
        repo_cache = _load_repo_cache_map(tf, root_data)

    candles_out: Dict[str, pd.DataFrame] = {}
    source_per_ticker: Dict[str, str] = {}
    rows_per_ticker: Dict[str, int] = {}
    skipped_tickers: Dict[str, str] = {}

    for sym in canon_tickers:
        df_loaded: Optional[pd.DataFrame] = None
        src_used = ""

        if source in ("auto", "yfinance"):
            csv_path = history_dir / f"{sym}.csv"
            if csv_path.exists() and not refresh:
                try:
                    cached_csv = _normalize_candle_df(pd.read_csv(csv_path), interval=tf)
                    if len(cached_csv) >= 300 or source == "yfinance":
                        df_loaded = cached_csv
                        src_used = "yfinance_history_csv"
                except Exception as exc:
                    logger.warning("Error leyendo CSV histórico %s: %s", csv_path, exc)

            if df_loaded is None:
                try:
                    yf_df = _fetch_yfinance_symbol(sym, interval=tf, period=resolved_period)
                    if not yf_df.empty:
                        yf_df.to_csv(csv_path, index=False)
                        if source == "yfinance" or len(yf_df) >= 300:
                            df_loaded = yf_df
                            src_used = "yfinance"
                        else:
                            logger.warning(
                                "[%s %s] yfinance devolvió solo %d filas (<300); intentando fallback caché.",
                                sym,
                                tf,
                                len(yf_df),
                            )
                except Exception as exc:
                    logger.warning("[%s %s] fallo descarga yfinance: %s", sym, tf, exc)

        if df_loaded is None and source in ("auto", "cache"):
            raw_list = (repo_cache or {}).get(sym)
            if raw_list:
                df_loaded = _normalize_candle_df(pd.DataFrame(raw_list), interval=tf)
                src_used = "cache"
            else:
                logger.warning(
                    "[%s %s] no encontrado en caché %s; omitiendo ticker.",
                    sym,
                    tf,
                    "replay_candles_1h.json" if tf == "1h" else "replay_candles.json",
                )

        if df_loaded is None or len(df_loaded) < min_rows:
            reason = f"filas={0 if df_loaded is None else len(df_loaded)} < minimo={min_rows}"
            skipped_tickers[sym] = reason
            logger.warning("[%s %s] omitido: %s", sym, tf, reason)
            continue

        candles_out[sym] = df_loaded
        source_per_ticker[sym] = src_used
        rows_per_ticker[sym] = int(len(df_loaded))

    all_starts = [str(df["open_time"].iloc[0]) for df in candles_out.values() if len(df) > 0]
    all_ends = [str(df["close_time"].iloc[-1]) for df in candles_out.values() if len(df) > 0]
    cache_only = bool(candles_out) and all(v == "cache" for v in source_per_ticker.values())

    meta: Dict[str, Any] = {
        "interval": tf,
        "requested_source": source,
        "period": resolved_period,
        "tickers_loaded": list(candles_out.keys()),
        "tickers_skipped": skipped_tickers,
        "source_per_ticker": source_per_ticker,
        "rows_per_ticker": rows_per_ticker,
        "start_utc": min(all_starts) if all_starts else "",
        "end_utc": max(all_ends) if all_ends else "",
        "dataset_sha256": _compute_dataset_sha256(candles_out) if candles_out else "",
        "cache_only": cache_only,
    }
    return candles_out, meta
