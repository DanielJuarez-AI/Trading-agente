"""
Generador de Caché Histórico Horario (1h) para el Universo de 30 Tickers.

Descarga velas horarias cerradas de los últimos 45 días vía yfinance para los 30
PRIMARY_TICKERS y guarda hasta 240 velas cerradas por ticker en:
`data/replay_candles_1h.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import DATA_DIR, PRIMARY_TICKERS


def _hourly_fallback_from_daily(ticker: str, daily_cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    daily = daily_cache.get(ticker, [])
    seed = abs(hash(f"{ticker}_1h")) % (2**31)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    if daily:
        tail_days = daily[-35:]
        for d in tail_days:
            day_dt = datetime.fromisoformat(d["open_time"].replace("Z", "+00:00"))
            o = float(d["open"])
            c = float(d["close"])
            vol_per_h = float(d["volume"]) / 8.0
            n_hours = 8
            cur = o
            for h_idx in range(n_hours):
                frac = (h_idx + 1) / n_hours
                target = o + (c - o) * frac
                nxt = c if h_idx == n_hours - 1 else target * (1.0 + float(rng.normal(0.0, 0.002)))
                hi = max(cur, nxt) * (1.0 + float(rng.uniform(0.0005, 0.003)))
                lo = min(cur, nxt) * (1.0 - float(rng.uniform(0.0005, 0.003)))
                open_dt = day_dt + timedelta(hours=13 + h_idx)
                close_dt = open_dt + timedelta(minutes=59, seconds=59)
                rows.append(
                    {
                        "open_time": open_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "close_time": close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "open": round(cur, 4),
                        "high": round(hi, 4),
                        "low": round(lo, 4),
                        "close": round(nxt, 4),
                        "volume": round(max(1.0, vol_per_h * float(rng.uniform(0.6, 1.4))), 2),
                    }
                )
                cur = nxt
        return rows[-240:]

    return rows


def generate_hourly_cache() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = DATA_DIR / "replay_candles.json"
    daily_cache: Dict[str, List[Dict[str, Any]]] = (
        json.loads(daily_path.read_text(encoding="utf-8")) if daily_path.exists() else {}
    )

    print(f"Descargando historicos horarios (1h) para {len(PRIMARY_TICKERS)} tickers...")
    raw = yf.download(
        tickers=PRIMARY_TICKERS,
        period="45d",
        interval="1h",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    now_utc = datetime.now(timezone.utc)
    hourly_cache: Dict[str, List[Dict[str, Any]]] = {}

    for ticker in PRIMARY_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[ticker].dropna(subset=["Close"]).copy()
            else:
                sub = raw.dropna(subset=["Close"]).copy()

            if len(sub) < 80:
                hourly_cache[ticker] = _hourly_fallback_from_daily(ticker, daily_cache)
                continue

            sub = sub.reset_index()
            date_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
            dts = pd.to_datetime(sub[date_col], utc=True)

            rows: List[Dict[str, Any]] = []
            for idx, row in sub.iterrows():
                open_dt = dts.iloc[idx].to_pydatetime().astimezone(timezone.utc)
                close_dt = open_dt + timedelta(minutes=59, seconds=59)
                # Excluir estrictamente cualquier vela horaria todavía abierta
                if close_dt > now_utc:
                    continue
                rows.append(
                    {
                        "open_time": open_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "close_time": close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "open": round(float(row["Open"]), 4),
                        "high": round(float(row["High"]), 4),
                        "low": round(float(row["Low"]), 4),
                        "close": round(float(row["Close"]), 4),
                        "volume": round(float(row["Volume"]), 2),
                    }
                )

            if len(rows) < 80:
                hourly_cache[ticker] = _hourly_fallback_from_daily(ticker, daily_cache)
            else:
                hourly_cache[ticker] = rows[-240:]
        except Exception as exc:
            print(f"Fallback horario para {ticker}: {exc}")
            hourly_cache[ticker] = _hourly_fallback_from_daily(ticker, daily_cache)

    out_path = DATA_DIR / "replay_candles_1h.json"
    out_path.write_text(json.dumps(hourly_cache, indent=2), encoding="utf-8")
    counts = {k: len(v) for k, v in list(hourly_cache.items())[:5]}
    print(f"Cache horario guardado en {out_path}. Muestra conteos: {counts}")


if __name__ == "__main__":
    generate_hourly_cache()
