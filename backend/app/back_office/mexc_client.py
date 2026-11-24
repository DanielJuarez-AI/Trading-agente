"""
Back Office - Cliente de Mercado Público (MEXC /api/v3/klines + Universo de 30 Tickers).

REGLAS ESTRICTAS DE SEGURIDAD Y DATOS:
1. Bloqueo explícito de órdenes reales: `if settings.enable_real_trading: raise RuntimeError(...)`.
2. Solo lectura pública de velas cerradas (`close_time <= current_utc_time`).
3. Reintentos con espera exponencial (máx. 3 intentos) ante 429 / 5xx / timeout;
   si agota reintentos, registra error y pausa el sistema automáticamente.
4. Validación de secuencia cronológica y antigüedad máxima (`MAX_DATA_AGE_SECONDS`).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd
import yfinance as yf

from backend.app.back_office.database import TradingDatabase
from backend.app.core.config import DATA_DIR, SYMBOL_ALIASES, Settings


class MarketDataUnavailableError(RuntimeError):
    """Error cuando la API de mercado falla tras agotar reintentos."""


class StaleDataError(ValueError):
    """Error cuando la vela más reciente supera MAX_DATA_AGE_SECONDS o es inválida."""


@dataclass
class CandleData:
    symbol: str
    interval: str
    open_time: str
    close_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source_mode": self.source_mode,
        }


def _parse_iso_utc(ts: str) -> datetime:
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ms_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class MEXCMarketClient:
    """
    Cliente público de datos de mercado para MEXC (/api/v3/klines) y los 30 tickers oficiales.
    """

    def __init__(
        self,
        settings: Settings,
        db: TradingDatabase,
        http_client: Optional[httpx.Client] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        replay_cache_path: Optional[Path] = None,
    ) -> None:
        if settings.enable_real_trading:
            raise RuntimeError(
                "CRITICAL SECURITY VIOLATION: Real trading is strictly disabled by policy."
            )
        self.settings = settings
        self.db = db
        self.http_client = http_client
        self.sleep_fn = sleep_fn
        self.replay_cache_path = replay_cache_path or (DATA_DIR / "replay_candles.json")
        self._cache_lock = threading.RLock()
        self._klines_cache: Dict[tuple, Dict[str, Any]] = {}
        self._mexc_ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._equity_quote_cache: Dict[str, Dict[str, Any]] = {}
        self._batch_refresh_inflight: bool = False

    def place_real_order(self, *args: Any, **kwargs: Any) -> None:
        """Ruta explícitamente bloqueada por normativa: cero órdenes reales."""
        self.db.log_error(
            error_code="REAL_TRADING_BLOCKED",
            component="MEXCMarketClient",
            message="Intento bloqueado de ejecutar orden real.",
            payload_excerpt=str(args)[:200],
        )
        raise RuntimeError("Real trading orders are strictly disabled (Paper Only).")

    def _resolve_mexc_symbol(self, symbol: str) -> str:
        if symbol == "BTC-USD":
            return SYMBOL_ALIASES["BTC-USD"]
        return symbol

    def _is_crypto_mexc_symbol(self, symbol: str) -> bool:
        return symbol.upper() in {"BTC-USD", "BTCUSDT"}

    def _fetch_mexc_klines_with_backoff(
        self, symbol: str, interval: str, limit: int = 250
    ) -> List[List[Any]]:
        """
        Consulta pública GET /api/v3/klines con máximo 3 intentos y espera exponencial.
        """
        mexc_symbol = self._resolve_mexc_symbol(symbol)
        mexc_interval = "60m" if interval == "1h" else interval
        url = f"{self.settings.mexc_public_base_url.rstrip('/')}/api/v3/klines"
        params = {"symbol": mexc_symbol, "interval": mexc_interval, "limit": limit}

        last_exc: Optional[Exception] = None
        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(timeout=10.0)

        try:
            for attempt in range(1, 4):
                try:
                    resp = client.get(url, params=params)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"HTTP {resp.status_code} desde MEXC /api/v3/klines",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    if not isinstance(data, list) or len(data) == 0:
                        raise ValueError("Respuesta vacía o inesperada de MEXC klines.")
                    return data
                except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
                    last_exc = exc
                    if attempt < 3:
                        backoff_seconds = 0.5 * (2 ** (attempt - 1))
                        self.sleep_fn(backoff_seconds)
        finally:
            if owns_client:
                client.close()

        # Si agotó los 3 reintentos: registrar error y pausar el sistema automáticamente
        self.db.log_error(
            error_code="MEXC_API_UNAVAILABLE",
            component="MEXCMarketClient",
            message=f"Fallo tras 3 intentos en /api/v3/klines para {mexc_symbol}: {last_exc}",
            payload_excerpt=f"symbol={mexc_symbol}, interval={interval}",
        )
        self.db.set_state("system_status", "PAUSED")
        raise MarketDataUnavailableError(
            f"API de mercado no disponible tras 3 reintentos para {symbol}: {last_exc}"
        )

    def fetch_mexc_realtime_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Consulta pública GET /api/v3/ticker/24hr en MEXC para obtener cotización spot
        y métricas de las últimas 24h en tiempo real (caché corta de 1.5s).
        """
        mexc_symbol = self._resolve_mexc_symbol(symbol)
        if not mexc_symbol.endswith("USDT"):
            mexc_symbol = "BTCUSDT"

        now_m = time.monotonic()
        with self._cache_lock:
            cached = self._mexc_ticker_cache.get(mexc_symbol)
            if (
                self.http_client is None
                and cached is not None
                and (now_m - float(cached.get("_ts", 0.0))) < 0.9
            ):
                return dict(cached["data"])

        url = f"{self.settings.mexc_public_base_url.rstrip('/')}/api/v3/ticker/24hr"
        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(timeout=3.5)
        try:
            resp = client.get(url, params={"symbol": mexc_symbol})
            resp.raise_for_status()
            raw = resp.json()
            if not isinstance(raw, dict) or "lastPrice" not in raw:
                raise ValueError("Respuesta sin lastPrice desde MEXC /api/v3/ticker/24hr.")

            last_price = float(raw["lastPrice"])
            prev_close = float(raw.get("prevClosePrice") or 0.0)
            if prev_close > 0:
                change_pct = ((last_price - prev_close) / prev_close) * 100.0
            else:
                change_pct = float(raw.get("priceChangePercent") or 0.0) * 100.0

            payload = {
                "mexc_symbol": mexc_symbol,
                "last_price": round(last_price, 6),
                "price_change_pct": round(change_pct, 4),
                "high_24h": round(float(raw.get("highPrice") or last_price), 6),
                "low_24h": round(float(raw.get("lowPrice") or last_price), 6),
                "volume_24h": round(float(raw.get("volume") or 0.0), 4),
                "bid_price": round(float(raw.get("bidPrice") or last_price), 6),
                "ask_price": round(float(raw.get("askPrice") or last_price), 6),
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mexc_online": True,
            }
            with self._cache_lock:
                self._mexc_ticker_cache[mexc_symbol] = {"_ts": now_m, "data": dict(payload)}
            return payload
        except Exception:
            with self._cache_lock:
                stale = self._mexc_ticker_cache.get(mexc_symbol)
                if stale is not None:
                    fallback = dict(stale["data"])
                    fallback["mexc_online"] = False
                    return fallback
            return {
                "mexc_symbol": mexc_symbol,
                "last_price": 0.0,
                "price_change_pct": 0.0,
                "high_24h": 0.0,
                "low_24h": 0.0,
                "volume_24h": 0.0,
                "bid_price": 0.0,
                "ask_price": 0.0,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mexc_online": False,
            }
        finally:
            if owns_client:
                client.close()

    def fetch_equity_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene último tick 1m (incluyendo pre/after-market) para acciones/ETFs vía yfinance
        con caché corta de 3.0s para evitar rate-limits durante el refresco en vivo.
        """
        if self._is_crypto_mexc_symbol(symbol):
            return None
        now_m = time.monotonic()
        with self._cache_lock:
            cached = self._equity_quote_cache.get(symbol)
            if cached is not None and (now_m - float(cached.get("_ts", 0.0))) < 3.0:
                return dict(cached["data"]) if cached["data"] is not None else None
        try:
            q_hist = yf.download(
                tickers=symbol,
                period="5d",
                interval="1m",
                prepost=True,
                auto_adjust=True,
                progress=False,
            )
            if q_hist.empty:
                return None
            if isinstance(q_hist.columns, pd.MultiIndex):
                q_hist.columns = q_hist.columns.get_level_values(0)
            last_row = q_hist.iloc[-1]
            last_px = float(last_row["Close"].iloc[0] if hasattr(last_row["Close"], "iloc") else last_row["Close"])
            if last_px <= 0:
                return None
            quote = {
                "symbol": symbol,
                "last_price": round(last_px, 4),
                "high": round(float(last_row["High"].iloc[0] if hasattr(last_row["High"], "iloc") else last_row["High"]), 4),
                "low": round(float(last_row["Low"].iloc[0] if hasattr(last_row["Low"], "iloc") else last_row["Low"]), 4),
                "timestamp_utc": pd.to_datetime(q_hist.index[-1], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with self._cache_lock:
                self._equity_quote_cache[symbol] = {"_ts": now_m, "data": dict(quote)}
            return quote
        except Exception:
            with self._cache_lock:
                stale = self._equity_quote_cache.get(symbol)
                return dict(stale["data"]) if stale and stale["data"] else None

    def _refresh_equity_batch_sync(self, symbols: List[str]) -> None:
        """Descarga cotizaciones 1m (con pre/after-market) en lote vía yfinance."""
        targets = [s for s in dict.fromkeys(symbols) if not self._is_crypto_mexc_symbol(s)]
        if not targets:
            return
        try:
            q_hist = yf.download(
                tickers=targets if len(targets) > 1 else targets[0],
                period="5d",
                interval="1m",
                prepost=True,
                auto_adjust=True,
                progress=False,
            )
            if q_hist is None or q_hist.empty:
                return
            now_m = time.monotonic()
            updates: Dict[str, Dict[str, Any]] = {}

            if isinstance(q_hist.columns, pd.MultiIndex):
                close_frame = q_hist.get("Close")
                high_frame = q_hist.get("High")
                low_frame = q_hist.get("Low")
                if close_frame is not None:
                    for sym in targets:
                        if sym not in close_frame.columns:
                            continue
                        series = close_frame[sym].dropna()
                        if series.empty:
                            continue
                        ts = series.index[-1]
                        px = float(series.iloc[-1])
                        if px <= 0:
                            continue
                        hi = float(high_frame[sym].loc[ts]) if high_frame is not None and sym in high_frame.columns else px
                        lo = float(low_frame[sym].loc[ts]) if low_frame is not None and sym in low_frame.columns else px
                        updates[sym] = {
                            "symbol": sym,
                            "last_price": round(px, 4),
                            "high": round(hi if pd.notna(hi) else px, 4),
                            "low": round(lo if pd.notna(lo) else px, 4),
                            "timestamp_utc": pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
            else:
                sym = targets[0]
                if "Close" in q_hist.columns:
                    series = q_hist["Close"].dropna()
                    if not series.empty:
                        ts = series.index[-1]
                        px = float(series.iloc[-1])
                        if px > 0:
                            hi = float(q_hist["High"].loc[ts]) if "High" in q_hist.columns else px
                            lo = float(q_hist["Low"].loc[ts]) if "Low" in q_hist.columns else px
                            updates[sym] = {
                                "symbol": sym,
                                "last_price": round(px, 4),
                                "high": round(hi if pd.notna(hi) else px, 4),
                                "low": round(lo if pd.notna(lo) else px, 4),
                                "timestamp_utc": pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            }

            if updates:
                with self._cache_lock:
                    for sym, quote in updates.items():
                        self._equity_quote_cache[sym] = {"_ts": now_m, "data": dict(quote)}
        except Exception:
            pass

    def fetch_realtime_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene precios actuales de múltiples símbolos sin bloquear el loop de 1s:
        - Cripto MEXC (`BTC-USD`, `BTCUSDT`) con TTL 0.9s.
        - Acciones/ETFs vía batch `yfinance` en caché + refresco asíncrono cada 3s.
        """
        ordered = []
        for s in symbols:
            clean = str(s).strip()
            if clean in self.settings.allowed_symbols and clean not in ordered:
                ordered.append(clean)

        result: Dict[str, Dict[str, Any]] = {}
        equities: List[str] = []
        for sym in ordered:
            if self._is_crypto_mexc_symbol(sym):
                tick = self.fetch_mexc_realtime_ticker(sym)
                px = float(tick.get("last_price") or 0.0)
                if px > 0:
                    result[sym] = {
                        "price": round(px, 4),
                        "change_pct_24h": tick.get("price_change_pct"),
                        "source": "MEXC_API",
                        "online": bool(tick.get("mexc_online")),
                        "updated_at_utc": tick.get("timestamp_utc"),
                    }
            else:
                equities.append(sym)

        if equities:
            now_m = time.monotonic()
            missing: List[str] = []
            stale: List[str] = []
            should_spawn_bg = False

            with self._cache_lock:
                for sym in equities:
                    entry = self._equity_quote_cache.get(sym)
                    if entry is None or entry.get("data") is None:
                        missing.append(sym)
                    elif (now_m - float(entry.get("_ts", 0.0))) >= 3.0:
                        stale.append(sym)

                if not missing and stale and not self._batch_refresh_inflight:
                    self._batch_refresh_inflight = True
                    should_spawn_bg = True

            if missing:
                self._refresh_equity_batch_sync(missing + stale)
            elif should_spawn_bg:
                def _bg_worker(targets: List[str]) -> None:
                    try:
                        self._refresh_equity_batch_sync(targets)
                    finally:
                        with self._cache_lock:
                            self._batch_refresh_inflight = False

                threading.Thread(target=_bg_worker, args=(stale,), daemon=True).start()

            with self._cache_lock:
                for sym in equities:
                    entry = self._equity_quote_cache.get(sym)
                    q = entry.get("data") if entry else None
                    if q and float(q.get("last_price") or 0.0) > 0:
                        result[sym] = {
                            "price": round(float(q["last_price"]), 4),
                            "change_pct_24h": None,
                            "source": "YAHOO_1M",
                            "online": True,
                            "updated_at_utc": q.get("timestamp_utc"),
                        }

        return result

    def _overlay_live_tick(
        self,
        df: pd.DataFrame,
        symbol: str,
        include_forming: bool,
        enforce_mexc_for_crypto: bool,
    ) -> pd.DataFrame:
        if not include_forming or df.empty:
            return df
        out = df.copy()
        last_idx = out.index[-1]
        # Solo se superpone el tick a una vela realmente en formación (nunca altera una vela ya cerrada)
        try:
            last_close_dt = _parse_iso_utc(str(out.at[last_idx, "close_time"]))
            if last_close_dt <= datetime.now(timezone.utc):
                return out
        except Exception:
            return out
        if self._is_crypto_mexc_symbol(symbol) and enforce_mexc_for_crypto:
            tick = self.fetch_mexc_realtime_ticker(symbol)
            px = float(tick.get("last_price") or 0.0)
            if tick.get("mexc_online") and px > 0.0:
                out.at[last_idx, "close"] = px
                out.at[last_idx, "high"] = max(float(out.at[last_idx, "high"]), px)
                out.at[last_idx, "low"] = min(float(out.at[last_idx, "low"]), px)
        elif not self._is_crypto_mexc_symbol(symbol):
            eq_tick = self.fetch_equity_realtime_quote(symbol)
            if eq_tick and float(eq_tick.get("last_price") or 0.0) > 0.0:
                px = float(eq_tick["last_price"])
                out.at[last_idx, "close"] = px
                out.at[last_idx, "high"] = max(float(out.at[last_idx, "high"]), px)
                out.at[last_idx, "low"] = min(float(out.at[last_idx, "low"]), px)
        return out

    def fetch_live_candles(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: int = 250,
        now_utc: Optional[datetime] = None,
        enforce_mexc_for_crypto: bool = True,
        *,
        include_forming: bool = False,
    ) -> pd.DataFrame:
        """
        Obtiene velas en modo 'live', validando orden cronológico y frescura.
        Por defecto (`include_forming=False`), descarta estrictamente velas abiertas
        para proteger la idempotencia del motor de trading. Si `include_forming=True`,
        conserva la vela en formación con su último tick en vivo para la gráfica.
        """
        if self.settings.enable_real_trading:
            raise RuntimeError("Real trading is strictly disabled.")
        if symbol not in self.settings.allowed_symbols:
            raise ValueError(f"Símbolo fuera de lista blanca: {symbol}")

        tf = interval or self.settings.candle_interval
        current_utc = now_utc or datetime.now(timezone.utc)
        cache_key = (symbol, tf, int(limit), bool(enforce_mexc_for_crypto))
        now_m = time.monotonic()

        if include_forming and self.http_client is None and now_utc is None:
            cached_df: Optional[pd.DataFrame] = None
            with self._cache_lock:
                cached = self._klines_cache.get(cache_key)
                if cached is not None and (now_m - float(cached.get("_ts", 0.0))) < 6.0:
                    cached_df = cached["df"].copy()
            if cached_df is not None:
                return self._overlay_live_tick(
                    cached_df,
                    symbol=symbol,
                    include_forming=True,
                    enforce_mexc_for_crypto=enforce_mexc_for_crypto,
                )

        rows: List[Dict[str, Any]] = []

        if self._is_crypto_mexc_symbol(symbol) and enforce_mexc_for_crypto:
            raw_klines = self._fetch_mexc_klines_with_backoff(symbol, tf, limit=limit)
            for k in raw_klines:
                open_ms = int(k[0])
                close_ms = int(k[6])
                open_dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
                close_dt = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc)
                if include_forming:
                    if open_dt > current_utc:
                        continue
                else:
                    # Descarta estrictamente velas abiertas
                    if close_dt > current_utc:
                        continue
                rows.append(
                    {
                        "symbol": symbol,
                        "interval": tf,
                        "open_time": _ms_to_iso_utc(open_ms),
                        "close_time": _ms_to_iso_utc(close_ms),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "source_mode": "live",
                    }
                )
        else:
            # Acciones / ETFs de los 30 PRIMARY_TICKERS vía datos públicos yfinance
            if tf not in {"1h", "1d"}:
                raise ValueError(f"Temporalidad no soportada: {tf}. Use '1h' o '1d'.")
            try:
                yf_period = "60d" if tf == "1h" else "2y"
                hist = yf.download(
                    tickers=symbol,
                    period=yf_period,
                    interval=tf,
                    auto_adjust=True,
                    progress=False,
                )
                if hist.empty:
                    raise ValueError(f"Sin velas para {symbol} desde proveedor público.")
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                hist = hist.reset_index()
                date_col = (
                    "Datetime"
                    if "Datetime" in hist.columns
                    else ("Date" if "Date" in hist.columns else hist.columns[0])
                )
                for _, row in hist.iterrows():
                    raw_dt = row[date_col]
                    if hasattr(raw_dt, "iloc"):
                        raw_dt = raw_dt.iloc[0]
                    dt = pd.to_datetime(raw_dt, utc=True)
                    if tf == "1h":
                        open_dt = dt
                        close_dt = dt + timedelta(minutes=59, seconds=59)
                        open_str = open_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        close_str = close_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        open_dt = dt
                        close_dt = dt + timedelta(hours=23, minutes=59, seconds=59)
                        open_str = dt.strftime("%Y-%m-%dT00:00:00Z")
                        close_str = dt.strftime("%Y-%m-%dT23:59:59Z")
                    if include_forming:
                        if open_dt > current_utc:
                            continue
                    else:
                        if close_dt > current_utc:
                            continue
                    def _cell(col: str) -> float:
                        v = row[col]
                        return float(v.iloc[0] if hasattr(v, "iloc") else v)

                    o_v, h_v, l_v, c_v = _cell("Open"), _cell("High"), _cell("Low"), _cell("Close")
                    vol_v = _cell("Volume")
                    if pd.isna(c_v) or c_v <= 0:
                        continue  # fila sin cierre válido (vela de hoy incompleta en Yahoo)
                    # Yahoo a veces entrega la vela del día en curso con O/H/L = NaN: se reconstruyen
                    if pd.isna(o_v) or o_v <= 0:
                        o_v = float(rows[-1]["close"]) if rows else c_v
                    if pd.isna(h_v) or h_v <= 0:
                        h_v = max(o_v, c_v)
                    if pd.isna(l_v) or l_v <= 0:
                        l_v = min(o_v, c_v)
                    h_v, l_v = max(h_v, o_v, c_v), min(l_v, o_v, c_v)
                    rows.append(
                        {
                            "symbol": symbol,
                            "interval": tf,
                            "open_time": open_str,
                            "close_time": close_str,
                            "open": o_v,
                            "high": h_v,
                            "low": l_v,
                            "close": c_v,
                            "volume": 0.0 if pd.isna(vol_v) else vol_v,
                            "source_mode": "live",
                        }
                    )
            except Exception as exc:
                self.db.log_error(
                    error_code="MARKET_DATA_FETCH_ERROR",
                    component="MEXCMarketClient",
                    message=f"Error obteniendo datos públicos para {symbol}: {exc}",
                    payload_excerpt=f"symbol={symbol}, interval={tf}",
                )
                self.db.set_state("system_status", "PAUSED")
                raise MarketDataUnavailableError(str(exc)) from exc

        df = self._validate_and_sort_candles(
            rows, symbol=symbol, current_utc=current_utc, check_freshness=True
        )
        if include_forming and self.http_client is None and now_utc is None:
            with self._cache_lock:
                self._klines_cache[cache_key] = {"_ts": now_m, "df": df.copy()}
            df = self._overlay_live_tick(
                df,
                symbol=symbol,
                include_forming=True,
                enforce_mexc_for_crypto=enforce_mexc_for_crypto,
            )
        elif include_forming:
            df = self._overlay_live_tick(
                df,
                symbol=symbol,
                include_forming=True,
                enforce_mexc_for_crypto=enforce_mexc_for_crypto,
            )
        return df

    def load_replay_candles(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """
        Carga velas históricas cerradas desde cache local (`replay_candles.json` para 1d
        o `replay_candles_1h.json` para 1h) para evaluación reproducible sin look-ahead.
        """
        if symbol not in self.settings.allowed_symbols:
            raise ValueError(f"Símbolo fuera de lista blanca: {symbol}")
        if interval not in {"1h", "1d"}:
            raise ValueError(f"Temporalidad no soportada: {interval}. Use '1h' o '1d'.")

        canonical = "BTC-USD" if symbol == "BTCUSDT" else symbol
        if interval == "1h":
            cache_path = self.replay_cache_path.parent / "replay_candles_1h.json"
            if cache_path.exists():
                cache_1h: Dict[str, List[Dict[str, Any]]] = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                if canonical in cache_1h and len(cache_1h[canonical]) >= 50:
                    rows = [
                        {
                            "symbol": symbol,
                            "interval": "1h",
                            "open_time": item["open_time"],
                            "close_time": item["close_time"],
                            "open": float(item["open"]),
                            "high": float(item["high"]),
                            "low": float(item["low"]),
                            "close": float(item["close"]),
                            "volume": float(item["volume"]),
                            "source_mode": "replay",
                        }
                        for item in cache_1h[canonical]
                    ]
                    return self._validate_and_sort_candles(
                        rows,
                        symbol=symbol,
                        current_utc=datetime.now(timezone.utc),
                        check_freshness=False,
                    )

        if not self.replay_cache_path.exists():
            raise FileNotFoundError(
                f"Cache de replay no encontrado en {self.replay_cache_path}. "
                "Ejecute notebooks/generate_phase1_artifacts.py."
            )

        cache: Dict[str, List[Dict[str, Any]]] = json.loads(
            self.replay_cache_path.read_text(encoding="utf-8")
        )
        if canonical not in cache:
            raise ValueError(f"Ticker {canonical} no encontrado en replay_candles.json")

        if interval == "1h":
            # Respaldo determinista intradía si no existe replay_candles_1h.json
            rows = []
            for item in cache[canonical][-35:]:
                day_dt = _parse_iso_utc(item["open_time"])
                o = float(item["open"])
                c = float(item["close"])
                vol_h = float(item["volume"]) / 8.0
                for h_idx in range(8):
                    frac0 = h_idx / 8.0
                    frac1 = (h_idx + 1) / 8.0
                    p0 = o + (c - o) * frac0
                    p1 = o + (c - o) * frac1
                    o_dt = day_dt + timedelta(hours=13 + h_idx)
                    c_dt = o_dt + timedelta(minutes=59, seconds=59)
                    rows.append(
                        {
                            "symbol": symbol,
                            "interval": "1h",
                            "open_time": o_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "close_time": c_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "open": p0,
                            "high": max(p0, p1, float(item["high"]) * 0.999 if h_idx == 4 else max(p0, p1)),
                            "low": min(p0, p1, float(item["low"]) * 1.001 if h_idx == 2 else min(p0, p1)),
                            "close": p1,
                            "volume": vol_h,
                            "source_mode": "replay",
                        }
                    )
        else:
            rows = [
                {
                    "symbol": symbol,
                    "interval": interval,
                    "open_time": item["open_time"],
                    "close_time": item["close_time"],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                    "source_mode": "replay",
                }
                for item in cache[canonical]
            ]
        return self._validate_and_sort_candles(
            rows,
            symbol=symbol,
            current_utc=datetime.now(timezone.utc),
            check_freshness=False,
        )

    def _validate_and_sort_candles(
        self,
        rows: List[Dict[str, Any]],
        symbol: str,
        current_utc: datetime,
        check_freshness: bool = True,
    ) -> pd.DataFrame:
        if not rows:
            self.db.log_error(
                error_code="NO_CLOSED_CANDLES",
                component="MEXCMarketClient",
                message=f"No se recibieron velas cerradas válidas para {symbol}.",
                payload_excerpt=f"symbol={symbol}",
            )
            raise StaleDataError(f"Sin velas cerradas para {symbol}.")

        df = pd.DataFrame(rows)
        df["_close_dt"] = df["close_time"].apply(_parse_iso_utc)
        df = df.sort_values("_close_dt", ascending=True).drop_duplicates(
            subset=["close_time"]
        ).reset_index(drop=True)

        # Validar secuencia cronológica estricta y precios positivos
        if (df["close"] <= 0).any() or (df["open"] <= 0).any():
            raise ValueError(f"Precios inválidos (<= 0) detectados en velas de {symbol}.")

        latest_close_dt = df["_close_dt"].iloc[-1]
        if check_freshness:
            age_seconds = (current_utc - latest_close_dt).total_seconds()
            effective_max_age = (
                self.settings.max_data_age_seconds
                if self._is_crypto_mexc_symbol(symbol)
                else max(self.settings.max_data_age_seconds, 5 * 86400)
            )
            if age_seconds > effective_max_age:
                self.db.log_error(
                    error_code="STALE_DATA_REJECTED",
                    component="MEXCMarketClient",
                    message=(
                        f"Dato atrasado rechazado para {symbol}: antigüedad={int(age_seconds)}s "
                        f"> máximo={effective_max_age}s"
                    ),
                    payload_excerpt=f"latest_close={df['close_time'].iloc[-1]}",
                )
                raise StaleDataError(
                    f"Dato atrasado para {symbol} (antigüedad {int(age_seconds)}s > {effective_max_age}s)."
                )

        return df.drop(columns=["_close_dt"])
