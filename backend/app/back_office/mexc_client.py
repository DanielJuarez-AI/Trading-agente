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
        self, symbol: str, interval: str, limit: int = 120
    ) -> List[List[Any]]:
        """
        Consulta pública GET /api/v3/klines con máximo 3 intentos y espera exponencial.
        """
        mexc_symbol = self._resolve_mexc_symbol(symbol)
        url = f"{self.settings.mexc_public_base_url.rstrip('/')}/api/v3/klines"
        params = {"symbol": mexc_symbol, "interval": interval, "limit": limit}

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

    def fetch_live_candles(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: int = 120,
        now_utc: Optional[datetime] = None,
        enforce_mexc_for_crypto: bool = True,
    ) -> pd.DataFrame:
        """
        Obtiene velas históricas cerradas en modo 'live', validando orden cronológico,
        descarte estricto de vela abierta y frescura (MAX_DATA_AGE_SECONDS).
        """
        if self.settings.enable_real_trading:
            raise RuntimeError("Real trading is strictly disabled.")
        if symbol not in self.settings.allowed_symbols:
            raise ValueError(f"Símbolo fuera de lista blanca: {symbol}")

        tf = interval or self.settings.candle_interval
        current_utc = now_utc or datetime.now(timezone.utc)
        rows: List[Dict[str, Any]] = []

        if self._is_crypto_mexc_symbol(symbol) and enforce_mexc_for_crypto:
            raw_klines = self._fetch_mexc_klines_with_backoff(symbol, tf, limit=limit)
            for k in raw_klines:
                open_ms = int(k[0])
                close_ms = int(k[6])
                close_dt = datetime.fromtimestamp(close_ms / 1000.0, tz=timezone.utc)
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
            try:
                hist = yf.download(
                    tickers=symbol,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if hist.empty:
                    raise ValueError(f"Sin velas para {symbol} desde proveedor público.")
                hist = hist.reset_index()
                date_col = "Date" if "Date" in hist.columns else hist.columns[0]
                for _, row in hist.iterrows():
                    dt = pd.to_datetime(row[date_col], utc=True)
                    close_dt = dt + timedelta(hours=23, minutes=59, seconds=59)
                    if close_dt > current_utc:
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "interval": "1d",
                            "open_time": dt.strftime("%Y-%m-%dT00:00:00Z"),
                            "close_time": close_dt.strftime("%Y-%m-%dT23:59:59Z"),
                            "open": float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"]),
                            "high": float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"]),
                            "low": float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"]),
                            "close": float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"]),
                            "volume": float(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"]),
                            "source_mode": "live",
                        }
                    )
            except Exception as exc:
                self.db.log_error(
                    error_code="MARKET_DATA_FETCH_ERROR",
                    component="MEXCMarketClient",
                    message=f"Error obteniendo datos públicos para {symbol}: {exc}",
                    payload_excerpt=f"symbol={symbol}",
                )
                self.db.set_state("system_status", "PAUSED")
                raise MarketDataUnavailableError(str(exc)) from exc

        df = self._validate_and_sort_candles(
            rows, symbol=symbol, current_utc=current_utc, check_freshness=True
        )
        return df

    def load_replay_candles(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """
        Carga velas históricas cerradas desde cache local (data/replay_candles.json)
        para evaluación reproducible en modo Historical Replay sin simular frecuencia artificial.
        """
        if symbol not in self.settings.allowed_symbols:
            raise ValueError(f"Símbolo fuera de lista blanca: {symbol}")

        canonical = "BTC-USD" if symbol == "BTCUSDT" else symbol
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

        rows = []
        for item in cache[canonical]:
            rows.append(
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
            )
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
            if age_seconds > self.settings.max_data_age_seconds:
                self.db.log_error(
                    error_code="STALE_DATA_REJECTED",
                    component="MEXCMarketClient",
                    message=(
                        f"Dato atrasado rechazado para {symbol}: antigüedad={int(age_seconds)}s "
                        f"> máximo={self.settings.max_data_age_seconds}s"
                    ),
                    payload_excerpt=f"latest_close={df['close_time'].iloc[-1]}",
                )
                raise StaleDataError(
                    f"Dato atrasado para {symbol} (antigüedad {int(age_seconds)}s > {self.settings.max_data_age_seconds}s)."
                )

        return df.drop(columns=["_close_dt"])
