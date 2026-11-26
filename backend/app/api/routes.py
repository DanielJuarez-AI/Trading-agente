"""
FastAPI API Routes - Endpoints de Observabilidad y Control Protegidos por Gateway.

Todos los endpoints de control (/api/control/*) pasan obligatoriamente por
`require_gateway_auth` (PrivateAuthGateway).
El frontend React NUNCA almacena GATEWAY_API_KEY en variables VITE_*.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.back_office.database import DuplicateCandleError
from backend.app.back_office.mexc_client import (
    MarketDataUnavailableError,
    StaleDataError,
    _parse_iso_utc,
)
from backend.app.back_office.paper_engine import (
    InsufficientFundsError,
    SystemStoppedError,
)
from backend.app.benchmark.comparator import run_three_lane_benchmark
from backend.app.core.config import EXPORTS_DIR, get_app_version_info
from backend.app.core.security import (
    SESSION_COOKIE_NAME,
    PrivateAuthGateway,
    require_gateway_auth,
)
from backend.app.front_office.features import _compute_rsi
from backend.app.front_office.laya_alpha import LLMServiceUnavailableError
from backend.app.middle_office.risk_engine import get_market_session_status
from backend.app.orchestrator.qwen_agent import QwenInvalidOutputError

router = APIRouter(prefix="/api", tags=["trading-agent"])


class LoginRequest(BaseModel):
    credential: str = Field(..., min_length=1, max_length=256)


class ControlPayload(BaseModel):
    symbol: Optional[str] = None
    mode: Optional[str] = None
    interval: Optional[str] = None
    qwen_mode: Optional[str] = None
    n_cycles: int = Field(default=25, ge=1, le=100)


@router.post("/auth/login")
def operator_login(body: LoginRequest, request: Request, response: Response) -> Dict[str, Any]:
    gateway: PrivateAuthGateway = request.app.state.gateway
    db = request.app.state.db
    if not gateway.authenticate_operator_secret(body.credential):
        db.log_error(
            error_code="AUTH_403_LOGIN_FAILED",
            component="PrivateAuthGateway",
            message="Intento de login con credencial de operador inválida.",
            payload_excerpt="endpoint=/api/auth/login",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Credencial de operador inválida.",
        )

    token = gateway.create_operator_session_token()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=86400,
    )
    return {"authenticated": True, "message": "Sesión de operador iniciada."}


@router.post("/auth/logout")
def operator_logout(response: Response) -> Dict[str, Any]:
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"authenticated": False}


@router.get("/auth/session")
def check_session(request: Request) -> Dict[str, Any]:
    gateway: PrivateAuthGateway = request.app.state.gateway
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    x_api_key = request.headers.get("X-API-Key")
    is_auth = gateway.verify_operator_session_token(cookie_token) or (
        x_api_key is not None and x_api_key == request.app.state.settings.gateway_api_key
    )
    return {"authenticated": bool(is_auth)}


def _parse_state_json(db: Any, key: str) -> Dict[str, Any]:
    raw = db.get_state(key)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


@router.get("/status")
def get_system_status(request: Request) -> Dict[str, Any]:
    settings = request.app.state.settings
    db = request.app.state.db
    paper_engine = request.app.state.paper_engine

    system_status = db.get_state("system_status", "RUNNING")
    active_ticker = db.get_state("active_ticker", settings.trading_symbol)
    source_mode = db.get_state("source_mode", "replay")
    candle_interval = db.get_state("candle_interval", settings.candle_interval) or settings.candle_interval
    qwen_mode = db.get_state("qwen_mode", settings.qwen_execution_mode) or settings.qwen_execution_mode
    portfolio_state = paper_engine.get_portfolio_state(symbol=active_ticker)
    metrics = paper_engine.compute_portfolio_metrics()
    laya1_feedback = _parse_state_json(db, "laya1_feedback_json")
    laya2_feedback = _parse_state_json(db, "laya2_feedback_json")
    stage_latencies_ms = _parse_state_json(db, "last_stage_latencies_json")
    ver_info = get_app_version_info()

    return {
        "app_version": ver_info["version"],
        "release_milestone": ver_info["release_milestone"],
        "commit_count": ver_info["commit_count"],
        "git_sha": ver_info["git_sha"],
        "system_status": system_status,
        "active_ticker": active_ticker,
        "source_mode": source_mode,
        "candle_interval": candle_interval,
        "qwen_mode": qwen_mode,
        "qwen_local_model": settings.qwen_local_model_name,
        "qwen_local_vram_gb": settings.qwen_local_vram_budget_gb,
        "laya1_drl_enabled": request.app.state.laya_alpha.drl_net is not None,
        "laya1_feedback": laya1_feedback,
        "laya2_feedback": laya2_feedback,
        "stage_latencies_ms": stage_latencies_ms,
        "daemon": request.app.state.scheduler.daemon_status(),
        "simulation_profile": settings.simulation_profile,
        "profile_config": {
            "name": settings.profile_config.name,
            "initial_capital": settings.profile_config.initial_capital,
            "fee_pct": settings.profile_config.fee_pct,
        },
        "enable_real_trading": settings.enable_real_trading,
        "primary_tickers": settings.primary_tickers,
        "allowed_symbols": settings.allowed_symbols,
        "hybrid_topology": settings.hybrid_topology_metadata(active_qwen_mode=qwen_mode),
        "portfolio": {
            "cash": round(portfolio_state.cash, 2),
            "position_qty": round(portfolio_state.position_qty, 6),
            "avg_entry_price": round(portfolio_state.avg_entry_price, 4),
            "last_price": round(portfolio_state.last_price, 4),
            "total_equity": round(portfolio_state.total_equity, 2),
            "peak_equity": round(portfolio_state.peak_equity, 2),
            "drawdown_pct": round(portfolio_state.drawdown_pct * 100.0, 4),
            "daily_pnl_pct": round(portfolio_state.daily_pnl_pct * 100.0, 4),
        },
        "metrics": metrics,
        "processed_candles_count": db.count_processed_candles(),
        "market_session": get_market_session_status(active_ticker),
        "us_equities_session": get_market_session_status("SPY"),
    }


@router.get("/decisions")
def get_decisions(request: Request, limit: int = 50) -> Dict[str, Any]:
    db = request.app.state.db
    return db.list_decisions_with_settlement(limit=min(limit, 200))


@router.get("/quotes")
def get_live_quotes(
    request: Request,
    symbols: Optional[str] = None,
) -> Dict[str, Any]:
    """Devuelve cotizaciones actuales en tiempo real (MEXC / Yahoo 1m) para la bitácora."""
    settings = request.app.state.settings
    db = request.app.state.db
    market_client = request.app.state.market_client

    sym_list: List[str] = []
    if symbols:
        for raw in symbols.split(","):
            s = raw.strip()
            if s in settings.allowed_symbols and s not in sym_list:
                sym_list.append(s)
    else:
        for p in db.list_positions():
            s = str(p["symbol"])
            if s in settings.allowed_symbols and s not in sym_list:
                sym_list.append(s)
        active = db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        if active in settings.allowed_symbols and active not in sym_list:
            sym_list.append(active)

    sym_list = sym_list[:30]
    quotes_map = market_client.fetch_realtime_quotes_batch(sym_list) if sym_list else {}

    for sym in sym_list:
        if sym not in quotes_map:
            pos = db.get_position(sym)
            if pos and float(pos.get("last_price") or 0.0) > 0:
                quotes_map[sym] = {
                    "price": round(float(pos["last_price"]), 4),
                    "change_pct_24h": None,
                    "source": "DB_POSITION",
                    "online": False,
                    "updated_at_utc": str(pos.get("updated_at") or ""),
                }

    # Recálculo periódico de TP/SL cada 10s para posiciones abiertas (minimización de pérdidas)
    recalculated: List[Dict[str, Any]] = []
    now_mono = time.monotonic()
    last_recalc = float(getattr(request.app.state, "_last_target_recalc_mono", 0.0) or 0.0)
    if (now_mono - last_recalc) >= 10.0:
        request.app.state._last_target_recalc_mono = now_mono
        try:
            tf = db.get_state("candle_interval", settings.candle_interval) or "1d"
            recalculated = request.app.state.paper_engine.recalculate_open_positions_targets(
                live_quotes=quotes_map,
                market_client=market_client,
                predictor=request.app.state.predictor,
                interval=tf,
            )
            changed = [r for r in recalculated if r.get("target_changed")]
            if changed and getattr(request.app.state, "event_bus", None) is not None:
                request.app.state.event_bus.publish(
                    "target_recalc",
                    {
                        "count": len(changed),
                        "symbols": [c["symbol"] for c in changed],
                        "updates": changed,
                    },
                )
        except Exception:
            recalculated = []

    return {
        "quotes": quotes_map,
        "recalculated_targets": recalculated,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@router.get("/portfolio")
def get_portfolio(request: Request) -> Dict[str, Any]:
    db = request.app.state.db
    paper_engine = request.app.state.paper_engine
    state = paper_engine.get_portfolio_state()
    return {
        "state": {
            "cash": round(state.cash, 2),
            "position_qty": round(state.position_qty, 6),
            "total_equity": round(state.total_equity, 2),
            "drawdown_pct": round(state.drawdown_pct * 100.0, 4),
            "active_symbol": state.active_symbol,
        },
        "metrics": paper_engine.compute_portfolio_metrics(),
        "snapshots": db.list_snapshots(limit=300),
        "orders": db.list_orders(limit=200),
    }


@router.get("/errors")
def get_errors(request: Request, limit: int = 100) -> Dict[str, Any]:
    db = request.app.state.db
    return {"items": db.list_errors(limit=min(limit, 300))}


def _compute_scanner_data(
    request: Request, interval: str, *, force_refresh: bool = False
) -> Dict[str, Any]:
    cache = getattr(request.app.state, "_scanner_cache", {})
    now_m = time.monotonic()
    cached = cache.get(interval)
    if not force_refresh and cached is not None and (now_m - cached["ts"]) < 15.0:
        return cached["data"]

    settings = request.app.state.settings
    market_client = request.app.state.market_client
    predictor = request.app.state.predictor
    laya_alpha = request.app.state.laya_alpha
    paper_engine = request.app.state.paper_engine

    rows: List[Dict[str, Any]] = []
    buy_cnt, hold_cnt, sell_cnt = 0, 0, 0
    active_mode = request.app.state.db.get_state("source_mode", "replay") or "replay"
    now_utc_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for sym in settings.primary_tickers:
        try:
            df = market_client.load_replay_candles(symbol=sym, interval=interval).copy()
            if len(df) < 35:
                continue
            if active_mode == "live":
                cached_px: Optional[float] = None
                with market_client._cache_lock:
                    if market_client._is_crypto_mexc_symbol(sym):
                        m_sym = market_client._resolve_mexc_symbol(sym)
                        m_entry = market_client._mexc_ticker_cache.get(m_sym)
                        if m_entry and m_entry.get("data"):
                            cached_px = float(m_entry["data"].get("last_price") or 0.0)
                    else:
                        eq_entry = market_client._equity_quote_cache.get(sym)
                        if eq_entry and eq_entry.get("data"):
                            cached_px = float(eq_entry["data"].get("last_price") or 0.0)
                if cached_px is not None and cached_px > 0:
                    last_i = df.index[-1]
                    df.at[last_i, "close"] = cached_px
                    df.at[last_i, "high"] = max(float(df.at[last_i, "high"]), cached_px)
                    df.at[last_i, "low"] = min(float(df.at[last_i, "low"]), cached_px)
            pr = predictor.predict_from_candles(df, symbol=sym, interval=interval)
            last_px = float(df["close"].iloc[-1])
            close_t = now_utc_iso if active_mode == "live" else str(df["close_time"].iloc[-1])
            port = paper_engine.get_portfolio_state(current_price=last_px, symbol=sym)
            pos_open = port.position_qty >= settings.min_lot_size
            l1 = laya_alpha.evaluate(
                prediction=pr,
                position_open=pos_open,
                gateway_key=settings.gateway_api_key,
                strict_live=False,
            )
            if l1.signal_action == "BUY":
                buy_cnt += 1
            elif l1.signal_action == "SELL":
                sell_cnt += 1
            else:
                hold_cnt += 1

            alpha_score = (pr.pred_return_t1 / max(pr.vol_20, 1e-4)) * l1.conviction_score
            rows.append(
                {
                    "symbol": sym,
                    "interval": interval,
                    "close_price": round(last_px, 4),
                    "candle_close_utc": close_t,
                    "pred_return_t1": round(pr.pred_return_t1, 6),
                    "pred_return_pct": round(pr.pred_return_t1 * 100.0, 4),
                    "signal_action": l1.signal_action,
                    "regime": l1.regime,
                    "conviction_score": round(l1.conviction_score, 4),
                    "kelly_alloc_pct": round(l1.recommended_exposure_pct * 100.0, 2),
                    "alpha_score": round(alpha_score, 4),
                    "position_open": bool(pos_open),
                    "model_detail": pr.model_detail,
                }
            )
        except Exception:
            continue

    rows.sort(
        key=lambda r: (
            2 if r["signal_action"] == "SELL" else (1 if r["signal_action"] == "BUY" else 0),
            r["alpha_score"],
        ),
        reverse=True,
    )
    data = {
        "interval": interval,
        "scanned_count": len(rows),
        "buy_signals_count": buy_cnt,
        "hold_signals_count": hold_cnt,
        "sell_signals_count": sell_cnt,
        "all_items": rows,
    }
    if not isinstance(getattr(request.app.state, "_scanner_cache", None), dict):
        request.app.state._scanner_cache = {}
    request.app.state._scanner_cache[interval] = {"ts": now_m, "data": data}
    return data


@router.get("/scanner")
def get_market_scanner(
    request: Request,
    interval: Optional[str] = None,
    top_n: int = 12,
) -> Dict[str, Any]:
    """Escáner transversal estilo Fincept Screener sobre los 30 tickers oficiales."""
    settings = request.app.state.settings
    db = request.app.state.db
    tf = interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
    if tf not in {"1h", "1d"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Temporalidad inválida.")
    snap = _compute_scanner_data(request, tf)
    clamped = max(1, min(int(top_n), 30))
    return {
        "interval": snap["interval"],
        "scanned_count": snap["scanned_count"],
        "buy_signals_count": snap["buy_signals_count"],
        "hold_signals_count": snap["hold_signals_count"],
        "sell_signals_count": snap["sell_signals_count"],
        "items": snap["all_items"][:clamped],
    }


@router.get("/audit/verify")
def verify_audit_chain(request: Request, limit: int = 1000) -> Dict[str, Any]:
    """Verifica la cadena criptográfica SHA-256 de decisiones (estilo autonomous-audit)."""
    return request.app.state.db.verify_decision_audit_chain(limit=min(limit, 5000))


@router.get("/audit/movements")
def get_audited_movements(request: Request, limit: int = 100) -> Dict[str, Any]:
    """Devuelve movimientos ejecutados auditados con TP/SL, PnL neto y hash SHA-256."""
    return request.app.state.db.list_audited_movements(limit=min(int(limit), 300))


@router.get("/news")
def get_ticker_news(
    request: Request,
    symbol: Optional[str] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Recupera noticias RSS reales, catalizadores y score de sentimiento para el ticker activo."""
    settings = request.app.state.settings
    db = request.app.state.db
    sym = symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
    if sym not in settings.allowed_symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Símbolo inválido '{sym}'. Permitidos: {settings.primary_tickers}",
        )
    active_mode = db.get_state("source_mode", "replay") or "replay"
    news_ctx = request.app.state.news_service.get_symbol_context(
        symbol=sym, source_mode="live", fetch_live=True, force_refresh=refresh
    )
    return {
        **news_ctx,
        "agent_source_mode": active_mode,
        "agent_injection_active": active_mode == "live",
    }


@router.get("/candles")
def get_ticker_candles(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = 200,
    mode: Optional[str] = None,
    interval: Optional[str] = None,
    include_forming: bool = True,
    realtime: bool = False,
) -> Dict[str, Any]:
    """Devuelve velas OHLCV (1h o 1d), EMA10/EMA20, RSI-14, marcadores, telemetría MEXC en vivo y pronóstico."""
    settings = request.app.state.settings
    db = request.app.state.db
    market_client = request.app.state.market_client

    sym = symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
    if sym not in settings.allowed_symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Símbolo inválido '{sym}'. Permitidos: {settings.primary_tickers}",
        )

    tf = interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
    if tf not in {"1h", "1d"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Temporalidad inválida '{tf}'. Opciones permitidas: '1h' o '1d'.",
        )

    clamped_limit = max(1, min(int(limit), 300))
    effective_mode = mode or db.get_state("source_mode", "replay") or "replay"
    use_live = (effective_mode == "live") or bool(realtime)
    is_mexc_native = sym.upper() in {"BTC-USD", "BTCUSDT"}
    data_source = "replay"
    mexc_realtime: Optional[Dict[str, Any]] = None

    if use_live:
        try:
            df = market_client.fetch_live_candles(
                symbol=sym,
                interval=tf,
                limit=max(clamped_limit + 50, 250),
                include_forming=bool(include_forming),
            )
            mexc_target = sym if is_mexc_native else "BTCUSDT"
            mexc_realtime = market_client.fetch_mexc_realtime_ticker(mexc_target)
            if is_mexc_native:
                data_source = "MEXC API /api/v3/klines (LIVE)"
            elif mexc_realtime and mexc_realtime.get("mexc_online"):
                data_source = "MEXC Live Stream + Market Feed (LIVE)"
            else:
                data_source = "Market Live Feed (LIVE)"
        except Exception:
            df = market_client.load_replay_candles(symbol=sym, interval=tf)
            data_source = "replay_fallback"
            use_live = False
    else:
        df = market_client.load_replay_candles(symbol=sym, interval=tf)

    df = df.copy()
    close_series = df["close"].astype(float)
    df["ema10"] = close_series.ewm(span=10, adjust=False).mean()
    df["ema20"] = close_series.ewm(span=20, adjust=False).mean()
    df["rsi_14"] = _compute_rsi(close_series, period=14).fillna(50.0)

    view_df = df.tail(clamped_limit).reset_index(drop=True)

    candles_by_time: Dict[Any, Dict[str, Any]] = {}
    close_to_chart_time: Dict[str, Any] = {}
    for _, row in view_df.iterrows():
        if any(pd.isna(row[k]) for k in ("open", "high", "low", "close")):
            continue
        open_iso = str(row["open_time"])
        close_iso = str(row["close_time"])
        if tf == "1h":
            chart_time: Any = int(_parse_iso_utc(open_iso).timestamp())
            label = open_iso[:16].replace("T", " ") + " UTC"
        else:
            chart_time = close_iso[:10]
            label = close_iso[:10]
        close_to_chart_time[close_iso] = chart_time
        candles_by_time[chart_time] = {
            "time": chart_time,
            "label": label,
            "open_time": open_iso,
            "close_time": close_iso,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": round(float(row["volume"]), 2),
            "ema10": round(float(row["ema10"]), 4),
            "ema20": round(float(row["ema20"]), 4),
            "rsi_14": round(float(row["rsi_14"]), 2),
        }
    candles = sorted(candles_by_time.values(), key=lambda c: c["time"])

    last_close = float(candles[-1]["close"]) if candles else 0.0
    prev_close = float(candles[-2]["close"]) if len(candles) >= 2 else last_close
    change_pct_1d = ((last_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    high_period = max((float(c["high"]) for c in candles), default=0.0)
    low_period = min((float(c["low"]) for c in candles), default=0.0)
    last_rsi_14 = float(candles[-1]["rsi_14"]) if candles else 50.0

    # Marcadores cronológicos de operaciones BUY/SELL del agente en el ticker y temporalidad
    markers_by_time: Dict[Any, Dict[str, Any]] = {}
    recent_decisions = db.list_decisions(limit=200)
    latest_coherence_prob = 0.91
    found_latest_coherence = False
    for dec in recent_decisions:
        if dec.get("symbol") != sym:
            continue
        if not found_latest_coherence:
            laya2_obj = dec.get("laya2_json")
            if isinstance(laya2_obj, dict) and laya2_obj.get("coherence_prob") is not None:
                latest_coherence_prob = float(laya2_obj["coherence_prob"])
                found_latest_coherence = True
        if dec.get("interval") != tf:
            continue
        act = dec.get("final_action")
        if act not in {"BUY", "SELL"}:
            continue
        c_utc = str(dec.get("candle_close_utc", ""))
        mapped_t = close_to_chart_time.get(c_utc)
        if mapped_t is not None and mapped_t not in markers_by_time:
            markers_by_time[mapped_t] = {
                "time": mapped_t,
                "action": act,
                "risk_status": dec.get("risk_status", "APPROVED"),
                "model_pred": round(float(dec.get("model_pred", 0.0)), 6),
            }
    markers = sorted(markers_by_time.values(), key=lambda m: m["time"])

    horizon_label = "+1H" if tf == "1h" else "+1D"

    # Pronóstico en vivo del agente (Quant v2 + LSTM T4 + Laya #1 + Coherencia Laya #2)
    agent_forecast: Optional[Dict[str, Any]] = None
    if len(df) >= 35 and candles:
        prediction = request.app.state.predictor.predict_from_candles(
            df, symbol=sym, interval=tf
        )
        portfolio = request.app.state.paper_engine.get_portfolio_state(
            current_price=last_close, symbol=sym
        )
        position_open = (
            portfolio.position_qty >= settings.min_lot_size
            and portfolio.active_symbol == sym
        )
        laya1 = request.app.state.laya_alpha.evaluate(
            prediction=prediction,
            position_open=position_open,
            gateway_key=settings.gateway_api_key,
            strict_live=False,
        )
        if tf == "1h":
            next_time: Any = int(candles[-1]["time"]) + 3600
        else:
            next_time = (
                datetime.strptime(str(candles[-1]["time"]), "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        projected_close_t1 = last_close * (1.0 + prediction.pred_return_t1)
        proj_rsi_series = _compute_rsi(
            pd.concat([close_series, pd.Series([projected_close_t1])], ignore_index=True),
            period=14,
        )
        proj_rsi_raw = proj_rsi_series.iloc[-1]
        proj_rsi = float(proj_rsi_raw) if pd.notna(proj_rsi_raw) else float(prediction.rsi_14)
        signal_strength = min(
            1.0, abs(prediction.pred_return_t1) / max(prediction.vol_20, 1e-6)
        )
        fc_upper = round(projected_close_t1 * (1.0 + prediction.vol_20), 4)
        fc_lower = round(max(0.0, projected_close_t1 * (1.0 - prediction.vol_20)), 4)
        agent_forecast = {
            "symbol": sym,
            "interval": tf,
            "horizon_label": horizon_label,
            "next_time": next_time,
            "pred_return_t1": round(prediction.pred_return_t1, 6),
            "pred_return_pct": round(prediction.pred_return_t1 * 100.0, 4),
            "direction": prediction.direction,
            "projected_close_t1": round(projected_close_t1, 4),
            "projected_rsi_t1": round(proj_rsi, 2),
            "band_upper_t1": fc_upper,
            "band_lower_t1": fc_lower,
            "forecast_candle": {
                "time": next_time,
                "open": round(last_close, 4),
                "high": round(max(last_close, projected_close_t1, fc_upper), 4),
                "low": round(min(last_close, projected_close_t1, fc_lower), 4),
                "close": round(projected_close_t1, 4),
            },
            "vol_20": round(prediction.vol_20, 6),
            "ema_ratio": round(prediction.ema_ratio, 6),
            "rsi_14": round(prediction.rsi_14, 2),
            "signal_action": laya1.signal_action,
            "signal_probs": laya1.signal_probs,
            "regime": laya1.regime,
            "regime_probs": laya1.regime_probs,
            "conviction_level": laya1.conviction_level,
            "conviction_score": round(laya1.conviction_score, 4),
            "coherence_prob": round(latest_coherence_prob, 4),
            "signal_strength": round(signal_strength, 4),
        }

    return {
        "symbol": sym,
        "interval": tf,
        "horizon_label": horizon_label,
        "data_source": data_source,
        "is_realtime": bool(use_live),
        "is_mexc_native": bool(is_mexc_native),
        "mexc_realtime": mexc_realtime,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_close": round(last_close, 4),
        "prev_close": round(prev_close, 4),
        "change_pct_1d": round(change_pct_1d, 4),
        "high_period": round(high_period, 4),
        "low_period": round(low_period, 4),
        "last_rsi_14": round(last_rsi_14, 2),
        "candles": candles,
        "markers": markers,
        "agent_forecast": agent_forecast,
    }


@router.get("/benchmark")
def get_benchmark(
    request: Request,
    interval: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    summary_path = EXPORTS_DIR / "benchmark_summary.json"
    if interval is None and symbol is None and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    settings = request.app.state.settings
    db = request.app.state.db
    active_ticker = symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
    tf = interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
    return run_three_lane_benchmark(
        symbol=active_ticker,
        n_cycles=25,
        settings=settings,
        predictor=request.app.state.predictor,
        export_csv=True,
        interval=tf,
    )


@router.post("/control/{action}")
async def control_action(
    action: str,
    request: Request,
    payload: Optional[ControlPayload] = None,
    _auth_mode: str = Depends(require_gateway_auth),
) -> Dict[str, Any]:
    settings = request.app.state.settings
    db = request.app.state.db
    scheduler = request.app.state.scheduler
    body = payload or ControlPayload()

    if action == "start":
        db.set_state("system_status", "RUNNING")
        if db.get_state("qwen_mode", settings.qwen_execution_mode) == "LOCAL_DEVICE":
            threading.Thread(
                target=request.app.state.qwen_agent.warmup_local_model, daemon=True
            ).start()
        daemon_st = await scheduler.start_daemon(interval_seconds=25)
        return {
            "status": "RUNNING",
            "daemon": daemon_st,
            "message": "Agente y daemon autónomo iniciados (RUNNING).",
        }

    if action == "pause":
        db.set_state("system_status", "PAUSED")
        daemon_st = await scheduler.stop_daemon()
        return {"status": "PAUSED", "daemon": daemon_st, "message": "Agente pausado (PAUSED)."}

    if action == "stop":
        # Detener bloquea nuevas órdenes pero NO liquida automáticamente la posición abierta
        db.set_state("system_status", "STOPPED")
        daemon_st = await scheduler.stop_daemon()
        state = request.app.state.paper_engine.get_portfolio_state()
        return {
            "status": "STOPPED",
            "daemon": daemon_st,
            "message": "Kill-Switch STOP activado: nuevas órdenes bloqueadas, posición preservada.",
            "preserved_position_qty": state.position_qty,
        }

    if action == "reset_portfolio":
        await scheduler.stop_daemon()
        reset_res = db.reset_paper_portfolio(settings.profile_config.initial_capital)
        if hasattr(request.app.state, "_scanner_cache"):
            request.app.state._scanner_cache = {}
        return {
            "status": "RUNNING",
            **reset_res,
            "message": (
                f"Cartera Paper restaurada a ${settings.profile_config.initial_capital:,.2f}, "
                "estado RUNNING y cadena SHA-256 reiniciada."
            ),
        }

    if action == "liquidate":
        sym = body.symbol
        if not sym or sym not in settings.allowed_symbols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Símbolo inválido para liquidar.",
            )
        pos = db.get_position(sym)
        if pos is None or float(pos.get("quantity") or 0.0) < settings.min_lot_size:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"No hay posición abierta en {sym}; ya fue liquidada.",
            )
        live_q = request.app.state.market_client.fetch_realtime_quotes_batch([sym]).get(sym)
        px = float(live_q["price"]) if live_q else float(pos.get("last_price") or 0.0)
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        try:
            res = scheduler.manual_liquidate(symbol=sym, market_price=px, interval=tf)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {
            **res,
            "message": (
                f"{sym} liquidado @ ${res['exec_price']:,.4f} "
                f"({res['quantity']:.4f} u.) • PnL realizado ${res['realized_pnl']:+,.2f}"
            ),
        }

    if action == "liquidate_all":
        open_positions = [
            p for p in db.list_positions()
            if float(p.get("quantity") or 0.0) >= settings.min_lot_size
        ]
        if not open_positions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No hay posiciones abiertas para liquidar.",
            )
        syms = [p["symbol"] for p in open_positions]
        live_quotes = request.app.state.market_client.fetch_realtime_quotes_batch(syms)
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        liquidated = []
        skipped = []
        for pos in open_positions:
            sym = pos["symbol"]
            live_q = live_quotes.get(sym)
            px = float(live_q["price"]) if live_q and live_q.get("price") else float(pos.get("last_price") or 0.0)
            try:
                res = scheduler.manual_liquidate(symbol=sym, market_price=px, interval=tf)
                liquidated.append(res)
            except ValueError as exc:
                skipped.append({"symbol": sym, "reason": str(exc)})
        if not liquidated and skipped:
            reasons = "; ".join(f"{s['symbol']}: {s['reason']}" for s in skipped)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"No se pudo liquidar ninguna posición abierta ({reasons}).",
            )
        total_pnl = round(sum(float(r.get("realized_pnl") or 0.0) for r in liquidated), 2)
        msg = f"{len(liquidated)} posición(es) liquidada(s) • PnL realizado ${total_pnl:+,.2f}"
        if skipped:
            msg += f" • Omitidas por mercado cerrado: {', '.join(s['symbol'] for s in skipped)}"
        return {
            "liquidated_count": len(liquidated),
            "skipped_count": len(skipped),
            "total_realized_pnl": total_pnl,
            "liquidated": liquidated,
            "skipped": skipped,
            "message": msg,
        }

    if action == "switch_ticker":
        if not body.symbol or body.symbol not in settings.allowed_symbols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ticker inválido. Permitidos: {settings.primary_tickers}",
            )
        db.set_state("active_ticker", body.symbol)
        return {"active_ticker": body.symbol, "message": f"Ticker cambiado a {body.symbol}."}

    if action == "switch_mode":
        if body.mode not in {"live", "replay"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Modo inválido. Opciones: 'live' o 'replay'.",
            )
        db.set_state("source_mode", body.mode)
        return {"source_mode": body.mode, "message": f"Modo de datos cambiado a {body.mode}."}

    if action == "switch_interval":
        if body.interval not in {"1h", "1d"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Temporalidad inválida. Opciones: '1h' (Hora) o '1d' (Día).",
            )
        db.set_state("candle_interval", body.interval)
        return {
            "candle_interval": body.interval,
            "message": f"Temporalidad cambiada a {'Hora (1h)' if body.interval == '1h' else 'Día (1d)'}.",
        }

    if action == "switch_qwen_mode":
        raw_qm = (body.qwen_mode or "").strip()
        norm_map = {
            "api": "EXTERNAL_API",
            "EXTERNAL_API": "EXTERNAL_API",
            "local": "LOCAL_DEVICE",
            "LOCAL_DEVICE": "LOCAL_DEVICE",
        }
        if raw_qm not in norm_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Modo de Qwen inválido. Opciones: 'EXTERNAL_API' ('api') o 'LOCAL_DEVICE' ('local').",
            )
        target_mode = norm_map[raw_qm]
        db.set_state("qwen_mode", target_mode)
        if target_mode == "LOCAL_DEVICE":
            threading.Thread(
                target=request.app.state.qwen_agent.warmup_local_model, daemon=True
            ).start()
        topo = settings.hybrid_topology_metadata(active_qwen_mode=target_mode)
        return {
            "qwen_mode": target_mode,
            "topology": topo["qwen_orchestrator"],
            "message": (
                f"Qwen cambiado a Local 6GB VRAM ({settings.qwen_local_model_name})."
                if target_mode == "LOCAL_DEVICE"
                else f"Qwen cambiado a Vía API Externa ({settings.qwen_model_name})."
            ),
        }

    if action == "recalculate_targets":
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        open_pos = db.list_positions()
        syms = [str(p["symbol"]) for p in open_pos]
        quotes_map = (
            request.app.state.market_client.fetch_realtime_quotes_batch(syms)
            if syms
            else {}
        )
        updated = request.app.state.paper_engine.recalculate_open_positions_targets(
            live_quotes=quotes_map,
            market_client=request.app.state.market_client,
            predictor=request.app.state.predictor,
            interval=tf,
        )
        request.app.state._last_target_recalc_mono = time.monotonic()
        if updated and getattr(request.app.state, "event_bus", None) is not None:
            request.app.state.event_bus.publish(
                "target_recalc",
                {
                    "count": len(updated),
                    "symbols": [u["symbol"] for u in updated],
                    "updates": updated,
                },
            )
        return {
            "recalculated_count": len(updated),
            "updates": updated,
            "message": f"Objetivos TP/SL recalculados en {len(updated)} posiciones abiertas.",
        }

    if action == "step":
        sym = body.symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        mode = body.mode or db.get_state("source_mode", "replay") or "replay"
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        try:
            if mode == "live":
                mkt = get_market_session_status(sym)
                if not mkt["is_open"]:
                    db.log_error(
                        error_code="RISK_MARKET_CLOSED",
                        component="ControlGateway",
                        message=f"Ciclo live bloqueado en {sym}: {mkt['session_label']}",
                        payload_excerpt=f"symbol={sym}, action=step",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Mercado cerrado para {sym} ({mkt['session_label']}). No se permiten operaciones.",
                    )
                res = scheduler.run_single_live_cycle(
                    symbol=sym, gateway_key=settings.gateway_api_key, interval=tf
                )
                return {"mode": "live", "interval": tf, "cycle": res}
            else:
                # En replay avanza 1 nueva vela no procesada dentro de la ventana visible de 200 velas
                full_df = request.app.state.market_client.load_replay_candles(
                    symbol=sym, interval=tf
                )
                start_search_idx = max(45, len(full_df) - 200)
                for idx in range(start_search_idx, len(full_df)):
                    close_t = str(full_df.iloc[idx]["close_time"])
                    if not db.is_candle_processed(sym, tf, close_t, "replay"):
                        res = scheduler.process_candle_window(
                            candles_df=full_df.iloc[: idx + 1].reset_index(drop=True),
                            symbol=sym,
                            interval=tf,
                            source_mode="replay",
                            gateway_key=settings.gateway_api_key,
                        )
                        return {"mode": "replay", "interval": tf, "cycle": res}
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Todas las velas ({tf}) de replay para {sym} ya fueron procesadas (idempotencia).",
                )
        except DuplicateCandleError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (StaleDataError, QwenInvalidOutputError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except (MarketDataUnavailableError, LLMServiceUnavailableError) as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except (InsufficientFundsError, SystemStoppedError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if action == "replay":
        sym = body.symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        cycles = scheduler.run_replay_cycles(
            symbol=sym,
            n_cycles=body.n_cycles,
            gateway_key=settings.gateway_api_key,
            interval=tf,
        )
        return {
            "mode": "replay",
            "symbol": sym,
            "interval": tf,
            "cycles_executed": len(cycles),
            "last_cycle": cycles[-1] if cycles else None,
        }

    if action == "scan_and_step":
        mode = body.mode or db.get_state("source_mode", "replay") or "replay"
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        snap = _compute_scanner_data(request, tf, force_refresh=True)
        items = snap["all_items"]
        if not items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sin datos suficientes en el scanner para temporalidad {tf}.",
            )

        chosen_item = items[0]
        if mode == "replay":
            for cand in items:
                if not db.is_candle_processed(cand["symbol"], tf, cand["candle_close_utc"], "replay"):
                    chosen_item = cand
                    break
        else:
            open_candidates = [c for c in items if get_market_session_status(c["symbol"])["is_open"]]
            if not open_candidates:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Todos los mercados de acciones US están cerrados; no se permiten operaciones live en este horario.",
                )
            chosen_item = open_candidates[0]

        chosen_sym = chosen_item["symbol"]
        db.set_state("active_ticker", chosen_sym)
        try:
            if mode == "live":
                res = scheduler.run_single_live_cycle(
                    symbol=chosen_sym, gateway_key=settings.gateway_api_key, interval=tf
                )
            else:
                full_df = request.app.state.market_client.load_replay_candles(
                    symbol=chosen_sym, interval=tf
                )
                start_search_idx = max(45, len(full_df) - 200)
                res = None
                for idx in range(len(full_df) - 1, start_search_idx - 1, -1):
                    close_t = str(full_df.iloc[idx]["close_time"])
                    if not db.is_candle_processed(chosen_sym, tf, close_t, "replay"):
                        res = scheduler.process_candle_window(
                            candles_df=full_df.iloc[: idx + 1].reset_index(drop=True),
                            symbol=chosen_sym,
                            interval=tf,
                            source_mode="replay",
                            gateway_key=settings.gateway_api_key,
                        )
                        break
                if res is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Velas recientes de {chosen_sym} ({tf}) ya procesadas en replay.",
                    )
            if hasattr(request.app.state, "_scanner_cache"):
                request.app.state._scanner_cache = {}
            return {
                "mode": mode,
                "interval": tf,
                "selected_symbol": chosen_sym,
                "scanner_summary": {
                    "buy_signals_count": snap["buy_signals_count"],
                    "hold_signals_count": snap["hold_signals_count"],
                    "sell_signals_count": snap["sell_signals_count"],
                },
                "cycle": res,
                "message": (
                    f"Auto-Scan eligió {chosen_sym} ({chosen_item['signal_action']}, "
                    f"r_hat={chosen_item['pred_return_pct']:+.2f}%) ➔ {res['risk']['final_action']}."
                ),
            }
        except DuplicateCandleError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (StaleDataError, QwenInvalidOutputError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except (MarketDataUnavailableError, LLMServiceUnavailableError) as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except (InsufficientFundsError, SystemStoppedError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if action == "benchmark":
        sym = body.symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        return run_three_lane_benchmark(
            symbol=sym,
            n_cycles=body.n_cycles,
            settings=settings,
            predictor=request.app.state.predictor,
            export_csv=True,
            interval=tf,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Acción de control desconocida: {action}",
    )


@router.get("/telemetry")
def get_system_telemetry(request: Request) -> Dict[str, Any]:
    """Devuelve telemetría hardware GPU/VRAM, modelos Ollama activos y estado de agentes."""
    telemetry = request.app.state.telemetry
    scheduler = request.app.state.scheduler
    return telemetry.get_snapshot(daemon_state=scheduler.daemon_status())


@router.get("/stream/events")
async def stream_pipeline_events(request: Request) -> StreamingResponse:
    """Flujo Server-Sent Events (SSE) con los pasos en vivo del pipeline multi-agente."""
    event_bus = request.app.state.event_bus

    async def event_generator():
        async for ev in event_bus.stream():
            if await request.is_disconnected():
                break
            ev_name = ev.get("event", "message")
            data_json = json.dumps(ev, ensure_ascii=False)
            yield f"event: {ev_name}\ndata: {data_json}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
