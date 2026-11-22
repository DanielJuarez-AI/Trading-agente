"""
FastAPI API Routes - Endpoints de Observabilidad y Control Protegidos por Gateway.

Todos los endpoints de control (/api/control/*) pasan obligatoriamente por
`require_gateway_auth` (PrivateAuthGateway).
El frontend React NUNCA almacena GATEWAY_API_KEY en variables VITE_*.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
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
from backend.app.core.config import EXPORTS_DIR
from backend.app.core.security import (
    SESSION_COOKIE_NAME,
    PrivateAuthGateway,
    require_gateway_auth,
)
from backend.app.front_office.laya_alpha import LLMServiceUnavailableError
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
    portfolio_state = paper_engine.get_portfolio_state()
    metrics = paper_engine.compute_portfolio_metrics()
    laya1_feedback = _parse_state_json(db, "laya1_feedback_json")
    laya2_feedback = _parse_state_json(db, "laya2_feedback_json")
    stage_latencies_ms = _parse_state_json(db, "last_stage_latencies_json")

    return {
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
    }


@router.get("/decisions")
def get_decisions(request: Request, limit: int = 50) -> Dict[str, Any]:
    db = request.app.state.db
    return {"items": db.list_decisions(limit=min(limit, 200))}


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


@router.get("/candles")
def get_ticker_candles(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = 140,
    mode: Optional[str] = None,
    interval: Optional[str] = None,
) -> Dict[str, Any]:
    """Devuelve velas OHLCV cerradas (1h o 1d), EMA10/EMA20, marcadores y pronóstico del agente."""
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

    effective_mode = mode or db.get_state("source_mode", "replay") or "replay"
    data_source = "replay"
    if effective_mode == "live":
        try:
            df = market_client.fetch_live_candles(symbol=sym, interval=tf, limit=max(limit, 140))
            data_source = "live"
        except Exception:
            df = market_client.load_replay_candles(symbol=sym, interval=tf)
            data_source = "replay_fallback"
    else:
        df = market_client.load_replay_candles(symbol=sym, interval=tf)

    df = df.copy()
    close_series = df["close"].astype(float)
    df["ema10"] = close_series.ewm(span=10, adjust=False).mean()
    df["ema20"] = close_series.ewm(span=20, adjust=False).mean()

    clamped_limit = max(1, min(int(limit), 300))
    view_df = df.tail(clamped_limit).reset_index(drop=True)

    candles = []
    close_to_chart_time: Dict[str, Any] = {}
    for _, row in view_df.iterrows():
        open_iso = str(row["open_time"])
        close_iso = str(row["close_time"])
        if tf == "1h":
            chart_time: Any = int(_parse_iso_utc(open_iso).timestamp())
            label = open_iso[:16].replace("T", " ") + " UTC"
        else:
            chart_time = close_iso[:10]
            label = close_iso[:10]
        close_to_chart_time[close_iso] = chart_time
        candles.append(
            {
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
            }
        )

    last_close = float(view_df["close"].iloc[-1]) if not view_df.empty else 0.0
    prev_close = float(view_df["close"].iloc[-2]) if len(view_df) >= 2 else last_close
    change_pct_1d = ((last_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    high_period = float(view_df["high"].max()) if not view_df.empty else 0.0
    low_period = float(view_df["low"].min()) if not view_df.empty else 0.0

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

    # Pronóstico en vivo del agente (LSTM T4 + Laya #1 + Coherencia Laya #2)
    agent_forecast: Optional[Dict[str, Any]] = None
    if len(df) >= 35 and candles:
        prediction = request.app.state.predictor.predict_from_candles(df, symbol=sym)
        portfolio = request.app.state.paper_engine.get_portfolio_state(current_price=last_close)
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
        "last_close": round(last_close, 4),
        "prev_close": round(prev_close, 4),
        "change_pct_1d": round(change_pct_1d, 4),
        "high_period": round(high_period, 4),
        "low_period": round(low_period, 4),
        "candles": candles,
        "markers": markers,
        "agent_forecast": agent_forecast,
    }


@router.get("/benchmark")
def get_benchmark(request: Request) -> Dict[str, Any]:
    summary_path = EXPORTS_DIR / "benchmark_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    settings = request.app.state.settings
    db = request.app.state.db
    active_ticker = db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
    return run_three_lane_benchmark(
        symbol=active_ticker,
        n_cycles=25,
        settings=settings,
        predictor=request.app.state.predictor,
        export_csv=True,
    )


@router.post("/control/{action}")
def control_action(
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
        return {"status": "RUNNING", "message": "Agente iniciado (RUNNING)."}

    if action == "pause":
        db.set_state("system_status", "PAUSED")
        return {"status": "PAUSED", "message": "Agente pausado (PAUSED)."}

    if action == "stop":
        # Detener bloquea nuevas órdenes pero NO liquida automáticamente la posición abierta
        db.set_state("system_status", "STOPPED")
        state = request.app.state.paper_engine.get_portfolio_state()
        return {
            "status": "STOPPED",
            "message": "Kill-Switch STOP activado: nuevas órdenes bloqueadas, posición preservada.",
            "preserved_position_qty": state.position_qty,
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

    if action == "step":
        sym = body.symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        mode = body.mode or db.get_state("source_mode", "replay") or "replay"
        tf = body.interval or db.get_state("candle_interval", settings.candle_interval) or "1d"
        try:
            if mode == "live":
                res = scheduler.run_single_live_cycle(
                    symbol=sym, gateway_key=settings.gateway_api_key, interval=tf
                )
                return {"mode": "live", "interval": tf, "cycle": res}
            else:
                # En replay avanza 1 nueva vela no procesada del histórico en la temporalidad activa
                full_df = request.app.state.market_client.load_replay_candles(
                    symbol=sym, interval=tf
                )
                for idx in range(45, len(full_df)):
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

    if action == "benchmark":
        sym = body.symbol or db.get_state("active_ticker", settings.trading_symbol) or settings.trading_symbol
        return run_three_lane_benchmark(
            symbol=sym,
            n_cycles=body.n_cycles,
            settings=settings,
            predictor=request.app.state.predictor,
            export_csv=True,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Acción de control desconocida: {action}",
    )
