"""
Core Scheduler - Orquestador del Ciclo Completo Multi-Agente + Eventos SSE + Daemon Autónomo.

Flujo:
1. Back Office: Ingesta de vela cerrada e idempotencia SQLite
2. Front Office: Features causales + Quant v2 (XGBoost Walk-Forward 1h/1d + LSTM T4)
2b. Front Office: Contexto de Noticias/Sentimiento + Memoria Episódica (FinMem)
3. Front Office [LOCAL]: Laya #1 (Dueling Double-DQN + A2C)
4. Orquestador [API EXTERNA o LOCAL 3B]: Qwen + Debate Bull/Bear + Feedback Real
4b. Retroalimentación Qwen -> Laya #1 y Laya #2
5. Middle Office [LOCAL]: Laya #2 Guardia Semántica (coherencia vs umbral + anti-inyección)
6. Middle Office [PYTHON DETERMINISTA]: Risk Engine + Stop-Loss / Trailing Stop / Take-Profit
7. Back Office: Simulación Paper Multi-Ticker + Memoria Post-Trade + Registro Atómico SQLite WAL
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.back_office.database import DuplicateCandleError, TradingDatabase, utc_now_iso
from backend.app.back_office.mexc_client import MEXCMarketClient
from backend.app.back_office.paper_engine import (
    InsufficientFundsError,
    PaperTradingEngine,
    SystemStoppedError,
)
from backend.app.core.config import Settings
from backend.app.core.events import PipelineEventBus
from backend.app.core.security import PrivateAuthGateway
from backend.app.core.telemetry import SystemTelemetryService
from backend.app.front_office.laya_alpha import (
    LAYA1_EXECUTION_MODE,
    LayaAlphaAgent,
)
from backend.app.front_office.news_sentiment import NewsSentimentService
from backend.app.front_office.predictor import QuantPredictor
from backend.app.middle_office.laya_guard import LayaGuardAgent
from backend.app.middle_office.risk_engine import (
    DeterministicRiskEngine,
    get_market_session_status,
)
from backend.app.orchestrator.qwen_agent import (
    QWEN_EXECUTION_LOCATION,
    QwenOrchestratorAgent,
)


class TradingScheduler:
    """Ejecutor del ciclo multi-agente en modo Live, Historical Replay y Daemon Autónomo."""

    def __init__(
        self,
        settings: Settings,
        db: TradingDatabase,
        gateway: PrivateAuthGateway,
        market_client: MEXCMarketClient,
        predictor: QuantPredictor,
        laya_alpha: LayaAlphaAgent,
        qwen_agent: QwenOrchestratorAgent,
        laya_guard: LayaGuardAgent,
        risk_engine: DeterministicRiskEngine,
        paper_engine: PaperTradingEngine,
        event_bus: Optional[PipelineEventBus] = None,
        news_service: Optional[NewsSentimentService] = None,
        telemetry: Optional[SystemTelemetryService] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.gateway = gateway
        self.market_client = market_client
        self.predictor = predictor
        self.laya_alpha = laya_alpha
        self.qwen_agent = qwen_agent
        self.laya_guard = laya_guard
        self.risk_engine = risk_engine
        self.paper_engine = paper_engine
        self.event_bus = event_bus
        self.news_service = news_service
        self.telemetry = telemetry

        self._bg_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._daemon_enabled: bool = False
        self._daemon_poll_seconds: int = 20
        self._daemon_last_tick_utc: Optional[str] = None
        self._daemon_cycles_run: int = 0
        self._daemon_last_error: Optional[str] = None

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_bus is not None:
            self.event_bus.publish(event_type, payload)

    def daemon_status(self) -> Dict[str, Any]:
        running = self._daemon_enabled and (self._bg_task is not None and not self._bg_task.done())
        return {
            "enabled": bool(self._daemon_enabled),
            "running": bool(running),
            "poll_seconds": int(self._daemon_poll_seconds),
            "cycles_run": int(self._daemon_cycles_run),
            "last_tick_utc": self._daemon_last_tick_utc,
            "last_error": self._daemon_last_error,
        }

    async def start_daemon(self, interval_seconds: int = 20) -> Dict[str, Any]:
        poll = max(5, min(int(interval_seconds), 3600))
        self._daemon_poll_seconds = poll
        if self._bg_task is not None and not self._bg_task.done():
            self._daemon_enabled = True
            return self.daemon_status()
        self._stop_event.clear()
        self._daemon_enabled = True
        self._daemon_last_error = None
        self._bg_task = asyncio.create_task(self._daemon_loop())
        self.db.set_state("daemon_enabled", "true")
        self._emit("daemon_status", self.daemon_status())
        return self.daemon_status()

    async def stop_daemon(self) -> Dict[str, Any]:
        self._daemon_enabled = False
        self._stop_event.set()
        if self._bg_task is not None:
            try:
                await asyncio.wait_for(self._bg_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._bg_task.cancel()
            self._bg_task = None
        self.db.set_state("daemon_enabled", "false")
        self._emit("daemon_status", self.daemon_status())
        return self.daemon_status()

    def _run_daemon_tick_sync(self) -> None:
        if self.db.get_state("system_status", "RUNNING") != "RUNNING":
            return
        sym = self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        mode = self.db.get_state("source_mode", "replay") or "replay"
        tf = self.db.get_state("candle_interval", self.settings.candle_interval) or "1d"
        if mode == "live":
            self.run_single_live_cycle(symbol=sym, gateway_key=self.settings.gateway_api_key, interval=tf)
            self._daemon_cycles_run += 1
        else:
            full_df = self.market_client.load_replay_candles(symbol=sym, interval=tf)
            for idx in range(45, len(full_df)):
                close_t = str(full_df.iloc[idx]["close_time"])
                if not self.db.is_candle_processed(sym, tf, close_t, "replay"):
                    self.process_candle_window(
                        candles_df=full_df.iloc[: idx + 1].reset_index(drop=True),
                        symbol=sym,
                        interval=tf,
                        source_mode="replay",
                        gateway_key=self.settings.gateway_api_key,
                    )
                    self._daemon_cycles_run += 1
                    break

    async def _daemon_loop(self) -> None:
        while self._daemon_enabled:
            try:
                self._daemon_last_tick_utc = utc_now_iso()
                await asyncio.to_thread(self._run_daemon_tick_sync)
                self._daemon_last_error = None
            except DuplicateCandleError:
                pass
            except Exception as exc:
                self._daemon_last_error = str(exc)[:200]
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=float(self._daemon_poll_seconds))
                break
            except asyncio.TimeoutError:
                continue

    def process_candle_window(
        self,
        candles_df: pd.DataFrame,
        symbol: str,
        interval: str,
        source_mode: str,
        gateway_key: str,
        strict_live_llm: bool = False,
        extra_metadata: str = "",
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo multi-agente sobre la última vela cerrada de `candles_df`.
        """
        t_cycle_start = time.perf_counter()
        if candles_df.empty:
            raise ValueError("El DataFrame de velas está vacío.")

        last_row = candles_df.iloc[-1]
        candle_dict = {
            "symbol": symbol,
            "interval": interval,
            "open_time": str(last_row["open_time"]),
            "close_time": str(last_row["close_time"]),
            "open": float(last_row["open"]),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "close": float(last_row["close"]),
            "volume": float(last_row["volume"]),
            "source_mode": source_mode,
        }

        # Etapa 1: Registro idempotente en SQLite (bloquea velas duplicadas)
        t_stage = time.perf_counter()
        candle_id = self.db.register_candle_idempotent(candle_dict)
        ingest_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_ingest",
            {
                "symbol": symbol,
                "interval": interval,
                "close": float(last_row["close"]),
                "candle_close_utc": candle_dict["close_time"],
                "latency_ms": round(ingest_ms, 2),
            },
        )

        # Etapa 2: Generación de Alpha Cuantitativo Multi-Temporalidad (Quant v2 XGB + LSTM)
        t_stage = time.perf_counter()
        prediction = self.predictor.predict_from_candles(candles_df, symbol=symbol, interval=interval)
        last_price = float(last_row["close"])
        portfolio = self.paper_engine.get_portfolio_state(
            current_price=last_price,
            symbol=symbol,
            candle_time=candle_dict["close_time"],
        )
        position_open = portfolio.position_qty >= self.settings.min_lot_size
        lstm_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_quant_model",
            {
                "symbol": symbol,
                "interval": interval,
                "pred_return_t1": round(prediction.pred_return_t1, 6),
                "pred_volatility_t1": round(prediction.pred_volatility_t1, 6),
                "abstain": bool(prediction.abstain),
                "direction_probs": dict(prediction.direction_probs),
                "atr_14": round(prediction.atr_14, 4),
                "model_detail": prediction.model_detail,
                "latency_ms": round(lstm_ms, 2),
            },
        )

        # Etapa 2b: Noticias / Sentimiento y Memoria Episódica Post-Trade
        t_stage = time.perf_counter()
        news_context: Dict[str, Any] = {}
        if self.news_service is not None:
            news_context = self.news_service.get_symbol_context(
                symbol=symbol, source_mode=source_mode
            )
        episodic_memories = self.db.list_episodic_memories(symbol=symbol, limit=3)
        news_memory_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_news_memory",
            {
                "symbol": symbol,
                "sentiment_label": news_context.get("sentiment_label", "NEUTRAL"),
                "memories_count": len(episodic_memories),
                "latency_ms": round(news_memory_ms, 2),
            },
        )

        # Etapa 3: Laya #1 [DISPOSITIVO LOCAL - DRL PyTorch / Ollama Fallback]
        t_stage = time.perf_counter()
        laya1_out = self.laya_alpha.evaluate(
            prediction=prediction,
            position_open=position_open,
            gateway_key=gateway_key,
            strict_live=strict_live_llm,
        )
        laya1_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_laya1_drl",
            {
                "symbol": symbol,
                "regime": laya1_out.regime,
                "signal_action": laya1_out.signal_action,
                "conviction_score": round(laya1_out.conviction_score, 4),
                "recommended_exposure_pct": round(laya1_out.recommended_exposure_pct, 4),
                "latency_ms": round(laya1_ms, 2),
            },
        )

        # Etapa 4: Qwen [API EXTERNA o LOCAL 3B] + Debate Bull/Bear + Retroalimentación
        t_stage = time.perf_counter()
        qwen_out = self.qwen_agent.orchestrate(
            prediction=prediction,
            laya1=laya1_out,
            portfolio=portfolio,
            gateway_key=gateway_key,
            strict_live=strict_live_llm,
            news_context=news_context,
            episodic_memories=episodic_memories,
        )
        qwen_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_qwen_orch",
            {
                "symbol": symbol,
                "action": qwen_out["action"],
                "execution_mode": qwen_out.get("execution_mode"),
                "model": qwen_out.get("model"),
                "is_fallback": bool(qwen_out.get("is_fallback", False)),
                "latency_ms": round(qwen_ms, 2),
            },
        )

        # Etapa 4b: Entrega y aplicación de retroalimentación de Qwen hacia Laya #1 y Laya #2
        orch_fb = qwen_out.get("orchestrator_feedback", {}) or {}
        t_fb1 = time.perf_counter()
        fb_laya1_applied = self.laya_alpha.apply_qwen_feedback(orch_fb.get("to_laya1", {}) or {})
        feedback_laya1_ms = (time.perf_counter() - t_fb1) * 1000.0

        t_fb2 = time.perf_counter()
        fb_laya2_applied = self.laya_guard.apply_qwen_feedback(orch_fb.get("to_laya2", {}) or {})
        feedback_laya2_ms = (time.perf_counter() - t_fb2) * 1000.0

        feedback_applied = {
            "to_laya1": fb_laya1_applied,
            "to_laya2": fb_laya2_applied,
            "feedback_source": orch_fb.get("feedback_source", "rule_fallback"),
        }
        qwen_out["feedback_applied"] = feedback_applied
        self._emit(
            "stage_feedback",
            {
                "symbol": symbol,
                "feedback_source": feedback_applied["feedback_source"],
                "feedback_laya1_ms": round(feedback_laya1_ms, 2),
                "feedback_laya2_ms": round(feedback_laya2_ms, 2),
            },
        )

        # Etapa 5: Laya #2 [DISPOSITIVO LOCAL - OLLAMA]
        t_stage = time.perf_counter()
        laya2_out = self.laya_guard.audit(
            prediction=prediction,
            qwen_decision=qwen_out,
            portfolio=portfolio,
            gateway_key=gateway_key,
            extra_metadata=extra_metadata,
            strict_live=strict_live_llm,
        )
        laya2_ms = (time.perf_counter() - t_stage) * 1000.0
        self._emit(
            "stage_laya2_guard",
            {
                "symbol": symbol,
                "approved": laya2_out.approved,
                "security_status": laya2_out.security_status,
                "coherence_prob": round(laya2_out.coherence_prob, 3),
                "min_coherence_threshold": round(laya2_out.min_coherence_threshold, 2),
                "is_fallback": laya2_out.is_fallback,
                "latency_ms": round(laya2_ms, 2),
            },
        )

        # Etapa 6 & 7: Motor Determinista de Riesgo (con SL/TP/TSL) + Liquidación Paper
        t_stage = time.perf_counter()
        system_status = self.db.get_state("system_status", "RUNNING") or "RUNNING"
        verdict = self.risk_engine.evaluate(
            proposed_action=str(qwen_out["action"]),
            symbol=symbol,
            market_price=last_price,
            portfolio=portfolio,
            system_status=system_status,
            laya2_approved=laya2_out.approved,
            laya2_reason=laya2_out.reason,
            atr_14=prediction.atr_14,
            candle_low=float(last_row["low"]),
            candle_high=float(last_row["high"]),
            source_mode=source_mode,
        )

        sim_exec = self.paper_engine.simulate_execution(
            action=verdict.final_action,
            symbol=symbol,
            market_price=last_price,
            atr_14=prediction.atr_14,
            conviction_score=laya1_out.conviction_score,
            vol_20=prediction.vol_20,
        )
        sim_exec["snapshot"]["candle_time"] = candle_dict["close_time"]
        sim_exec["snapshot"]["source_mode"] = source_mode
        declared_sl = sim_exec["snapshot"].get("stop_loss_price") or portfolio.stop_loss_price
        declared_tp = sim_exec["snapshot"].get("take_profit_price") or portfolio.take_profit_price
        risk_paper_ms = (time.perf_counter() - t_stage) * 1000.0

        total_latency_ms = (time.perf_counter() - t_cycle_start) * 1000.0
        stage_latencies_ms = {
            "ingest_ms": round(ingest_ms, 2),
            "lstm_ms": round(lstm_ms, 2),
            "news_memory_ms": round(news_memory_ms, 2),
            "laya1_ms": round(laya1_ms, 2),
            "qwen_ms": round(qwen_ms, 2),
            "feedback_laya1_ms": round(feedback_laya1_ms, 2),
            "feedback_laya2_ms": round(feedback_laya2_ms, 2),
            "laya2_ms": round(laya2_ms, 2),
            "risk_paper_ms": round(risk_paper_ms, 2),
            "total_cycle_ms": round(total_latency_ms, 2),
            "updated_at": candle_dict["close_time"],
        }
        qwen_out["stage_latencies_ms"] = stage_latencies_ms
        self.db.set_state("last_stage_latencies_json", json.dumps(stage_latencies_ms))

        decision_record = {
            "symbol": symbol,
            "interval": interval,
            "candle_close_utc": candle_dict["close_time"],
            "model_pred": prediction.pred_return_t1,
            "laya1_json": laya1_out.to_dict(),
            "qwen_json": qwen_out,
            "laya2_json": laya2_out.to_dict(),
            "risk_status": verdict.status,
            "risk_reason": verdict.reason,
            "final_action": verdict.final_action,
            "stop_loss_price": declared_sl,
            "take_profit_price": declared_tp,
            "latency_ms": round(total_latency_ms, 2),
            "qwen_execution_mode": str(qwen_out.get("execution_mode", QWEN_EXECUTION_LOCATION)),
            "laya_execution_mode": LAYA1_EXECUTION_MODE,
            "source_mode": source_mode,
        }

        decision_id = self.db.record_cycle_result(
            candle_id=candle_id,
            decision_data=decision_record,
            order_data=sim_exec["order"],
            snapshot_data=sim_exec["snapshot"],
        )
        self.db.set_state("peak_equity", str(sim_exec["peak_equity"]))

        # Memoria Episódica Post-Trade al cerrar operación SELL
        order_obj = sim_exec["order"]
        if order_obj is not None and order_obj.get("side") == "SELL":
            entry_px = float(order_obj.get("avg_entry_price") or last_price)
            exit_px = float(order_obj["exec_price"])
            realized = float(order_obj.get("realized_pnl") or 0.0)
            ret_pct = ((exit_px - entry_px) / entry_px * 100.0) if entry_px > 0 else 0.0
            outcome = "WIN" if realized > 0.0 else ("LOSS" if realized < 0.0 else "BREAKEVEN")
            trigger = verdict.rule_triggered if verdict.rule_triggered != "NONE" else "QWEN_SIGNAL"
            lesson = (
                f"[{outcome} {ret_pct:+.2f}%] Cierre en {symbol} vía {trigger} "
                f"(régimen={laya1_out.regime}, RSI={prediction.rsi_14:.1f}, vol={prediction.vol_20:.4f}, PnL=${realized:+.2f})."
            )
            self.db.add_episodic_memory(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "candle_time": candle_dict["close_time"],
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "realized_pnl": realized,
                    "return_pct": round(ret_pct, 4),
                    "regime": laya1_out.regime,
                    "rsi_at_entry": float(prediction.rsi_14),
                    "vol_at_entry": float(prediction.vol_20),
                    "outcome": outcome,
                    "lesson": lesson,
                }
            )

        if self.telemetry is not None:
            self.telemetry.record_agent_status(
                qwen_mode=str(qwen_out.get("execution_mode", QWEN_EXECUTION_LOCATION)),
                qwen_model=str(qwen_out.get("model", "")),
                qwen_transport=str(qwen_out.get("transport", "")),
                qwen_is_fallback=bool(qwen_out.get("is_fallback", False)),
                qwen_latency_ms=qwen_ms,
                laya2_model=laya2_out.model,
                laya2_transport=laya2_out.transport,
                laya2_is_fallback=laya2_out.is_fallback,
                laya2_latency_ms=laya2_ms,
            )

        self._emit(
            "stage_risk_paper",
            {
                "symbol": symbol,
                "risk_status": verdict.status,
                "final_action": verdict.final_action,
                "rule_triggered": verdict.rule_triggered,
                "stop_loss_price": declared_sl,
                "take_profit_price": declared_tp,
                "equity": round(float(sim_exec["snapshot"]["total_equity"]), 2),
                "latency_ms": round(risk_paper_ms, 2),
            },
        )
        self._emit(
            "cycle_complete",
            {
                "decision_id": decision_id,
                "symbol": symbol,
                "interval": interval,
                "final_action": verdict.final_action,
                "risk_status": verdict.status,
                "total_equity": round(float(sim_exec["snapshot"]["total_equity"]), 2),
                "total_cycle_ms": round(total_latency_ms, 2),
            },
        )

        return {
            "decision_id": decision_id,
            "candle_id": candle_id,
            "symbol": symbol,
            "candle_close_utc": candle_dict["close_time"],
            "close_price": last_price,
            "source_mode": source_mode,
            "pred_return_t1": prediction.pred_return_t1,
            "model_detail": prediction.model_detail,
            "atr_14": round(prediction.atr_14, 4),
            "indicators": {
                "ema_ratio": prediction.ema_ratio,
                "vol_20": prediction.vol_20,
                "rsi_14": prediction.rsi_14,
                "momentum_5": prediction.momentum_5,
                "atr_14": prediction.atr_14,
            },
            "news_context": news_context,
            "laya1": laya1_out.to_dict(),
            "qwen": qwen_out,
            "laya2": laya2_out.to_dict(),
            "risk": verdict.to_dict(),
            "order": sim_exec["order"],
            "snapshot": sim_exec["snapshot"],
            "latency_ms": round(total_latency_ms, 2),
            "stage_latencies_ms": stage_latencies_ms,
            "feedback_applied": feedback_applied,
        }

    def manual_liquidate(
        self,
        symbol: str,
        market_price: float,
        interval: str = "1d",
        *,
        enforce_market_hours: bool = True,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Liquidación manual (Paper): cierra al precio actual toda la posición abierta de `symbol`
        registrando decisión SELL auditable (hash-chain), orden y snapshot atómicamente.
        Bloquea la operación si el mercado del símbolo está cerrado (`enforce_market_hours=True`).
        """
        if enforce_market_hours:
            mkt = get_market_session_status(symbol, now_utc=now_utc)
            if not mkt["is_open"]:
                self.db.log_error(
                    error_code="RISK_MARKET_CLOSED",
                    component="TradingScheduler",
                    message=f"Liquidación bloqueada en {symbol}: {mkt['session_label']}",
                    payload_excerpt=f"symbol={symbol}, et_time={mkt['et_time']}",
                )
                raise ValueError(
                    f"Mercado cerrado para {symbol}: {mkt['session_label']}. No se pueden ejecutar liquidaciones hasta la apertura."
                )

        pos = self.db.get_position(symbol)
        if pos is None or float(pos.get("quantity") or 0.0) < self.settings.min_lot_size:
            raise ValueError(f"No hay posición abierta para liquidar en {symbol}.")
        if market_price <= 0:
            raise ValueError("Precio de mercado inválido para liquidar.")

        exit_sl = pos.get("stop_loss_price")
        exit_tp = pos.get("take_profit_price")

        # Se permite liquidar aunque el sistema esté STOPPED: es una orden explícita del operador
        prev_status = self.db.get_state("system_status", "RUNNING")
        sim_exec = None
        try:
            if prev_status == "STOPPED":
                self.db.set_state("system_status", "PAUSED")
            sim_exec = self.paper_engine.simulate_execution(
                action="SELL", symbol=symbol, market_price=market_price
            )
        finally:
            if prev_status == "STOPPED":
                self.db.set_state("system_status", "STOPPED")

        now_iso = utc_now_iso()
        sim_exec["snapshot"]["candle_time"] = now_iso
        sim_exec["snapshot"]["source_mode"] = "live"
        candle_id = self.db.register_candle_idempotent(
            {
                "symbol": symbol,
                "interval": interval,
                "open_time": now_iso,
                "close_time": now_iso,
                "open": market_price,
                "high": market_price,
                "low": market_price,
                "close": market_price,
                "volume": 0.0,
                "source_mode": "live",
            }
        )
        manual_meta = {"manual": True, "action": "SELL", "is_local_device": True,
                       "justification": "Liquidación manual solicitada por el operador."}
        decision_id = self.db.record_cycle_result(
            candle_id=candle_id,
            decision_data={
                "symbol": symbol,
                "interval": interval,
                "candle_close_utc": now_iso,
                "model_pred": 0.0,
                "laya1_json": {**manual_meta, "signal_action": "SELL", "regime": "manual"},
                "qwen_json": manual_meta,
                "laya2_json": {**manual_meta, "approved": True, "security_status": "SAFE"},
                "risk_status": "APPROVED",
                "risk_reason": "MANUAL_LIQUIDATION",
                "final_action": "SELL",
                "stop_loss_price": exit_sl,
                "take_profit_price": exit_tp,
                "latency_ms": 0.0,
                "qwen_execution_mode": "MANUAL",
                "laya_execution_mode": "MANUAL",
                "source_mode": "live",
            },
            order_data=sim_exec["order"],
            snapshot_data=sim_exec["snapshot"],
        )
        self.db.set_state("peak_equity", str(sim_exec["peak_equity"]))
        order = sim_exec["order"]
        realized = float(order.get("realized_pnl") or 0.0)
        self._emit(
            "cycle_complete",
            {
                "decision_id": decision_id,
                "symbol": symbol,
                "interval": interval,
                "final_action": "SELL",
                "risk_status": "APPROVED",
                "total_cycle_ms": 0.0,
            },
        )
        return {
            "decision_id": decision_id,
            "symbol": symbol,
            "exec_price": float(order["exec_price"]),
            "quantity": float(order["quantity"]),
            "realized_pnl": realized,
        }

    def run_single_live_cycle(
        self,
        symbol: Optional[str] = None,
        gateway_key: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Consulta mercado público en modo live y ejecuta un ciclo de decisión."""
        sym = symbol or self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        tf = interval or self.db.get_state("candle_interval", self.settings.candle_interval) or self.settings.candle_interval
        key = gateway_key or self.settings.gateway_api_key
        df = self.market_client.fetch_live_candles(
            symbol=sym, interval=tf
        )
        return self.process_candle_window(
            candles_df=df,
            symbol=sym,
            interval=tf,
            source_mode="live",
            gateway_key=key,
        )

    def run_replay_cycles(
        self,
        symbol: Optional[str] = None,
        n_cycles: int = 25,
        gateway_key: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta `n_cycles` (>=20 requeridos) secuenciales sobre velas históricas cerradas
        en modo 'replay', preservando estricto orden cronológico y cero look-ahead.
        """
        sym = symbol or self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        tf = interval or self.db.get_state("candle_interval", self.settings.candle_interval) or self.settings.candle_interval
        key = gateway_key or self.settings.gateway_api_key
        full_df = self.market_client.load_replay_candles(symbol=sym, interval=tf)

        min_warmup = 45
        if len(full_df) < min_warmup + n_cycles:
            raise ValueError(
                f"Histórico insuficiente para {n_cycles} ciclos en {sym} ({tf}): {len(full_df)} filas."
            )

        start_end_idx = len(full_df) - n_cycles
        results: List[Dict[str, Any]] = []

        for end_idx in range(start_end_idx, len(full_df)):
            window_df = full_df.iloc[: end_idx + 1].reset_index(drop=True)
            try:
                res = self.process_candle_window(
                    candles_df=window_df,
                    symbol=sym,
                    interval=tf,
                    source_mode="replay",
                    gateway_key=key,
                )
                results.append(res)
            except DuplicateCandleError:
                continue
            except (InsufficientFundsError, SystemStoppedError):
                continue

        return results
