"""
Core Scheduler - Orquestador del Ciclo Completo de 7 Etapas.

Flujo estricto:
1. Back Office: Ingesta de vela cerrada e idempotencia SQLite
2. Front Office: Features sin leakage + Modelo LSTM T4 congelado (r_hat_{t+1})
3. Front Office [DISPOSITIVO LOCAL]: Laya #1 (/v1/systemone)
4. Orquestador [API EXTERNA - NO LOCAL]: Qwen 14B (/chat/completions) + Pydantic
5. Middle Office [DISPOSITIVO LOCAL]: Laya #2 (/v1/systemone) Anti-Injection & Coherencia
6. Middle Office [PYTHON DETERMINISTA]: Risk Engine (Hard Gate inmutable)
7. Back Office: Simulación Paper + Registro Atómico SQLite WAL
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.back_office.database import DuplicateCandleError, TradingDatabase
from backend.app.back_office.mexc_client import MEXCMarketClient
from backend.app.back_office.paper_engine import (
    InsufficientFundsError,
    PaperTradingEngine,
    SystemStoppedError,
)
from backend.app.core.config import Settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.laya_alpha import (
    LAYA1_EXECUTION_MODE,
    LayaAlphaAgent,
)
from backend.app.front_office.predictor import QuantPredictor
from backend.app.middle_office.laya_guard import LayaGuardAgent
from backend.app.middle_office.risk_engine import DeterministicRiskEngine
from backend.app.orchestrator.qwen_agent import (
    QWEN_EXECUTION_LOCATION,
    QwenOrchestratorAgent,
)


class TradingScheduler:
    """Ejecutor del ciclo multi-agente en modo Live o Historical Replay."""

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
        self._bg_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

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
        Ejecuta las 7 etapas del ciclo sobre la última vela cerrada de `candles_df`.
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
        candle_id = self.db.register_candle_idempotent(candle_dict)

        # Etapa 2: Generación de Alpha Cuantitativo (T4 LSTM congelado)
        prediction = self.predictor.predict_from_candles(candles_df, symbol=symbol)
        last_price = float(last_row["close"])
        portfolio = self.paper_engine.get_portfolio_state(current_price=last_price)
        position_open = portfolio.position_qty >= self.settings.min_lot_size

        # Etapa 3: Laya #1 [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
        laya1_out = self.laya_alpha.evaluate(
            prediction=prediction,
            position_open=position_open,
            gateway_key=gateway_key,
            strict_live=strict_live_llm,
        )

        # Etapa 4: Qwen 14B [API EXTERNA - NO LOCAL] + Validación Pydantic
        qwen_out = self.qwen_agent.orchestrate(
            prediction=prediction,
            laya1=laya1_out,
            portfolio=portfolio,
            gateway_key=gateway_key,
            strict_live=strict_live_llm,
        )

        # Etapa 5: Laya #2 [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
        laya2_out = self.laya_guard.audit(
            prediction=prediction,
            qwen_decision=qwen_out,
            portfolio=portfolio,
            gateway_key=gateway_key,
            extra_metadata=extra_metadata,
            strict_live=strict_live_llm,
        )

        # Etapa 6: Motor Determinista de Riesgo (Hard Gate en Python puro)
        system_status = self.db.get_state("system_status", "RUNNING") or "RUNNING"
        verdict = self.risk_engine.evaluate(
            proposed_action=str(qwen_out["action"]),
            symbol=symbol,
            market_price=last_price,
            portfolio=portfolio,
            system_status=system_status,
            laya2_approved=laya2_out.approved,
            laya2_reason=laya2_out.reason,
        )

        # Etapa 7: Liquidación Paper y Persistencia Atómica SQLite
        sim_exec = self.paper_engine.simulate_execution(
            action=verdict.final_action,
            symbol=symbol,
            market_price=last_price,
        )
        sim_exec["snapshot"]["candle_time"] = candle_dict["close_time"]
        sim_exec["snapshot"]["source_mode"] = source_mode

        total_latency_ms = (time.perf_counter() - t_cycle_start) * 1000.0
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
            "latency_ms": round(total_latency_ms, 2),
            "qwen_execution_mode": QWEN_EXECUTION_LOCATION,
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

        return {
            "decision_id": decision_id,
            "candle_id": candle_id,
            "symbol": symbol,
            "candle_close_utc": candle_dict["close_time"],
            "close_price": last_price,
            "source_mode": source_mode,
            "pred_return_t1": prediction.pred_return_t1,
            "indicators": {
                "ema_ratio": prediction.ema_ratio,
                "vol_20": prediction.vol_20,
                "rsi_14": prediction.rsi_14,
                "momentum_5": prediction.momentum_5,
            },
            "laya1": laya1_out.to_dict(),
            "qwen": qwen_out,
            "laya2": laya2_out.to_dict(),
            "risk": verdict.to_dict(),
            "order": sim_exec["order"],
            "snapshot": sim_exec["snapshot"],
            "latency_ms": round(total_latency_ms, 2),
        }

    def run_single_live_cycle(
        self, symbol: Optional[str] = None, gateway_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Consulta mercado público en modo live y ejecuta un ciclo de decisión."""
        sym = symbol or self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        key = gateway_key or self.settings.gateway_api_key
        df = self.market_client.fetch_live_candles(
            symbol=sym, interval=self.settings.candle_interval
        )
        return self.process_candle_window(
            candles_df=df,
            symbol=sym,
            interval=self.settings.candle_interval,
            source_mode="live",
            gateway_key=key,
        )

    def run_replay_cycles(
        self,
        symbol: Optional[str] = None,
        n_cycles: int = 25,
        gateway_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta `n_cycles` (>=20 requeridos) secuenciales sobre velas históricas cerradas
        en modo 'replay', preservando estricto orden cronológico y cero look-ahead.
        """
        sym = symbol or self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        key = gateway_key or self.settings.gateway_api_key
        full_df = self.market_client.load_replay_candles(symbol=sym, interval="1d")

        min_warmup = 45
        if len(full_df) < min_warmup + n_cycles:
            raise ValueError(
                f"Histórico insuficiente para {n_cycles} ciclos en {sym}: {len(full_df)} filas."
            )

        start_end_idx = len(full_df) - n_cycles
        results: List[Dict[str, Any]] = []

        for end_idx in range(start_end_idx, len(full_df)):
            # Si fue pausado/detenido a mitad de un replay, respetamos el estado excepto si queremos verificar bloqueo STOP
            window_df = full_df.iloc[: end_idx + 1].reset_index(drop=True)
            try:
                res = self.process_candle_window(
                    candles_df=window_df,
                    symbol=sym,
                    interval="1d",
                    source_mode="replay",
                    gateway_key=key,
                )
                results.append(res)
            except DuplicateCandleError:
                # Idempotencia: vela ya procesada previamente en este modo
                continue
            except (InsufficientFundsError, SystemStoppedError):
                continue

        return results
