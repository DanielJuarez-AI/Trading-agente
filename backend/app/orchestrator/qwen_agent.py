"""
Orquestador Central System-2 - Agente Qwen 14B.

# ==============================================================================
# [API EXTERNA - NO EN DISPOSITIVO LOCAL]
# IMPORTANTE: Qwen 14B NO se ejecuta en Ollama ni en el dispositivo local.
# Se invoca exclusivamente mediante peticiones HTTP REST a una API externa
# compatible con OpenAI / DashScope / OpenRouter (QWEN_API_BASE_URL + QWEN_API_KEY).
# Todo lo demás del pipeline (Laya #1, Laya #2, Predictor T4, Riesgo, SQLite)
# sí corre en el dispositivo local.
# ==============================================================================
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.paper_engine import PortfolioState
from backend.app.core.config import Settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.laya_alpha import (
    LayaAlphaOutput,
    LLMServiceUnavailableError,
)
from backend.app.front_office.predictor import PredictionResult

# Marcadores explícitos de inferencia por API externa (NO local)
QWEN_EXECUTION_LOCATION: str = "EXTERNAL_API"
QWEN_IS_LOCAL_DEVICE: bool = False


class QwenInvalidOutputError(ValueError):
    """Error lanzado cuando la respuesta de Qwen viola el esquema JSON obligatorio."""


class QwenDecisionOutput(BaseModel):
    """Contrato JSON estricto obligatorio de la Sección 5.2.B."""

    action: Literal["BUY", "HOLD", "SELL"]
    symbol: str = Field(..., min_length=1, max_length=20)
    candle_date: str = Field(..., min_length=10, max_length=40)
    justification: str = Field(..., min_length=1, max_length=280)

    @field_validator("justification")
    @classmethod
    def justification_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("La justificación no puede estar vacía.")
        if len(cleaned) > 280:
            raise ValueError("La justificación excede el límite de 280 caracteres.")
        return cleaned


class QwenOrchestratorAgent:
    """
    [API EXTERNA - NO EN DISPOSITIVO LOCAL]
    Coordina señales de Front Office (LSTM T4 + Laya #1 Local) y produce el dictamen JSON
    estricto mediante llamada HTTPS al endpoint remoto de Qwen 14B.
    """

    def __init__(
        self,
        settings: Settings,
        db: TradingDatabase,
        gateway: PrivateAuthGateway,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.gateway = gateway
        self.http_client = http_client

    def _has_real_api_key(self) -> bool:
        key = (self.settings.qwen_api_key or "").strip()
        if not key:
            return False
        placeholders = {
            "replace_with_external_qwen_api_key",
            "your_api_key_here",
            "changeme",
        }
        return key.lower() not in placeholders

    def _build_messages(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
    ) -> list[Dict[str, str]]:
        system_prompt = (
            "Eres el Orquestador Cuantitativo System-2 (Qwen 14B vía API externa) de un fondo Long-Only. "
            "Responde EXCLUSIVAMENTE con un objeto JSON válido con exactamente las claves: "
            "action ('BUY'|'HOLD'|'SELL'), symbol, candle_date, justification (<=280 chars, en español y basada en los números reales). "
            "Nunca propongas SELL si position_qty == 0. No inventes métricas ni alteres símbolos o fechas."
        )
        user_payload = {
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "quant_model_t4": {
                "pred_return_t1": round(prediction.pred_return_t1, 6),
                "direction": prediction.direction,
                "ema_ratio": round(prediction.ema_ratio, 6),
                "vol_20": round(prediction.vol_20, 6),
                "rsi_14": round(prediction.rsi_14, 2),
            },
            "laya1_local_system1": {
                "regime": laya1.regime,
                "signal_action": laya1.signal_action,
                "conviction_level": laya1.conviction_level,
                "conviction_score": round(laya1.conviction_score, 3),
            },
            "portfolio": {
                "cash": round(portfolio.cash, 2),
                "position_qty": round(portfolio.position_qty, 6),
                "total_equity": round(portfolio.total_equity, 2),
                "drawdown_pct": round(portfolio.drawdown_pct, 4),
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    @staticmethod
    def _extract_json_text(raw_content: str) -> str:
        text = raw_content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return text

    def _validate_decision_contract(
        self,
        raw_content: str,
        expected_symbol: str,
        expected_candle_date: str,
    ) -> QwenDecisionOutput:
        try:
            cleaned = self._extract_json_text(raw_content)
            parsed = json.loads(cleaned)
            decision = QwenDecisionOutput.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            self.db.log_error(
                error_code="QWEN_INVALID_JSON_SCHEMA",
                component="QwenOrchestratorAgent",
                message=f"JSON inválido o fuera de esquema devuelto por API externa Qwen: {exc}",
                payload_excerpt=str(raw_content)[:300],
            )
            raise QwenInvalidOutputError(f"Respuesta JSON inválida de Qwen: {exc}") from exc

        if decision.symbol != expected_symbol:
            self.db.log_error(
                error_code="QWEN_SYMBOL_MISMATCH",
                component="QwenOrchestratorAgent",
                message=f"Qwen devolvió símbolo '{decision.symbol}' distinto de '{expected_symbol}'.",
                payload_excerpt=str(raw_content)[:250],
            )
            raise QwenInvalidOutputError(
                f"Símbolo inconsistente en respuesta Qwen: {decision.symbol} != {expected_symbol}"
            )

        if decision.candle_date != expected_candle_date:
            self.db.log_error(
                error_code="QWEN_DATE_MISMATCH",
                component="QwenOrchestratorAgent",
                message=f"Qwen devolvió candle_date '{decision.candle_date}' != '{expected_candle_date}'.",
                payload_excerpt=str(raw_content)[:250],
            )
            raise QwenInvalidOutputError(
                f"Fecha de vela inconsistente en respuesta Qwen: {decision.candle_date} != {expected_candle_date}"
            )

        return decision

    def _dry_run_external_api_simulation(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
    ) -> Dict[str, Any]:
        """
        Simulación estructurada cuando aún no se configura QWEN_API_KEY real en .env.
        Mantiene explícitamente marcado `execution_mode = EXTERNAL_API` y `is_local_device = False`.
        """
        r_pct = prediction.pred_return_t1 * 100.0
        has_pos = portfolio.position_qty >= self.settings.min_lot_size

        if (
            laya1.signal_action == "BUY"
            and prediction.pred_return_t1 > 0.0005
            and laya1.regime != "high_vol_chop"
            and not has_pos
        ):
            action = "BUY"
            just = (
                f"[Qwen API Externa] Pronóstico T4 positivo ({r_pct:+.2f}%), "
                f"EMA ratio {prediction.ema_ratio:.3f} y Laya #1 local confirma {laya1.regime} "
                f"(convicción {laya1.conviction_score:.2f})."
            )
        elif (
            (laya1.signal_action == "SELL" or prediction.pred_return_t1 < -0.0008)
            and has_pos
        ):
            action = "SELL"
            just = (
                f"[Qwen API Externa] Pronóstico T4 negativo ({r_pct:+.2f}%), "
                f"Laya #1 local reporta {laya1.regime}; se cierra posición Long para proteger capital."
            )
        else:
            action = "HOLD"
            just = (
                f"[Qwen API Externa] Pronóstico T4 {r_pct:+.2f}%, régimen {laya1.regime} "
                f"(vol_20={prediction.vol_20:.4f}); se mantiene espera prudente ({'con posición' if has_pos else 'en efectivo'})."
            )

        return {
            "action": action,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "justification": just[:280],
        }

    def orchestrate(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        gateway_key: str,
        strict_live: bool = False,
    ) -> Dict[str, Any]:
        """
        [API EXTERNA - NO EN DISPOSITIVO LOCAL]
        Invoca a Qwen 14B vía API HTTP remota autenticado por PrivateAuthGateway.
        """
        self.gateway.authorize_llm_call(
            target_agent="qwen_orchestrator",
            execution_mode=QWEN_EXECUTION_LOCATION,
            gateway_key=gateway_key,
        )

        endpoint_url = f"{self.settings.qwen_api_base_url.rstrip('/')}/chat/completions"
        t0 = time.perf_counter()

        use_http = (self.http_client is not None) or self._has_real_api_key() or strict_live

        if not use_http:
            raw_obj = self._dry_run_external_api_simulation(prediction, laya1, portfolio)
            decision = self._validate_decision_contract(
                json.dumps(raw_obj, ensure_ascii=False),
                expected_symbol=prediction.symbol,
                expected_candle_date=prediction.candle_close_utc,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                **decision.model_dump(),
                "execution_mode": QWEN_EXECUTION_LOCATION,
                "is_local_device": QWEN_IS_LOCAL_DEVICE,
                "api_endpoint": endpoint_url,
                "model": self.settings.qwen_model_name,
                "temperature": self.settings.qwen_temperature,
                "latency_ms": round(latency_ms, 2),
                "transport": "external_api_simulated_no_key",
                "api_key_configured": False,
            }

        # Invocación HTTP real al endpoint externo de Qwen 14B (NO LOCAL)
        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.qwen_model_name,
            "temperature": self.settings.qwen_temperature,
            "max_tokens": self.settings.qwen_max_tokens,
            "response_format": {"type": "json_object"},
            "messages": self._build_messages(prediction, laya1, portfolio),
        }

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(timeout=float(self.settings.llm_timeout_seconds))
        try:
            resp = client.post(endpoint_url, headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()
            raw_content = payload["choices"][0]["message"]["content"]
        except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError, TypeError) as exc:
            self.db.log_error(
                error_code="QWEN_EXTERNAL_API_UNAVAILABLE",
                component="QwenOrchestratorAgent",
                message=f"Fallo conectando con API externa de Qwen 14B: {exc}",
                payload_excerpt=f"endpoint={endpoint_url}, symbol={prediction.symbol}",
            )
            self.db.set_state("system_status", "PAUSED")
            raise LLMServiceUnavailableError(
                f"API externa de Qwen 14B inaccesible: {exc}"
            ) from exc
        finally:
            if owns_client:
                client.close()

        decision = self._validate_decision_contract(
            raw_content,
            expected_symbol=prediction.symbol,
            expected_candle_date=prediction.candle_close_utc,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            **decision.model_dump(),
            "execution_mode": QWEN_EXECUTION_LOCATION,
            "is_local_device": QWEN_IS_LOCAL_DEVICE,
            "api_endpoint": endpoint_url,
            "model": self.settings.qwen_model_name,
            "temperature": self.settings.qwen_temperature,
            "latency_ms": round(latency_ms, 2),
            "transport": "external_https_api",
            "api_key_configured": self._has_real_api_key(),
        }
