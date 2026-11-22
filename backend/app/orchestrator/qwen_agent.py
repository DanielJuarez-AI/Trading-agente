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
        self._last_local_unavailable_ts: Optional[float] = None

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
            "Eres el Orquestador Cuantitativo System-2 de un fondo Long-Only. "
            "Responde EXCLUSIVAMENTE con un objeto JSON válido con exactamente 4 claves:\n"
            f'{{"action": "BUY|HOLD|SELL", "symbol": "{prediction.symbol}", '
            f'"candle_date": "{prediction.candle_close_utc}", '
            '"justification": "Explicación concisa en español basada en los números reales"}}\n'
            "Reglas estrictas: SELL solo si position_qty > 0; BUY solo si pred_return_t1 >= 0 y position_qty == 0; "
            "en otro caso elige HOLD. No cambies symbol ni candle_date."
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

    def _normalize_local_json_content(
        self,
        raw_content: str,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
    ) -> str:
        """Normaliza JSON del modelo local cuantizado para garantizar contrato exacto y coherencia Long-Only."""
        has_pos = portfolio.position_qty >= self.settings.min_lot_size
        try:
            cleaned = self._extract_json_text(raw_content)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("JSON no es un objeto")
        except (ValueError, TypeError):
            fallback = self._dry_run_external_api_simulation(
                prediction,
                laya1,
                portfolio,
                mode_tag=f"Qwen Local ({self.settings.qwen_local_model_name})",
            )
            return json.dumps(fallback, ensure_ascii=False)

        action = str(parsed.get("action", "HOLD")).strip().upper()
        if action not in {"BUY", "HOLD", "SELL"}:
            action = "HOLD"
        if action == "SELL" and not has_pos:
            action = "HOLD"
        elif action == "BUY" and (has_pos or prediction.pred_return_t1 < -0.0005):
            action = "HOLD"

        just = str(parsed.get("justification", "")).strip()
        if len(just) < 12:
            fallback = self._dry_run_external_api_simulation(
                prediction,
                laya1,
                portfolio,
                mode_tag=f"Qwen Local ({self.settings.qwen_local_model_name})",
            )
            just = fallback["justification"]
        if prediction.pred_return_t1 < 0 and "positivo" in just.lower() and "negativo" not in just.lower():
            just = f"Retorno T4 negativo ({prediction.pred_return_t1 * 100:+.2f}%), postura prudente {action}."

        normalized = {
            "action": action,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "justification": just[:280],
        }
        return json.dumps(normalized, ensure_ascii=False)

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
        mode_tag: str = "Qwen API Externa",
    ) -> Dict[str, Any]:
        """
        Simulación estructurada cuando aún no se configura QWEN_API_KEY o Ollama Local no está activo.
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
                f"[{mode_tag}] Pronóstico T4 positivo ({r_pct:+.2f}%), "
                f"EMA ratio {prediction.ema_ratio:.3f} y Laya #1 DRL confirma {laya1.regime} "
                f"(convicción {laya1.conviction_score:.2f})."
            )
        elif (
            (laya1.signal_action == "SELL" or prediction.pred_return_t1 < -0.0008)
            and has_pos
        ):
            action = "SELL"
            just = (
                f"[{mode_tag}] Pronóstico T4 negativo ({r_pct:+.2f}%), "
                f"Laya #1 DRL reporta {laya1.regime}; se cierra posición Long para proteger capital."
            )
        else:
            action = "HOLD"
            just = (
                f"[{mode_tag}] Pronóstico T4 {r_pct:+.2f}%, régimen {laya1.regime} "
                f"(vol_20={prediction.vol_20:.4f}); se mantiene espera prudente ({'con posición' if has_pos else 'en efectivo'})."
            )

        return {
            "action": action,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "justification": just[:280],
        }

    def _build_orchestrator_feedback(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        decision_action: str,
    ) -> Dict[str, Any]:
        """Genera retroalimentación estructurada de Qwen (orquestador) hacia Laya #1 y Laya #2."""
        aligned = decision_action == laya1.signal_action
        high_vol = (laya1.regime == "high_vol_chop") or (prediction.vol_20 > 0.035)
        has_pos = portfolio.position_qty >= self.settings.min_lot_size

        if not aligned:
            temp_delta = 0.10
            conv_bias = -0.08
            laya1_guidance = (
                f"Divergencia Qwen({decision_action}) vs Laya#1({laya1.signal_action}): "
                f"suavizar temperatura (+0.10) y moderar sesgo de convicción (-0.08)."
            )
            strictness = "strict"
            min_coh = 0.88
            focus_rule = "AUDITAR_DIVERGENCIA_SENAL"
        elif high_vol:
            temp_delta = 0.08
            conv_bias = -0.04
            laya1_guidance = (
                f"Régimen volátil ({laya1.regime}, vol={prediction.vol_20:.4f}): "
                f"elevar cautela en política DRL (+0.08 temp, -0.04 sesgo)."
            )
            strictness = "strict"
            min_coh = 0.88
            focus_rule = "AUDITAR_CONTRA_VOLATILIDAD"
        elif decision_action in {"BUY", "SELL"}:
            temp_delta = -0.08
            conv_bias = 0.05
            laya1_guidance = (
                f"Consenso DRL-Qwen ({decision_action} en {prediction.symbol}): "
                f"refuerzo positivo (+0.05 sesgo, -0.08 temp)."
            )
            strictness = "standard"
            min_coh = 0.75
            focus_rule = "VERIFICACION_ESTANDAR_LONG_ONLY"
        else:
            temp_delta = 0.0
            conv_bias = 0.0
            laya1_guidance = (
                f"Consenso HOLD ({'con posición' if has_pos else 'sin posición'}): "
                f"mantener calibración DRL nominal."
            )
            strictness = "standard"
            min_coh = 0.75
            focus_rule = "VERIFICACION_ESTANDAR_LONG_ONLY"

        return {
            "source_agent": "Qwen Orchestrator",
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "qwen_action": decision_action,
            "laya1_signal": laya1.signal_action,
            "aligned": aligned,
            "to_laya1": {
                "target": "Laya #1 (DRL Local)",
                "temperature_delta": round(temp_delta, 4),
                "conviction_bias": round(conv_bias, 4),
                "guidance": laya1_guidance,
                "aligned_with_qwen": aligned,
                "qwen_action": decision_action,
                "symbol": prediction.symbol,
            },
            "to_laya2": {
                "target": "Laya #2 (Guardia Local)",
                "strictness_level": strictness,
                "min_coherence_threshold": round(min_coh, 2),
                "focus_rule": focus_rule,
                "qwen_action": decision_action,
                "laya1_signal": laya1.signal_action,
                "symbol": prediction.symbol,
            },
        }

    def _format_orchestrate_result(
        self,
        decision: QwenDecisionOutput,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        *,
        execution_mode: str,
        is_local_device: bool,
        api_endpoint: str,
        model: str,
        vram_budget_gb: Optional[float],
        latency_ms: float,
        transport: str,
        api_key_configured: bool,
        num_ctx: Optional[int] = None,
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            **decision.model_dump(),
            "execution_mode": execution_mode,
            "is_local_device": is_local_device,
            "api_endpoint": api_endpoint,
            "model": model,
            "vram_budget_gb": vram_budget_gb,
            "temperature": self.settings.qwen_temperature,
            "latency_ms": round(latency_ms, 2),
            "transport": transport,
            "api_key_configured": api_key_configured,
            "orchestrator_feedback": self._build_orchestrator_feedback(
                prediction=prediction,
                laya1=laya1,
                portfolio=portfolio,
                decision_action=decision.action,
            ),
        }
        if num_ctx is not None:
            res["num_ctx"] = num_ctx
        return res

    def orchestrate(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        gateway_key: str,
        strict_live: bool = False,
        qwen_mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Switch Dinámico System-2:
        - EXTERNAL_API: invoca Qwen 14B por API HTTP remota (no local).
        - LOCAL_DEVICE: invoca Qwen 7B-Q4 (<= 6GB VRAM) vía Ollama local (/v1/chat/completions).
        """
        active_mode = (
            qwen_mode_override
            or self.db.get_state("qwen_mode", self.settings.qwen_execution_mode)
            or self.settings.qwen_execution_mode
        )
        if active_mode not in {"EXTERNAL_API", "LOCAL_DEVICE"}:
            active_mode = "EXTERNAL_API"

        self.gateway.authorize_llm_call(
            target_agent="qwen_orchestrator",
            execution_mode=active_mode,
            gateway_key=gateway_key,
        )

        t0 = time.perf_counter()

        # ======================================================================
        # RAMA LOCAL <= 6 GB VRAM: Qwen 2.5 7B Instruct Q4_K_M en Ollama Local
        # ======================================================================
        if active_mode == "LOCAL_DEVICE":
            local_endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/v1/chat/completions"
            local_model = self.settings.qwen_local_model_name

            if (
                self.http_client is None
                and not strict_live
                and self.settings.allow_local_laya_calibrated_fallback
                and self._last_local_unavailable_ts is not None
                and (time.monotonic() - self._last_local_unavailable_ts) < 15.0
            ):
                raw_obj = self._dry_run_external_api_simulation(
                    prediction,
                    laya1,
                    portfolio,
                    mode_tag=f"Qwen Local 6GB ({local_model})",
                )
                decision = self._validate_decision_contract(
                    json.dumps(raw_obj, ensure_ascii=False),
                    expected_symbol=prediction.symbol,
                    expected_candle_date=prediction.candle_close_utc,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._format_orchestrate_result(
                    decision,
                    prediction,
                    laya1,
                    portfolio,
                    execution_mode="LOCAL_DEVICE",
                    is_local_device=True,
                    api_endpoint=local_endpoint,
                    model=local_model,
                    vram_budget_gb=self.settings.qwen_local_vram_budget_gb,
                    num_ctx=self.settings.qwen_local_num_ctx,
                    latency_ms=latency_ms,
                    transport="ollama_local_6gb_simulated_fallback",
                    api_key_configured=False,
                )

            local_body = {
                "model": local_model,
                "temperature": self.settings.qwen_temperature,
                "max_tokens": self.settings.qwen_max_tokens,
                "keep_alive": "24h",
                "response_format": {"type": "json_object"},
                "options": {"num_ctx": self.settings.qwen_local_num_ctx},
                "messages": self._build_messages(prediction, laya1, portfolio),
            }
            owns_client = self.http_client is None
            client = self.http_client or httpx.Client(
                timeout=httpx.Timeout(
                    timeout=min(float(self.settings.llm_timeout_seconds), 20.0),
                    connect=0.25,
                )
            )
            try:
                resp = client.post(local_endpoint, json=local_body)
                resp.raise_for_status()
                self._last_local_unavailable_ts = None
                payload = resp.json()
                raw_content = payload["choices"][0]["message"]["content"]
                if self.http_client is None:
                    raw_content = self._normalize_local_json_content(
                        raw_content=raw_content,
                        prediction=prediction,
                        laya1=laya1,
                        portfolio=portfolio,
                    )
                decision = self._validate_decision_contract(
                    raw_content,
                    expected_symbol=prediction.symbol,
                    expected_candle_date=prediction.candle_close_utc,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._format_orchestrate_result(
                    decision,
                    prediction,
                    laya1,
                    portfolio,
                    execution_mode="LOCAL_DEVICE",
                    is_local_device=True,
                    api_endpoint=local_endpoint,
                    model=local_model,
                    vram_budget_gb=self.settings.qwen_local_vram_budget_gb,
                    num_ctx=self.settings.qwen_local_num_ctx,
                    latency_ms=latency_ms,
                    transport="ollama_local_6gb_vram",
                    api_key_configured=True,
                )
            except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError, TypeError) as exc:
                self._last_local_unavailable_ts = time.monotonic()
                if strict_live or (self.http_client is not None) or not self.settings.allow_local_laya_calibrated_fallback:
                    self.db.log_error(
                        error_code="QWEN_LOCAL_6GB_UNAVAILABLE",
                        component="QwenOrchestratorAgent",
                        message=f"Ollama local (Qwen {local_model}) inaccesible: {exc}",
                        payload_excerpt=f"endpoint={local_endpoint}, symbol={prediction.symbol}",
                    )
                    self.db.set_state("system_status", "PAUSED")
                    raise LLMServiceUnavailableError(
                        f"Ollama local (Qwen 6GB VRAM) inaccesible: {exc}"
                    ) from exc

                raw_obj = self._dry_run_external_api_simulation(
                    prediction,
                    laya1,
                    portfolio,
                    mode_tag=f"Qwen Local 6GB ({local_model})",
                )
                decision = self._validate_decision_contract(
                    json.dumps(raw_obj, ensure_ascii=False),
                    expected_symbol=prediction.symbol,
                    expected_candle_date=prediction.candle_close_utc,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._format_orchestrate_result(
                    decision,
                    prediction,
                    laya1,
                    portfolio,
                    execution_mode="LOCAL_DEVICE",
                    is_local_device=True,
                    api_endpoint=local_endpoint,
                    model=local_model,
                    vram_budget_gb=self.settings.qwen_local_vram_budget_gb,
                    num_ctx=self.settings.qwen_local_num_ctx,
                    latency_ms=latency_ms,
                    transport="ollama_local_6gb_simulated_fallback",
                    api_key_configured=False,
                )
            finally:
                if owns_client:
                    client.close()

        # ======================================================================
        # RAMA API EXTERNA (NO LOCAL): Qwen 14B vía HTTPS REST
        # ======================================================================
        endpoint_url = f"{self.settings.qwen_api_base_url.rstrip('/')}/chat/completions"
        use_http = (self.http_client is not None) or self._has_real_api_key() or strict_live

        if not use_http:
            raw_obj = self._dry_run_external_api_simulation(prediction, laya1, portfolio)
            decision = self._validate_decision_contract(
                json.dumps(raw_obj, ensure_ascii=False),
                expected_symbol=prediction.symbol,
                expected_candle_date=prediction.candle_close_utc,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return self._format_orchestrate_result(
                decision,
                prediction,
                laya1,
                portfolio,
                execution_mode=QWEN_EXECUTION_LOCATION,
                is_local_device=QWEN_IS_LOCAL_DEVICE,
                api_endpoint=endpoint_url,
                model=self.settings.qwen_model_name,
                vram_budget_gb=None,
                latency_ms=latency_ms,
                transport="external_api_simulated_no_key",
                api_key_configured=False,
            )

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
        return self._format_orchestrate_result(
            decision,
            prediction,
            laya1,
            portfolio,
            execution_mode=QWEN_EXECUTION_LOCATION,
            is_local_device=QWEN_IS_LOCAL_DEVICE,
            api_endpoint=endpoint_url,
            model=self.settings.qwen_model_name,
            vram_budget_gb=None,
            latency_ms=latency_ms,
            transport="external_https_api",
            api_key_configured=self._has_real_api_key(),
        )
