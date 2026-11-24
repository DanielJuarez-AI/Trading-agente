"""
Orquestador Central System-2 - Agente Qwen (API Externa 14B <-> Ollama Local 3B <= 6GB VRAM).

Incluye:
- Debate estructurado Bull vs. Bear en una sola llamada JSON.
- Contexto ortogonal de Noticias/Sentimiento y Memoria Episódica Post-Trade (FinMem).
- Emisión real de retroalimentación calibrada hacia Laya #1 (DRL) y Laya #2 (Guardia),
  con fallback determinista auditado y trazabilidad explícita Real LLM vs Fallback.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal, Optional

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

# Marcadores por defecto de topología (sobreescritos cuando el switch está en LOCAL_DEVICE)
QWEN_EXECUTION_LOCATION: str = "EXTERNAL_API"
QWEN_IS_LOCAL_DEVICE: bool = False


class QwenInvalidOutputError(ValueError):
    """Error lanzado cuando la respuesta de Qwen viola el esquema JSON obligatorio."""


class QwenDecisionOutput(BaseModel):
    """Contrato JSON estricto obligatorio (4 claves core + debate & feedback estructurado)."""

    action: Literal["BUY", "HOLD", "SELL"]
    symbol: str = Field(..., min_length=1, max_length=20)
    candle_date: str = Field(..., min_length=10, max_length=40)
    justification: str = Field(..., min_length=1, max_length=280)
    bull_case: str = Field(default="", max_length=260)
    bear_case: str = Field(default="", max_length=260)
    feedback_to_laya1: Optional[Dict[str, Any]] = None
    feedback_to_laya2: Optional[Dict[str, Any]] = None

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
    Orquestador System-2: coordina Front Office (Quant v2 + Laya #1 DRL),
    Noticias/Sentimiento y Memoria Episódica, generando dictamen JSON,
    debate Bull/Bear y retroalimentación hacia Laya #1 y Laya #2.
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

    def warmup_local_model(self) -> Dict[str, Any]:
        """Precarga en VRAM el modelo local de Ollama sin bloquear el flujo principal."""
        local_model = self.settings.qwen_local_model_name
        if self.http_client is not None:
            return {"warmed_up": False, "model": local_model}
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        body = {
            "model": local_model,
            "prompt": "",
            "keep_alive": "24h",
            "stream": False,
            "options": {"num_ctx": self.settings.qwen_local_num_ctx},
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout=15.0, connect=0.35)) as client:
                resp = client.post(url, json=body)
                resp.raise_for_status()
            self._last_local_unavailable_ts = None
            return {"warmed_up": True, "model": local_model}
        except Exception as exc:
            return {"warmed_up": False, "model": local_model, "error": str(exc)[:160]}

    def _build_messages(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        news_context: Optional[Dict[str, Any]] = None,
        episodic_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> list[Dict[str, str]]:
        system_prompt = (
            "Eres el Orquestador Cuantitativo System-2 de un fondo Long-Only. "
            "Evalúa los indicadores, el triaje DRL de Laya #1, noticias y memoria de operaciones pasadas. "
            "Responde EXCLUSIVAMENTE con un objeto JSON válido con estas claves:\n"
            f'{{"action": "BUY|HOLD|SELL", "symbol": "{prediction.symbol}", '
            f'"candle_date": "{prediction.candle_close_utc}", '
            '"justification": "Explicación concisa en español max 280 chars", '
            '"bull_case": "Argumento alcista breve", "bear_case": "Argumento bajista breve", '
            '"feedback_to_laya1": {"temperature_delta": 0.0, "conviction_bias": 0.0, "guidance": "Instrucción corta para calibrar DRL"}, '
            '"feedback_to_laya2": {"strictness_level": "standard|strict", "min_coherence_threshold": 0.75, "focus_rule": "REGLA_AUDITORIA"}}\n'
            "Reglas estrictas: SELL solo si position_qty > 0; BUY solo si pred_return_t1 >= 0 y position_qty == 0; "
            "en otro caso elige HOLD. No cambies symbol ni candle_date."
        )
        memories_compact = [
            {
                "candle_time": m.get("candle_time"),
                "outcome": m.get("outcome"),
                "return_pct": round(float(m.get("return_pct", 0.0)), 2),
                "regime": m.get("regime"),
                "lesson": str(m.get("lesson", ""))[:140],
            }
            for m in (episodic_memories or [])[:3]
        ]
        user_payload = {
            "symbol": prediction.symbol,
            "interval": getattr(prediction, "interval", "1d"),
            "candle_date": prediction.candle_close_utc,
            "quant_model": {
                "model_detail": getattr(prediction, "model_detail", prediction.model_version),
                "pred_return_t1": round(prediction.pred_return_t1, 6),
                "direction": prediction.direction,
                "ema_ratio": round(prediction.ema_ratio, 6),
                "vol_20": round(prediction.vol_20, 6),
                "atr_14": round(getattr(prediction, "atr_14", 0.0), 4),
                "rsi_14": round(prediction.rsi_14, 2),
            },
            "laya1_local_system1": {
                "regime": laya1.regime,
                "signal_action": laya1.signal_action,
                "conviction_level": laya1.conviction_level,
                "conviction_score": round(laya1.conviction_score, 3),
                "recommended_exposure_pct": round(getattr(laya1, "recommended_exposure_pct", 0.35), 3),
            },
            "portfolio": {
                "cash": round(portfolio.cash, 2),
                "position_qty": round(portfolio.position_qty, 6),
                "avg_entry_price": round(portfolio.avg_entry_price, 4),
                "stop_loss_price": portfolio.stop_loss_price,
                "take_profit_price": portfolio.take_profit_price,
                "total_equity": round(portfolio.total_equity, 2),
                "total_exposure_pct": round(getattr(portfolio, "total_exposure_pct", 0.0), 4),
                "drawdown_pct": round(portfolio.drawdown_pct, 4),
            },
            "news_sentiment": {
                "label": (news_context or {}).get("sentiment_label", "NEUTRAL"),
                "score": (news_context or {}).get("sentiment_score", 0.0),
                "catalysts": (news_context or {}).get("catalysts", []),
                "headlines": ((news_context or {}).get("headlines") or [])[:3],
            },
            "episodic_memories": memories_compact,
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
        news_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Normaliza JSON del modelo local para garantizar contrato exacto y coherencia Long-Only."""
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
                news_context=news_context,
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
                news_context=news_context,
            )
            just = fallback["justification"]
        if prediction.pred_return_t1 < 0 and "positivo" in just.lower() and "negativo" not in just.lower():
            just = f"Retorno cuantitativo negativo ({prediction.pred_return_t1 * 100:+.2f}%), postura prudente {action}."

        normalized: Dict[str, Any] = {
            "action": action,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "justification": just[:280],
            "bull_case": str(parsed.get("bull_case", "")).strip()[:260],
            "bear_case": str(parsed.get("bear_case", "")).strip()[:260],
        }
        if isinstance(parsed.get("feedback_to_laya1"), dict):
            normalized["feedback_to_laya1"] = parsed["feedback_to_laya1"]
        if isinstance(parsed.get("feedback_to_laya2"), dict):
            normalized["feedback_to_laya2"] = parsed["feedback_to_laya2"]
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

    @staticmethod
    def _default_bull_bear(
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        news_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        r_pct = prediction.pred_return_t1 * 100.0
        sent_label = (news_context or {}).get("sentiment_label", "NEUTRAL")
        bull = (
            f"Retorno T+1={r_pct:+.2f}%, EMA ratio={prediction.ema_ratio:.3f}, "
            f"régimen DRL={laya1.regime}, sentimiento={sent_label}."
        )
        bear = (
            f"Volatilidad vol_20={prediction.vol_20:.4f}, RSI={prediction.rsi_14:.1f}, "
            f"convicción={laya1.conviction_score:.2f}."
        )
        return bull[:260], bear[:260]

    def _dry_run_external_api_simulation(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        mode_tag: str = "Qwen API Externa",
        news_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulación estructurada cuando no hay API Key o Ollama Local está inaccesible."""
        r_pct = prediction.pred_return_t1 * 100.0
        has_pos = portfolio.position_qty >= self.settings.min_lot_size
        sent_score = float((news_context or {}).get("sentiment_score", 0.0))
        bearish_news_veto = sent_score <= -0.35

        if (
            laya1.signal_action == "BUY"
            and prediction.pred_return_t1 > 0.0005
            and laya1.regime != "high_vol_chop"
            and not has_pos
            and not bearish_news_veto
        ):
            action = "BUY"
            just = (
                f"[{mode_tag}] Pronóstico {r_pct:+.2f}%, EMA {prediction.ema_ratio:.3f} y Laya #1 DRL "
                f"{laya1.regime} (convicción {laya1.conviction_score:.2f})."
            )
        elif (
            (laya1.signal_action == "SELL" or prediction.pred_return_t1 < -0.0008 or bearish_news_veto)
            and has_pos
        ):
            action = "SELL"
            just = (
                f"[{mode_tag}] Señal de salida ({r_pct:+.2f}%, sentimiento={sent_score:+.2f}); "
                f"Laya #1 DRL {laya1.regime}, cierre Long protector."
            )
        else:
            action = "HOLD"
            reason_veto = " + veto noticias bajistas" if bearish_news_veto else ""
            just = (
                f"[{mode_tag}] Pronóstico {r_pct:+.2f}%, régimen {laya1.regime}{reason_veto} "
                f"(vol_20={prediction.vol_20:.4f}); espera ({'con posición' if has_pos else 'efectivo'})."
            )

        bull, bear = self._default_bull_bear(prediction, laya1, news_context)
        return {
            "action": action,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "justification": just[:280],
            "bull_case": bull,
            "bear_case": bear,
        }

    def _build_orchestrator_feedback(
        self,
        prediction: PredictionResult,
        laya1: LayaAlphaOutput,
        portfolio: PortfolioState,
        decision_action: str,
        decision: Optional[QwenDecisionOutput] = None,
        is_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Extrae retroalimentación real generada por Qwen LLM cuando está presente y válida,
        o usa reglas calibradas deterministas etiquetadas como `rule_fallback`.
        """
        aligned = decision_action == laya1.signal_action
        high_vol = (laya1.regime == "high_vol_chop") or (prediction.vol_20 > 0.035)
        has_pos = portfolio.position_qty >= self.settings.min_lot_size

        llm_fb1 = (decision.feedback_to_laya1 if decision else None) or {}
        llm_fb2 = (decision.feedback_to_laya2 if decision else None) or {}
        used_llm_fb1 = (not is_fallback) and isinstance(llm_fb1, dict) and bool(llm_fb1.get("guidance"))
        used_llm_fb2 = (not is_fallback) and isinstance(llm_fb2, dict) and bool(llm_fb2.get("focus_rule"))

        if used_llm_fb1:
            try:
                temp_delta = float(max(-0.20, min(0.25, float(llm_fb1.get("temperature_delta", 0.0)))))
            except (TypeError, ValueError):
                temp_delta = 0.0
            try:
                conv_bias = float(max(-0.20, min(0.20, float(llm_fb1.get("conviction_bias", 0.0)))))
            except (TypeError, ValueError):
                conv_bias = 0.0
            laya1_guidance = str(llm_fb1.get("guidance", "")).strip()[:220]
        elif not aligned:
            temp_delta = 0.10
            conv_bias = -0.08
            laya1_guidance = (
                f"Divergencia Qwen({decision_action}) vs Laya#1({laya1.signal_action}): "
                f"suavizar temperatura (+0.10) y moderar sesgo de convicción (-0.08)."
            )
        elif high_vol:
            temp_delta = 0.08
            conv_bias = -0.04
            laya1_guidance = (
                f"Régimen volátil ({laya1.regime}, vol={prediction.vol_20:.4f}): "
                f"elevar cautela en política DRL (+0.08 temp, -0.04 sesgo)."
            )
        elif decision_action in {"BUY", "SELL"}:
            temp_delta = -0.08
            conv_bias = 0.05
            laya1_guidance = (
                f"Consenso DRL-Qwen ({decision_action} en {prediction.symbol}): "
                f"refuerzo positivo (+0.05 sesgo, -0.08 temp)."
            )
        else:
            temp_delta = 0.0
            conv_bias = 0.0
            laya1_guidance = (
                f"Consenso HOLD ({'con posición' if has_pos else 'sin posición'}): "
                f"mantener calibración DRL nominal."
            )

        if used_llm_fb2:
            strictness = str(llm_fb2.get("strictness_level", "standard")).strip().lower()
            if strictness not in {"standard", "strict"}:
                strictness = "strict" if (not aligned or high_vol) else "standard"
            try:
                min_coh = float(
                    max(0.55, min(0.95, float(llm_fb2.get("min_coherence_threshold", 0.75))))
                )
            except (TypeError, ValueError):
                min_coh = 0.85 if strictness == "strict" else 0.75
            focus_rule = str(llm_fb2.get("focus_rule", "VERIFICACION_LLM")).strip()[:80]
        elif not aligned:
            strictness = "strict"
            min_coh = 0.88
            focus_rule = "AUDITAR_DIVERGENCIA_SENAL"
        elif high_vol:
            strictness = "strict"
            min_coh = 0.88
            focus_rule = "AUDITAR_CONTRA_VOLATILIDAD"
        else:
            strictness = "standard"
            min_coh = 0.75
            focus_rule = "VERIFICACION_ESTANDAR_LONG_ONLY"

        fb_source = "llm_generated" if (used_llm_fb1 or used_llm_fb2) else "rule_fallback"
        return {
            "source_agent": "Qwen Orchestrator",
            "feedback_source": fb_source,
            "symbol": prediction.symbol,
            "candle_date": prediction.candle_close_utc,
            "qwen_action": decision_action,
            "laya1_signal": laya1.signal_action,
            "aligned": aligned,
            "to_laya1": {
                "target": "Laya #1 (DRL Local)",
                "feedback_source": "llm_generated" if used_llm_fb1 else "rule_fallback",
                "temperature_delta": round(temp_delta, 4),
                "conviction_bias": round(conv_bias, 4),
                "guidance": laya1_guidance,
                "aligned_with_qwen": aligned,
                "qwen_action": decision_action,
                "symbol": prediction.symbol,
            },
            "to_laya2": {
                "target": "Laya #2 (Guardia Local)",
                "feedback_source": "llm_generated" if used_llm_fb2 else "rule_fallback",
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
        raw_llm_response: str = "",
        prompt_summary: str = "",
        news_context: Optional[Dict[str, Any]] = None,
        episodic_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        is_fallback = ("simulated" in transport) or ("fallback" in transport)
        default_bull, default_bear = self._default_bull_bear(prediction, laya1, news_context)
        bull_case = (decision.bull_case or "").strip() or default_bull
        bear_case = (decision.bear_case or "").strip() or default_bear

        fb = self._build_orchestrator_feedback(
            prediction=prediction,
            laya1=laya1,
            portfolio=portfolio,
            decision_action=decision.action,
            decision=decision,
            is_fallback=is_fallback,
        )

        res: Dict[str, Any] = {
            "action": decision.action,
            "symbol": decision.symbol,
            "candle_date": decision.candle_date,
            "justification": decision.justification,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "execution_mode": execution_mode,
            "is_local_device": is_local_device,
            "is_fallback": is_fallback,
            "api_endpoint": api_endpoint,
            "model": model,
            "vram_budget_gb": vram_budget_gb,
            "temperature": self.settings.qwen_temperature,
            "latency_ms": round(latency_ms, 2),
            "transport": transport,
            "api_key_configured": api_key_configured,
            "orchestrator_feedback": fb,
            "raw_llm_response": raw_llm_response[:1200],
            "prompt_summary": prompt_summary[:1200],
            "news_context": news_context or {},
            "memories_used_count": len(episodic_memories or []),
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
        news_context: Optional[Dict[str, Any]] = None,
        episodic_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Switch Dinámico System-2:
        - EXTERNAL_API: invoca Qwen 14B por API HTTP remota (no local).
        - LOCAL_DEVICE: invoca Qwen 2.5 3B (<= 6GB VRAM) vía Ollama local (/v1/chat/completions).
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
        messages = self._build_messages(
            prediction, laya1, portfolio, news_context=news_context, episodic_memories=episodic_memories
        )
        prompt_summary = messages[1]["content"][:1200]

        # ======================================================================
        # RAMA LOCAL <= 6 GB VRAM: Qwen Local Real en Ollama (/api/chat)
        # ======================================================================
        if active_mode == "LOCAL_DEVICE":
            use_ollama_native = self.http_client is None
            local_endpoint = (
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
                if use_ollama_native
                else f"{self.settings.ollama_base_url.rstrip('/')}/v1/chat/completions"
            )
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
                    news_context=news_context,
                )
                raw_json_str = json.dumps(raw_obj, ensure_ascii=False)
                decision = self._validate_decision_contract(
                    raw_json_str,
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
                    raw_llm_response=raw_json_str,
                    prompt_summary=prompt_summary,
                    news_context=news_context,
                    episodic_memories=episodic_memories,
                )

            if use_ollama_native:
                local_body = {
                    "model": local_model,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "keep_alive": "24h",
                    "options": {
                        "temperature": self.settings.qwen_temperature,
                        "num_ctx": self.settings.qwen_local_num_ctx,
                        "num_predict": self.settings.qwen_max_tokens,
                    },
                    "messages": messages,
                }
            else:
                local_body = {
                    "model": local_model,
                    "temperature": self.settings.qwen_temperature,
                    "max_tokens": self.settings.qwen_max_tokens,
                    "keep_alive": "24h",
                    "response_format": {"type": "json_object"},
                    "options": {"num_ctx": self.settings.qwen_local_num_ctx},
                    "messages": messages,
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
                if isinstance(payload.get("message"), dict):
                    raw_llm = str(payload["message"].get("content", ""))
                else:
                    msg = payload["choices"][0]["message"]
                    raw_llm = str(msg.get("content") or msg.get("reasoning") or "")
                normalized_content = (
                    self._normalize_local_json_content(
                        raw_content=raw_llm,
                        prediction=prediction,
                        laya1=laya1,
                        portfolio=portfolio,
                        news_context=news_context,
                    )
                    if self.http_client is None
                    else raw_llm
                )
                decision = self._validate_decision_contract(
                    normalized_content,
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
                    raw_llm_response=raw_llm,
                    prompt_summary=prompt_summary,
                    news_context=news_context,
                    episodic_memories=episodic_memories,
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
                    news_context=news_context,
                )
                raw_json_str = json.dumps(raw_obj, ensure_ascii=False)
                decision = self._validate_decision_contract(
                    raw_json_str,
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
                    raw_llm_response=raw_json_str,
                    prompt_summary=prompt_summary,
                    news_context=news_context,
                    episodic_memories=episodic_memories,
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
            raw_obj = self._dry_run_external_api_simulation(
                prediction, laya1, portfolio, news_context=news_context
            )
            raw_json_str = json.dumps(raw_obj, ensure_ascii=False)
            decision = self._validate_decision_contract(
                raw_json_str,
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
                raw_llm_response=raw_json_str,
                prompt_summary=prompt_summary,
                news_context=news_context,
                episodic_memories=episodic_memories,
            )

        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.qwen_model_name,
            "temperature": self.settings.qwen_temperature,
            "max_tokens": self.settings.qwen_max_tokens,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(timeout=float(self.settings.llm_timeout_seconds))
        try:
            resp = client.post(endpoint_url, headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()
            raw_content = str(payload["choices"][0]["message"]["content"])
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
            raw_llm_response=raw_content,
            prompt_summary=prompt_summary,
            news_context=news_context,
            episodic_memories=episodic_memories,
        )
