"""
Middle Office - Agente Laya #2 (Guardia Semántica, Coherencia y Anti-Inyección).

Ejecutado 100% en el dispositivo local vía Ollama (/api/generate o /v1/systemone),
auditando la salida de Qwen contra prompt injection, alucinaciones numéricas
y cumplimiento estricto de `min_coherence_threshold` inyectado por el orquestador.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.paper_engine import PortfolioState
from backend.app.core.config import Settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.laya_alpha import LLMServiceUnavailableError
from backend.app.front_office.predictor import PredictionResult

# Marcadores explícitos de ejecución en dispositivo local
LAYA2_EXECUTION_MODE: str = "LOCAL_DEVICE"
LAYA2_IS_LOCAL_DEVICE: bool = True

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"ignore\s+all\s+rules",
    r"system\s*prompt",
    r"override\s+risk",
    r"enable_real_trading\s*=\s*true",
    r"max_exposure",
    r"drop\s+table",
    r"<\s*script",
    r"bypass\s+gateway",
]


@dataclass
class LayaGuardOutput:
    approved: bool
    security_status: str
    coherence_status: str
    reason: str
    injection_prob: float
    coherence_prob: float
    latency_ms: float
    execution_mode: str
    is_local_device: bool
    transport: str
    model: str
    qwen_feedback: Optional[Dict[str, Any]] = None
    is_fallback: bool = False
    min_coherence_threshold: float = 0.75
    raw_llm_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": "Laya #2 (Middle Office Semantic Guard)",
            "execution_mode": self.execution_mode,
            "is_local_device": self.is_local_device,
            "transport": self.transport,
            "model": self.model,
            "approved": self.approved,
            "security_status": self.security_status,
            "coherence_status": self.coherence_status,
            "reason": self.reason,
            "injection_prob": round(self.injection_prob, 3),
            "coherence_prob": round(self.coherence_prob, 3),
            "latency_ms": round(self.latency_ms, 2),
            "qwen_feedback": self.qwen_feedback or {},
            "is_fallback": self.is_fallback,
            "min_coherence_threshold": round(self.min_coherence_threshold, 2),
            "raw_llm_response": self.raw_llm_response[:600],
        }


class LayaGuardAgent:
    """
    [DISPOSITIVO LOCAL - OLLAMA]
    Audita entradas/salidas frente a prompt injection, umbral dinámico `min_coherence_threshold`
    y contradicciones numéricas con Quant v2 o el estado real del portafolio.
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
        self._last_unavailable_ts: Optional[float] = None
        self.strictness_level: str = "standard"
        self.min_coherence_threshold: float = 0.75
        self.last_qwen_feedback: Dict[str, Any] = {}
        try:
            saved_fb = self.db.get_state("laya2_feedback_json")
            if saved_fb:
                parsed = json.loads(saved_fb)
                if isinstance(parsed, dict):
                    self.last_qwen_feedback = parsed
                    self.strictness_level = str(parsed.get("strictness_level", "standard"))
                    self.min_coherence_threshold = float(parsed.get("min_coherence_threshold", 0.75))
        except Exception:
            pass

    def apply_qwen_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica retroalimentación de rigor del orquestador Qwen hacia Laya #2."""
        if not feedback:
            return dict(self.last_qwen_feedback)
        strictness = str(feedback.get("strictness_level", "standard")).lower()
        if strictness not in {"standard", "strict"}:
            strictness = "standard"
        min_coh = float(feedback.get("min_coherence_threshold", 0.88 if strictness == "strict" else 0.75))
        self.strictness_level = strictness
        self.min_coherence_threshold = round(max(0.5, min(0.98, min_coh)), 2)
        applied = {
            **feedback,
            "strictness_level": self.strictness_level,
            "min_coherence_threshold": self.min_coherence_threshold,
        }
        self.last_qwen_feedback = applied
        self.db.set_state("laya2_feedback_json", json.dumps(applied, ensure_ascii=False))
        return applied

    @staticmethod
    def _detect_injection_patterns(text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in INJECTION_PATTERNS)

    def _deterministic_semantic_audit(
        self,
        prediction: PredictionResult,
        qwen_decision: Dict[str, Any],
        portfolio: PortfolioState,
        extra_metadata: str = "",
    ) -> Dict[str, Any]:
        justification = str(qwen_decision.get("justification", ""))
        action = str(qwen_decision.get("action", "HOLD"))

        if self._detect_injection_patterns(justification) or self._detect_injection_patterns(
            extra_metadata
        ):
            return {
                "approved": False,
                "security_status": "PROMPT_INJECTION_DETECTED",
                "coherence_status": "REJECTED",
                "reason": "Patrón de prompt injection detectado en metadatos o justificación.",
                "injection_prob": 0.96,
                "coherence_prob": 0.05,
            }

        # Coherencia con posición abierta (Long-Only)
        if action == "SELL" and portfolio.position_qty < self.settings.min_lot_size:
            return {
                "approved": False,
                "security_status": "SAFE",
                "coherence_status": "CONTRADICTION",
                "reason": "Contradicción: Qwen propone SELL sin posición abierta en portafolio.",
                "injection_prob": 0.02,
                "coherence_prob": 0.08,
            }

        # Coherencia direccional contra r_hat_{t+1} (ajustada por feedback Qwen)
        buy_neg_limit = -0.0005 if self.strictness_level == "strict" else -0.0015
        if action == "BUY" and prediction.pred_return_t1 < buy_neg_limit:
            return {
                "approved": False,
                "security_status": "SAFE",
                "coherence_status": "CONTRADICTION",
                "reason": (
                    f"Contradicción ({self.strictness_level}): Qwen propone BUY pero modelo predice "
                    f"retorno negativo ({prediction.pred_return_t1:.5f} < {buy_neg_limit:.4f})."
                ),
                "injection_prob": 0.02,
                "coherence_prob": 0.14,
            }

        just_lower = justification.lower()
        if prediction.pred_return_t1 < 0 and "positivo" in just_lower and "negativo" not in just_lower:
            return {
                "approved": False,
                "security_status": "SAFE",
                "coherence_status": "CONTRADICTION",
                "reason": "Alucinación textual: Qwen cita pronóstico positivo cuando r_hat < 0.",
                "injection_prob": 0.03,
                "coherence_prob": 0.12,
            }

        return {
            "approved": True,
            "security_status": "SAFE",
            "coherence_status": "COHERENT",
            "reason": "Auditoría semántica y anti-inyección aprobada por Laya #2 local.",
            "injection_prob": 0.02,
            "coherence_prob": 0.92,
        }

    def audit(
        self,
        prediction: PredictionResult,
        qwen_decision: Dict[str, Any],
        portfolio: PortfolioState,
        gateway_key: str,
        extra_metadata: str = "",
        strict_live: bool = False,
    ) -> LayaGuardOutput:
        """
        [DISPOSITIVO LOCAL - OLLAMA]
        Ejecuta Laya #2 vía Gateway y valida `min_coherence_threshold` antes de pasar al Motor de Riesgo.
        """
        self.gateway.authorize_llm_call(
            target_agent="laya_guard",
            execution_mode=LAYA2_EXECUTION_MODE,
            gateway_key=gateway_key,
        )

        det = self._deterministic_semantic_audit(
            prediction, qwen_decision, portfolio, extra_metadata=extra_metadata
        )

        payload = {
            "model": self.settings.laya_model_name,
            "state": {
                "symbol": prediction.symbol,
                "pred_return_t1": round(prediction.pred_return_t1, 6),
                "proposed_action": qwen_decision.get("action", "HOLD"),
                "position_open": portfolio.position_qty >= self.settings.min_lot_size,
                "min_coherence_threshold": self.min_coherence_threshold,
                "justification_excerpt": str(qwen_decision.get("justification", ""))[:180],
            },
            "questions": {
                "injection_risk": {
                    "type": "choice",
                    "options": ["safe", "suspicious", "prompt_injection", "none"],
                },
                "coherence": {
                    "type": "choice",
                    "options": ["coherent", "contradiction", "none"],
                },
            },
        }
        url = f"{self.settings.ollama_base_url.rstrip('/')}/v1/systemone"
        t0 = time.perf_counter()

        if (
            self.http_client is None
            and not strict_live
            and self.settings.allow_local_laya_calibrated_fallback
            and self._last_unavailable_ts is not None
            and (time.monotonic() - self._last_unavailable_ts) < 15.0
        ):
            latency_ms = (time.perf_counter() - t0) * 1000.0
            det_ok = det["approved"] and (det["coherence_prob"] >= self.min_coherence_threshold)
            reason = det["reason"]
            if det["approved"] and not det_ok:
                reason = (
                    f"Veto de Laya #2 (fallback): coherence_prob={det['coherence_prob']:.2f} "
                    f"< umbral={self.min_coherence_threshold:.2f}."
                )
            if not det_ok:
                self.db.log_error(
                    error_code="LAYA2_SEMANTIC_VETO",
                    component="LayaGuardAgent",
                    message=reason,
                    payload_excerpt=str(qwen_decision.get("justification", ""))[:200],
                )
            return LayaGuardOutput(
                approved=det_ok,
                security_status=det["security_status"],
                coherence_status=det["coherence_status"] if det_ok else "BELOW_THRESHOLD",
                reason=reason,
                injection_prob=det["injection_prob"],
                coherence_prob=det["coherence_prob"],
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
                qwen_feedback=dict(self.last_qwen_feedback),
                is_fallback=True,
                min_coherence_threshold=self.min_coherence_threshold,
            )

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=httpx.Timeout(
                timeout=min(float(self.settings.llm_timeout_seconds), 8.0),
                connect=0.25,
            )
        )
        try:
            if self.http_client is None:
                gen_url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
                gen_prompt = (
                    "Eres un auditor de seguridad y coherencia financiera Long-Only. "
                    "Evalúa objetivamente el estado y responde SOLO JSON con el esquema: "
                    '{"injection_risk": "<safe|suspicious|prompt_injection>", '
                    '"coherence": "<coherent|contradiction>", '
                    '"injection_score": <float_0_a_1>, "coherence_score": <float_0_a_1>}. '
                    f"Estado a auditar: {json.dumps(payload['state'], ensure_ascii=False)}"
                )
                resp = client.post(
                    gen_url,
                    json={
                        "model": self.settings.laya_model_name,
                        "prompt": gen_prompt,
                        "format": "json",
                        "stream": False,
                        "keep_alive": "24h",
                        "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 80},
                    },
                )
            else:
                resp = client.post(url, json=payload)
                if resp.status_code == 404:
                    gen_url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
                    resp = client.post(
                        gen_url,
                        json={
                            "model": self.settings.laya_model_name,
                            "prompt": json.dumps(payload["state"], ensure_ascii=False),
                            "format": "json",
                            "stream": False,
                            "keep_alive": "24h",
                            "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 80},
                        },
                    )
            resp.raise_for_status()
            self._last_unavailable_ts = None
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw_llm_resp = ""
            if isinstance(data.get("response"), str):
                raw_llm_resp = data["response"]
                try:
                    parsed_gen = json.loads(raw_llm_resp)
                    if not isinstance(parsed_gen, dict):
                        parsed_gen = {}
                except ValueError:
                    parsed_gen = {}
                inj_label = str(parsed_gen.get("injection_risk", "safe")).strip().lower()
                coh_label = str(parsed_gen.get("coherence", "coherent")).strip().lower()
                if inj_label not in {"safe", "suspicious", "prompt_injection"}:
                    inj_label = "safe"
                if coh_label not in {"coherent", "contradiction"}:
                    coh_label = "coherent"
                try:
                    inj_score = float(parsed_gen.get("injection_score", det["injection_prob"]))
                except (TypeError, ValueError):
                    inj_score = det["injection_prob"]
                try:
                    coh_score = float(parsed_gen.get("coherence_score", det["coherence_prob"]))
                except (TypeError, ValueError):
                    coh_score = det["coherence_prob"]
                inj_score = max(0.0, min(1.0, inj_score))
                coh_score = max(0.0, min(1.0, coh_score))
            else:
                raw_llm_resp = json.dumps(data, ensure_ascii=False)
                answers = data.get("answers", data)
                inj_label = str(answers.get("injection_risk", {}).get("label", "safe"))
                coh_label = str(answers.get("coherence", {}).get("label", "coherent"))
                inj_score = float(answers.get("injection_risk", {}).get("score", det["injection_prob"]))
                coh_score = float(answers.get("coherence", {}).get("score", det["coherence_prob"]))

            threshold_ok = coh_score >= self.min_coherence_threshold
            inj_ok = (inj_label == "safe") and (inj_score < 0.50)
            coh_ok = (coh_label == "coherent") and threshold_ok
            ollama_ok = inj_ok and coh_ok
            approved = det["approved"] and ollama_ok

            if not det["approved"]:
                reason = det["reason"]
            elif not inj_ok:
                reason = f"Veto de Laya #2 local: riesgo de seguridad {inj_label} (score={inj_score:.2f})."
            elif not threshold_ok:
                reason = (
                    f"Veto de Laya #2 local: coherence_score={coh_score:.2f} < "
                    f"umbral requerido={self.min_coherence_threshold:.2f} ({self.strictness_level})."
                )
            elif not coh_ok:
                reason = f"Veto de Laya #2 local: coherencia={coh_label}."
            else:
                reason = (
                    f"Auditoría semántica aprobada por Laya #2 local "
                    f"(coherencia={coh_score:.2f} >= {self.min_coherence_threshold:.2f})."
                )

            if not approved:
                self.db.log_error(
                    error_code="LAYA2_SEMANTIC_VETO",
                    component="LayaGuardAgent",
                    message=reason,
                    payload_excerpt=str(qwen_decision.get("justification", ""))[:200],
                )
            return LayaGuardOutput(
                approved=approved,
                security_status=det["security_status"] if not det["approved"] else inj_label.upper(),
                coherence_status=(
                    det["coherence_status"]
                    if not det["approved"]
                    else ("BELOW_THRESHOLD" if not threshold_ok else coh_label.upper())
                ),
                reason=reason,
                injection_prob=inj_score,
                coherence_prob=coh_score,
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="ollama_local_systemone",
                model=self.settings.laya_model_name,
                qwen_feedback=dict(self.last_qwen_feedback),
                is_fallback=False,
                min_coherence_threshold=self.min_coherence_threshold,
                raw_llm_response=raw_llm_resp,
            )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as exc:
            self._last_unavailable_ts = time.monotonic()
            if strict_live or not self.settings.allow_local_laya_calibrated_fallback:
                self.db.log_error(
                    error_code="OLLAMA_LOCAL_UNAVAILABLE",
                    component="LayaGuardAgent",
                    message=f"Ollama local (/v1/systemone) inaccesible para Laya #2: {exc}",
                    payload_excerpt=f"url={url}, symbol={prediction.symbol}",
                )
                self.db.set_state("system_status", "PAUSED")
                raise LLMServiceUnavailableError(
                    f"Ollama local detenido o inaccesible para Laya #2: {exc}"
                ) from exc

            latency_ms = (time.perf_counter() - t0) * 1000.0
            det_ok = det["approved"] and (det["coherence_prob"] >= self.min_coherence_threshold)
            reason = det["reason"]
            if det["approved"] and not det_ok:
                reason = (
                    f"Veto de Laya #2 (fallback): coherence_prob={det['coherence_prob']:.2f} "
                    f"< umbral={self.min_coherence_threshold:.2f}."
                )
            if not det_ok:
                self.db.log_error(
                    error_code="LAYA2_SEMANTIC_VETO",
                    component="LayaGuardAgent",
                    message=reason,
                    payload_excerpt=str(qwen_decision.get("justification", ""))[:200],
                )
            return LayaGuardOutput(
                approved=det_ok,
                security_status=det["security_status"],
                coherence_status=det["coherence_status"] if det_ok else "BELOW_THRESHOLD",
                reason=reason,
                injection_prob=det["injection_prob"],
                coherence_prob=det["coherence_prob"],
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
                qwen_feedback=dict(self.last_qwen_feedback),
                is_fallback=True,
                min_coherence_threshold=self.min_coherence_threshold,
            )
        finally:
            if owns_client:
                client.close()
