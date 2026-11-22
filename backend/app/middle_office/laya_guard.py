"""
Middle Office - Agente Laya #2 (Guardia Semántica, Coherencia y Anti-Inyección).

# ==============================================================================
# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
# Este agente se ejecuta 100% en el dispositivo local vía Ollama (/v1/systemone),
# auditando la salida de Qwen 14B (API Externa) antes de llegar al Motor de Riesgo.
# ==============================================================================
"""

from __future__ import annotations

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
        }


class LayaGuardAgent:
    """
    [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
    Audita entradas/salidas frente a prompt injection y contradicciones numéricas
    con el modelo cuantitativo T4 o el estado real del portafolio.
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

        # Coherencia direccional contra r_hat_{t+1}
        if action == "BUY" and prediction.pred_return_t1 < -0.0015:
            return {
                "approved": False,
                "security_status": "SAFE",
                "coherence_status": "CONTRADICTION",
                "reason": (
                    f"Contradicción: Qwen propone BUY pero modelo T4 predice retorno negativo "
                    f"({prediction.pred_return_t1:.5f})."
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
            "coherence_prob": 0.91,
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
        [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
        Ejecuta Laya #2 vía Gateway antes de permitir el paso al Motor de Riesgo.
        """
        self.gateway.authorize_llm_call(
            target_agent="laya_guard",
            execution_mode=LAYA2_EXECUTION_MODE,
            gateway_key=gateway_key,
        )

        # Auditoría determinista previa siempre activa como primera barrera
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
            if not det["approved"]:
                self.db.log_error(
                    error_code="LAYA2_SEMANTIC_VETO",
                    component="LayaGuardAgent",
                    message=det["reason"],
                    payload_excerpt=str(qwen_decision.get("justification", ""))[:200],
                )
            return LayaGuardOutput(
                approved=det["approved"],
                security_status=det["security_status"],
                coherence_status=det["coherence_status"],
                reason=det["reason"],
                injection_prob=det["injection_prob"],
                coherence_prob=det["coherence_prob"],
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
            )

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=httpx.Timeout(
                timeout=min(float(self.settings.llm_timeout_seconds), 3.0),
                connect=0.25,
            )
        )
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            self._last_unavailable_ts = None
            data = resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            answers = data.get("answers", data)
            inj_label = str(answers.get("injection_risk", {}).get("label", "safe"))
            coh_label = str(answers.get("coherence", {}).get("label", "coherent"))

            ollama_ok = (inj_label == "safe") and (coh_label == "coherent")
            approved = det["approved"] and ollama_ok
            reason = (
                det["reason"]
                if not det["approved"]
                else (
                    "Auditoría semántica aprobada por Laya #2 local (Ollama)."
                    if ollama_ok
                    else f"Veto de Laya #2 local: injection={inj_label}, coherence={coh_label}"
                )
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
                coherence_status=det["coherence_status"] if not det["approved"] else coh_label.upper(),
                reason=reason,
                injection_prob=float(answers.get("injection_risk", {}).get("score", det["injection_prob"])),
                coherence_prob=float(answers.get("coherence", {}).get("score", det["coherence_prob"])),
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="ollama_local_systemone",
                model=self.settings.laya_model_name,
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
            if not det["approved"]:
                self.db.log_error(
                    error_code="LAYA2_SEMANTIC_VETO",
                    component="LayaGuardAgent",
                    message=det["reason"],
                    payload_excerpt=str(qwen_decision.get("justification", ""))[:200],
                )
            return LayaGuardOutput(
                approved=det["approved"],
                security_status=det["security_status"],
                coherence_status=det["coherence_status"],
                reason=det["reason"],
                injection_prob=det["injection_prob"],
                coherence_prob=det["coherence_prob"],
                latency_ms=latency_ms,
                execution_mode=LAYA2_EXECUTION_MODE,
                is_local_device=LAYA2_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
            )
        finally:
            if owns_client:
                client.close()
