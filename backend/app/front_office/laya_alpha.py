"""
Front Office - Agente Laya #1 (Triaje de Alpha, Régimen y Convicción System-1).

# ==============================================================================
# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
# Este agente SÍ se ejecuta 100% en el dispositivo local vía Ollama (/v1/systemone),
# separado de Qwen 14B (que se invoca por API externa).
# ==============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from backend.app.back_office.database import TradingDatabase
from backend.app.core.config import Settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.predictor import PredictionResult

# Marcadores explícitos de ejecución en dispositivo local
LAYA1_EXECUTION_MODE: str = "LOCAL_DEVICE"
LAYA1_IS_LOCAL_DEVICE: bool = True


class LLMServiceUnavailableError(RuntimeError):
    """Error lanzado cuando Ollama local o la API externa de Qwen está inaccesible."""


@dataclass
class LayaAlphaOutput:
    regime: str
    regime_probs: Dict[str, float]
    signal_action: str
    signal_probs: Dict[str, float]
    conviction_level: str
    conviction_score: float
    latency_ms: float
    execution_mode: str
    is_local_device: bool
    transport: str
    model: str
    payload_token_estimate: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": "Laya #1 (Front Office Alpha Triage)",
            "execution_mode": self.execution_mode,
            "is_local_device": self.is_local_device,
            "transport": self.transport,
            "model": self.model,
            "regime": self.regime,
            "regime_probs": self.regime_probs,
            "signal_action": self.signal_action,
            "signal_probs": self.signal_probs,
            "conviction_level": self.conviction_level,
            "conviction_score": self.conviction_score,
            "latency_ms": round(self.latency_ms, 2),
            "payload_token_estimate": self.payload_token_estimate,
        }


class LayaAlphaAgent:
    """
    [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
    Agente System-1 para clasificación de régimen de mercado y convicción en <35 ms.
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

    def build_systemone_payload(
        self, prediction: PredictionResult, position_open: bool
    ) -> Dict[str, Any]:
        """Construye payload JSON ultracompacto (< 300 tokens) para /v1/systemone."""
        return {
            "model": self.settings.laya_model_name,
            "state": {
                "symbol": prediction.symbol,
                "candle_close_utc": prediction.candle_close_utc,
                "pred_return_t1": round(prediction.pred_return_t1, 6),
                "ema_ratio": round(prediction.ema_ratio, 6),
                "vol_20": round(prediction.vol_20, 6),
                "position_open": bool(position_open),
            },
            "questions": {
                "regime": {
                    "type": "choice",
                    "options": [
                        "bull_trend",
                        "bear_trend",
                        "sideways",
                        "high_vol_chop",
                        "none",
                    ],
                },
                "signal_action": {
                    "type": "choice",
                    "options": ["BUY", "HOLD", "SELL"],
                },
                "conviction": {
                    "type": "score",
                    "levels": ["very_low", "low", "moderate", "high"],
                },
            },
        }

    def _calibrated_local_rlcd(
        self, prediction: PredictionResult, position_open: bool
    ) -> Dict[str, Any]:
        """Inferencia probabilística calibrada cuando Ollama local no está en daemon."""
        r = prediction.pred_return_t1
        ema_r = prediction.ema_ratio
        vol = prediction.vol_20

        if vol > 0.045:
            regime = "high_vol_chop"
            regime_probs = {
                "bull_trend": 0.12,
                "bear_trend": 0.15,
                "sideways": 0.10,
                "high_vol_chop": 0.58,
                "none": 0.05,
            }
        elif ema_r >= 1.005 and r > 0:
            regime = "bull_trend"
            regime_probs = {
                "bull_trend": 0.72,
                "bear_trend": 0.05,
                "sideways": 0.15,
                "high_vol_chop": 0.05,
                "none": 0.03,
            }
        elif ema_r <= 0.995 and r < 0:
            regime = "bear_trend"
            regime_probs = {
                "bull_trend": 0.06,
                "bear_trend": 0.70,
                "sideways": 0.14,
                "high_vol_chop": 0.07,
                "none": 0.03,
            }
        else:
            regime = "sideways"
            regime_probs = {
                "bull_trend": 0.18,
                "bear_trend": 0.17,
                "sideways": 0.55,
                "high_vol_chop": 0.06,
                "none": 0.04,
            }

        if r > 0.0008 and ema_r >= 0.998 and not position_open:
            action = "BUY"
            sig_probs = {"BUY": 0.76, "HOLD": 0.20, "SELL": 0.04}
        elif r < -0.0008 and position_open:
            action = "SELL"
            sig_probs = {"BUY": 0.05, "HOLD": 0.22, "SELL": 0.73}
        else:
            action = "HOLD"
            sig_probs = {"BUY": 0.18, "HOLD": 0.68, "SELL": 0.14}

        abs_r = abs(r)
        if abs_r >= 0.008 and regime != "high_vol_chop":
            conv_level, conv_score = "high", 0.84
        elif abs_r >= 0.0025:
            conv_level, conv_score = "moderate", 0.68
        elif abs_r >= 0.0008:
            conv_level, conv_score = "low", 0.45
        else:
            conv_level, conv_score = "very_low", 0.22

        return {
            "regime": regime,
            "regime_probs": regime_probs,
            "signal_action": action,
            "signal_probs": sig_probs,
            "conviction_level": conv_level,
            "conviction_score": conv_score,
        }

    def evaluate(
        self,
        prediction: PredictionResult,
        position_open: bool,
        gateway_key: str,
        strict_live: bool = False,
    ) -> LayaAlphaOutput:
        """
        [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
        Invoca a Laya #1 pasando obligatoriamente por el PrivateAuthGateway.
        """
        self.gateway.authorize_llm_call(
            target_agent="laya_alpha",
            execution_mode=LAYA1_EXECUTION_MODE,
            gateway_key=gateway_key,
        )

        payload = self.build_systemone_payload(prediction, position_open)
        token_est = max(50, len(str(payload)) // 4)
        url = f"{self.settings.ollama_base_url.rstrip('/')}/v1/systemone"

        t0 = time.perf_counter()
        if (
            self.http_client is None
            and not strict_live
            and self.settings.allow_local_laya_calibrated_fallback
            and self._last_unavailable_ts is not None
            and (time.monotonic() - self._last_unavailable_ts) < 15.0
        ):
            cal = self._calibrated_local_rlcd(prediction, position_open)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return LayaAlphaOutput(
                regime=cal["regime"],
                regime_probs=cal["regime_probs"],
                signal_action=cal["signal_action"],
                signal_probs=cal["signal_probs"],
                conviction_level=cal["conviction_level"],
                conviction_score=cal["conviction_score"],
                latency_ms=latency_ms,
                execution_mode=LAYA1_EXECUTION_MODE,
                is_local_device=LAYA1_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
                payload_token_estimate=token_est,
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

            regime_block = answers.get("regime", {})
            signal_block = answers.get("signal_action", {})
            conv_block = answers.get("conviction", {})

            return LayaAlphaOutput(
                regime=str(regime_block.get("label", "none")),
                regime_probs=dict(regime_block.get("probabilities", {})),
                signal_action=str(signal_block.get("label", "HOLD")),
                signal_probs=dict(signal_block.get("probabilities", {})),
                conviction_level=str(conv_block.get("label", "low")),
                conviction_score=float(conv_block.get("score", 0.5)),
                latency_ms=latency_ms,
                execution_mode=LAYA1_EXECUTION_MODE,
                is_local_device=LAYA1_IS_LOCAL_DEVICE,
                transport="ollama_local_systemone",
                model=self.settings.laya_model_name,
                payload_token_estimate=token_est,
            )
        except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as exc:
            self._last_unavailable_ts = time.monotonic()
            if strict_live or not self.settings.allow_local_laya_calibrated_fallback:
                self.db.log_error(
                    error_code="OLLAMA_LOCAL_UNAVAILABLE",
                    component="LayaAlphaAgent",
                    message=f"Ollama local (/v1/systemone) inaccesible para Laya #1: {exc}",
                    payload_excerpt=f"url={url}, symbol={prediction.symbol}",
                )
                self.db.set_state("system_status", "PAUSED")
                raise LLMServiceUnavailableError(
                    f"Ollama local detenido o inaccesible para Laya #1: {exc}"
                ) from exc

            # Fallback calibrado determinista cuando el daemon local de Ollama no está activo en desarrollo
            cal = self._calibrated_local_rlcd(prediction, position_open)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return LayaAlphaOutput(
                regime=cal["regime"],
                regime_probs=cal["regime_probs"],
                signal_action=cal["signal_action"],
                signal_probs=cal["signal_probs"],
                conviction_level=cal["conviction_level"],
                conviction_score=cal["conviction_score"],
                latency_ms=latency_ms,
                execution_mode=LAYA1_EXECUTION_MODE,
                is_local_device=LAYA1_IS_LOCAL_DEVICE,
                transport="local_rlcd_calibrated_fallback",
                model=self.settings.laya_model_name,
                payload_token_estimate=token_est,
            )
        finally:
            if owns_client:
                client.close()
