"""
Front Office - Agente Laya #1 (Triaje de Alpha, Régimen y Convicción System-1).

# ==============================================================================
# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
# Este agente SÍ se ejecuta 100% en el dispositivo local vía Ollama (/v1/systemone),
# separado de Qwen 14B (que se invoca por API externa).
# ==============================================================================
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import torch
import torch.nn as nn

from backend.app.back_office.database import TradingDatabase
from backend.app.core.config import ARTIFACTS_DIR, Settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.predictor import PredictionResult

# Marcadores explícitos de ejecución en dispositivo local
LAYA1_EXECUTION_MODE: str = "LOCAL_DEVICE"
LAYA1_IS_LOCAL_DEVICE: bool = True

DRL_ACTIONS: List[str] = ["BUY", "HOLD", "SELL"]
DRL_REGIMES: List[str] = ["bull_trend", "bear_trend", "sideways", "high_vol_chop", "none"]
DRL_STATE_DIM: int = 10


def build_drl_state_vector(
    pred_return_t1: float,
    ret_lag_1: float,
    ema_ratio: float,
    vol_20: float,
    rsi_14: float,
    momentum_5: float,
    position_open: bool,
) -> np.ndarray:
    """Construye el vector de estado de 10 dimensiones para Laya #1 DRL."""
    r = float(pred_return_t1)
    v = max(float(vol_20), 1e-4)
    signed_sqrt_r = math.copysign(math.sqrt(min(abs(r) * 100.0, 25.0)), r)
    sharpe_like = float(np.clip(r / v, -5.0, 5.0))
    trend_sign = 1.0 if ema_ratio >= 1.0 else -1.0

    return np.asarray(
        [
            float(np.clip(r * 50.0, -5.0, 5.0)),
            float(np.clip(float(ret_lag_1) * 50.0, -5.0, 5.0)),
            float(np.clip((float(ema_ratio) - 1.0) * 50.0, -5.0, 5.0)),
            float(np.clip(v * 25.0, 0.0, 5.0)),
            float(np.clip((float(rsi_14) - 50.0) / 50.0, -1.0, 1.0)),
            float(np.clip(float(momentum_5) * 20.0, -5.0, 5.0)),
            1.0 if position_open else 0.0,
            signed_sqrt_r,
            sharpe_like,
            trend_sign,
        ],
        dtype=np.float32,
    )


class Laya1DRLPolicyNet(nn.Module):
    """
    Red Dueling Double-DQN + Advantage Actor-Critic (A2C) + Regime Head para Laya #1.
    Entrenada offline sobre splits temporales TR/VA (nunca TE) para decisiones
    de triaje de alpha, régimen y convicción calibrada.
    """

    def __init__(self, state_dim: int = DRL_STATE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, len(DRL_ACTIONS))
        self.policy_head = nn.Linear(hidden_dim, len(DRL_ACTIONS))
        self.regime_head = nn.Linear(hidden_dim, len(DRL_REGIMES))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        value = self.value_head(h)
        advantage = self.advantage_head(h)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        policy_logits = self.policy_head(h)
        regime_logits = self.regime_head(h)
        return q_values, policy_logits, regime_logits, value.squeeze(-1)


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
    qwen_feedback: Optional[Dict[str, Any]] = None
    q_values: Optional[Dict[str, float]] = None
    recommended_exposure_pct: float = 0.35
    is_fallback: bool = False

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
            "qwen_feedback": self.qwen_feedback or {},
            "q_values": self.q_values or {},
            "recommended_exposure_pct": round(self.recommended_exposure_pct, 4),
            "is_fallback": self.is_fallback,
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
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.gateway = gateway
        self.http_client = http_client
        self._last_unavailable_ts: Optional[float] = None
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.drl_net: Optional[Laya1DRLPolicyNet] = None
        self.drl_temperature: float = 1.0
        self.drl_model_tag: str = "laya1_drl_v1"
        self.conviction_bias: float = 0.0
        self.last_qwen_feedback: Dict[str, Any] = {}
        self._load_drl_model()
        try:
            saved_fb = self.db.get_state("laya1_feedback_json")
            if saved_fb:
                parsed = json.loads(saved_fb)
                if isinstance(parsed, dict):
                    self.last_qwen_feedback = parsed
                    self.conviction_bias = float(parsed.get("applied_conviction_bias", 0.0))
        except Exception:
            pass

    def apply_qwen_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica retroalimentación del orquestador Qwen sobre temperatura DRL y sesgo de convicción."""
        if not feedback:
            return dict(self.last_qwen_feedback)
        temp_delta = float(feedback.get("temperature_delta", 0.0))
        conv_bias = float(feedback.get("conviction_bias", 0.0))
        self.drl_temperature = round(max(0.35, min(2.0, self.drl_temperature + temp_delta)), 4)
        self.conviction_bias = round(max(-0.20, min(0.20, conv_bias)), 4)
        applied = {
            **feedback,
            "applied_drl_temperature": self.drl_temperature,
            "applied_conviction_bias": self.conviction_bias,
        }
        self.last_qwen_feedback = applied
        self.db.set_state("laya1_feedback_json", json.dumps(applied, ensure_ascii=False))
        return applied

    def _load_drl_model(self) -> None:
        weights_path = self.artifacts_dir / "laya1_drl_v1.pt"
        manifest_path = self.artifacts_dir / "laya1_drl_manifest.json"
        if not weights_path.exists():
            return
        try:
            hidden_dim = 64
            if manifest_path.exists():
                man = json.loads(manifest_path.read_text(encoding="utf-8"))
                hidden_dim = int(man.get("architecture", {}).get("hidden_dim", 64))
                self.drl_temperature = float(man.get("calibration_temperature", 1.0))
                self.drl_model_tag = str(man.get("model_version", "laya1_drl_v1"))
            net = Laya1DRLPolicyNet(state_dim=DRL_STATE_DIM, hidden_dim=hidden_dim)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state_dict)
            net.eval()
            self.drl_net = net
        except Exception:
            self.drl_net = None

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
        """
        Inferencia local mediante Deep Reinforcement Learning (Dueling Double-DQN + A2C)
        cuando el modelo `laya1_drl_v1.pt` está cargado, o fallback determinista si no existe.
        """
        if self.drl_net is not None:
            state_vec = build_drl_state_vector(
                pred_return_t1=prediction.pred_return_t1,
                ret_lag_1=prediction.ret_lag_1,
                ema_ratio=prediction.ema_ratio,
                vol_20=prediction.vol_20,
                rsi_14=prediction.rsi_14,
                momentum_5=prediction.momentum_5,
                position_open=position_open,
            )
            state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals, policy_logits, regime_logits, _ = self.drl_net(state_t)

                temp = max(0.25, float(self.drl_temperature))
                # Máscara Long-Only para evitar acciones inválidas en distribución
                masked_logits = policy_logits.clone()
                valid_buy = (prediction.pred_return_t1 > 0.0) and (not position_open)
                valid_sell = bool(position_open)
                if not valid_buy:
                    masked_logits[0, 0] -= 4.0
                if not valid_sell:
                    masked_logits[0, 2] -= 4.0

                act_probs_t = torch.softmax(masked_logits / temp, dim=-1)[0]
                reg_probs_t = torch.softmax(regime_logits / temp, dim=-1)[0]

            sig_probs = {
                act: round(float(act_probs_t[i].item()), 4)
                for i, act in enumerate(DRL_ACTIONS)
            }
            regime_probs = {
                reg: round(float(reg_probs_t[i].item()), 4)
                for i, reg in enumerate(DRL_REGIMES)
            }

            # Selección estricta sobre acciones permitidas en Long-Only
            valid_indices = [1]  # HOLD siempre permitido
            if valid_buy:
                valid_indices.append(0)
            if valid_sell:
                valid_indices.append(2)

            best_idx = max(
                valid_indices,
                key=lambda idx: float(act_probs_t[idx].item()) + 0.15 * float(q_vals[0, idx].item()),
            )
            action = DRL_ACTIONS[best_idx]
            regime_idx = int(torch.argmax(reg_probs_t).item())
            regime = DRL_REGIMES[regime_idx]

            sorted_p = sorted(sig_probs.values(), reverse=True)
            margin = sorted_p[0] - sorted_p[1]
            q_adv = float((q_vals[0, best_idx] - q_vals[0, 1]).clamp(-2.0, 2.0).item())
            conv_score = float(
                np.clip(
                    0.35 + 0.50 * margin + 0.15 * (1.0 / (1.0 + math.exp(-q_adv))) + self.conviction_bias,
                    0.15,
                    0.96,
                )
            )
            if conv_score >= 0.78:
                conv_level = "high"
            elif conv_score >= 0.58:
                conv_level = "moderate"
            elif conv_score >= 0.40:
                conv_level = "low"
            else:
                conv_level = "very_low"

            vol_scale = float(np.clip(0.018 / max(prediction.vol_20, 0.005), 0.35, 1.0))
            max_single = float(getattr(self.settings, "max_single_position_pct", 0.45))
            rec_exp = float(np.clip(max_single * vol_scale * conv_score * 1.35, 0.12, max_single))
            q_dict = {
                act: round(float(q_vals[0, i].item()), 4)
                for i, act in enumerate(DRL_ACTIONS)
            }

            return {
                "regime": regime,
                "regime_probs": regime_probs,
                "signal_action": action,
                "signal_probs": sig_probs,
                "conviction_level": conv_level,
                "conviction_score": round(conv_score, 4),
                "transport": "local_drl_dueling_a2c_v1",
                "model": f"{self.settings.laya_model_name} ({self.drl_model_tag})",
                "q_values": q_dict,
                "recommended_exposure_pct": round(rec_exp, 4),
                "is_fallback": False,
            }

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

        conv_score = round(float(np.clip(conv_score + self.conviction_bias, 0.15, 0.96)), 4)

        return {
            "regime": regime,
            "regime_probs": regime_probs,
            "signal_action": action,
            "signal_probs": sig_probs,
            "conviction_level": conv_level,
            "conviction_score": conv_score,
            "transport": "local_rlcd_calibrated_fallback",
            "model": self.settings.laya_model_name,
            "q_values": {},
            "recommended_exposure_pct": round(
                min(0.35, float(getattr(self.settings, "max_single_position_pct", 0.45))), 4
            ),
            "is_fallback": True,
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
            and (
                self.drl_net is not None
                or (
                    self.settings.allow_local_laya_calibrated_fallback
                    and self._last_unavailable_ts is not None
                    and (time.monotonic() - self._last_unavailable_ts) < 15.0
                )
            )
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
                transport=str(cal.get("transport", "local_rlcd_calibrated_fallback")),
                model=str(cal.get("model", self.settings.laya_model_name)),
                payload_token_estimate=token_est,
                qwen_feedback=dict(self.last_qwen_feedback),
                q_values=cal.get("q_values", {}),
                recommended_exposure_pct=float(cal.get("recommended_exposure_pct", 0.35)),
                is_fallback=bool(cal.get("is_fallback", False)),
            )

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=httpx.Timeout(
                timeout=min(float(self.settings.llm_timeout_seconds), 5.0),
                connect=0.25,
            )
        )
        try:
            resp = client.post(url, json=payload)
            if resp.status_code == 404:
                gen_url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
                gen_prompt = (
                    "Analiza el siguiente estado System-1 y devuelve SOLO JSON con claves "
                    "regime, signal_action, conviction_level, conviction_score: "
                    + json.dumps(payload["state"], ensure_ascii=False)
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
                resp.raise_for_status()
                self._last_unavailable_ts = None
                gen_data = resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                cal = self._calibrated_local_rlcd(prediction, position_open)
                raw_resp = gen_data.get("response", "{}")
                try:
                    parsed = json.loads(raw_resp) if isinstance(raw_resp, str) else {}
                except ValueError:
                    parsed = {}
                regime = str(parsed.get("regime", cal["regime"]))
                if regime not in {"bull_trend", "bear_trend", "sideways", "high_vol_chop", "none"}:
                    regime = cal["regime"]
                sig = str(parsed.get("signal_action", cal["signal_action"])).upper()
                if sig not in {"BUY", "HOLD", "SELL"}:
                    sig = cal["signal_action"]
                conv_score = float(parsed.get("conviction_score", cal["conviction_score"]))
                conv_score = max(0.0, min(1.0, conv_score))
                conv_level = str(parsed.get("conviction_level", cal["conviction_level"]))
                if conv_level not in {"very_low", "low", "moderate", "high"}:
                    conv_level = cal["conviction_level"]
                return LayaAlphaOutput(
                    regime=regime,
                    regime_probs=cal["regime_probs"],
                    signal_action=sig,
                    signal_probs=cal["signal_probs"],
                    conviction_level=conv_level,
                    conviction_score=round(conv_score, 4),
                    latency_ms=latency_ms,
                    execution_mode=LAYA1_EXECUTION_MODE,
                    is_local_device=LAYA1_IS_LOCAL_DEVICE,
                    transport="ollama_local_systemone",
                    model=self.settings.laya_model_name,
                    payload_token_estimate=token_est,
                    qwen_feedback=dict(self.last_qwen_feedback),
                )

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
                qwen_feedback=dict(self.last_qwen_feedback),
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
                transport=str(cal.get("transport", "local_rlcd_calibrated_fallback")),
                model=str(cal.get("model", self.settings.laya_model_name)),
                payload_token_estimate=token_est,
                qwen_feedback=dict(self.last_qwen_feedback),
                q_values=cal.get("q_values", {}),
                recommended_exposure_pct=float(cal.get("recommended_exposure_pct", 0.35)),
                is_fallback=bool(cal.get("is_fallback", False)),
            )
        finally:
            if owns_client:
                client.close()
