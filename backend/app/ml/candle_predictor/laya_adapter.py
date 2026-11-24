"""
Adaptador de estado latente para Laya #1 (`Laya1DRLPolicyNet`) sin dependencias
circulares hacia `predictor.py` ni `laya_alpha.py`.

Esquema extendido de `state_dim = 17 + latent_dim` (= 33 cuando `latent_dim = 16`):
  [0..9]   : 10 dimensiones legacy (`build_drl_state_vector`)
  [10]     : clip(candle_v2_return * 50, -5, 5)
  [11]     : clip(pred_volatility_t1 / max(atr_pct, 1e-4), 0, 5)
  [12..14] : P(DOWN), P(NEUTRAL), P(UP) calibradas
  [15]     : edge = P(UP) - P(DOWN)
  [16]     : abstain_flag (1.0 si HOLD/baja confianza, 0.0 si señal activa)
  [17..]   : vector latente `h_t` en [-1, 1] emitido por `NextCandleGRU.encode_state`
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

LEGACY_STATE_DIM: int = 10
CANDLE_V2_SIGNAL_DIM: int = 7


def laya_state_dim_v2(latent_dim: int = 16) -> int:
    """Devuelve la dimensión total del vector de estado extendido para Laya #1."""
    return LEGACY_STATE_DIM + CANDLE_V2_SIGNAL_DIM + int(latent_dim)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extend_state_with_candle_v2(
    legacy_vec: np.ndarray,
    prediction: Any,
    latent_dim: int = 16,
) -> np.ndarray:
    """
    Concatena de forma retrocompatible las 10 dimensiones legacy de `build_drl_state_vector`
    con las 7 señales calibradas + `latent_dim` componentes latentes de Candle Predictor v2.
    """
    base = np.asarray(legacy_vec, dtype=np.float32).reshape(-1)
    if len(base) != LEGACY_STATE_DIM:
        raise ValueError(
            f"legacy_vec debe tener exactamente {LEGACY_STATE_DIM} dimensiones; recibidas={len(base)}"
        )

    ldim = int(latent_dim)
    probs: Dict[str, Any] = _get_field(prediction, "direction_probs", {}) or {}
    latent_raw: Sequence[Any] = _get_field(prediction, "candle_v2_latent", ()) or ()

    has_v2 = bool(probs) and bool(latent_raw)
    if not has_v2:
        signals = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        latent_arr = np.zeros(ldim, dtype=np.float32)
        return np.concatenate([base, signals, latent_arr]).astype(np.float32)

    c2_ret = float(_get_field(prediction, "candle_v2_return", 0.0) or 0.0)
    pred_vol = float(_get_field(prediction, "pred_volatility_t1", 0.0) or 0.0)
    atr_pct = max(float(_get_field(prediction, "atr_pct", 0.015) or 0.015), 1e-4)

    p_down = float(probs.get("DOWN", 0.0))
    p_neutral = float(probs.get("NEUTRAL", 1.0))
    p_up = float(probs.get("UP", 0.0))
    edge = p_up - p_down
    abstain_flag = 1.0 if bool(_get_field(prediction, "abstain", False)) else 0.0

    signals = np.asarray(
        [
            float(np.clip(c2_ret * 50.0, -5.0, 5.0)),
            float(np.clip(pred_vol / atr_pct, 0.0, 5.0)),
            p_down,
            p_neutral,
            p_up,
            edge,
            abstain_flag,
        ],
        dtype=np.float32,
    )

    lat_vec = np.asarray(list(latent_raw), dtype=np.float32).reshape(-1)
    if len(lat_vec) >= ldim:
        latent_arr = lat_vec[:ldim]
    else:
        latent_arr = np.pad(lat_vec, (0, ldim - len(lat_vec)), mode="constant")

    return np.concatenate([base, signals, latent_arr]).astype(np.float32)
