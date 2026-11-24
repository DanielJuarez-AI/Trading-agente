"""
Métricas Cuantitativas Institucionales & Dimensionamiento de Posición
(Inspirado en ML4T 3rd Ed., Awesome-Quant y FinceptTerminal).

Incluye:
- Probabilistic Sharpe Ratio (PSR) y Deflated Sharpe Ratio (DSR)
- Value-at-Risk (VaR) y Conditional VaR / Expected Shortfall (CVaR 95%)
- Spearman Rank Information Coefficient (Rank IC)
- Dimensionamiento Half-Kelly Fraccional combinado con Volatility Targeting
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

EULER_GAMMA: float = 0.5772156649015329


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inversa de la CDF Normal estándar mediante bisección robusta (sin SciPy)."""
    prob = max(1e-9, min(1.0 - 1e-9, float(p)))
    lo, hi = -8.0, 8.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def probabilistic_sharpe_ratio(
    returns: Sequence[float] | np.ndarray, benchmark_sr: float = 0.0
) -> float:
    """
    Calcula el Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012)
    corregido por asimetría (skewness) y curtosis muestral.
    """
    r = _finite_array(returns)
    n = len(r)
    if n < 3:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std <= 1e-10:
        return 0.0

    sr = float(np.mean(r)) / std
    z = (r - float(np.mean(r))) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))

    denom_sq = max(1e-8, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2))
    stat = ((sr - float(benchmark_sr)) * math.sqrt(float(n - 1))) / math.sqrt(denom_sq)
    return round(float(_norm_cdf(stat)), 4)


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray, n_trials: int = 30
) -> float:
    """
    Calcula el Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
    ajustando el umbral contra sesgo de selección múltiple en `n_trials` pruebas.
    """
    r = _finite_array(returns)
    n = len(r)
    if n < 3:
        return 0.0
    if int(n_trials) <= 1:
        return probabilistic_sharpe_ratio(r, benchmark_sr=0.0)

    std_sr = math.sqrt(1.0 / max(float(n - 1), 1.0))
    trials = float(max(2, int(n_trials)))
    z1 = _norm_ppf(1.0 - (1.0 / trials))
    z2 = _norm_ppf(1.0 - (1.0 / (trials * math.e)))
    sr_star = std_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return probabilistic_sharpe_ratio(r, benchmark_sr=sr_star)


def historical_var_cvar(
    returns: Sequence[float] | np.ndarray, alpha: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula VaR y CVaR (Expected Shortfall) históricos al nivel `alpha` (ej. 0.95).
    Devuelve magnitudes positivas en fracción decimal (ej. 0.021 = 2.1% de pérdida).
    """
    r = _finite_array(returns)
    if len(r) < 3:
        return (0.0, 0.0)
    q = float(np.quantile(r, max(0.001, min(0.5, 1.0 - float(alpha)))))
    var_val = max(0.0, -q)
    tail = r[r <= q]
    cvar_val = max(var_val, float(-np.mean(tail)) if len(tail) > 0 else var_val)
    return (round(var_val, 6), round(cvar_val, 6))


def spearman_rank_ic(preds: Sequence[float], realized: Sequence[float]) -> float:
    """Correlación de Spearman (Rank IC) sin dependencia externa de SciPy."""
    p_arr = np.asarray(preds, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(realized, dtype=np.float64).reshape(-1)
    if len(p_arr) != len(y_arr) or len(p_arr) < 3:
        return 0.0
    mask = np.isfinite(p_arr) & np.isfinite(y_arr)
    p_clean = p_arr[mask]
    y_clean = y_arr[mask]
    if len(p_clean) < 3 or float(np.std(p_clean)) <= 1e-12 or float(np.std(y_clean)) <= 1e-12:
        return 0.0
    corr = pd.Series(p_clean).rank().corr(pd.Series(y_clean).rank())
    return round(0.0 if (corr is None or np.isnan(corr)) else float(corr), 4)


def fractional_kelly_allocation(
    p_win: float,
    payoff_ratio: float = 1.75,
    kelly_fraction: float = 0.5,
    vol_20: float = 0.02,
    target_vol: float = 0.018,
    min_alloc: float = 0.10,
    max_alloc: float = 0.45,
) -> float:
    """
    Dimensionamiento Half-Kelly fraccional escalado por objetivo de volatilidad.
    Garantiza asignación acotada dentro de `[min_alloc, max_alloc]`.
    """
    p = float(np.clip(p_win, 0.01, 0.99))
    b = max(float(payoff_ratio), 1e-4)
    raw_kelly = max(0.0, p - ((1.0 - p) / b)) * float(kelly_fraction)
    vol_scale = float(np.clip(float(target_vol) / max(float(vol_20), 0.005), 0.35, 1.15))

    if raw_kelly <= 0.0:
        alloc = float(min_alloc) * min(vol_scale, 1.0)
    else:
        alloc = max(float(min_alloc), raw_kelly * vol_scale)
    return round(float(np.clip(alloc, float(min_alloc), float(max_alloc))), 4)
