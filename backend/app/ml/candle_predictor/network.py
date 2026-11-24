"""
Arquitectura PyTorch compacta multi-cabezal (`NextCandleGRU`) y utilidades
de calibración probabilística (Temperature Scaling + ECE) para Candle Predictor v2.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.app.ml.candle_predictor.config import CandleV2Config


class NextCandleGRU(nn.Module):
    """
    Predictor recurrente compacto (< 2 ms CPU) compartido entre todos los tickers:
    - Backbone GRU (por defecto) o LSTM con LayerNorm + Dropout
    - Cuello de botella latente `tanh` (`latent_dim`, exportable para Laya #1)
    - Tres cabezales desacoplados:
        1) `head_return`: retorno logarítmico normalizado r_{t+1}
        2) `head_volatility`: rango esperado normalizado TR_{t+1} (con Softplus)
        3) `head_direction`: logits 3 clases (`DOWN`, `NEUTRAL`, `UP`)
    """

    def __init__(self, n_features: int, cfg: CandleV2Config) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.cfg = cfg
        self.cell = cfg.cell
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_layers)
        self.latent_dim = int(cfg.latent_dim)

        rnn_cls = nn.GRU if cfg.cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(cfg.dropout) if self.num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.drop = nn.Dropout(float(cfg.dropout))
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_dim),
            nn.Tanh(),
        )
        self.head_return = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.head_volatility = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        self.head_direction = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def encode_state(
        self, x_seq: torch.Tensor, h_prev: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Extrae el vector latente compacto `h_t` (shape `(B, latent_dim)`) y el estado
        oculto recurrente siguiente `h_next` para consumo externo (ej. Laya #1 DRL).
        """
        out, h_next = self.rnn(x_seq, h_prev)
        last_hidden = out[:, -1, :]
        normed = self.drop(self.norm(last_hidden))
        latent = self.bottleneck(normed)
        return latent, h_next

    def forward(
        self, x_seq: torch.Tensor, h_prev: Any = None
    ) -> Dict[str, Any]:
        """
        Forward completo sobre ventanas `(B, seq_len, n_features)`.
        Nota: `ret` y `tr` se devuelven en escala normalizada por el escalador causal.
        """
        latent, h_next = self.encode_state(x_seq, h_prev=h_prev)
        z = self.drop(latent)
        ret = self.head_return(z).squeeze(-1)
        tr = self.head_volatility(z).squeeze(-1)
        logits = self.head_direction(z)
        return {
            "ret": ret,
            "tr": tr,
            "logits": logits,
            "latent": latent,
            "h_next": h_next,
        }


def calibrated_proba(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax numéricamente estable escalado por `temperature`."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    temp = max(float(temperature), 1e-4)
    scaled = arr / temp
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)


def _cross_entropy_nll(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    temp = max(float(temperature), 1e-4)
    scaled = logits / temp
    max_l = np.max(scaled, axis=-1, keepdims=True)
    log_sum_exp = np.squeeze(max_l, axis=-1) + np.log(
        np.sum(np.exp(scaled - max_l), axis=-1)
    )
    row_idx = np.arange(len(y_true))
    target_logits = scaled[row_idx, y_true.astype(np.int64)]
    return float(np.mean(log_sum_exp - target_logits))


def fit_temperature(logits: np.ndarray, y_dir: np.ndarray) -> float:
    """
    Ajusta Temperature Scaling sobre la malla log-espaciada `[0.25, 5.0]`
    minimizando NLL estrictamente sobre el conjunto de validación.
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y_dir, dtype=np.int64)
    if len(logits_arr) == 0 or len(y_arr) == 0:
        return 1.0

    grid = np.exp(np.linspace(np.log(0.25), np.log(5.0), 60))
    best_t = 1.0
    best_nll = float("inf")
    for temp in grid:
        nll = _cross_entropy_nll(logits_arr, y_arr, float(temp))
        if np.isfinite(nll) and nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    return round(best_t, 6)


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, bins: int = 10
) -> float:
    """Calcula Expected Calibration Error (ECE) sobre las probabilidades máximas."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)
    if len(probs_arr) == 0 or len(y_arr) == 0:
        return 0.0

    confidences = np.max(probs_arr, axis=-1)
    preds = np.argmax(probs_arr, axis=-1)
    accuracies = (preds == y_arr).astype(np.float64)

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    n = float(len(y_arr))
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)
