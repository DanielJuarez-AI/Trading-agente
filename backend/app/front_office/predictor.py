"""
Front Office - Predictor Cuantitativo (Modelo MLP / LSTM entrenado en Fase 1 / T4).

Carga los artefactos congelados (model_manifest.json, scaler_v1.joblib y lstm_v1.pt),
verifica su integridad SHA-256 y el contrato exacto de columnas antes de emitir
el rendimiento estimado r_hat_{t+1} en escala original.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.app.core.config import ARTIFACTS_DIR
from backend.app.front_office.features import (
    DEFAULT_SEQ_LEN,
    FEATURE_COLUMNS,
    build_sequences,
    compute_technical_features,
)


class LSTMRegressor(nn.Module):
    """Red LSTM para predicción cuantitativa de rendimiento siguiente r_{t+1}."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class MLPRegressor(nn.Module):
    """Perceptrón Multicapa para ventana temporal aplanada."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        seq_len: int = DEFAULT_SEQ_LEN,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        flat_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        return self.net(flat).squeeze(-1)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PredictionResult:
    symbol: str
    candle_close_utc: str
    pred_return_t1: float
    direction: str
    ema_ratio: float
    vol_20: float
    rsi_14: float
    momentum_5: float
    ret_lag_1: float
    model_version: str
    features_verified: bool


class QuantPredictor:
    """Cargador e inferencia de producción para el modelo T4 congelado."""

    def __init__(self, artifacts_dir: Path | None = None, verify_hashes: bool = True) -> None:
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.manifest_path = self.artifacts_dir / "model_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No se encontró model_manifest.json en {self.manifest_path}. "
                "Ejecute primero notebooks/generate_phase1_artifacts.py."
            )

        self.manifest: Dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.feature_columns: List[str] = list(
            self.manifest.get("feature_columns", FEATURE_COLUMNS)
        )
        self.seq_len: int = int(self.manifest.get("seq_len", DEFAULT_SEQ_LEN))
        self.model_version: str = str(self.manifest.get("model_version", "v1"))

        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Contrato de features inconsistente: manifest={self.feature_columns} "
                f"vs runtime={FEATURE_COLUMNS}"
            )

        scaler_path = self.artifacts_dir / self.manifest.get("scaler_file", "scaler_v1.joblib")
        model_path = self.artifacts_dir / self.manifest.get("model_file", "lstm_v1.pt")
        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Faltan artefactos de modelo en {self.artifacts_dir}: "
                f"scaler={scaler_path.exists()}, model={model_path.exists()}"
            )

        if verify_hashes:
            expected_hashes = self.manifest.get("sha256", {})
            if "scaler" in expected_hashes:
                actual_scaler_hash = compute_sha256(scaler_path)
                if actual_scaler_hash != expected_hashes["scaler"]:
                    raise RuntimeError("Hash SHA-256 inválido para scaler_v1.joblib")
            if "model" in expected_hashes:
                actual_model_hash = compute_sha256(model_path)
                if actual_model_hash != expected_hashes["model"]:
                    raise RuntimeError("Hash SHA-256 inválido para lstm_v1.pt")

        self.scaler = joblib.load(scaler_path)
        arch = self.manifest.get("architecture", {})
        hidden_dim = int(arch.get("hidden_dim", 32))
        num_layers = int(arch.get("num_layers", 1))
        dropout = float(arch.get("dropout", 0.1))

        self.model = LSTMRegressor(
            input_dim=len(self.feature_columns),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_from_candles(self, candles_df: pd.DataFrame, symbol: str) -> PredictionResult:
        """
        Recibe un DataFrame histórico de velas cerradas ordenado ascendentemente,
        calcula features sin leakage y predice r_hat_{t+1} para la última vela cerrada.
        """
        if len(candles_df) < 35:
            raise ValueError(
                f"Se requieren al menos 35 velas cerradas para inferencia segura; recibidas={len(candles_df)}"
            )

        feat_df = compute_technical_features(candles_df, drop_na=True)
        if len(feat_df) < self.seq_len:
            raise ValueError(
                f"Filas válidas tras warmup ({len(feat_df)}) < seq_len ({self.seq_len})."
            )

        x_raw = feat_df[self.feature_columns].to_numpy(dtype=np.float32)
        x_scaled = self.scaler.transform(x_raw)
        seqs, _ = build_sequences(x_scaled, targets=None, seq_len=self.seq_len)
        last_seq = torch.tensor(seqs[-1:], dtype=torch.float32)

        with torch.no_grad():
            pred_val = float(self.model(last_seq).item())

        last_row = feat_df.iloc[-1]
        close_time = str(last_row.get("close_time", last_row.get("timestamp", "")))
        direction = "UP" if pred_val > 0 else ("DOWN" if pred_val < 0 else "FLAT")

        return PredictionResult(
            symbol=symbol,
            candle_close_utc=close_time,
            pred_return_t1=pred_val,
            direction=direction,
            ema_ratio=float(last_row["ema_ratio"]),
            vol_20=float(last_row["vol_20"]),
            rsi_14=float(last_row["rsi_14"]),
            momentum_5=float(last_row["momentum_5"]),
            ret_lag_1=float(last_row["ret_lag_1"]),
            model_version=self.model_version,
            features_verified=True,
        )
