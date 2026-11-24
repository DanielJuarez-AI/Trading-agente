"""
Configuración central y utilidades de universo para Candle Predictor v2 (1h / 1d).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from backend.app.core.config import CONFIG_DIR, PRIMARY_TICKERS, SYMBOL_ALIASES

CLASSES: Tuple[str, ...] = ("DOWN", "NEUTRAL", "UP")
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
WARMUP_BARS: int = 61
CRYPTO_SUFFIXES: Tuple[str, ...] = ("-USD", "USDT")
VALID_INTERVALS: Tuple[str, ...] = ("1d", "1h")
VALID_CELLS: Tuple[str, ...] = ("gru", "lstm")

DEFAULT_PILOT_TICKERS: Tuple[str, ...] = ("BTCUSDT", "NVDA", "SPY", "AAPL", "ETHUSDT")


def sha256_file(file_path: Path) -> str:
    """Calcula el hash SHA-256 hexadecimal de un fichero binario o texto."""
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_symbol(symbol: str) -> str:
    """
    Normaliza alias de símbolos (ej. BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD)
    para unificar históricos multi-fuente sin duplicar series.
    """
    sym = str(symbol).strip().upper()
    if sym in SYMBOL_ALIASES and sym.endswith("USDT"):
        return SYMBOL_ALIASES[sym]
    if sym == "ETHUSDT":
        return "ETH-USD"
    if sym.endswith("USDT") and len(sym) > 4:
        return f"{sym[:-4]}-USD"
    return sym


def is_crypto(symbol: str) -> bool:
    """Devuelve True si el activo cotiza 24/7 en mercado cripto."""
    sym = str(symbol).strip().upper()
    return any(sym.endswith(suffix) for suffix in CRYPTO_SUFFIXES)


def deduplicate_tickers(tickers: Iterable[str]) -> List[str]:
    """Convierte a símbolos canónicos preservando orden y eliminando duplicados."""
    seen: set[str] = set()
    out: List[str] = []
    for raw in tickers:
        canon = canonical_symbol(raw)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def default_universe_tickers() -> List[str]:
    """Universo completo de 30 PRIMARY_TICKERS + BTCUSDT deduplicado (30 series únicas)."""
    return deduplicate_tickers([*PRIMARY_TICKERS, "BTCUSDT"])


@dataclass(frozen=True)
class CandleV2Config:
    interval: str = "1d"
    cell: str = "gru"
    seq_len: int = 24
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.25
    latent_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    threshold_bps: float = 25.0
    n_folds: int = 4
    te_frac: float = 0.15
    embargo_bars: Optional[int] = None
    loss_w_ret: float = 1.0
    loss_w_tr: float = 0.5
    loss_w_dir: float = 1.0
    grad_clip: float = 1.0
    seed: int = 42
    ensemble_weight: float = 0.35
    min_coverage: float = 0.02
    conf_grid: Tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

    @property
    def resolved_embargo_bars(self) -> int:
        return int(self.embargo_bars if self.embargo_bars is not None else self.seq_len)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["conf_grid"] = list(self.conf_grid)
        d["resolved_embargo_bars"] = self.resolved_embargo_bars
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandleV2Config":
        allowed = {f for f in cls.__dataclass_fields__}
        clean: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in allowed:
                continue
            if k == "conf_grid" and isinstance(v, (list, tuple)):
                clean[k] = tuple(float(x) for x in v)
            else:
                clean[k] = v
        return cls(**clean)


_INTERVAL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "1d": {
        "interval": "1d",
        "cell": "gru",
        "seq_len": 24,
        "hidden_size": 48,
        "num_layers": 1,
        "max_epochs": 60,
        "patience": 8,
        "threshold_bps": 25.0,
    },
    "1h": {
        "interval": "1h",
        "cell": "gru",
        "seq_len": 48,
        "hidden_size": 64,
        "num_layers": 2,
        "max_epochs": 30,
        "patience": 6,
        "threshold_bps": 15.0,
    },
}


def validate_config(cfg: CandleV2Config) -> CandleV2Config:
    """Valida cotas de complejidad y rangos permitidos para evitar sobreajuste."""
    if cfg.interval not in VALID_INTERVALS:
        raise ValueError(f"interval inválido: {cfg.interval}. Permitidos: {VALID_INTERVALS}")
    if cfg.cell not in VALID_CELLS:
        raise ValueError(f"cell inválido: {cfg.cell}. Permitidos: {VALID_CELLS}")
    if not (16 <= int(cfg.seq_len) <= 48):
        raise ValueError(f"seq_len={cfg.seq_len} fuera de rango permitido [16, 48].")
    if not (48 <= int(cfg.hidden_size) <= 64):
        raise ValueError(f"hidden_size={cfg.hidden_size} fuera de rango permitido [48, 64].")
    if not (1 <= int(cfg.num_layers) <= 2):
        raise ValueError(f"num_layers={cfg.num_layers} fuera de rango permitido [1, 2].")
    if cfg.latent_dim < 4:
        raise ValueError(f"latent_dim={cfg.latent_dim} debe ser >= 4.")
    if cfg.embargo_bars is not None and cfg.embargo_bars < 0:
        raise ValueError(f"embargo_bars={cfg.embargo_bars} no puede ser negativo.")
    return cfg


def load_config(
    interval: str = "1d",
    overrides: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> CandleV2Config:
    """
    Carga configuración por temporalidad combinando:
    1) Valores por defecto por intervalo (`1d` vs `1h`)
    2) `config/candle_predictor.yaml` (si existe)
    3) Overrides explícitos de CLI / tests
    """
    tf = str(interval).strip().lower()
    if tf not in _INTERVAL_DEFAULTS:
        raise ValueError(f"Temporalidad no soportada: {interval}. Use '1d' o '1h'.")

    merged: Dict[str, Any] = dict(_INTERVAL_DEFAULTS[tf])

    yaml_path = config_path or (CONFIG_DIR / "candle_predictor.yaml")
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw.get("common"), dict):
            merged.update({k: v for k, v in raw["common"].items() if v is not None})
        intervals_block = raw.get("intervals", {})
        if isinstance(intervals_block, dict) and isinstance(intervals_block.get(tf), dict):
            merged.update({k: v for k, v in intervals_block[tf].items() if v is not None})

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v

    if "conf_grid" in merged and isinstance(merged["conf_grid"], list):
        merged["conf_grid"] = tuple(float(x) for x in merged["conf_grid"])
    merged["interval"] = tf

    cfg = CandleV2Config.from_dict(merged)
    return validate_config(replace(cfg, interval=tf))
