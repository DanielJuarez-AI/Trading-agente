"""
Candle Predictor v2: extractor secuencial multi-cabezal (GRU/LSTM) y puente para Laya #1.
"""

from backend.app.ml.candle_predictor.config import (
    CLASSES,
    DEFAULT_PILOT_TICKERS,
    WARMUP_BARS,
    CandleV2Config,
    canonical_symbol,
    default_universe_tickers,
    is_crypto,
    load_config,
)
from backend.app.ml.candle_predictor.features import (
    CANDLE_V2_FEATURES,
    CausalRobustScaler,
    MultiTickerSequenceDataset,
    TickerFrame,
    build_index,
    build_ticker_frame,
    compute_candle_v2_features,
)
from backend.app.ml.candle_predictor.inference import CandleV2Output, CandleV2Runtime
from backend.app.ml.candle_predictor.laya_adapter import (
    CANDLE_V2_SIGNAL_DIM,
    LEGACY_STATE_DIM,
    extend_state_with_candle_v2,
    laya_state_dim_v2,
)
from backend.app.ml.candle_predictor.network import (
    NextCandleGRU,
    calibrated_proba,
    expected_calibration_error,
    fit_temperature,
)

__all__ = [
    "CANDLE_V2_FEATURES",
    "CANDLE_V2_SIGNAL_DIM",
    "CLASSES",
    "DEFAULT_PILOT_TICKERS",
    "LEGACY_STATE_DIM",
    "WARMUP_BARS",
    "CandleV2Config",
    "CandleV2Output",
    "CandleV2Runtime",
    "CausalRobustScaler",
    "MultiTickerSequenceDataset",
    "NextCandleGRU",
    "TickerFrame",
    "build_index",
    "build_ticker_frame",
    "calibrated_proba",
    "canonical_symbol",
    "compute_candle_v2_features",
    "default_universe_tickers",
    "expected_calibration_error",
    "extend_state_with_candle_v2",
    "fit_temperature",
    "is_crypto",
    "laya_state_dim_v2",
    "load_config",
]
