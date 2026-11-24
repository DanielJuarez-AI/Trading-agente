"""
Pruebas unitarias e integración para Candle Predictor v2 (GRU/LSTM multi-ticker, 1h/1d):
- Cero cruce de fronteras entre tickers en Dataset
- Cero look-ahead bias bit a bit en features, escalado e inferencia
- Embargo Walk-Forward estricto sin shuffle
- Calibración de temperatura y latencia CPU
- Integridad SHA-256 en QuantPredictor y retrocompatibilidad con Laya #1 / Risk Engine
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.mexc_client import MEXCMarketClient
from backend.app.back_office.paper_engine import PaperTradingEngine
from backend.app.core.config import ARTIFACTS_DIR, get_settings
from backend.app.core.security import PrivateAuthGateway
from backend.app.front_office.laya_alpha import (
    DRL_STATE_DIM,
    Laya1DRLPolicyNet,
    LayaAlphaAgent,
)
from backend.app.front_office.predictor import PredictionResult, QuantPredictor
from backend.app.middle_office.risk_engine import DeterministicRiskEngine
from backend.app.ml.candle_predictor.config import load_config
from backend.app.ml.candle_predictor.features import (
    CANDLE_V2_FEATURES,
    CausalRobustScaler,
    MultiTickerSequenceDataset,
    TickerFrame,
    build_index,
    build_ticker_frame,
    compute_candle_v2_features,
)
from backend.app.ml.candle_predictor.inference import CandleV2Runtime
from backend.app.ml.candle_predictor.laya_adapter import (
    extend_state_with_candle_v2,
    laya_state_dim_v2,
)
from backend.app.ml.candle_predictor.network import (
    NextCandleGRU,
    _cross_entropy_nll,
    fit_temperature,
)
from backend.app.ml.candle_predictor.splits import make_walk_forward_splits
from backend.app.ml.candle_predictor.trainer import (
    measure_inference_latency_ms,
    save_candle_v2_artifacts,
)


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test_candle_v2.db"))
    monkeypatch.setenv("SIMULATION_PROFILE", "agent_paper")
    monkeypatch.setenv("GATEWAY_API_KEY", "test_gateway_secret_12345")
    monkeypatch.setenv("ENABLE_REAL_TRADING", "false")
    return get_settings()


def _synthetic_ohlcv(
    n: int = 300,
    interval: str = "1d",
    seed: int = 42,
    base_price: float = 150.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = timedelta(days=1) if interval == "1d" else timedelta(hours=1)
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    log_rets = rng.normal(0.0003, 0.015 if interval == "1d" else 0.006, size=n)
    closes = base_price * np.exp(np.cumsum(log_rets))
    opens = np.concatenate([[base_price], closes[:-1]]) * (1.0 + rng.normal(0.0, 0.002, size=n))
    spread = np.abs(rng.normal(0.008, 0.003, size=n))
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)
    vols = rng.uniform(1e5, 5e6, size=n)

    rows = []
    for i in range(n):
        odt = start + i * step
        cdt = odt + step - timedelta(seconds=1)
        rows.append(
            {
                "open_time": odt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "close_time": cdt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(vols[i]),
            }
        )
    return pd.DataFrame(rows)


def _copy_legacy_artifacts(dst_dir: Path) -> None:
    for name in (
        "model_manifest.json",
        "lstm_v1.pt",
        "scaler_v1.joblib",
        "quant_v2_manifest.json",
        "quant_xgb_1d.joblib",
        "quant_xgb_1h.joblib",
        "laya1_drl_v1.pt",
        "laya1_drl_manifest.json",
    ):
        src = ARTIFACTS_DIR / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def test_windows_never_cross_ticker_boundaries() -> None:
    n_rows = 30
    f_count = len(CANDLE_V2_FEATURES)
    rows_a = np.tile(np.arange(n_rows, dtype=np.float32)[:, None], (1, f_count)) + 1000.0
    rows_b = np.tile(np.arange(n_rows, dtype=np.float32)[:, None], (1, f_count)) + 2000.0
    ts_arr = np.arange(n_rows, dtype=np.int64) * 86_400_000_000_000

    frame_a = TickerFrame(
        symbol="AAPL",
        interval="1d",
        X=rows_a,
        y_ret=np.full(n_rows, 0.001, dtype=np.float32),
        y_tr=np.full(n_rows, 0.02, dtype=np.float32),
        y_dir=np.full(n_rows, 1.0, dtype=np.float32),
        ts=ts_arr,
        valid=np.ones(n_rows, dtype=bool),
    )
    frame_b = TickerFrame(
        symbol="NVDA",
        interval="1d",
        X=rows_b,
        y_ret=np.full(n_rows, -0.002, dtype=np.float32),
        y_tr=np.full(n_rows, 0.03, dtype=np.float32),
        y_dir=np.full(n_rows, 0.0, dtype=np.float32),
        ts=ts_arr,
        valid=np.ones(n_rows, dtype=bool),
    )

    seq_len = 8
    idx_list = build_index([frame_a, frame_b], seq_len=seq_len, require_target=True)
    scaler = CausalRobustScaler.identity(n_features=f_count)
    ds = MultiTickerSequenceDataset([frame_a, frame_b], scaler=scaler, seq_len=seq_len, index=idx_list)

    assert len(ds) == (n_rows - seq_len + 1) * 2
    for i in range(len(ds)):
        t_idx, end_idx = ds.index[i]
        assert 0 <= end_idx < n_rows
        assert end_idx >= seq_len - 1
        x_win, _, _, _ = ds[i]
        arr = x_win.numpy()
        assert arr.shape == (seq_len, f_count)
        expected_base = 1000.0 if t_idx == 0 else 2000.0
        assert float(arr.min()) >= expected_base
        assert float(arr.max()) < expected_base + n_rows
        # Fila consecutiva estricta +1 sin saltos entre símbolos
        col0 = arr[:, 0]
        assert np.allclose(np.diff(col0), 1.0)


@pytest.mark.parametrize("interval", ["1d", "1h"])
def test_features_no_lookahead(interval: str) -> None:
    df_orig = _synthetic_ohlcv(n=300, interval=interval, seed=7)
    df_mod = df_orig.copy()
    rng = np.random.default_rng(999)
    df_mod.loc[201:, "close"] = rng.uniform(5.0, 5000.0, size=99)
    df_mod.loc[201:, "open"] = df_mod.loc[201:, "close"] * 0.97
    df_mod.loc[201:, "high"] = df_mod.loc[201:, "close"] * 1.15
    df_mod.loc[201:, "low"] = df_mod.loc[201:, "close"] * 0.85
    df_mod.loc[201:, "volume"] = rng.uniform(10.0, 1e9, size=99)

    feat_orig = compute_candle_v2_features(df_orig, symbol="NVDA", interval=interval)
    feat_mod = compute_candle_v2_features(df_mod, symbol="NVDA", interval=interval)

    cols = CANDLE_V2_FEATURES + ["valid_mask"]
    a = feat_orig.loc[:200, cols].to_numpy(dtype=np.float64)
    b = feat_mod.loc[:200, cols].to_numpy(dtype=np.float64)
    assert np.array_equal(a, b, equal_nan=True)


def test_scaler_and_prediction_no_lookahead() -> None:
    df_orig = _synthetic_ohlcv(n=300, interval="1d", seed=11)
    df_mod = df_orig.copy()
    rng = np.random.default_rng(2025)
    df_mod.loc[201:, "close"] = rng.uniform(10.0, 2000.0, size=99)
    df_mod.loc[201:, "high"] = df_mod.loc[201:, "close"] * 1.05
    df_mod.loc[201:, "low"] = df_mod.loc[201:, "close"] * 0.95
    df_mod.loc[201:, "open"] = df_mod.loc[201:, "close"] * 0.99

    f_orig = compute_candle_v2_features(df_orig, symbol="AAPL", interval="1d")
    f_mod = compute_candle_v2_features(df_mod, symbol="AAPL", interval="1d")

    m_orig = f_orig.loc[:150]
    m_mod = f_mod.loc[:150]
    s1 = CausalRobustScaler().fit(
        m_orig.loc[m_orig["valid_mask"], CANDLE_V2_FEATURES].to_numpy(),
        y_ret=m_orig.loc[m_orig["valid_mask"], "target_logret"].to_numpy(),
        y_tr=m_orig.loc[m_orig["valid_mask"], "target_tr"].to_numpy(),
    )
    s2 = CausalRobustScaler().fit(
        m_mod.loc[m_mod["valid_mask"], CANDLE_V2_FEATURES].to_numpy(),
        y_ret=m_mod.loc[m_mod["valid_mask"], "target_logret"].to_numpy(),
        y_tr=m_mod.loc[m_mod["valid_mask"], "target_tr"].to_numpy(),
    )
    assert np.array_equal(s1.center, s2.center)
    assert np.array_equal(s1.scale, s2.scale)
    assert s1.ret_scale == s2.ret_scale
    assert s1.tr_scale == s2.tr_scale

    cfg = load_config("1d", overrides={"seq_len": 16, "hidden_size": 48, "num_layers": 1})
    torch.manual_seed(123)
    model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=cfg)
    rt = CandleV2Runtime(
        interval="1d",
        model=model,
        scaler=s1,
        cfg=cfg,
        temperature=1.2,
        conf_threshold=0.45,
        ensemble_weight=0.35,
        production_enabled=True,
    )

    out_orig = rt.predict_batch(df_orig, symbol="AAPL")
    out_mod = rt.predict_batch(df_mod, symbol="AAPL")
    for key in ("pred_logret", "pred_return", "pred_tr", "probs", "abstain", "latent"):
        arr_o = out_orig[key][:201]
        arr_m = out_mod[key][:201]
        if arr_o.dtype == bool:
            assert np.array_equal(arr_o, arr_m)
        else:
            assert np.array_equal(arr_o, arr_m, equal_nan=True)


def test_walk_forward_embargo() -> None:
    frames: List[TickerFrame] = []
    symbols = ["BTC-USD", "NVDA", "SPY", "AAPL"]
    for i, sym in enumerate(symbols):
        df = _synthetic_ohlcv(n=680, interval="1d", seed=100 + i, base_price=50.0 + 40.0 * i)
        frames.append(build_ticker_frame(df, symbol=sym, interval="1d", threshold_bps=25.0))

    seq_len = 24
    embargo = 24
    plan = make_walk_forward_splits(
        frames=frames, seq_len=seq_len, n_folds=4, te_frac=0.15, embargo=embargo
    )
    assert len(plan.folds) == 4

    for fold in plan.folds:
        # Orden cronológico estricto sin shuffle
        train_ts = [int(frames[ti].ts[ei]) for ti, ei in fold.train_index]
        assert train_ts == sorted(train_ts)

        max_train_label_ts = max(int(frames[ti].ts[ei + 1]) for ti, ei in fold.train_index)
        min_val_win_start_ts = min(
            int(frames[ti].ts[ei - seq_len + 1]) for ti, ei in fold.val_index
        )
        assert max_train_label_ts < min_val_win_start_ts
        assert all(int(frames[ti].ts[ei]) < plan.te_start_ns for ti, ei in fold.val_index)

    max_cal_label_ts = max(int(frames[ti].ts[ei + 1]) for ti, ei in plan.final_cal_index)
    min_te_win_start_ts = min(
        int(frames[ti].ts[ei - seq_len + 1]) for ti, ei in plan.final_te_index
    )
    assert max_cal_label_ts < min_te_win_start_ts
    assert all(int(frames[ti].ts[ei]) >= plan.te_start_ns for ti, ei in plan.final_te_index)


def test_temperature_scaling_reduces_nll() -> None:
    rng = np.random.default_rng(77)
    n = 600
    y = rng.integers(0, 3, size=n)
    # Logits excesivamente confiados en dirección aleatoria + algo de señal real
    logits = rng.normal(0.0, 2.5, size=(n, 3))
    logits[np.arange(n), y] += 1.2
    raw_nll = _cross_entropy_nll(logits, y, temperature=1.0)
    t_opt = fit_temperature(logits, y)
    cal_nll = _cross_entropy_nll(logits, y, temperature=t_opt)
    assert t_opt > 1.0
    assert cal_nll < raw_nll


def test_quantpredictor_candle_v2_sha256_and_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_settings(tmp_path, monkeypatch)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    _copy_legacy_artifacts(art_dir)

    for tf in ("1d", "1h"):
        c2_cfg = load_config(tf)
        torch.manual_seed(10 if tf == "1d" else 20)
        model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=c2_cfg)
        scaler = CausalRobustScaler.identity(n_features=len(CANDLE_V2_FEATURES))
        scaler.ret_scale = 0.015
        scaler.tr_scale = 0.02
        save_candle_v2_artifacts(
            model=model,
            scaler=scaler,
            cfg=c2_cfg,
            out_dir=art_dir,
            interval=tf,
            extra_manifest={
                "temperature": 1.15,
                "conf_threshold": 0.40,
                "production_enabled": True,
                "gate_reason": "test_enabled",
            },
        )

    db = TradingDatabase(cfg.sqlite_db_path)
    market = MEXCMarketClient(cfg, db)
    predictor = QuantPredictor(artifacts_dir=art_dir, verify_hashes=True)

    for tf in ("1d", "1h"):
        df = market.load_replay_candles("NVDA", interval=tf)
        res = predictor.predict_from_candles(df, symbol="NVDA", interval=tf)
        assert res.candle_v2_enabled is True
        assert "candlev2" in res.model_detail
        assert len(res.candle_v2_latent) == 16
        assert res.pred_volatility_t1 >= 0.0
        assert abs(sum(res.direction_probs.values()) - 1.0) < 1e-3

    # Corromper 1 byte del archivo de pesos 1d -> debe lanzar RuntimeError
    weights_1d = art_dir / "lstm_v2_1d.pt"
    raw = bytearray(weights_1d.read_bytes())
    raw[-1] ^= 0xFF
    weights_1d.write_bytes(bytes(raw))

    with pytest.raises(RuntimeError, match="SHA-256"):
        QuantPredictor(artifacts_dir=art_dir, verify_hashes=True)


def test_laya_backward_compat_and_v2_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_settings(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    gateway = PrivateAuthGateway(cfg, db)

    # Retrocompatibilidad con artefactos oficiales (v2 33D si existe o v1 10D)
    agent_legacy = LayaAlphaAgent(cfg, db, gateway, artifacts_dir=ARTIFACTS_DIR)
    assert agent_legacy.drl_net is not None
    assert agent_legacy.drl_state_dim in (DRL_STATE_DIM, 33)

    # Validar explícitamente carga v1 de 10 dims cuando solo existen artefactos v1
    v1_only_dir = tmp_path / "laya_v1_only"
    v1_only_dir.mkdir()
    shutil.copy2(ARTIFACTS_DIR / "laya1_drl_v1.pt", v1_only_dir / "laya1_drl_v1.pt")
    shutil.copy2(ARTIFACTS_DIR / "laya1_drl_manifest.json", v1_only_dir / "laya1_drl_manifest.json")
    agent_v1_only = LayaAlphaAgent(cfg, db, gateway, artifacts_dir=v1_only_dir)
    assert agent_v1_only.drl_state_dim == DRL_STATE_DIM == 10

    pred_legacy = PredictionResult(
        symbol="NVDA",
        candle_close_utc="2026-11-20T23:59:59Z",
        pred_return_t1=0.004,
        direction="UP",
        ema_ratio=1.01,
        vol_20=0.02,
        rsi_14=56.0,
        momentum_5=0.015,
        ret_lag_1=0.003,
        model_version="lstm_v1",
        features_verified=True,
    )
    out_legacy = agent_legacy.evaluate(
        pred_legacy, position_open=False, gateway_key=cfg.gateway_api_key
    )
    assert out_legacy.signal_action in {"BUY", "HOLD", "SELL"}

    # Adaptador 33 dims + red Laya1DRLPolicyNet(state_dim=33)
    v2_dim = laya_state_dim_v2(16)
    assert v2_dim == 33
    base_10 = np.zeros(10, dtype=np.float32)

    fallback_33 = extend_state_with_candle_v2(base_10, pred_legacy, latent_dim=16)
    assert fallback_33.shape == (33,)
    assert fallback_33[16] == 1.0  # abstain fallback

    pred_v2 = PredictionResult(
        symbol="NVDA",
        candle_close_utc="2026-11-20T23:59:59Z",
        pred_return_t1=0.004,
        direction="UP",
        ema_ratio=1.01,
        vol_20=0.02,
        rsi_14=56.0,
        momentum_5=0.015,
        ret_lag_1=0.003,
        model_version="lstm_v1",
        features_verified=True,
        atr_pct=0.02,
        pred_volatility_t1=0.03,
        candle_v2_return=0.005,
        direction_probs={"DOWN": 0.15, "NEUTRAL": 0.25, "UP": 0.60},
        direction_confidence=0.60,
        abstain=False,
        candle_v2_latent=[0.1 * i - 0.5 for i in range(16)],
        candle_v2_enabled=True,
    )
    vec_33 = extend_state_with_candle_v2(base_10, pred_v2, latent_dim=16)
    assert vec_33.shape == (33,)
    assert vec_33.dtype == np.float32
    assert vec_33[16] == 0.0
    assert np.isclose(vec_33[15], 0.45)  # edge = P(UP) - P(DOWN)

    # Simular Laya1DRLPolicyNet reentrenada con state_dim=33 en artefacto temporal
    laya_dir = tmp_path / "laya_v2_art"
    laya_dir.mkdir()
    torch.manual_seed(55)
    net_33 = Laya1DRLPolicyNet(state_dim=33, hidden_dim=64)
    q_vals, p_logits, r_logits, val = net_33(torch.from_numpy(vec_33).unsqueeze(0))
    assert q_vals.shape == (1, 3) and p_logits.shape == (1, 3) and r_logits.shape == (1, 5)
    assert val.shape == (1,)

    torch.save(net_33.state_dict(), laya_dir / "laya1_drl_v1.pt")
    (laya_dir / "laya1_drl_manifest.json").write_text(
        json.dumps(
            {
                "model_version": "laya1_drl_v3_test",
                "architecture": {"state_dim": 33, "hidden_dim": 64},
                "calibration_temperature": 1.0,
            }
        ),
        encoding="utf-8",
    )
    agent_v2 = LayaAlphaAgent(cfg, db, gateway, artifacts_dir=laya_dir)
    assert agent_v2.drl_state_dim == 33
    out_v2 = agent_v2.evaluate(pred_v2, position_open=False, gateway_key=cfg.gateway_api_key)
    assert out_v2.signal_action in {"BUY", "HOLD", "SELL"}


def test_risk_engine_unaffected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_settings(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    paper = PaperTradingEngine(cfg, db)
    risk = DeterministicRiskEngine(cfg, db)
    portfolio = paper.get_portfolio_state(current_price=140.0, symbol="NVDA")

    verdict_buy = risk.evaluate("BUY", "NVDA", 140.0, portfolio, system_status="RUNNING")
    assert verdict_buy.approved is True
    assert verdict_buy.final_action == "BUY"

    verdict_sell = risk.evaluate("SELL", "NVDA", 140.0, portfolio, system_status="RUNNING")
    assert verdict_sell.approved is False
    assert verdict_sell.rule_triggered == "LONG_ONLY_NO_SHORT"


@pytest.mark.parametrize("interval", ["1d", "1h"])
def test_inference_latency(interval: str) -> None:
    cfg = load_config(interval)
    model = NextCandleGRU(n_features=len(CANDLE_V2_FEATURES), cfg=cfg)
    p50_ms = measure_inference_latency_ms(
        model=model, seq_len=cfg.seq_len, n_features=len(CANDLE_V2_FEATURES), runs=100
    )
    assert p50_ms < 10.0


def test_quant_metrics_psr_dsr_var_kelly() -> None:
    from backend.app.core.quant_metrics import (
        deflated_sharpe_ratio,
        fractional_kelly_allocation,
        historical_var_cvar,
        probabilistic_sharpe_ratio,
        spearman_rank_ic,
    )

    rets = np.array([0.012, 0.008, -0.004, 0.015, 0.009, -0.002, 0.011, 0.007, 0.006, 0.013])
    psr = probabilistic_sharpe_ratio(rets, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(rets, n_trials=30)
    assert 0.5 < psr <= 1.0
    assert 0.0 <= dsr <= psr

    var_95, cvar_95 = historical_var_cvar(rets, alpha=0.95)
    assert var_95 >= 0.0
    assert cvar_95 >= var_95

    ic = spearman_rank_ic([0.01, 0.02, -0.01, 0.03], [0.005, 0.015, -0.02, 0.04])
    assert ic > 0.9

    k_low = fractional_kelly_allocation(p_win=0.50, vol_20=0.04, min_alloc=0.10, max_alloc=0.45)
    k_high = fractional_kelly_allocation(p_win=0.80, vol_20=0.015, min_alloc=0.10, max_alloc=0.45)
    assert 0.10 <= k_low <= k_high <= 0.45


def test_sha256_decision_audit_chain_and_reset_portfolio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_settings(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    db.initialize_defaults_if_empty("RUNNING", "NVDA", "replay", "agent_paper", 50000.0)

    for idx in range(3):
        cid = db.register_candle_idempotent(
            {
                "symbol": "NVDA",
                "interval": "1d",
                "open_time": f"2026-11-1{idx}T00:00:00Z",
                "close_time": f"2026-11-1{idx}T23:59:59Z",
                "open": 140.0 + idx,
                "high": 142.0 + idx,
                "low": 139.0 + idx,
                "close": 141.0 + idx,
                "volume": 1000000.0,
                "source_mode": "replay",
            }
        )
        db.record_cycle_result(
            candle_id=cid,
            decision_data={
                "symbol": "NVDA",
                "interval": "1d",
                "candle_close_utc": f"2026-11-1{idx}T23:59:59Z",
                "model_pred": 0.003 + 0.001 * idx,
                "laya1_json": {"signal_action": "BUY"},
                "qwen_json": {"action": "BUY"},
                "laya2_json": {"approved": True},
                "risk_status": "APPROVED",
                "risk_reason": "OK",
                "final_action": "BUY",
                "latency_ms": 12.0,
                "qwen_execution_mode": "LOCAL_DEVICE",
                "laya_execution_mode": "LOCAL_DEVICE",
                "source_mode": "replay",
            },
            order_data=None,
            snapshot_data={
                "candle_time": f"2026-11-1{idx}T23:59:59Z",
                "symbol": "NVDA",
                "cash": 50000.0,
                "position_qty": 0.0,
                "position_value": 0.0,
                "total_equity": 50000.0,
                "daily_pnl_pct": 0.0,
                "drawdown_pct": 0.0,
                "source_mode": "replay",
            },
        )

    chain_ok = db.verify_decision_audit_chain()
    assert chain_ok["valid"] is True
    assert chain_ok["checked_count"] == 3
    assert chain_ok["broken_at_id"] is None

    # Simular manipulación externa en la segunda decisión -> cadena debe detectar rotura
    with db.atomic_tx() as conn:
        conn.execute("UPDATE decisions SET final_action = 'SELL' WHERE id = 2;")
    chain_bad = db.verify_decision_audit_chain()
    assert chain_bad["valid"] is False
    assert chain_bad["broken_at_id"] == 2

    # Reset de cartera limpia historial, restaura capital y deja cadena limpia
    res_reset = db.reset_paper_portfolio(50000.0)
    assert res_reset["initial_capital"] == 50000.0
    assert db.count_processed_candles() == 0
    assert db.list_orders() == []
    assert db.verify_decision_audit_chain()["checked_count"] == 0


def test_scanner_and_mexc_1h_60m_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    db = TradingDatabase(cfg.sqlite_db_path)

    captured_params = {}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    def mexc_handler(request: httpx.Request) -> httpx.Response:
        captured_params["interval"] = request.url.params.get("interval")
        klines = [
            [now_ms - 7200_000, "90000", "90500", "89800", "90200", "15.0", now_ms - 3600_000],
        ]
        return httpx.Response(200, json=klines)

    mc = MEXCMarketClient(cfg, db, http_client=httpx.Client(transport=httpx.MockTransport(mexc_handler)))
    df_live = mc.fetch_live_candles("BTC-USD", interval="1h", enforce_mexc_for_crypto=True)
    assert captured_params["interval"] == "60m"
    assert len(df_live) == 1

    app = create_app(settings_override=cfg)
    with TestClient(app) as client:
        sc = client.get("/api/scanner?interval=1h&top_n=8")
        assert sc.status_code == 200
        body = sc.json()
        assert body["scanned_count"] == 30
        assert len(body["items"]) == 8
        assert "alpha_score" in body["items"][0]
        assert "kelly_alloc_pct" in body["items"][0]

        au = client.get("/api/audit/verify")
        assert au.status_code == 200
        assert au.json()["valid"] is True


def test_news_endpoint_returns_structured_articles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    from backend.app.front_office.news_sentiment import NewsSentimentService
    from backend.app.main import create_app

    scored = NewsSentimentService.score_headlines(
        ["NVDA beats earnings expectations with record surge"]
    )
    assert scored["sentiment_label"] == "BULLISH"
    assert "EARNINGS" in scored["catalysts"]

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    app = create_app(settings_override=cfg)

    def fake_get_symbol_context(symbol: str, source_mode: str = "replay", fetch_live: bool = False, force_refresh: bool = False):
        return {
            "symbol": symbol,
            "source": "google_news_rss",
            "sentiment_score": 0.65,
            "sentiment_label": "BULLISH",
            "catalysts": ["EARNINGS"],
            "headlines": [f"{symbol} beats earnings estimates"],
            "articles": [
                {
                    "title": f"{symbol} beats earnings estimates",
                    "link": "https://example.com/news/nvda",
                    "pub_date": "Mon, 23 Nov 2026 15:00:00 GMT",
                    "source": "Reuters",
                    "sentiment_label": "BULLISH",
                    "sentiment_score": 1.0,
                    "catalysts": ["EARNINGS"],
                }
            ],
            "headline_count": 1,
            "high_impact_event": True,
            "fetched_at_utc": "2026-11-23T15:05:00Z",
        }

    monkeypatch.setattr(app.state.news_service, "get_symbol_context", fake_get_symbol_context)
    with TestClient(app) as client:
        resp = client.get("/api/news?symbol=NVDA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "NVDA"
        assert data["sentiment_label"] == "BULLISH"
        assert len(data["articles"]) == 1
        assert data["articles"][0]["source"] == "Reuters"
        assert "EARNINGS" in data["articles"][0]["catalysts"]
        assert "agent_injection_active" in data


def test_mexc_realtime_forming_candles_and_ticker_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    db = TradingDatabase(cfg.sqlite_db_path)

    now = datetime.now(timezone.utc)
    closed_open_ms = int((now - timedelta(hours=2)).timestamp() * 1000)
    closed_close_ms = int((now - timedelta(hours=1)).timestamp() * 1000)
    forming_open_ms = int((now - timedelta(minutes=15)).timestamp() * 1000)
    forming_close_ms = int((now + timedelta(minutes=45)).timestamp() * 1000)

    def mexc_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v3/ticker/24hr"):
            return httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "lastPrice": "91234.56",
                    "prevClosePrice": "90000.00",
                    "priceChangePercent": "0.0137",
                    "highPrice": "92000.00",
                    "lowPrice": "89500.00",
                    "volume": "12345.67",
                    "bidPrice": "91234.00",
                    "askPrice": "91235.00",
                },
            )
        klines = [
            [closed_open_ms, "90000", "90500", "89800", "90200", "15.0", closed_close_ms],
            [forming_open_ms, "90200", "91000", "90100", "90900", "8.5", forming_close_ms],
        ]
        return httpx.Response(200, json=klines)

    mock_http = httpx.Client(transport=httpx.MockTransport(mexc_handler))
    mc = MEXCMarketClient(cfg, db, http_client=mock_http)

    # Por defecto (include_forming=False): descarta estrictamente la vela abierta
    df_closed = mc.fetch_live_candles("BTC-USD", interval="1h", now_utc=now, include_forming=False)
    assert len(df_closed) == 1
    assert float(df_closed["close"].iloc[-1]) == pytest.approx(90200.0)

    # Con include_forming=True: incluye la vela en formación y superpone el tick 24hr de MEXC
    df_rt = mc.fetch_live_candles("BTC-USD", interval="1h", now_utc=now, include_forming=True)
    assert len(df_rt) == 2
    assert float(df_rt["close"].iloc[-1]) == pytest.approx(91234.56)
    assert float(df_rt["high"].iloc[-1]) >= 91234.56

    app = create_app(settings_override=cfg)
    app.state.market_client = MEXCMarketClient(cfg, app.state.db, http_client=mock_http)
    with TestClient(app) as client:
        resp = client.get("/api/candles?symbol=BTC-USD&interval=1h&realtime=true&include_forming=true")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_realtime"] is True
        assert body["is_mexc_native"] is True
        assert "MEXC API /api/v3/klines" in body["data_source"]
        assert body["mexc_realtime"]["mexc_online"] is True
        assert body["mexc_realtime"]["last_price"] == pytest.approx(91234.56)
        assert body["candles"][-1]["close"] == pytest.approx(91234.56)


def test_decisions_endpoint_tracks_movement_and_liquidation_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    app = create_app(settings_override=cfg)
    db = app.state.db

    def _add_entry(
        sym: str,
        close_t: str,
        action: str,
        order_data: Optional[Dict[str, Any]],
        pos_qty: float,
        pos_px: float,
    ) -> int:
        cid = db.register_candle_idempotent(
            {
                "symbol": sym,
                "interval": "1d",
                "open_time": close_t[:10] + "T00:00:00Z",
                "close_time": close_t,
                "open": pos_px,
                "high": pos_px,
                "low": pos_px,
                "close": pos_px,
                "volume": 10000.0,
                "source_mode": "replay",
            }
        )
        return db.record_cycle_result(
            candle_id=cid,
            decision_data={
                "symbol": sym,
                "interval": "1d",
                "candle_close_utc": close_t,
                "model_pred": 0.005,
                "laya1_json": {"signal_action": action, "regime": "bull_trend"},
                "qwen_json": {"action": action},
                "laya2_json": {"approved": True},
                "risk_status": "APPROVED",
                "risk_reason": "OK",
                "final_action": action,
                "latency_ms": 15.0,
                "qwen_execution_mode": "LOCAL_DEVICE",
                "laya_execution_mode": "LOCAL_DEVICE",
                "source_mode": "replay",
            },
            order_data=order_data,
            snapshot_data={
                "candle_time": close_t,
                "symbol": sym,
                "cash": 40000.0,
                "position_qty": pos_qty,
                "avg_entry_price": pos_px if pos_qty > 0 else 0.0,
                "entry_cost_total": pos_qty * pos_px,
                "position_value": pos_qty * pos_px,
                "total_equity": 40000.0 + pos_qty * pos_px,
                "daily_pnl_pct": 0.0,
                "drawdown_pct": 0.0,
                "last_price": pos_px,
                "highest_price": pos_px,
                "source_mode": "replay",
            },
        )

    # 1) HOLD -> sin movimiento
    id_hold = _add_entry("AAPL", "2026-11-18T23:59:59Z", "HOLD", None, 0.0, 220.0)
    # 2) BUY NVDA -> sin liquidar (OPEN) con ganancia latente
    id_nvda_buy = _add_entry(
        "NVDA",
        "2026-11-19T23:59:59Z",
        "BUY",
        {
            "side": "BUY",
            "symbol": "NVDA",
            "exec_price": 140.0,
            "quantity": 50.0,
            "fee": 5.0,
            "slippage_bps": 5.0,
            "notional": 7000.0,
            "realized_pnl": None,
            "executed_at": "2026-11-19T23:59:59Z",
        },
        50.0,
        145.0,
    )
    # 3) BUY SPY luego SELL SPY -> liquidada FIFO (LIQUIDATED + LIQUIDATED_EXIT)
    id_spy_buy = _add_entry(
        "SPY",
        "2026-11-19T23:59:59Z",
        "BUY",
        {
            "side": "BUY",
            "symbol": "SPY",
            "exec_price": 500.0,
            "quantity": 20.0,
            "fee": 5.0,
            "slippage_bps": 5.0,
            "notional": 10000.0,
            "realized_pnl": None,
            "executed_at": "2026-11-19T23:59:59Z",
        },
        20.0,
        500.0,
    )
    id_spy_sell = _add_entry(
        "SPY",
        "2026-11-20T23:59:59Z",
        "SELL",
        {
            "side": "SELL",
            "symbol": "SPY",
            "exec_price": 515.0,
            "quantity": 20.0,
            "fee": 5.0,
            "slippage_bps": 5.0,
            "notional": 10300.0,
            "realized_pnl": 290.0,
            "executed_at": "2026-11-20T23:59:59Z",
        },
        0.0,
        515.0,
    )

    with TestClient(app) as client:
        resp = client.get("/api/decisions?limit=20")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_movements"] == 3
        assert body["summary"]["open_unliquidated_count"] == 1
        assert body["summary"]["liquidated_count"] == 2

        by_id = {it["id"]: it for it in body["items"]}
        assert by_id[id_hold]["movement"]["has_movement"] is False
        assert by_id[id_hold]["movement"]["settlement_status"] == "NONE"

        m_nvda = by_id[id_nvda_buy]["movement"]
        assert m_nvda["has_movement"] is True
        assert m_nvda["is_liquidated"] is False
        assert m_nvda["settlement_status"] == "OPEN"
        assert m_nvda["unrealized_pnl"] == pytest.approx(245.0)

        m_spy_buy = by_id[id_spy_buy]["movement"]
        assert m_spy_buy["has_movement"] is True
        assert m_spy_buy["is_liquidated"] is True
        assert m_spy_buy["settlement_status"] == "LIQUIDATED"
        assert m_spy_buy["realized_pnl"] == pytest.approx(290.0)
        assert m_spy_buy["exit_price"] == pytest.approx(515.0)

        m_spy_sell = by_id[id_spy_sell]["movement"]
        assert m_spy_sell["has_movement"] is True
        assert m_spy_sell["is_liquidated"] is True
        assert m_spy_sell["settlement_status"] == "LIQUIDATED_EXIT"
        assert m_spy_sell["realized_pnl"] == pytest.approx(290.0)


def test_quotes_endpoint_returns_mexc_and_equity_live_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from fastapi.testclient import TestClient
    import backend.app.back_office.mexc_client as mexc_mod
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""

    def mexc_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "lastPrice": "61500.25",
                "prevClosePrice": "60000.00",
                "priceChangePercent": "0.025",
            },
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(mexc_handler))

    def fake_yf_download(*args, **kwargs):
        idx = pd.to_datetime(["2026-11-23T22:00:00Z"])
        return pd.DataFrame(
            {"Open": [165.0], "High": [167.0], "Low": [164.5], "Close": [166.42], "Volume": [1000.0]},
            index=idx,
        )

    monkeypatch.setattr(mexc_mod.yf, "download", fake_yf_download)

    app = create_app(settings_override=cfg)
    app.state.market_client = MEXCMarketClient(cfg, app.state.db, http_client=mock_http)

    with TestClient(app) as client:
        r = client.get("/api/quotes?symbols=BTC-USD,NVDA")
        assert r.status_code == 200
        body = r.json()
        assert "updated_at_utc" in body
        quotes = body["quotes"]
        assert quotes["BTC-USD"]["price"] == pytest.approx(61500.25)
        assert quotes["BTC-USD"]["source"] == "MEXC_API"
        assert quotes["BTC-USD"]["online"] is True
        assert quotes["NVDA"]["price"] == pytest.approx(166.42)
        assert quotes["NVDA"]["source"] == "YAHOO_1M"







def test_forming_daily_row_with_nan_ohl_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backend.app.back_office.mexc_client as mexc_mod

    cfg = _make_settings(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    idx = pd.DatetimeIndex([today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today], name="Date")

    def fake_download(*args, **kwargs):
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, np.nan],
                "High": [102.0, 103.0, np.nan],
                "Low": [99.0, 100.0, np.nan],
                "Close": [101.0, 102.5, 104.9],
                "Volume": [10.0, 11.0, 12.0],
            },
            index=idx,
        )

    monkeypatch.setattr(mexc_mod.yf, "download", fake_download)
    mc = MEXCMarketClient(cfg, db)
    df = mc.fetch_live_candles(
        "LULU", interval="1d", now_utc=datetime.now(timezone.utc), include_forming=True
    )
    last = df.iloc[-1]
    assert not df[["open", "high", "low", "close"]].isna().any().any()
    assert float(last["open"]) == pytest.approx(102.5)
    assert float(last["high"]) >= float(last["close"]) >= float(last["low"]) > 0


def test_manual_liquidate_closes_position_and_logs_sell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    app = create_app(settings_override=cfg)
    db = app.state.db
    close_t = "2026-11-20T23:59:59Z"
    cid = db.register_candle_idempotent(
        {
            "symbol": "NVDA", "interval": "1d", "open_time": "2026-11-20T00:00:00Z",
            "close_time": close_t, "open": 140.0, "high": 141.0, "low": 139.0,
            "close": 140.0, "volume": 1000.0, "source_mode": "replay",
        }
    )
    db.record_cycle_result(
        candle_id=cid,
        decision_data={
            "symbol": "NVDA", "interval": "1d", "candle_close_utc": close_t, "model_pred": 0.004,
            "laya1_json": {"signal_action": "BUY"}, "qwen_json": {"action": "BUY"},
            "laya2_json": {"approved": True}, "risk_status": "APPROVED", "risk_reason": "OK",
            "final_action": "BUY", "latency_ms": 5.0, "qwen_execution_mode": "LOCAL_DEVICE",
            "laya_execution_mode": "LOCAL_DEVICE", "source_mode": "replay",
        },
        order_data={
            "side": "BUY", "symbol": "NVDA", "exec_price": 140.0, "quantity": 50.0, "fee": 5.0,
            "slippage_bps": 5.0, "notional": 7000.0, "realized_pnl": None, "executed_at": close_t,
        },
        snapshot_data={
            "candle_time": close_t, "symbol": "NVDA", "cash": 43000.0, "position_qty": 50.0,
            "avg_entry_price": 140.0, "entry_cost_total": 7005.0, "position_value": 7000.0,
            "total_equity": 50000.0, "daily_pnl_pct": 0.0, "drawdown_pct": 0.0,
            "last_price": 140.0, "highest_price": 140.0, "stop_loss_price": 133.0,
            "take_profit_price": 150.0, "source_mode": "replay",
        },
    )
    import backend.app.core.scheduler as sched_mod

    monkeypatch.setattr(
        app.state.market_client, "fetch_realtime_quotes_batch",
        lambda syms: {"NVDA": {"price": 148.0, "source": "TEST", "online": True}},
    )
    hdr = {"X-API-Key": cfg.gateway_api_key}
    with TestClient(app) as client:
        before = client.get("/api/decisions").json()
        open_mov = [i for i in before["items"] if i["symbol"] == "NVDA"][0]["movement"]
        assert open_mov["settlement_status"] == "OPEN"
        assert open_mov["take_profit_price"] == pytest.approx(150.0)
        assert open_mov["declared_tp"] == pytest.approx(150.0)
        assert open_mov["declared_sl"] == pytest.approx(133.0)

        # Con mercado cerrado se bloquea la liquidación de acciones
        monkeypatch.setattr(
            sched_mod,
            "get_market_session_status",
            lambda sym, now_utc=None: {"symbol": sym, "is_open": False, "et_time": "2026-11-23 20:00 ET", "session_label": "CERRADO"},
        )
        r_closed = client.post("/api/control/liquidate", headers=hdr, json={"symbol": "NVDA"})
        assert r_closed.status_code == 409

        # Con mercado abierto procede la liquidación
        monkeypatch.setattr(
            sched_mod,
            "get_market_session_status",
            lambda sym, now_utc=None: {"symbol": sym, "is_open": True, "et_time": "2026-11-23 11:00 ET", "session_label": "ABIERTO"},
        )
        r = client.post("/api/control/liquidate", headers=hdr, json={"symbol": "NVDA"})
        assert r.status_code == 200, r.text
        assert r.json()["realized_pnl"] > 0
        assert db.get_position("NVDA") is None

        after = client.get("/api/decisions").json()
        assert after["summary"]["open_unliquidated_count"] == 0
        assert after["summary"]["liquidated_count"] == 2
        assert client.get("/api/audit/verify").json()["valid"] is True

        r2 = client.post("/api/control/liquidate", headers=hdr, json={"symbol": "NVDA"})
        assert r2.status_code == 409


def test_market_closed_blocks_live_equities_and_allows_crypto_24_7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.middle_office.risk_engine import (
        DeterministicRiskEngine,
        get_market_session_status,
    )

    cfg = _make_settings(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    paper = PaperTradingEngine(cfg, db)
    risk = DeterministicRiskEngine(cfg, db)
    port = paper.get_portfolio_state(current_price=150.0, symbol="NVDA")

    # Sábado 15:00 UTC -> acciones cerradas, BTC 24/7 abierto
    weekend_utc = datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc)
    # Lunes 16:00 UTC (11:00 AM ET) -> NYSE/NASDAQ abierto
    weekday_open_utc = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)
    # Lunes 22:30 UTC (17:30 ET After-Hours) -> acciones cerradas
    after_hours_utc = datetime(2026, 11, 23, 22, 30, tzinfo=timezone.utc)

    assert get_market_session_status("NVDA", now_utc=weekend_utc)["is_open"] is False
    assert get_market_session_status("NVDA", now_utc=after_hours_utc)["is_open"] is False
    assert get_market_session_status("NVDA", now_utc=weekday_open_utc)["is_open"] is True
    assert get_market_session_status("BTC-USD", now_utc=weekend_utc)["is_open"] is True

    v_closed = risk.evaluate(
        proposed_action="BUY",
        symbol="NVDA",
        market_price=150.0,
        portfolio=port,
        system_status="RUNNING",
        laya2_approved=True,
        source_mode="live",
        now_utc=after_hours_utc,
    )
    assert v_closed.approved is False
    assert v_closed.final_action == "HOLD"
    assert v_closed.rule_triggered == "MARKET_CLOSED"

    v_crypto = risk.evaluate(
        proposed_action="BUY",
        symbol="BTC-USD",
        market_price=61000.0,
        portfolio=paper.get_portfolio_state(current_price=61000.0, symbol="BTC-USD"),
        system_status="RUNNING",
        laya2_approved=True,
        source_mode="live",
        now_utc=weekend_utc,
    )
    assert v_crypto.approved is True
    assert v_crypto.final_action == "BUY"

    # En modo replay histórico no bloquea por horario actual
    v_replay = risk.evaluate(
        proposed_action="BUY",
        symbol="NVDA",
        market_price=150.0,
        portfolio=port,
        system_status="RUNNING",
        laya2_approved=True,
        source_mode="replay",
        now_utc=weekend_utc,
    )
    assert v_replay.approved is True


def test_periodic_tp_sl_recalculation_and_movements_audit_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    cfg = _make_settings(tmp_path, monkeypatch)
    cfg.qwen_api_key = ""
    app = create_app(settings_override=cfg)
    db = app.state.db
    close_t = "2026-11-20T23:59:59Z"

    cid = db.register_candle_idempotent(
        {
            "symbol": "SPY", "interval": "1d", "open_time": "2026-11-20T00:00:00Z",
            "close_time": close_t, "open": 680.0, "high": 692.0, "low": 679.0,
            "close": 690.0, "volume": 100000.0, "source_mode": "replay",
        }
    )
    db.record_cycle_result(
        candle_id=cid,
        decision_data={
            "symbol": "SPY", "interval": "1d", "candle_close_utc": close_t, "model_pred": 0.006,
            "laya1_json": {"signal_action": "BUY"}, "qwen_json": {"action": "BUY"},
            "laya2_json": {"approved": True}, "risk_status": "APPROVED", "risk_reason": "OK",
            "final_action": "BUY", "stop_loss_price": 660.0, "take_profit_price": 730.0,
            "latency_ms": 8.0, "qwen_execution_mode": "LOCAL_DEVICE",
            "laya_execution_mode": "LOCAL_DEVICE", "source_mode": "replay",
        },
        order_data={
            "side": "BUY", "symbol": "SPY", "exec_price": 690.0, "quantity": 18.0, "fee": 6.21,
            "slippage_bps": 5.0, "notional": 12420.0, "realized_pnl": None, "executed_at": close_t,
        },
        snapshot_data={
            "candle_time": close_t, "symbol": "SPY", "cash": 37573.79, "position_qty": 18.0,
            "avg_entry_price": 690.0, "entry_cost_total": 12426.21, "position_value": 12420.0,
            "total_equity": 49993.79, "daily_pnl_pct": 0.0, "drawdown_pct": 0.0,
            "last_price": 690.0, "highest_price": 690.0, "stop_loss_price": 660.0,
            "take_profit_price": 730.0, "source_mode": "replay",
        },
    )

    # Recalcular con precio actual subiendo a 724.0 -> SL debe subir (Break-Even/Trailing > 660.0)
    updates = app.state.paper_engine.recalculate_open_positions_targets(
        live_quotes={"SPY": {"price": 724.0}},
        market_client=app.state.market_client,
        predictor=app.state.predictor,
        interval="1d",
    )
    assert len(updates) == 1
    assert updates[0]["new_sl"] > 690.0  # ratchet por encima del entry, minimizando pérdidas
    assert updates[0]["new_tp"] > 724.0

    # Verificar endpoint /api/audit/movements
    with TestClient(app) as client:
        audit_res = client.get("/api/audit/movements?limit=50")
        assert audit_res.status_code == 200
        body = audit_res.json()
        assert body["summary"]["total_movements"] == 1
        assert body["summary"]["open_count"] == 1
        assert body["summary"]["chain_valid"] is True
        row = body["items"][0]
        assert row["symbol"] == "SPY"
        assert row["declared_sl"] == pytest.approx(660.0)
        assert row["declared_tp"] == pytest.approx(730.0)
        assert row["stop_loss_price"] == pytest.approx(updates[0]["new_sl"])
        assert row["take_profit_price"] == pytest.approx(updates[0]["new_tp"])
        assert row["hash_verified"] is True

