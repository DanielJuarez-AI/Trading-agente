"""
Suite Automatizada de los 9 Escenarios de Error Obligatorios (Fase 5 / Sección 15)
+ Verificación de >= 20 Ciclos Válidos y Comparativa de 3 Carriles.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.back_office.database import DuplicateCandleError, TradingDatabase
from backend.app.back_office.mexc_client import (
    MEXCMarketClient,
    MarketDataUnavailableError,
    StaleDataError,
)
from backend.app.back_office.paper_engine import (
    InsufficientFundsError,
    PaperTradingEngine,
    SystemStoppedError,
)
from backend.app.benchmark.comparator import run_three_lane_benchmark
from backend.app.core.config import EXPORTS_DIR, get_settings
from backend.app.core.security import GatewayAuthError, PrivateAuthGateway
from backend.app.front_office.laya_alpha import (
    LLMServiceUnavailableError,
    LayaAlphaAgent,
    LayaAlphaOutput,
)
from backend.app.front_office.predictor import PredictionResult
from backend.app.main import create_app
from backend.app.orchestrator.qwen_agent import (
    QwenInvalidOutputError,
    QwenOrchestratorAgent,
)


def _setup_cfg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db_name: str = "nine_errors.db",
    qwen_api_key: str = "",
):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / db_name))
    monkeypatch.setenv("GATEWAY_API_KEY", "secret_gateway_key_for_tests")
    monkeypatch.setenv("ADMIN_CONTROL_TOKEN", "secret_admin_token_for_tests")
    monkeypatch.setenv("QWEN_API_KEY", qwen_api_key)
    monkeypatch.setenv("SIMULATION_PROFILE", "agent_paper")
    monkeypatch.setenv("ENABLE_REAL_TRADING", "false")
    return get_settings()


# ==============================================================================
# ESCENARIO 1: API MEXC sin respuesta o con rate limit (429) -> backoff + pausa
# ==============================================================================
def test_error_1_mexc_api_rate_limit_backoff_and_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    db.set_state("system_status", "RUNNING")

    attempts = {"count": 0}
    sleep_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(429, json={"msg": "Too many requests"})

    mock_http = httpx.Client(transport=httpx.MockTransport(handler))
    client = MEXCMarketClient(
        cfg, db, http_client=mock_http, sleep_fn=lambda s: sleep_calls.append(s)
    )

    with pytest.raises(MarketDataUnavailableError):
        client.fetch_live_candles("BTC-USD", interval="1d", enforce_mexc_for_crypto=True)

    assert attempts["count"] == 3
    assert sleep_calls == [0.5, 1.0]
    assert db.get_state("system_status") == "PAUSED"
    errors = db.list_errors()
    assert any(e["error_code"] == "MEXC_API_UNAVAILABLE" for e in errors)


# ==============================================================================
# ESCENARIO 2: Dato atrasado (stale data) -> rechazo sin operar
# ==============================================================================
def test_error_2_stale_data_rejected_without_trading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    cfg.max_data_age_seconds = 3600  # 1 hora
    db = TradingDatabase(cfg.sqlite_db_path)

    now = datetime.now(timezone.utc)
    stale_close = now - timedelta(hours=10)
    stale_open = stale_close - timedelta(hours=1)
    future_close = now + timedelta(minutes=30)

    klines = [
        [
            int(stale_open.timestamp() * 1000),
            "90000", "91000", "89500", "90500", "100",
            int(stale_close.timestamp() * 1000),
        ],
        # Vela abierta en formación que debe descartarse estrictamente
        [
            int(now.timestamp() * 1000),
            "90500", "90800", "90400", "90700", "10",
            int(future_close.timestamp() * 1000),
        ],
    ]

    mock_http = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=klines))
    )
    client = MEXCMarketClient(cfg, db, http_client=mock_http)

    with pytest.raises(StaleDataError):
        client.fetch_live_candles(
            "BTC-USD", interval="1h", now_utc=now, enforce_mexc_for_crypto=True
        )

    assert len(db.list_orders()) == 0
    assert any(e["error_code"] == "STALE_DATA_REJECTED" for e in db.list_errors())


# ==============================================================================
# ESCENARIO 3: Ollama local / API Qwen inaccesible -> captura, log y pausa
# ==============================================================================
def test_error_3_ollama_local_and_qwen_api_unavailable_pauses_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    db.set_state("system_status", "RUNNING")
    gateway = PrivateAuthGateway(cfg, db)
    paper = PaperTradingEngine(cfg, db)
    portfolio = paper.get_portfolio_state(current_price=90000.0)

    pred = PredictionResult(
        symbol="BTC-USD",
        candle_close_utc="2026-11-20T23:59:59Z",
        pred_return_t1=0.005,
        direction="UP",
        ema_ratio=1.01,
        vol_20=0.02,
        rsi_14=58.0,
        momentum_5=0.015,
        ret_lag_1=0.002,
        model_version="lstm_v1",
        features_verified=True,
    )

    # 3A: Ollama local detenido para Laya #1
    def fail_conn(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    laya1 = LayaAlphaAgent(
        cfg, db, gateway, http_client=httpx.Client(transport=httpx.MockTransport(fail_conn))
    )
    with pytest.raises(LLMServiceUnavailableError):
        laya1.evaluate(pred, position_open=False, gateway_key=cfg.gateway_api_key, strict_live=True)

    assert db.get_state("system_status") == "PAUSED"
    assert any(e["error_code"] == "OLLAMA_LOCAL_UNAVAILABLE" for e in db.list_errors())

    # 3B: API externa de Qwen caída (HTTP 503)
    db.set_state("system_status", "RUNNING")
    qwen = QwenOrchestratorAgent(
        cfg,
        db,
        gateway,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(503, json={"error": "down"}))
        ),
    )
    laya1_ok = LayaAlphaOutput(
        regime="bull_trend",
        regime_probs={"bull_trend": 0.8},
        signal_action="BUY",
        signal_probs={"BUY": 0.8},
        conviction_level="high",
        conviction_score=0.85,
        latency_ms=12.0,
        execution_mode="LOCAL_DEVICE",
        is_local_device=True,
        transport="ollama_local_systemone",
        model="laya:latest",
        payload_token_estimate=120,
    )
    with pytest.raises(LLMServiceUnavailableError):
        qwen.orchestrate(pred, laya1_ok, portfolio, gateway_key=cfg.gateway_api_key, strict_live=True)

    assert db.get_state("system_status") == "PAUSED"
    assert any(e["error_code"] == "QWEN_EXTERNAL_API_UNAVAILABLE" for e in db.list_errors())


# ==============================================================================
# ESCENARIO 4: JSON inválido devuelto por Qwen -> rechazo por validador Pydantic
# ==============================================================================
def test_error_4_qwen_invalid_json_rejected_by_pydantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    db = TradingDatabase(cfg.sqlite_db_path)
    gateway = PrivateAuthGateway(cfg, db)
    paper = PaperTradingEngine(cfg, db)
    portfolio = paper.get_portfolio_state(current_price=90000.0)

    pred = PredictionResult(
        symbol="NVDA",
        candle_close_utc="2026-11-20T23:59:59Z",
        pred_return_t1=0.005,
        direction="UP",
        ema_ratio=1.01,
        vol_20=0.02,
        rsi_14=58.0,
        momentum_5=0.015,
        ret_lag_1=0.002,
        model_version="lstm_v1",
        features_verified=True,
    )
    laya1_ok = LayaAlphaOutput(
        regime="bull_trend",
        regime_probs={"bull_trend": 0.8},
        signal_action="BUY",
        signal_probs={"BUY": 0.8},
        conviction_level="high",
        conviction_score=0.85,
        latency_ms=12.0,
        execution_mode="LOCAL_DEVICE",
        is_local_device=True,
        transport="ollama_local_systemone",
        model="laya:latest",
        payload_token_estimate=120,
    )

    bad_payload = {
        "choices": [
            {
                "message": {
                    # Falta JSON válido y contiene acción inventada
                    "content": '{"action": "LEVERAGE_10X", "symbol": "NVDA"}'
                }
            }
        ]
    }
    qwen = QwenOrchestratorAgent(
        cfg,
        db,
        gateway,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bad_payload))
        ),
    )

    with pytest.raises(QwenInvalidOutputError):
        qwen.orchestrate(pred, laya1_ok, portfolio, gateway_key=cfg.gateway_api_key, strict_live=True)

    assert len(db.list_orders()) == 0
    assert any(e["error_code"] == "QWEN_INVALID_JSON_SCHEMA" for e in db.list_errors())


# ==============================================================================
# ESCENARIO 5: API Key incorrecta o ausente en el Gateway -> HTTP 401/403 + log
# ==============================================================================
def test_error_5_gateway_rejects_missing_or_wrong_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    app = create_app(settings_override=cfg)

    with TestClient(app) as client:
        # Sin clave -> 401
        r_missing = client.post("/api/control/start", json={})
        assert r_missing.status_code == 401

        # Clave incorrecta -> 403
        r_wrong = client.post(
            "/api/control/start",
            headers={"X-API-Key": "wrong_attacker_key"},
            json={},
        )
        assert r_wrong.status_code == 403

        # Invocación directa de LLM bloqueada sin clave válida
        with pytest.raises(GatewayAuthError) as exc_info:
            app.state.gateway.authorize_llm_call("qwen_orchestrator", "EXTERNAL_API", "bad_key")
        assert exc_info.value.status_code == 403

        errors = client.get("/api/errors").json()["items"]
        codes = {e["error_code"] for e in errors}
        assert "AUTH_401_MISSING_CREDENTIALS" in codes
        assert "AUTH_403_INVALID_KEY" in codes
        # Verificar que jamás se exponga la clave secreta real en logs
        for e in errors:
            assert "secret_gateway_key_for_tests" not in e["message"]
            assert "secret_gateway_key_for_tests" not in e["payload_excerpt"]


# ==============================================================================
# ESCENARIO 6: Saldo insuficiente en compra -> rechazo en risk_engine / paper_engine
# ==============================================================================
def test_error_6_insufficient_funds_rejected_on_buy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    app = create_app(settings_override=cfg)
    db = app.state.db
    paper = app.state.paper_engine
    risk = app.state.risk_engine

    # Simular efectivo casi agotado ($2.00 frente a BTC a $90,000 con min_lot 0.0001 = $9.00)
    db.set_state("daily_start_equity", "2.0")
    db.set_state("peak_equity", "2.0")
    with db.atomic_tx() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_snapshots (
                candle_time, symbol, cash, position_qty, avg_entry_price,
                entry_cost_total, position_value, total_equity,
                daily_pnl_pct, drawdown_pct, source_mode, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "2026-11-20T23:59:59Z", "BTC-USD", 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, "replay", "2026-11-20T23:59:59Z"
            ),
        )

    state = paper.get_portfolio_state(current_price=90000.0)
    verdict = risk.evaluate("BUY", "BTC-USD", 90000.0, state, system_status="RUNNING")
    assert verdict.approved is False
    assert verdict.rule_triggered == "INSUFFICIENT_FUNDS"
    assert verdict.final_action == "HOLD"

    with pytest.raises(InsufficientFundsError):
        paper.simulate_execution("BUY", "BTC-USD", 90000.0)

    assert any(
        e["error_code"] in {"RISK_INSUFFICIENT_FUNDS", "INSUFFICIENT_FUNDS_BUY"}
        for e in db.list_errors()
    )


# ==============================================================================
# ESCENARIO 7: Evento/vela duplicada -> bloqueo por idempotencia SQLite
# ==============================================================================
def test_error_7_duplicate_candle_blocked_by_sqlite_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    app = create_app(settings_override=cfg)
    scheduler = app.state.scheduler
    df = app.state.market_client.load_replay_candles("AAPL", interval="1d").iloc[:50].reset_index(drop=True)

    # Primer procesamiento válido
    first = scheduler.process_candle_window(
        candles_df=df,
        symbol="AAPL",
        interval="1d",
        source_mode="replay",
        gateway_key=cfg.gateway_api_key,
    )
    assert first["symbol"] == "AAPL"
    assert app.state.db.count_processed_candles() == 1

    # Segundo procesamiento de la misma vela -> bloqueado por UNIQUE constraint
    with pytest.raises(DuplicateCandleError):
        scheduler.process_candle_window(
            candles_df=df,
            symbol="AAPL",
            interval="1d",
            source_mode="replay",
            gateway_key=cfg.gateway_api_key,
        )

    assert app.state.db.count_processed_candles() == 1
    assert any(e["error_code"] == "DUPLICATE_CANDLE" for e in app.state.db.list_errors())


# ==============================================================================
# ESCENARIO 8: Reinicio abrupto del backend -> recuperación exacta sin duplicar
# ==============================================================================
def test_error_8_abrupt_restart_recovers_exact_state_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg1 = _setup_cfg(tmp_path, monkeypatch, db_name="restart_recovery.db")
    app1 = create_app(settings_override=cfg1)

    df = app1.state.market_client.load_replay_candles("MSFT", interval="1d").iloc[:50].reset_index(drop=True)
    last_close = float(df.iloc[-1]["close"])

    # Registrar 1 vela y ejecutar una compra real en Paper en la primera instancia
    candle_id = app1.state.db.register_candle_idempotent(
        {
            "symbol": "MSFT",
            "interval": "1d",
            "open_time": str(df.iloc[-1]["open_time"]),
            "close_time": str(df.iloc[-1]["close_time"]),
            "open": float(df.iloc[-1]["open"]),
            "high": float(df.iloc[-1]["high"]),
            "low": float(df.iloc[-1]["low"]),
            "close": last_close,
            "volume": float(df.iloc[-1]["volume"]),
            "source_mode": "replay",
        }
    )
    sim = app1.state.paper_engine.simulate_execution("BUY", "MSFT", last_close)
    sim["snapshot"]["candle_time"] = str(df.iloc[-1]["close_time"])
    sim["snapshot"]["source_mode"] = "replay"
    app1.state.db.record_cycle_result(
        candle_id=candle_id,
        decision_data={
            "symbol": "MSFT",
            "interval": "1d",
            "candle_close_utc": str(df.iloc[-1]["close_time"]),
            "model_pred": 0.004,
            "laya1_json": {"regime": "bull_trend"},
            "qwen_json": {"action": "BUY"},
            "laya2_json": {"approved": True},
            "risk_status": "APPROVED",
            "risk_reason": "OK",
            "final_action": "BUY",
            "latency_ms": 25.0,
            "qwen_execution_mode": "EXTERNAL_API",
            "laya_execution_mode": "LOCAL_DEVICE",
            "source_mode": "replay",
        },
        order_data=sim["order"],
        snapshot_data=sim["snapshot"],
    )
    app1.state.db.set_state("active_ticker", "MSFT")
    app1.state.db.set_state("system_status", "PAUSED")
    cash_before = sim["snapshot"]["cash"]
    qty_before = sim["snapshot"]["position_qty"]
    orders_before = len(app1.state.db.list_orders())

    # Simular reinicio abrupto creando una instancia completamente nueva sobre el mismo SQLite
    del app1
    cfg2 = _setup_cfg(tmp_path, monkeypatch, db_name="restart_recovery.db")
    app2 = create_app(settings_override=cfg2)

    recovered = app2.state.paper_engine.get_portfolio_state(current_price=last_close)
    assert recovered.cash == pytest.approx(cash_before, rel=1e-6)
    assert recovered.position_qty == pytest.approx(qty_before, rel=1e-6)
    assert app2.state.db.get_state("active_ticker") == "MSFT"
    assert app2.state.db.get_state("system_status") == "PAUSED"
    assert len(app2.state.db.list_orders()) == orders_before

    # Re-ingresar la misma vela tras el reinicio sigue bloqueado
    with pytest.raises(DuplicateCandleError):
        app2.state.scheduler.process_candle_window(
            candles_df=df,
            symbol="MSFT",
            interval="1d",
            source_mode="replay",
            gateway_key=cfg2.gateway_api_key,
        )
    assert len(app2.state.db.list_orders()) == orders_before


# ==============================================================================
# ESCENARIO 9: Activación de STOP -> bloquea nuevas órdenes preservando posición
# ==============================================================================
def test_error_9_stop_button_blocks_new_orders_and_preserves_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch)
    app = create_app(settings_override=cfg)

    with TestClient(app) as client:
        # Simular compra previa dejando posición abierta en NVDA
        sim_buy = app.state.paper_engine.simulate_execution("BUY", "NVDA", 140.0)
        c_id = app.state.db.register_candle_idempotent(
            {
                "symbol": "NVDA",
                "interval": "1d",
                "open_time": "2026-11-18T00:00:00Z",
                "close_time": "2026-11-18T23:59:59Z",
                "open": 139.0,
                "high": 141.0,
                "low": 138.0,
                "close": 140.0,
                "volume": 1000000.0,
                "source_mode": "replay",
            }
        )
        sim_buy["snapshot"]["candle_time"] = "2026-11-18T23:59:59Z"
        sim_buy["snapshot"]["source_mode"] = "replay"
        app.state.db.record_cycle_result(
            candle_id=c_id,
            decision_data={
                "symbol": "NVDA",
                "interval": "1d",
                "candle_close_utc": "2026-11-18T23:59:59Z",
                "model_pred": 0.006,
                "laya1_json": {"regime": "bull_trend"},
                "qwen_json": {"action": "BUY"},
                "laya2_json": {"approved": True},
                "risk_status": "APPROVED",
                "risk_reason": "OK",
                "final_action": "BUY",
                "latency_ms": 20.0,
                "qwen_execution_mode": "EXTERNAL_API",
                "laya_execution_mode": "LOCAL_DEVICE",
                "source_mode": "replay",
            },
            order_data=sim_buy["order"],
            snapshot_data=sim_buy["snapshot"],
        )
        pos_before_stop = app.state.paper_engine.get_portfolio_state(140.0).position_qty
        assert pos_before_stop > 0

        # Activar botón STOP vía API autenticada
        resp = client.post(
            "/api/control/stop",
            headers={"X-API-Key": cfg.gateway_api_key},
            json={},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "STOPPED"
        assert resp.json()["preserved_position_qty"] == pytest.approx(pos_before_stop, rel=1e-6)

        # Risk engine y Paper engine deben bloquear cualquier nueva orden (BUY o SELL)
        state = app.state.paper_engine.get_portfolio_state(145.0)
        verdict = app.state.risk_engine.evaluate(
            "SELL", "NVDA", 145.0, state, system_status="STOPPED"
        )
        assert verdict.approved is False
        assert verdict.rule_triggered == "KILL_SWITCH_STOPPED"
        assert verdict.final_action == "HOLD"

        with pytest.raises(SystemStoppedError):
            app.state.paper_engine.simulate_execution("SELL", "NVDA", 145.0)

        # La posición abierta NO fue liquidada automáticamente
        pos_after_stop = app.state.paper_engine.get_portfolio_state(145.0).position_qty
        assert pos_after_stop == pytest.approx(pos_before_stop, rel=1e-6)


# ==============================================================================
# INTEGRACIÓN FASE 5: >= 20 CICLOS VÁLIDOS + COMPARATIVA DE 3 CARRILES
# ==============================================================================
def test_twenty_five_cycles_and_three_lane_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _setup_cfg(tmp_path, monkeypatch, db_name="cycles_25.db")
    app = create_app(settings_override=cfg)

    with TestClient(app) as client:
        resp = client.post(
            "/api/control/replay",
            headers={"X-API-Key": cfg.gateway_api_key},
            json={"symbol": "BTC-USD", "n_cycles": 25},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cycles_executed"] == 25

        decisions = client.get("/api/decisions?limit=50").json()["items"]
        assert len(decisions) == 25
        # Verificar trazabilidad de topología híbrida en cada decisión
        for d in decisions:
            assert d["qwen_execution_mode"] == "EXTERNAL_API"
            assert d["laya_execution_mode"] == "LOCAL_DEVICE"
            assert d["qwen_json"]["is_local_device"] is False
            assert d["laya1_json"]["is_local_device"] is True
            assert d["laya2_json"]["is_local_device"] is True

    summary = run_three_lane_benchmark(
        symbol="BTC-USD",
        n_cycles=25,
        settings=cfg,
        predictor=app.state.predictor,
        export_csv=True,
    )
    assert summary["cycles_evaluated"] == 25
    assert len(summary["lanes"]) == 3
    assert (EXPORTS_DIR / "simulacion.csv").exists()
    assert (EXPORTS_DIR / "benchmark_summary.json").exists()
