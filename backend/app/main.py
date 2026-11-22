"""
FastAPI Entrypoint - Orquestación Multi-Agente para Trading Cuantitativo.

Topología explícita:
- Front Office: Predictor LSTM T4 + Laya #1 [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]
- Orquestador:  Qwen 14B [API EXTERNA - NO LOCAL]
- Middle Office: Laya #2 [DISPOSITIVO LOCAL] + Risk Engine [PYTHON DETERMINISTA]
- Back Office:  MEXC Public / 30 Tickers + Paper Engine + SQLite WAL + Auth Gateway
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from backend.app.api.routes import router as api_router
from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.mexc_client import MEXCMarketClient
from backend.app.back_office.paper_engine import PaperTradingEngine
from backend.app.core.config import Settings, get_app_version_info, get_settings
from backend.app.core.events import PipelineEventBus
from backend.app.core.scheduler import TradingScheduler
from backend.app.core.security import PrivateAuthGateway
from backend.app.core.telemetry import SystemTelemetryService
from backend.app.front_office.laya_alpha import LayaAlphaAgent
from backend.app.front_office.news_sentiment import NewsSentimentService
from backend.app.front_office.predictor import QuantPredictor
from backend.app.middle_office.laya_guard import LayaGuardAgent
from backend.app.middle_office.risk_engine import DeterministicRiskEngine
from backend.app.orchestrator.qwen_agent import QwenOrchestratorAgent


def create_app(settings_override: Optional[Settings] = None) -> FastAPI:
    cfg = settings_override or get_settings()
    if cfg.enable_real_trading:
        raise RuntimeError(
            "CRITICAL: ENABLE_REAL_TRADING debe permanecer false en todo momento."
        )

    db = TradingDatabase(cfg.sqlite_db_path)
    db.initialize_defaults_if_empty(
        default_status="RUNNING",
        default_ticker=cfg.trading_symbol,
        default_mode="replay",
        profile_name=cfg.simulation_profile,
        initial_capital=cfg.profile_config.initial_capital,
        default_interval=cfg.candle_interval,
        default_qwen_mode=cfg.qwen_execution_mode,
    )

    gateway = PrivateAuthGateway(cfg, db)
    event_bus = PipelineEventBus()
    telemetry = SystemTelemetryService(cfg, db)
    news_service = NewsSentimentService()
    market_client = MEXCMarketClient(cfg, db)
    predictor = QuantPredictor()
    laya_alpha = LayaAlphaAgent(cfg, db, gateway)
    qwen_agent = QwenOrchestratorAgent(cfg, db, gateway)
    laya_guard = LayaGuardAgent(cfg, db, gateway)
    risk_engine = DeterministicRiskEngine(cfg, db)
    paper_engine = PaperTradingEngine(cfg, db)
    scheduler = TradingScheduler(
        settings=cfg,
        db=db,
        gateway=gateway,
        market_client=market_client,
        predictor=predictor,
        laya_alpha=laya_alpha,
        qwen_agent=qwen_agent,
        laya_guard=laya_guard,
        risk_engine=risk_engine,
        paper_engine=paper_engine,
        event_bus=event_bus,
        news_service=news_service,
        telemetry=telemetry,
    )

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        # Al arrancar, recuperar estado exacto desde SQLite WAL y enlazar event_bus
        fastapi_app.state.recovered_status = db.get_state("system_status", "RUNNING")
        event_bus.set_loop(asyncio.get_running_loop())
        try:
            yield
        finally:
            await scheduler.stop_daemon()

    ver_info = get_app_version_info()
    application = FastAPI(
        title="Quant Multi-Agent Trader (Qwen API Externa + Dual Laya Local)",
        version=ver_info["semver"],
        description=(
            "Sistema cuantitativo híbrido Front/Middle/Back Office con Qwen 14B por API externa "
            "y Dual Laya en dispositivo local, sobre universo de 30 tickers."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.state.settings = cfg
    application.state.version_info = ver_info
    application.state.db = db
    application.state.gateway = gateway
    application.state.event_bus = event_bus
    application.state.telemetry = telemetry
    application.state.news_service = news_service
    application.state.market_client = market_client
    application.state.predictor = predictor
    application.state.laya_alpha = laya_alpha
    application.state.qwen_agent = qwen_agent
    application.state.laya_guard = laya_guard
    application.state.risk_engine = risk_engine
    application.state.paper_engine = paper_engine
    application.state.scheduler = scheduler

    application.include_router(api_router)

    static_index = Path(__file__).parent / "static" / "index.html"

    @application.get("/", include_in_schema=False)
    def serve_dashboard():
        if static_index.exists():
            return FileResponse(static_index)
        return {"message": "Quant Multi-Agent Trader API. Visite /docs para la documentación interactiva."}

    @application.get("/health", tags=["health"])
    def healthcheck():
        v = get_app_version_info()
        return {
            "status": "ok",
            "app_version": v["version"],
            "release_milestone": v["release_milestone"],
            "commit_count": v["commit_count"],
            "git_sha": v["git_sha"],
            "system_status": db.get_state("system_status", "RUNNING"),
            "qwen_execution_mode": cfg.qwen_execution_mode,
            "qwen_is_local": cfg.qwen_is_local,
            "laya_execution_mode": cfg.laya_execution_mode,
            "laya_is_local": cfg.laya_is_local,
        }

    return application


app = create_app()
