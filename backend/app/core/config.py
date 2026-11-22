"""
Configuración Central del Sistema Multi-Agente de Trading Cuantitativo.

ARQUITECTURA HÍBRIDA EXPLÍCITA:
- [API EXTERNA - NO LOCAL]: Qwen 14B (Orquestador System-2) se consume mediante
  peticiones HTTP a una API externa remota compatible con OpenAI/DashScope.
  NO corre en el dispositivo local.
- [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]: Los agentes Laya #1 (Front Office)
  y Laya #2 (Middle Office) se ejecutan 100% en el dispositivo local vía Ollama.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Raíz del repositorio: c:\Users\danjd\Trading-agente
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
ARTIFACTS_DIR = REPO_ROOT / "notebooks" / "artifacts"
EXPORTS_DIR = REPO_ROOT / "exports"
DATA_DIR = REPO_ROOT / "data"
BASE_RELEASE_VERSION: str = "0.1"


def get_app_version_info() -> Dict[str, Any]:
    """
    Calcula la versión dinámica del sistema basada en el hito V.0.1 y el conteo
    real de commits en Git (formato v0.1.<commits>).
    """
    commit_count = 10
    git_sha = "local"
    try:
        cnt_proc = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if cnt_proc.returncode == 0 and cnt_proc.stdout.strip().isdigit():
            commit_count = int(cnt_proc.stdout.strip())
        sha_proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if sha_proc.returncode == 0 and sha_proc.stdout.strip():
            git_sha = sha_proc.stdout.strip()
    except Exception:
        pass

    semver = f"{BASE_RELEASE_VERSION}.{commit_count}"
    return {
        "version": f"v{semver}",
        "semver": semver,
        "release_milestone": "V.0.1",
        "commit_count": commit_count,
        "git_sha": git_sha,
    }

# ==============================================================================
# UNIVERSO OFICIAL DE 30 TICKERS SOLICITADO POR EL USUARIO
# ==============================================================================
PRIMARY_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "TSLA", "HWM", "JPM", "GOOGL", "AMZN", "META", "LMT",
    "WM", "LULU", "DIS", "V", "MA", "BTC-USD", "SPY", "QQQ", "NFLX", "AMD",
    "CRM", "ADBE", "PYPL", "COST", "HD", "UNH", "JNJ", "PG", "KO", "WMT",
]

# Alias de MEXC para compatibilidad con /api/v3/klines
SYMBOL_ALIASES: Dict[str, str] = {
    "BTC-USD": "BTCUSDT",
    "BTCUSDT": "BTC-USD",
}

ALLOWED_SYMBOLS: List[str] = PRIMARY_TICKERS + ["BTCUSDT"]

# ==============================================================================
# MARCADORES EXPLÍCITOS DE TOPOLOGÍA DE INFERENCIA
# ==============================================================================
# [API EXTERNA - NO LOCAL]: Qwen 14B NO se ejecuta en local.
QWEN_EXECUTION_MODE: str = "EXTERNAL_API"
QWEN_IS_LOCAL: bool = False

# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]: Laya #1 y Laya #2 SÍ corren en local.
LAYA_EXECUTION_MODE: str = "LOCAL_DEVICE"
LAYA_IS_LOCAL: bool = True


def _load_dotenv_if_present(env_path: Path) -> None:
    """Carga variables de .env sin sobreescribir variables ya exportadas."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    initial_capital: float
    fee_pct: float


@dataclass
class Settings:
    # Seguridad (NUNCA exponer en frontend / VITE_*)
    gateway_api_key: str
    admin_control_token: str

    # --------------------------------------------------------------------------
    # [SWITCH QWEN: API EXTERNA 14B <-> LOCAL 6 GB VRAM (qwen2.5-3b-quant:latest)]
    # --------------------------------------------------------------------------
    qwen_execution_mode: str
    qwen_is_local: bool
    qwen_api_base_url: str
    qwen_api_key: str
    qwen_model_name: str
    qwen_local_model_name: str
    qwen_local_vram_budget_gb: float
    qwen_local_num_ctx: int
    qwen_temperature: float
    qwen_max_tokens: int

    # --------------------------------------------------------------------------
    # [DISPOSITIVO LOCAL - OLLAMA /v1/systemone] Dual Laya #1 y #2 (System-1)
    # --------------------------------------------------------------------------
    laya_execution_mode: str
    laya_is_local: bool
    ollama_base_url: str
    laya_model_name: str
    llm_timeout_seconds: int
    allow_local_laya_calibrated_fallback: bool

    # Mercado y simulación
    mexc_public_base_url: str
    trading_symbol: str
    candle_interval: str
    enable_real_trading: bool
    simulation_profile: str
    profile_config: ProfileConfig
    slippage_bps: float
    max_exposure_pct: float
    max_daily_loss_pct: float
    min_lot_size: float
    step_size: float
    long_only: bool
    allow_leverage: bool
    max_data_age_seconds: int
    sqlite_db_path: Path
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.5
    trailing_stop_pct: float = 0.035
    max_single_position_pct: float = 0.45

    # Tickers y configuraciones cargadas
    primary_tickers: List[str] = field(default_factory=lambda: list(PRIMARY_TICKERS))
    allowed_symbols: List[str] = field(default_factory=lambda: list(ALLOWED_SYMBOLS))
    risk_limits_raw: Dict[str, Any] = field(default_factory=dict)
    llm_config_raw: Dict[str, Any] = field(default_factory=dict)

    def hybrid_topology_metadata(
        self, active_qwen_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Devuelve metadatos explícitos de ubicación de cómputo de los agentes."""
        mode = active_qwen_mode or self.qwen_execution_mode
        if mode == "LOCAL_DEVICE":
            qwen_meta = {
                "execution_mode": "LOCAL_DEVICE",
                "is_local_device": True,
                "location_label": f"DISPOSITIVO LOCAL (Ollama {self.qwen_local_vram_budget_gb:.0f}GB VRAM)",
                "endpoint": f"{self.ollama_base_url.rstrip('/')}/v1/chat/completions",
                "model": self.qwen_local_model_name,
                "vram_budget_gb": self.qwen_local_vram_budget_gb,
                "num_ctx": self.qwen_local_num_ctx,
            }
        else:
            qwen_meta = {
                "execution_mode": "EXTERNAL_API",
                "is_local_device": False,
                "location_label": "API EXTERNA (No Local)",
                "endpoint": self.qwen_api_base_url,
                "model": self.qwen_model_name,
                "vram_budget_gb": None,
                "num_ctx": None,
            }
        return {
            "qwen_orchestrator": qwen_meta,
            "laya_alpha": {
                "execution_mode": self.laya_execution_mode,
                "is_local_device": self.laya_is_local,
                "location_label": "DISPOSITIVO LOCAL (DRL Dueling A2C PyTorch)",
                "endpoint": "pytorch://laya1_drl_v1.pt (Ollama /v1/systemone fallback en strict_live)",
                "model": f"laya1_drl_v1 + {self.laya_model_name}",
            },
            "laya_guard": {
                "execution_mode": self.laya_execution_mode,
                "is_local_device": self.laya_is_local,
                "location_label": "DISPOSITIVO LOCAL (Ollama /api/generate + /v1/systemone)",
                "endpoint": f"{self.ollama_base_url.rstrip('/')}/api/generate",
                "model": self.laya_model_name,
            },
        }


def get_settings() -> Settings:
    """Construye Settings desde .env, entorno, risk_limits.yaml y llm_config.yaml."""
    _load_dotenv_if_present(REPO_ROOT / ".env")
    risk_yaml = _load_yaml(CONFIG_DIR / "risk_limits.yaml")
    llm_yaml = _load_yaml(CONFIG_DIR / "llm_config.yaml")

    profiles_yaml = risk_yaml.get("profiles", {})
    risk_controls = risk_yaml.get("risk_controls", {})

    profile_name = os.getenv("SIMULATION_PROFILE", "agent_paper")
    if profile_name not in profiles_yaml:
        profile_name = "agent_paper"

    raw_prof = profiles_yaml.get(
        profile_name,
        {"initial_capital": 50000.0, "fee_pct": 0.0}
        if profile_name == "agent_paper"
        else {"initial_capital": 100000.0, "fee_pct": 0.001},
    )
    profile_cfg = ProfileConfig(
        name=profile_name,
        initial_capital=float(raw_prof["initial_capital"]),
        fee_pct=float(raw_prof["fee_pct"]),
    )

    qwen_yaml = llm_yaml.get("qwen_orchestrator", {})
    qwen_local_yaml = llm_yaml.get("qwen_local_6gb", {})
    laya_yaml = llm_yaml.get("laya_alpha_agent", {})

    db_env = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "trading_agent.db"))
    db_path = Path(db_env)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    # En runtime se carga la lista de tickers desde YAML o PRIMARY_TICKERS
    yaml_primary = risk_yaml.get("primary_tickers") or list(PRIMARY_TICKERS)
    yaml_allowed = risk_yaml.get("allowed_tickers") or list(ALLOWED_SYMBOLS)

    # Forzar lectura booleana segura para bloqueo de órdenes reales
    env_real_trading = _as_bool(os.getenv("ENABLE_REAL_TRADING"), default=False)
    yaml_real_trading = bool(risk_controls.get("enable_real_trading", False))
    enable_real = env_real_trading or yaml_real_trading

    return Settings(
        gateway_api_key=os.getenv("GATEWAY_API_KEY", "dev_local_gateway_key_change_me"),
        admin_control_token=os.getenv("ADMIN_CONTROL_TOKEN", "dev_admin_control_token"),
        # [SWITCH QWEN: API EXTERNA <-> LOCAL 6 GB VRAM]
        qwen_execution_mode=os.getenv("QWEN_EXECUTION_MODE", QWEN_EXECUTION_MODE),
        qwen_is_local=(os.getenv("QWEN_EXECUTION_MODE", QWEN_EXECUTION_MODE) == "LOCAL_DEVICE"),
        qwen_api_base_url=os.getenv(
            "QWEN_API_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        qwen_api_key=os.getenv("QWEN_API_KEY", ""),
        qwen_model_name=os.getenv(
            "QWEN_MODEL_NAME",
            str(qwen_yaml.get("model", "qwen2.5-14b-instruct")),
        ),
        qwen_local_model_name=os.getenv(
            "QWEN_LOCAL_MODEL_NAME",
            str(qwen_local_yaml.get("model", "qwen2.5-3b-quant:latest")),
        ),
        qwen_local_vram_budget_gb=float(
            os.getenv(
                "QWEN_LOCAL_VRAM_BUDGET_GB",
                str(qwen_local_yaml.get("vram_budget_gb", 6.0)),
            )
        ),
        qwen_local_num_ctx=int(
            os.getenv(
                "QWEN_LOCAL_NUM_CTX",
                str(qwen_local_yaml.get("num_ctx", 4096)),
            )
        ),
        qwen_temperature=float(qwen_yaml.get("temperature", 0.1)),
        qwen_max_tokens=int(qwen_yaml.get("max_tokens", 450)),
        # [DISPOSITIVO LOCAL - OLLAMA /v1/systemone] Dual Laya
        laya_execution_mode=os.getenv("LAYA_EXECUTION_MODE", LAYA_EXECUTION_MODE),
        laya_is_local=True,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        laya_model_name=os.getenv(
            "LAYA_MODEL_NAME",
            str(laya_yaml.get("model", "laya:latest")),
        ),
        llm_timeout_seconds=int(
            os.getenv("LLM_TIMEOUT_SECONDS", str(qwen_yaml.get("timeout_seconds", 30)))
        ),
        allow_local_laya_calibrated_fallback=_as_bool(
            os.getenv("ALLOW_LAYA_CALIBRATED_FALLBACK"), default=True
        ),
        # Mercado y reglas de riesgo
        mexc_public_base_url=os.getenv("MEXC_PUBLIC_BASE_URL", "https://api.mexc.com"),
        trading_symbol=os.getenv(
            "TRADING_SYMBOL", os.getenv("DEFAULT_TICKER", "BTC-USD")
        ),
        candle_interval=os.getenv("CANDLE_INTERVAL", "1d"),
        enable_real_trading=enable_real,
        simulation_profile=profile_name,
        profile_config=profile_cfg,
        slippage_bps=float(os.getenv("SLIPPAGE_BPS", str(risk_controls.get("slippage_bps", 5)))),
        max_exposure_pct=float(
            os.getenv("MAX_EXPOSURE_PCT", str(risk_controls.get("max_exposure_pct", 1.0)))
        ),
        max_daily_loss_pct=float(
            os.getenv("MAX_DAILY_LOSS_PCT", str(risk_controls.get("max_daily_loss_pct", 0.03)))
        ),
        min_lot_size=float(risk_controls.get("min_lot_size", 0.0001)),
        step_size=float(risk_controls.get("step_size", 0.0001)),
        long_only=bool(risk_controls.get("long_only", True)),
        allow_leverage=bool(risk_controls.get("allow_leverage", False)),
        max_data_age_seconds=int(
            os.getenv(
                "MAX_DATA_AGE_SECONDS",
                str(risk_controls.get("max_data_age_seconds", 172800)),
            )
        ),
        stop_loss_atr_mult=float(
            os.getenv(
                "STOP_LOSS_ATR_MULT",
                str(risk_controls.get("stop_loss_atr_mult", 2.0)),
            )
        ),
        take_profit_atr_mult=float(
            os.getenv(
                "TAKE_PROFIT_ATR_MULT",
                str(risk_controls.get("take_profit_atr_mult", 3.5)),
            )
        ),
        trailing_stop_pct=float(
            os.getenv(
                "TRAILING_STOP_PCT",
                str(risk_controls.get("trailing_stop_pct", 0.035)),
            )
        ),
        max_single_position_pct=float(
            os.getenv(
                "MAX_SINGLE_POSITION_PCT",
                str(risk_controls.get("max_single_position_pct", 0.45)),
            )
        ),
        sqlite_db_path=db_path,
        primary_tickers=list(yaml_primary),
        allowed_symbols=list(yaml_allowed),
        risk_limits_raw=risk_yaml,
        llm_config_raw=llm_yaml,
    )
