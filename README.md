# Quant Multi-Agent Trader (System-1 / System-2 Híbrido) — `v0.1.10`

> **Objetivo / Entrega:** `V.0.1` — Bot Cuantitativo Multi-Agente de Trading (Paper Trading)  
> **Versión Actual:** **`v0.1.10`** *(generada automáticamente a partir de `10` commits en Git — formato `v0.1.<commits>`)*  
> **Repositorio Git (Público):** [https://github.com/DanielJuarez-AI/Trading-agente](https://github.com/DanielJuarez-AI/Trading-agente)

Plataforma cuantitativa multi-agente de grado terminal para trading algorítmico en modo **Paper Trading estricto** (`ENABLE_REAL_TRADING=false`, cero órdenes reales). Diseñada bajo separación institucional de **Front Office**, **Orquestador System-2**, **Middle Office** y **Back Office**, con soporte nativo para operación **Diaria (`1d` — 200 Días)** y **Horaria (`1h` — 200 Horas)**, sub-gráfica **RSI (14) Wilder**, persistencia atómica **SQLite WAL**, streaming **Server-Sent Events (SSE)** en tiempo real y compatibilidad optimizada para GPUs de **6 GB VRAM** (vía Ollama).

---

## Tecnologías Aplicadas

| Área | Tecnologías, Librerías y Estándares |
| :--- | :--- |
| **Lenguaje & Backend API** | Python 3.12, FastAPI, Uvicorn (ASGI), Pydantic v2, Server-Sent Events (SSE), `httpx`, PyYAML |
| **Machine Learning, Deep Learning & DRL** | PyTorch (LSTM T4 + Red DRL Dueling Double-DQN / A2C), XGBoost (Walk-Forward `1h`/`1d`), scikit-learn (`StandardScaler`, Ridge), pandas, NumPy, Joblib |
| **LLMs & Orquestación Multi-Agente** | Ollama (inferencia local GPU 6 GB VRAM: `qwen2.5-3b-quant:latest` y `laya:latest`), API compatible OpenAI / DashScope (`qwen2.5-14b-instruct`) |
| **Datos de Mercado & Persistencia** | SQLite en modo WAL (`PRAGMA journal_mode=WAL`), MEXC Public REST API (`/api/v3/klines`), `yfinance`, Google News RSS (`xml.etree.ElementTree`) |
| **Frontend & Visualización Cuantitativa** | TradingView LightweightCharts v4.2.0 (Velas OHLCV 200D/200H + RSI-14 Wilder sincronizado), HTML5 Canvas (Mapa de Constelaciones Dinámico), Tailwind CSS, JavaScript ES6+ (`EventSource` SSE), React 18 + TypeScript + Vite |
| **Calidad, Seguridad & Control de Versiones** | `pytest`, `pytest-asyncio`, `PrivateAuthGateway` (`X-API-Key` / cookies `HttpOnly`), verificación de integridad SHA-256 de artefactos, Git y GitHub (repositorio público) |

---

## 1. Arquitectura Híbrida System-1 / System-2

El sistema separa las inferencias rápidas locales (*System-1*) del razonamiento estratégico (*System-2*), protegiendo toda ejecución mediante una compuerta determinista en Python puro (*Hard Gate*):

| Capa | Componente / Archivo | Ejecución | Modelo / Motor |
| :--- | :--- | :--- | :--- |
| **Front Office (Alpha)** | `backend/app/front_office/features.py` & `predictor.py` | Local (PyTorch / XGBoost) | **Quant v2**: XGBoost Walk-Forward (`quant_xgb_1d.joblib` + `quant_xgb_1h.joblib`) & LSTM T4 (`lstm_v1.pt`) |
| **Front Office (Contexto)** | `backend/app/front_office/news_sentiment.py` & `database.py` | Local | Google News RSS (catalizadores FOMC/SEC/Earnings) + **Memoria Episódica (FinMem)** |
| **Front Office (Laya #1)** | `backend/app/front_office/laya_alpha.py` | Local (`LOCAL_DEVICE`) | Red DRL **Dueling Double-DQN + A2C** (`laya1_drl_v1.pt`) + fallback Ollama (`laya:latest`) |
| **Orquestador (System-2)** | `backend/app/orchestrator/qwen_agent.py` | **Switch Dinámico** (`EXTERNAL_API` ↔ `LOCAL_DEVICE`) | API Externa (`qwen2.5-14b-instruct`) ↔ Local 6GB VRAM Ollama (`qwen2.5-3b-quant:latest`) |
| **Middle Office (Laya #2)** | `backend/app/middle_office/laya_guard.py` | Local (`LOCAL_DEVICE`) | Guardia Semántica & Anti-Prompt Injection vía Ollama (`laya:latest`) |
| **Middle Office (Hard Gate)** | `backend/app/middle_office/risk_engine.py` | Local (Python Puro) | Motor Determinista inmutable (`config/risk_limits.yaml`): Kill-Switch, Long-Only, ATR Stop-Loss, Trailing Stop & Take-Profit |
| **Back Office** | `backend/app/back_office/` (`mexc_client.py`, `paper_engine.py`, `database.py`) | Local | Ingesta MEXC/yfinance (solo velas cerradas), Paper Multi-Ticker y SQLite WAL idempotente |
| **Observabilidad & Stream** | `backend/app/core/events.py` & `telemetry.py` | Local (FastAPI) | Bus **SSE** (`/api/stream/events`) + telemetría GPU `nvidia-smi` y `/api/ps` de Ollama |

---

## 2. Capacidades Principales

* **Operación Multi-Temporalidad (`1h` y `1d`):**
  * **Trading por Horas (`1h`):** Evalúa velas horarias con modelo dedicado `quant_xgb_1h.joblib`, horizonte `+1H` y ventana gráfica continua de las últimas **200 horas**.
  * **Trading Diario (`1d`):** Ensamble ponderado (`0.65 * XGBoost 1d + 0.35 * LSTM T4`), horizonte `+1D` y ventana de los últimos **200 días**.
* **Gráfica Cuantitativa Unificada + Sub-Gráfica RSI (14):**
  * Ventana única de **200 velas** (`200 Días` o `200 Horas`) con EMA 10, EMA 20, Volumen, marcadores históricos `BUY`/`SELL` y vela proyectada `t+1`.
  * Sub-gráfica sincronizada **Relative Strength Index — RSI (14) Wilder** directamente debajo de la gráfica principal con bandas **70 (Sobrecompra)**, **50 (Neutral)**, **30 (Sobreventa)** y punto proyectado `RSI t+1`.
* **Mapa de Constelaciones Dinámico en Vivo (SSE):**
  * 10 estaciones interactivas arrastrables con órbitas armónicas, flujos de plasma multi-cometa proporcionales a las latencias reales en ms, ondas de choque al recibir eventos SSE (`/api/stream/events`), nebulosa reactiva al régimen (`bull_trend`, `bear_trend`, `high_vol_chop`, `sideways`) y selector de velocidad (`1x`, `2x`, `3x Turbo`).
* **Optimizado para 6 GB de VRAM:**
  * Ejecución simultánea 100% en GPU local mediante `laya:latest` (~442 MB VRAM) y `qwen2.5-3b-quant:latest` (~2.2 GB VRAM, contexto `4096`), consumiendo ~2.7 GB de 6 GB totales.
* **Universo Oficial de 30 Tickers (+ alias `BTCUSDT`):**
  `AAPL`, `MSFT`, `NVDA`, `TSLA`, `HWM`, `JPM`, `GOOGL`, `AMZN`, `META`, `LMT`, `WM`, `LULU`, `DIS`, `V`, `MA`, `BTC-USD`, `SPY`, `QQQ`, `NFLX`, `AMD`, `CRM`, `ADBE`, `PYPL`, `COST`, `HD`, `UNH`, `JNJ`, `PG`, `KO`, `WMT`.

---

## 3. Estructura del Repositorio

```text
Trading-agente/
├── .env.example                      # Plantilla de variables de entorno sin secretos
├── .gitignore                        # Exclusiones (secretos, SQLite WAL, caches, venv)
├── README.md                         # Esta guía principal
├── config/
│   ├── Modelfile.qwen_local          # Definición Ollama para qwen2.5-3b-quant:latest (ctx 4096)
│   ├── llm_config.yaml               # Topología Qwen API / Local 6GB y Dual Laya Local
│   └── risk_limits.yaml              # Perfiles de capital/comisión, límites y 30 tickers
├── data/
│   ├── replay_candles.json           # Cache histórico cerrado 1d (250 velas × 30 tickers)
│   └── replay_candles_1h.json        # Cache histórico cerrado 1h (250 velas × 30 tickers)
├── docs/
│   ├── ARQUITECTURA_Y_AGENTES.md     # Arquitectura detallada, modelos ML/DRL, contratos y SQLite
│   └── API_Y_OPERACION.md            # Endpoints REST/SSE, trading 1h/1d y matriz de 9 errores
├── exports/                          # Resultados CSV/JSON de experimentos y benchmark 3 carriles
├── notebooks/
│   ├── 01_colab_t4_training.ipynb    # Notebook de entrenamiento base T4
│   ├── generate_phase1_artifacts.py  # Entrenamiento LSTM/MLP/Ridge y cache replay
│   ├── train_quant_v2.py             # Entrenamiento Walk-Forward XGBoost 1d y 1h
│   ├── train_laya1_drl.py            # Entrenamiento DRL Dueling Double-DQN + A2C
│   └── artifacts/                    # Pesos congelados + manifiestos SHA-256
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                   # App FastAPI + Lifespan + Bus SSE + Static Dashboard
│   │   ├── static/index.html         # Terminal Web UI (LightweightCharts + RSI + Constelación)
│   │   ├── api/routes.py             # Endpoints REST y /api/stream/events
│   │   ├── core/                     # config.py, security.py, scheduler.py, events.py, telemetry.py
│   │   ├── front_office/             # features.py, predictor.py, laya_alpha.py, news_sentiment.py
│   │   ├── orchestrator/             # qwen_agent.py (Debate Bull/Bear + Feedback bidireccional)
│   │   ├── middle_office/            # laya_guard.py, risk_engine.py
│   │   ├── back_office/              # database.py, mexc_client.py, paper_engine.py
│   │   └── benchmark/                # comparator.py (Evaluación 3 Carriles)
│   └── tests/                        # Suite pytest (22+ pruebas unitarias y de integración)
└── frontend/                         # Proyecto React/Vite complementario
```

---

## 4. Instalación y Puesta en Marcha

### 4.1 Variables de Entorno
```powershell
copy .env.example .env
```
* Configura `GATEWAY_API_KEY` (por defecto en desarrollo: `dev_local_gateway_key_change_me`) y `ADMIN_CONTROL_TOKEN`.
* Si usas **Qwen API Externa (14B)**, define `QWEN_API_BASE_URL` y `QWEN_API_KEY`.
* Mantén `OLLAMA_BASE_URL=http://127.0.0.1:11434` y `ENABLE_REAL_TRADING=false`.

### 4.2 Dependencias Python y Modelos Ollama Locales
```powershell
pip install -r backend/requirements.txt
```
Para habilitar los modelos locales en Ollama (GPU 6 GB VRAM):
```powershell
ollama pull qwen2.5:3b-instruct
ollama create qwen2.5-3b-quant -f config/Modelfile.qwen_local
ollama list
```
*(Verifica que aparezcan `qwen2.5-3b-quant:latest` y `laya:latest` en Ollama).*

### 4.3 Iniciar el Servidor y Abrir el Dashboard
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Luego abre en tu navegador:
* **Dashboard Principal:** `http://127.0.0.1:8000/`
* **Documentación Swagger / OpenAPI:** `http://127.0.0.1:8000/docs`

### 4.4 Ejecutar Pruebas Automatizadas
```powershell
python -m pytest -v backend/tests
```

---

## 5. Documentación Técnica Completa

* **[docs/ARQUITECTURA_Y_AGENTES.md](docs/ARQUITECTURA_Y_AGENTES.md):** Diseño de las 3 oficinas + orquestador, matemáticas del predictor LSTM + XGBoost (`1d`/`1h`), DRL Dueling A2C, cálculo de RSI (14) y ATR (14), retroalimentación dinámica, memoria episódica FinMem y esquema SQLite WAL completo.
* **[docs/API_Y_OPERACION.md](docs/API_Y_OPERACION.md):** Manual de API REST y Server-Sent Events (SSE), autenticación por `X-API-Key` / cookie `HttpOnly`, guía de trading por horas (`1h`) vs días (`1d`), comparativa de 3 carriles y matriz de los 9 escenarios de error.
