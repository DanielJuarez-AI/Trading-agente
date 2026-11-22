# Quant Multi-Agent Trader (Orquestación Híbrida System-1 / System-2)

Sistema multi-agente institucional (*Front Office*, *Middle Office* y *Back Office*) para trading cuantitativo en modo **Paper Trading** (0 órdenes reales), diseñado bajo cumplimiento estricto de los contratos JSON, persistencia SQLite WAL, controles deterministas de riesgo y matriz de los 9 escenarios de error.

---

## 1. Arquitectura Híbrida Explícita (Local vs. API Externa)

Tal como se definió explícitamente para este despliegue, los agentes operan en una topología híbrida marcada en todo el código (`backend/app/core/config.py`, `config/llm_config.yaml`, `qwen_agent.py`, `laya_alpha.py`, `laya_guard.py` y UI):

| Agente / Componente | Ubicación de Ejecución | Marcador en Código | Protocolo / Endpoint |
| :--- | :--- | :--- | :--- |
| **Orquestador Qwen 14B** (`System-2`) | **API EXTERNA REMOTA (NO Local)** | `# [API EXTERNA - NO EN DISPOSITIVO LOCAL]` (`QWEN_EXECUTION_MODE="EXTERNAL_API"`, `QWEN_IS_LOCAL=False`) | HTTPS REST compatible OpenAI / DashScope (`{QWEN_API_BASE_URL}/chat/completions`) |
| **Agente Laya #1** (`Front Office` - Alpha/Régimen) | **DISPOSITIVO LOCAL** | `# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]` (`LAYA_EXECUTION_MODE="LOCAL_DEVICE"`, `LAYA_IS_LOCAL=True`) | Ollama Local (`http://127.0.0.1:11434/v1/systemone`) |
| **Agente Laya #2** (`Middle Office` - Anti-Injection/Coherencia) | **DISPOSITIVO LOCAL** | `# [DISPOSITIVO LOCAL - OLLAMA /v1/systemone]` (`LAYA_EXECUTION_MODE="LOCAL_DEVICE"`, `LAYA_IS_LOCAL=True`) | Ollama Local (`http://127.0.0.1:11434/v1/systemone`) |
| **Predictor Cuantitativo LSTM T4** (`Front Office`) | **DISPOSITIVO LOCAL** | Artefactos congelados verificados por SHA-256 (`lstm_v1.pt`, `scaler_v1.joblib`) | PyTorch CPU / CUDA |
| **Motor Determinista de Riesgo** (`Middle Office`) | **DISPOSITIVO LOCAL (Python Puro)** | Inmutable por LLMs (`config/risk_limits.yaml`) | Hard Gate Determinista |
| **Back Office (Gateway, MEXC Público, Paper, SQLite WAL)** | **DISPOSITIVO LOCAL** | `PrivateAuthGateway`, `TradingDatabase` | FastAPI + SQLite WAL |

---

## 2. Universo Oficial de 30 Tickers

El sistema está configurado y validado (tanto en entrenamiento `notebooks/`, lista blanca en `config/risk_limits.yaml`, ingesta de mercado y selector del Frontend) para operar sobre los **30 activos oficiales** (más el alias `BTCUSDT` para `/api/v3/klines` de MEXC):

```python
PRIMARY_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'HWM', 'JPM', 'GOOGL', 'AMZN', 'META', 'LMT',
    'WM', 'LULU', 'DIS', 'V', 'MA', 'BTC-USD', 'SPY', 'QQQ', 'NFLX', 'AMD',
    'CRM', 'ADBE', 'PYPL', 'COST', 'HD', 'UNH', 'JNJ', 'PG', 'KO', 'WMT'
]
```

---

## 3. Estructura del Repositorio

```text
Trading-agente/
├── .env.example                      # Plantilla limpia sin secretos reales
├── .gitignore
├── config/
│   ├── risk_limits.yaml              # Perfiles (agent_paper / practice_backtest), límites y 30 tickers
│   └── llm_config.yaml               # Topología híbrida (Qwen API Externa + Dual Laya Local)
├── notebooks/
│   ├── 01_colab_t4_training.ipynb    # Entrenamiento T4, splits TR/VA/TE, 3x3 semillas, 4 fallos
│   ├── generate_phase1_artifacts.py  # Script reproducible de entrenamiento y exportación
│   └── artifacts/
│       ├── model_manifest.json       # SHA-256, contrato de features y métricas TE
│       ├── scaler_v1.joblib          # Ajustado exclusivamente en TR
│       └── lstm_v1.pt                # Pesos congelados del modelo ganador en VA
├── exports/
│   ├── experimentos.csv
│   ├── predicciones_TE.csv
│   ├── prediccion_futura.csv         # Pronósticos t+1 para los 30 tickers (real vacío)
│   ├── simulacion.csv                # Comparativa oficial 3 carriles
│   └── benchmark_summary.json
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── core/ (config.py, security.py, scheduler.py)
│   │   ├── back_office/ (database.py, mexc_client.py, paper_engine.py)
│   │   ├── front_office/ (features.py, predictor.py, laya_alpha.py)
│   │   ├── orchestrator/ (qwen_agent.py)
│   │   ├── middle_office/ (laya_guard.py, risk_engine.py)
│   │   ├── benchmark/ (comparator.py)
│   │   └── api/ (routes.py)
│   └── tests/
│       ├── test_risk_controls.py
│       └── test_nine_errors.py
└── frontend/                         # React 18 + TypeScript + Vite + TailwindCSS
```

---

## 4. Instalación y Ejecución desde Sesión Limpia

### 4.1 Variables de Entorno
Copia `.env.example` a `.env` y completa tus claves privadas:
```bash
cp .env.example .env
```
* Configura `GATEWAY_API_KEY` y `ADMIN_CONTROL_TOKEN` con secretos propios.
* Configura `QWEN_API_BASE_URL` y `QWEN_API_KEY` para el endpoint remoto externo de Qwen 14B.
* Mantén `OLLAMA_BASE_URL=http://127.0.0.1:11434` y el modelo `laya:latest` en tu Ollama local.
* Mantén siempre `ENABLE_REAL_TRADING=false`.

### 4.2 Dependencias Python y Artefactos de Fase 1
```bash
pip install -r backend/requirements.txt
python notebooks/generate_phase1_artifacts.py
```

### 4.3 Ejecutar Backend FastAPI
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Documentación interactiva OpenAPI/Swagger disponible en `http://127.0.0.1:8000/docs`.

### 4.4 Ejecutar Frontend React + TypeScript
```bash
cd frontend
npm install
npm run dev
```
Accede a `http://localhost:5173`, ingresa tu `ADMIN_CONTROL_TOKEN` o `GATEWAY_API_KEY` en el panel superior para iniciar sesión `HttpOnly` (cero secretos almacenados en `VITE_*`).

### 4.5 Pruebas Automatizadas (Controles + 9 Escenarios de Error + 25 Ciclos + Benchmark)
```bash
python -m pytest -v backend/tests
```
