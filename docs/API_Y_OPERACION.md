# Referencia de API, Operación por Horas/Días y Seguridad

Guía operativa completa para el consumo de endpoints REST/SSE, configuración de trading en **1 Hora (`1h`)** vs **1 Día (`1d`)**, benchmark comparativo de 3 carriles y matriz de resiliencia ante los 9 escenarios de error obligatorios.

---

## 1. Autenticación y Gateway de Seguridad (`PrivateAuthGateway`)

* Todos los endpoints de mutación (`/api/control/*`) exigen autenticación mediante:
  * Header **`X-API-Key: <GATEWAY_API_KEY>`**, **o**
  * Cookie segura **`HttpOnly`** emitida por **`POST /api/auth/login`** usando `GATEWAY_API_KEY` o `ADMIN_CONTROL_TOKEN`.
* **Cero secretos en navegador:** Ninguna clave privada del backend se incrusta en variables `VITE_*` ni en logs (`error_logs` enmascara credenciales automáticamente).

---

## 2. Catálogo de Endpoints REST & Streaming

### 2.1 Autenticación
* `POST /api/auth/login` — Valida `{"credential": "..."}` e instala cookie `HttpOnly` de sesión (24h).
* `POST /api/auth/logout` — Revoca la cookie de sesión.
* `GET /api/auth/session` — Verifica si la petición actual está autenticada.

### 2.2 Observabilidad, Gráficas y Telemetría
* `GET /health` — Estado de vida del servidor, `system_status`, modo activo de Qwen y Laya.
* `GET /api/status` — Estado consolidado del sistema (`RUNNING | PAUSED | STOPPED`), `candle_interval` (`1h` o `1d`), `qwen_mode`, métricas de portafolio, feedback activo hacia Laya #1/#2, latencias por etapa en ms y conteo de velas procesadas.
* `GET /api/candles?symbol=BTC-USD&limit=200&mode=replay&interval=1d`
  * Devuelve por defecto las **últimas 200 velas** cerradas (`1d` = 200 días, `1h` = 200 horas) de `data/replay_candles.json` o `data/replay_candles_1h.json` (o mercado en vivo en modo `live`).
  * Cada vela incluye `time`, `open`, `high`, `low`, `close`, `volume`, **`ema10`**, **`ema20`** y **`rsi_14`** (Wilder).
  * Incluye `markers` cronológicos (`BUY`/`SELL`), `last_rsi_14` y `agent_forecast` con `pred_return_t1`, `projected_close_t1`, **`projected_rsi_t1`**, `band_upper_t1`, `band_lower_t1`, `forecast_candle`, `signal_probs`, `regime_probs` y `coherence_prob`.
* `GET /api/decisions?limit=50` — Bitácora de decisiones almacenadas en SQLite WAL (máx. 200).
* `GET /api/portfolio` — Posición actual, métricas de rendimiento, snapshots e historial de `paper_orders`.
* `GET /api/errors?limit=100` — Bitácora de errores auditados (`error_logs`).
* `GET /api/benchmark?symbol=BTC-USD&interval=1d` — Devuelve o ejecuta la comparativa de 3 carriles.
* `GET /api/telemetry` — Snapshot hardware (`nvidia-smi`: uso VRAM MB/%, temperatura, GPU %), modelos cargados en Ollama (`/api/ps`) y estado Real vs Fallback de Qwen y Laya #2.
* `GET /api/stream/events` — Flujo **Server-Sent Events (SSE)** en tiempo real con los eventos del ciclo:
  * `connected`, `heartbeat`, `daemon_status`
  * `stage_ingest`, `stage_quant_model`, `stage_news_memory`, `stage_laya1_drl`, `stage_qwen_orch`, `stage_feedback`, `stage_laya2_guard`, `stage_risk_paper`, `cycle_complete`.

### 2.3 Control Protegido (`POST /api/control/{action}`)
Requiere `X-API-Key` o cookie de sesión válida:
* `POST /api/control/start` — Pasa `system_status` a `RUNNING`.
* `POST /api/control/pause` — Pasa `system_status` a `PAUSED` (no abre ni cierra posiciones).
* `POST /api/control/stop` — Activa **Kill-Switch `STOPPED`** (bloquea nuevas órdenes preservando posiciones abiertas).
* `POST /api/control/switch_ticker` — Body `{"symbol": "NVDA"}`. Cambia el activo activo.
* `POST /api/control/switch_mode` — Body `{"mode": "live" | "replay"}`. Alterna entre MEXC/yfinance y Historical Replay.
* `POST /api/control/switch_interval` — Body `{"interval": "1h" | "1d"}`. Cambia la temporalidad de trading y gráficas.
* `POST /api/control/switch_qwen_mode` — Body `{"qwen_mode": "EXTERNAL_API" | "LOCAL_DEVICE"}`. Alterna Qwen entre API Externa 14B y Ollama Local 6GB VRAM (`qwen2.5-3b-quant:latest`).
* `POST /api/control/step` — Body `{"symbol": "...", "interval": "1h|1d", "mode": "..."}`. Avanza **1 ciclo completo** sobre la siguiente vela cerrada no procesada dentro de la ventana de 200 velas.
* `POST /api/control/replay` — Body `{"symbol": "...", "interval": "1h|1d", "n_cycles": 25}`. Ejecuta en lote `n_cycles` cronológicos sin look-ahead.
* `POST /api/control/benchmark` — Body `{"symbol": "...", "interval": "1h|1d", "n_cycles": 25}`. Ejecuta la comparativa de 3 carriles sobre la temporalidad elegida.

---

## 3. Guía de Trading Horario (`1h`) vs Diario (`1d`)

1. En la barra de control superior o directamente en la barra de la gráfica, selecciona **`⏱️ Trading 1H (Horas)` / `Ventana por Horas • 200H (1H)`** para activar el trading horario, o **`📅 Trading 1D (Días)` / `Últimos 200 Días (1D)`** para el modo diario.
2. El dashboard sincroniza automáticamente:
   * Gráfica OHLCV + EMA10/20 + vela proyectada `t+1` de las últimas **200 horas** (timestamps UTC por hora) o **200 días**.
   * Sub-gráfica sincronizada **RSI (14)** debajo de la gráfica principal con líneas 70/50/30 y su valor proyectado `t+1`.
   * Modelo cuantitativo dedicado (`quant_xgb_1h` en `1h` vs ensamble `XGBoost 1d + LSTM T4` en `1d`).
   * Ejecución de ciclos individuales (`⚡ Ejecutar 1 Ciclo (1H)`) o en lote (`🔄 Replay 25 Ciclos (1H)`), registrando cada operación en la bitácora con su etiqueta `1H` o `1D`.

---

## 4. Benchmark Comparativo de 3 Carriles (`backend/app/benchmark/comparator.py`)

Sobre exactamente la misma ventana histórica, costos de perfil y slippage (`5 bps`), compara:
1. **`1_Modelo_Sin_LLM`:** Señal directa de Quant v2 / LSTM + filtro técnico `EMA10/EMA20` + reglas de capital.
2. **`2_Agente_DualLaya_Qwen`:** Pipeline completo (Quant v2 + Laya #1 DRL + Qwen Debate/Feedback + Laya #2 Guardia + Risk Engine con SL/TP/TSL).
3. **`3_Buy_And_Hold`:** Inversión pasiva desde la primera vela de la ventana.

Reporta: Patrimonio Final, Retorno Neto %, Max Drawdown %, Win Rate, Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Órdenes, Acciones Rechazadas, Latencia Promedio y exporta `exports/simulacion.csv` y `exports/benchmark_summary.json`.

---

## 5. Matriz de los 9 Escenarios de Error Validados (`backend/tests/test_nine_errors.py`)

| # | Escenario de Fallo | Comportamiento Verificado | Código / Regla Registrada |
| :---: | :--- | :--- | :--- |
| **1** | API MEXC `429` / `5xx` / timeout | 3 reintentos con backoff exponencial (`0.5s`, `1.0s`); si agota reintentos, pausa el sistema (`PAUSED`) | `MEXC_API_UNAVAILABLE` |
| **2** | Datos atrasados (*stale data*) o vela abierta | Descarta velas abiertas (`close_time > now_utc`); rechaza histórico viejo sin operar | `STALE_DATA_REJECTED` |
| **3** | Ollama local o API Qwen caída | Captura fallo en modo estricto, registra error y pasa el sistema a `PAUSED` | `OLLAMA_LOCAL_UNAVAILABLE` / `QWEN_EXTERNAL_API_UNAVAILABLE` |
| **4** | JSON inválido o acción inventada por LLM | Validador Pydantic (`QwenDecisionOutput`) rechaza la salida con HTTP 422; 0 órdenes emitidas | `QWEN_INVALID_JSON_SCHEMA` |
| **5** | Clave `X-API-Key` ausente o errónea | Gateway bloquea con HTTP `401` / `403` sin exponer secretos en bitácora ni logs | `AUTH_401_MISSING_CREDENTIALS` / `AUTH_403_INVALID_KEY` |
| **6** | Saldo insuficiente en compra | Rechazo determinista antes de operar (`InsufficientFundsError` / `HOLD`) | `INSUFFICIENT_FUNDS` |
| **7** | Vela duplicada (`replay` o `live`) | Restricción `UNIQUE(symbol, interval, close_time, source_mode)` en SQLite WAL bloquea duplicados | `DuplicateCandleError` / `DUPLICATE_CANDLE` |
| **8** | Reinicio abrupto del servidor | Recuperación exacta de efectivo, posiciones y estado operativo desde SQLite WAL sin reprocesar | Idempotencia WAL verificada |
| **9** | Activación del Kill-Switch `STOP` | Bloquea cualquier nueva orden `BUY`/`SELL` (`HOLD`) preservando intacta la posición abierta | `KILL_SWITCH_STOPPED` |
