# Comparativa Técnica con Ecosistema Quant & Corrección del Bloqueo en `HOLD`

Este documento analiza nuestra arquitectura (**Quant Multi-Agent Trader v0.1.10+**) frente a tres referentes de código abierto seleccionados y documenta las mejoras institucionales integradas junto con la resolución del bloqueo persistente en `HOLD`.

---

## 1. Comparativa frente a Repositorios Referentes

| Dimensión | **FinceptTerminal** | **Awesome-Quant Ecosystem** | **ML4T (Stefan Jansen 3rd Ed.)** | **Quant Multi-Agent Trader (Nuestro Sistema)** |
| :--- | :--- | :--- | :--- | :--- |
| **Stack / Arquitectura** | C++20 + Qt6 + Python embebido (41 módulos, 6 desks) | Librerías modulares (`purgedcv`, `midas-core`, `autonomous-audit`, `Deadlatch`) | Pipeline investigativo modular (Cap. 2–26, backtesting + agentes) | FastAPI + PyTorch + XGBoost + Dual Laya Local + Qwen (`qwen3:4b` / 14B) + SQLite WAL + SSE |
| **Modelos Predictivos** | AI Quant Lab + factor discovery + RL | LightGBM/XGBoost + Transformers + RL (`Orallexa`, `DeepAlpha`) | Ridge/ElasticNet, GBDT, Deep Time Series, RL y Debate Bull/Bear | `NextCandleGRU` (26 features adimensionales) + XGBoost `1h`/`1d` + `Laya #1` DRL (33D) |
| **Dimensionamiento de Posición** | Optimización de portafolio / QuantLib | `Kelly-Criterion` + Risk Parity | Half-Kelly fraccional, Conformal Sizing, HRP | **Integrado:** Half-Kelly fraccional + Volatility Targeting acotado por riesgo duro (`fractional_kelly_allocation`) |
| **Evaluación Estadística & Cola** | VaR, Sharpe, Derivados QuantLib | **PSR / DSR**, PBO (`purgedcv`, `stochastly-calculators`) | **PSR, DSR**, **VaR/CVaR**, **Rank IC** (Spearman), Deflated Sharpe | **Integrado:** `PSR`, `DSR`, `VaR 95%`, `CVaR 95%`, `Rank IC`, Sortino y Profit Factor en vivo |
| **Despliegue / Auditoría / Seguridad** | Broker routing + Enterprise RBAC | **Cadena SHA-256** (`autonomous-audit`), rieles deterministas pre-trade (`Deadlatch`, `midas-core`) | Drift detection, circuit breakers y debate auditable | **Integrado:** Cadena criptográfica SHA-256 + `git_sha` por decisión + Hard Gate Python inmutable |
| **Descubrimiento Multi-Activo** | **Equity Screener** multi-activo | Ranking cross-sectional por Rank IC (`Inalpha`) | Ranking transversal de universo por señales | **Integrado:** **Alpha Scanner Multi-Ticker** (`/api/scanner`) + botón `Auto-Scan & Operar Top Alpha` |

---

## 2. Diagnóstico del Bloqueo Persistente en `HOLD` (5 Causas Raíz Corregidas)

1. **Estado Patrimonial Corrupto en `data/trading_agent.db` (`MAX_DAILY_LOSS` permanente):**
   - Una compra antigua de `91.8318` `MSFT` (`$47,499.98`) quedó registrada sin registro en `positions`. Al conmutar a `BTC-USD`, esas acciones se valuaron a precio de BTC inflando `peak_equity = $5,531,916.77`. Al cambiar a `NVDA` (`position_qty = 0`), el patrimonio reportó `$2,500.02` con `daily_pnl_pct = -95.0%` y `drawdown_pct = 99.95%`, disparando el Hard Gate `MAX_DAILY_LOSS` (`<= -3%`) y bloqueando toda compra futura.
   - **Solución:** Añadido `POST /api/control/reset_portfolio` (`TradingDatabase.reset_paper_portfolio`) para limpiar posiciones, snapshots y reiniciar la cartera a `$50,000.00`.
2. **Botón `Iniciar` sin Arrancar el Daemon Autónomo:**
   - `POST /api/control/start` cambiaba `system_status = "RUNNING"` pero no invocaba `scheduler.start_daemon()`.
   - **Solución:** Conectado `start_daemon(interval_seconds=25)` en `start` y `stop_daemon()` en `pause`, `stop` y `reset_portfolio`.
3. **Bug MEXC Live `1h` → `60m`:**
   - La API pública `/api/v3/klines` de MEXC rechaza `interval=1h` con HTTP `400 Bad Request` y exige `interval=60m`. Tras 3 intentos fallidos, `MEXCMarketClient` pausaba el sistema automáticamente (`PAUSED`).
   - **Solución:** Mapeo explícito `mexc_interval = "60m" if interval == "1h" else interval` en [`mexc_client.py`](../backend/app/back_office/mexc_client.py).
4. **Reentrenamiento de `Laya #1` sobre Estado 33D de `Candle Predictor v2` (`laya1_drl_v2.pt`):**
   - El modelo `v1` (`state_dim=10`, TE Sharpe `-0.1063`) presentaba alta inercia hacia `HOLD`. Se reentrenó [`train_laya1_drl.py`](../notebooks/train_laya1_drl.py) sobre las **33 dimensiones** (`10` base + `7` señales calibradas + `16` latentes $h_t$) logrando balance real (`action_rate = 0.4997` en TE y mejora de Sharpe OOS a `-0.0676`).
5. **Visibilidad Multi-Ticker mediante `Alpha Scanner`:**
   - En modo *Long-Only*, observar un único ticker con pronóstico negativo o neutro (ej. `NVDA`) produce correctamente `HOLD` mientras otros 20+ activos del universo presentan señal `BUY`. El nuevo endpoint `GET /api/scanner` y el botón `Auto-Scan & Operar Top Alpha` automatizan la detección y ejecución sobre la mejor oportunidad del universo oficial de 30 tickers.
