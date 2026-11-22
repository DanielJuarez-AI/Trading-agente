from backend.app.core.config import get_settings
from backend.app.main import create_app
from fastapi.testclient import TestClient

cfg = get_settings()
cfg.qwen_api_key = ""
app = create_app(settings_override=cfg)
client = TestClient(app)

tickers_to_test = ["NVDA", "AAPL", "MSFT", "SPY"]
print("Ejecutando ciclos de prueba para múltiples tickers...")
for ticker in tickers_to_test:
    r = client.post(
        "/api/control/replay",
        headers={"X-API-Key": cfg.gateway_api_key},
        json={"symbol": ticker, "n_cycles": 5},
    )
    executed = r.json().get("cycles_executed")
    print(f"Ticker {ticker}: HTTP {r.status_code} - {executed} ciclos ejecutados")

total_candles = client.get("/api/status").json()["processed_candles_count"]
print(f"\nTotal de velas registradas e idempotentes en SQLite: {total_candles}")
