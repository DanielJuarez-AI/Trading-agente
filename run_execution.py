import sys
from backend.app.core.config import get_settings
from backend.app.main import create_app
from fastapi.testclient import TestClient

cfg = get_settings()
cfg.qwen_api_key = ""  # Sin llamada externa a Qwen
app = create_app(settings_override=cfg)
client = TestClient(app)

print("=== INICIANDO EJECUCIÓN DEL SISTEMA ===")
status_resp = client.get("/api/status").json()
print("1. Estado del Sistema:", status_resp["system_status"])
print("   Activo predeterminado:", status_resp["active_ticker"])
print("   Capital inicial:", status_resp["portfolio"]["cash"])
print("   Total Tickers disponibles:", len(status_resp["primary_tickers"]))

print("\n2. Ejecutando 25 ciclos de Replay para BTC-USD (T4 + Dual Laya + Riesgo)...")
resp = client.post(
    "/api/control/replay",
    headers={"X-API-Key": cfg.gateway_api_key},
    json={"symbol": "BTC-USD", "n_cycles": 25},
)
print("   Status HTTP:", resp.status_code)
print("   Ciclos ejecutados:", resp.json().get("cycles_executed"))

print("\n3. Consultando bitácora de decisiones guardadas en SQLite WAL...")
decs = client.get("/api/decisions?limit=5").json()["items"]
print(f"   Decisiones registradas (muestra de {len(decs)}):")
for d in decs[:3]:
    print(f"   * Vela: {d['candle_close_utc']} | {d['symbol']} | r_hat: {d['model_pred']:.5f} | Acción: {d['final_action']} | Latencia: {d['latency_ms']:.1f}ms")

print("\n4. Ejecutando Comparativa de 3 Carriles (Cap. 16)...")
bm = client.get("/api/benchmark").json()
print(f"   Evaluación de {bm['cycles_evaluated']} ciclos sobre {bm['symbol']}:")
for lane in bm["lanes"]:
    print(f"   - {lane['strategy']}: Eq: ${lane['final_equity']} | Ret: {lane['net_return_pct']}% | MDD: {lane['max_drawdown_pct']}% | Trades: {lane['total_orders']}")
print("   Interpretación:", bm["interpretation"])

print("\n5. Estado final del Portafolio:")
p = client.get("/api/portfolio").json()["state"]
print(f"   Efectivo: ${p['cash']} | Posición: {p['position_qty']} | Patrimonio: ${p['total_equity']}")
print("=== EJECUCIÓN COMPLETADA EXITOSAMENTE ===")
