"""
Core Telemetry - Telemetría GPU/VRAM, Ollama Live Models y Estado Real vs Fallback.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from backend.app.back_office.database import TradingDatabase, utc_now_iso
from backend.app.core.config import Settings


class SystemTelemetryService:
    """Recolecta telemetría hardware (nvidia-smi), runtime Ollama (/api/ps) y estado de agentes."""

    def __init__(self, settings: Settings, db: TradingDatabase, ttl_seconds: float = 3.0) -> None:
        self.settings = settings
        self.db = db
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        self._cache_ts: float = 0.0
        self._cache_payload: Optional[Dict[str, Any]] = None

        self._qwen_status: Dict[str, Any] = self._load_state_dict("telemetry_qwen_status") or {
            "agent": "Qwen Orchestrator",
            "mode": settings.qwen_execution_mode,
            "model": settings.qwen_model_name,
            "transport": "pending",
            "is_fallback": False,
            "status_badge": "PENDING",
            "latency_ms": 0.0,
            "updated_at": None,
        }
        self._laya2_status: Dict[str, Any] = self._load_state_dict("telemetry_laya2_status") or {
            "agent": "Laya #2 Semantic Guard",
            "mode": settings.laya_execution_mode,
            "model": settings.laya_model_name,
            "transport": "pending",
            "is_fallback": False,
            "status_badge": "PENDING",
            "latency_ms": 0.0,
            "updated_at": None,
        }

    def _load_state_dict(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.db.get_state(key)
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass
        return None

    def record_agent_status(
        self,
        *,
        qwen_mode: str,
        qwen_model: str,
        qwen_transport: str,
        qwen_is_fallback: bool,
        qwen_latency_ms: float,
        laya2_model: str,
        laya2_transport: str,
        laya2_is_fallback: bool,
        laya2_latency_ms: float,
    ) -> None:
        now = utc_now_iso()
        if qwen_is_fallback:
            q_badge = "FALLBACK_SIMULADO"
        elif qwen_mode == "LOCAL_DEVICE":
            q_badge = "LIVE_OLLAMA_LOCAL"
        else:
            q_badge = "LIVE_EXTERNAL_API"

        l2_badge = "FALLBACK_SIMULADO" if laya2_is_fallback else "LIVE_OLLAMA_LOCAL"

        with self._lock:
            self._qwen_status = {
                "agent": "Qwen Orchestrator",
                "mode": qwen_mode,
                "model": qwen_model,
                "transport": qwen_transport,
                "is_fallback": bool(qwen_is_fallback),
                "status_badge": q_badge,
                "latency_ms": round(float(qwen_latency_ms), 2),
                "updated_at": now,
            }
            self._laya2_status = {
                "agent": "Laya #2 Semantic Guard",
                "mode": "LOCAL_DEVICE",
                "model": laya2_model,
                "transport": laya2_transport,
                "is_fallback": bool(laya2_is_fallback),
                "status_badge": l2_badge,
                "latency_ms": round(float(laya2_latency_ms), 2),
                "updated_at": now,
            }
            self._cache_ts = 0.0
        try:
            self.db.set_state("telemetry_qwen_status", json.dumps(self._qwen_status, ensure_ascii=False))
            self.db.set_state("telemetry_laya2_status", json.dumps(self._laya2_status, ensure_ascii=False))
        except Exception:
            pass

    @staticmethod
    def _query_nvidia_smi() -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return {
                    "available": False,
                    "gpu_name": "Unavailable",
                    "vram_used_mb": 0.0,
                    "vram_total_mb": 0.0,
                    "vram_used_pct": 0.0,
                    "gpu_util_pct": 0.0,
                    "temperature_c": None,
                }
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) < 5:
                raise ValueError(f"Salida nvidia-smi inesperada: {first_line}")
            used_mb = float(parts[1])
            total_mb = max(float(parts[2]), 1.0)
            return {
                "available": True,
                "gpu_name": parts[0],
                "vram_used_mb": round(used_mb, 1),
                "vram_total_mb": round(total_mb, 1),
                "vram_used_pct": round((used_mb / total_mb) * 100.0, 1),
                "gpu_util_pct": round(float(parts[3]), 1),
                "temperature_c": round(float(parts[4]), 1),
            }
        except Exception:
            return {
                "available": False,
                "gpu_name": "Unavailable",
                "vram_used_mb": 0.0,
                "vram_total_mb": 0.0,
                "vram_used_pct": 0.0,
                "gpu_util_pct": 0.0,
                "temperature_c": None,
            }

    def _query_ollama_ps(self) -> Dict[str, Any]:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/ps"
        try:
            with httpx.Client(timeout=0.5) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            raw_models = data.get("models", []) or []
            models: List[Dict[str, Any]] = []
            total_vram_mb = 0.0
            for m in raw_models:
                size_vram = float(m.get("size_vram", 0) or 0)
                vram_mb = round(size_vram / (1024.0 * 1024.0), 1)
                total_vram_mb += vram_mb
                models.append(
                    {
                        "name": str(m.get("name", m.get("model", "unknown"))),
                        "size_mb": round(float(m.get("size", 0) or 0) / (1024.0 * 1024.0), 1),
                        "vram_mb": vram_mb,
                        "expires_at": m.get("expires_at"),
                        "context_length": m.get("context_length"),
                    }
                )
            return {
                "reachable": True,
                "endpoint": self.settings.ollama_base_url,
                "loaded_models": models,
                "total_vram_mb": round(total_vram_mb, 1),
            }
        except Exception as exc:
            return {
                "reachable": False,
                "endpoint": self.settings.ollama_base_url,
                "loaded_models": [],
                "total_vram_mb": 0.0,
                "error": str(exc)[:160],
            }

    def get_snapshot(self, daemon_state: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, Any]:
        now_m = time.monotonic()
        with self._lock:
            if not force and self._cache_payload is not None and (now_m - self._cache_ts) < self.ttl_seconds:
                cached = dict(self._cache_payload)
                if daemon_state is not None:
                    cached["daemon"] = daemon_state
                return cached

        gpu = self._query_nvidia_smi()
        ollama = self._query_ollama_ps()

        with self._lock:
            payload = {
                "timestamp_utc": utc_now_iso(),
                "gpu": gpu,
                "ollama": ollama,
                "qwen_live_status": dict(self._qwen_status),
                "laya2_live_status": dict(self._laya2_status),
                "daemon": daemon_state or {"enabled": False, "running": False, "poll_seconds": 20},
            }
            self._cache_ts = time.monotonic()
            self._cache_payload = dict(payload)
            return payload
