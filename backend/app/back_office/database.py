"""
Back Office - Persistencia SQLite en Modo WAL con Idempotencia y Transacciones Atómicas.

Implementa exactamente las 6 tablas obligatorias de la Sección 5.3:
1. processed_candles (UNIQUE(symbol, interval, close_time, source_mode))
2. decisions
3. paper_orders
4. portfolio_snapshots
5. system_state
6. error_logs
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class DuplicateCandleError(RuntimeError):
    """Lanzada cuando una vela ya fue procesada (bloqueo por idempotencia SQLite)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TradingDatabase:
    """Repositorio SQLite WAL thread-safe y recuperable tras reinicios abruptos."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def atomic_tx(self) -> Iterator[sqlite3.Connection]:
        """Context manager de transacción atómica BEGIN IMMEDIATE ... COMMIT."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                yield conn
                if conn.in_transaction:
                    conn.execute("COMMIT;")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, interval, close_time, source_mode)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candle_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    candle_close_utc TEXT NOT NULL,
                    model_pred REAL NOT NULL,
                    laya1_json TEXT NOT NULL,
                    qwen_json TEXT NOT NULL,
                    laya2_json TEXT NOT NULL,
                    risk_status TEXT NOT NULL,
                    risk_reason TEXT NOT NULL,
                    final_action TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    qwen_execution_mode TEXT NOT NULL,
                    laya_execution_mode TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(candle_id) REFERENCES processed_candles(id)
                );

                CREATE TABLE IF NOT EXISTS paper_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exec_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    fee REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    notional REAL NOT NULL,
                    realized_pnl REAL,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(id)
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candle_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cash REAL NOT NULL,
                    position_qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL DEFAULT 0.0,
                    entry_cost_total REAL NOT NULL DEFAULT 0.0,
                    position_value REAL NOT NULL,
                    total_equity REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    source_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_code TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_excerpt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
                )
            finally:
                conn.close()

    # --------------------------------------------------------------------------
    # SYSTEM STATE (RUNNING | PAUSED | STOPPED + recuperación tras reinicio)
    # --------------------------------------------------------------------------
    def set_state(self, key: str, value: Any) -> None:
        now = utc_now_iso()
        val_str = str(value)
        with self.atomic_tx() as conn:
            conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at;
                """,
                (key, val_str, now),
            )

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = ?;", (key,)
                ).fetchone()
                return str(row["value"]) if row else default
            finally:
                conn.close()

    def initialize_defaults_if_empty(
        self,
        default_status: str,
        default_ticker: str,
        default_mode: str,
        profile_name: str,
        initial_capital: float,
        default_interval: str = "1d",
        default_qwen_mode: str = "EXTERNAL_API",
    ) -> None:
        if self.get_state("system_status") is None:
            self.set_state("system_status", default_status)
        if self.get_state("active_ticker") is None:
            self.set_state("active_ticker", default_ticker)
        if self.get_state("source_mode") is None:
            self.set_state("source_mode", default_mode)
        if self.get_state("candle_interval") is None:
            self.set_state("candle_interval", default_interval)
        if self.get_state("qwen_mode") is None:
            self.set_state("qwen_mode", default_qwen_mode)
        if self.get_state("simulation_profile") is None:
            self.set_state("simulation_profile", profile_name)
        if self.get_state("initial_capital") is None:
            self.set_state("initial_capital", str(initial_capital))
        if self.get_state("peak_equity") is None:
            self.set_state("peak_equity", str(initial_capital))
        if self.get_state("daily_start_equity") is None:
            self.set_state("daily_start_equity", str(initial_capital))

    # --------------------------------------------------------------------------
    # LOG FORENSE DE ERRORES
    # --------------------------------------------------------------------------
    def log_error(
        self,
        error_code: str,
        component: str,
        message: str,
        payload_excerpt: str = "",
    ) -> int:
        now = utc_now_iso()
        excerpt = str(payload_excerpt)[:500]
        with self.atomic_tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO error_logs (error_code, component, message, payload_excerpt, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (error_code, component, str(message)[:500], excerpt, now),
            )
            return int(cur.lastrowid)

    # --------------------------------------------------------------------------
    # IDEMPOTENCIA DE VELAS
    # --------------------------------------------------------------------------
    def is_candle_processed(
        self, symbol: str, interval: str, close_time: str, source_mode: str
    ) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id FROM processed_candles
                    WHERE symbol = ? AND interval = ? AND close_time = ? AND source_mode = ?;
                    """,
                    (symbol, interval, close_time, source_mode),
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    def register_candle_idempotent(self, candle: Dict[str, Any]) -> int:
        now = utc_now_iso()
        try:
            with self.atomic_tx() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO processed_candles
                    (symbol, interval, open_time, close_time, open, high, low, close, volume, source_mode, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        candle["symbol"],
                        candle["interval"],
                        candle["open_time"],
                        candle["close_time"],
                        float(candle["open"]),
                        float(candle["high"]),
                        float(candle["low"]),
                        float(candle["close"]),
                        float(candle["volume"]),
                        candle["source_mode"],
                        now,
                    ),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            self.log_error(
                error_code="DUPLICATE_CANDLE",
                component="TradingDatabase",
                message=f"Vela duplicada bloqueada por idempotencia: {candle['symbol']} @ {candle['close_time']}",
                payload_excerpt=json.dumps(
                    {
                        "symbol": candle["symbol"],
                        "interval": candle["interval"],
                        "close_time": candle["close_time"],
                        "source_mode": candle["source_mode"],
                    }
                ),
            )
            raise DuplicateCandleError(
                f"Vela ya procesada previamente: {candle['symbol']} {candle['interval']} {candle['close_time']} ({candle['source_mode']})"
            ) from exc

    # --------------------------------------------------------------------------
    # REGISTRO ATÓMICO DE DECISIÓN, ORDEN PAPER Y SNAPSHOT PATRIMONIAL
    # --------------------------------------------------------------------------
    def record_cycle_result(
        self,
        candle_id: int,
        decision_data: Dict[str, Any],
        order_data: Optional[Dict[str, Any]],
        snapshot_data: Dict[str, Any],
    ) -> int:
        now = utc_now_iso()
        with self.atomic_tx() as conn:
            cur_dec = conn.execute(
                """
                INSERT INTO decisions (
                    candle_id, symbol, interval, candle_close_utc, model_pred,
                    laya1_json, qwen_json, laya2_json, risk_status, risk_reason,
                    final_action, latency_ms, qwen_execution_mode, laya_execution_mode,
                    source_mode, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    candle_id,
                    decision_data["symbol"],
                    decision_data["interval"],
                    decision_data["candle_close_utc"],
                    float(decision_data["model_pred"]),
                    json.dumps(decision_data["laya1_json"], ensure_ascii=False),
                    json.dumps(decision_data["qwen_json"], ensure_ascii=False),
                    json.dumps(decision_data["laya2_json"], ensure_ascii=False),
                    decision_data["risk_status"],
                    decision_data["risk_reason"],
                    decision_data["final_action"],
                    float(decision_data["latency_ms"]),
                    decision_data["qwen_execution_mode"],
                    decision_data["laya_execution_mode"],
                    decision_data["source_mode"],
                    now,
                ),
            )
            decision_id = int(cur_dec.lastrowid)

            if order_data is not None:
                conn.execute(
                    """
                    INSERT INTO paper_orders (
                        decision_id, side, symbol, exec_price, quantity, fee,
                        slippage_bps, notional, realized_pnl, executed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        decision_id,
                        order_data["side"],
                        order_data["symbol"],
                        float(order_data["exec_price"]),
                        float(order_data["quantity"]),
                        float(order_data["fee"]),
                        float(order_data["slippage_bps"]),
                        float(order_data["notional"]),
                        order_data.get("realized_pnl"),
                        order_data.get("executed_at", now),
                    ),
                )

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
                    snapshot_data["candle_time"],
                    snapshot_data["symbol"],
                    float(snapshot_data["cash"]),
                    float(snapshot_data["position_qty"]),
                    float(snapshot_data.get("avg_entry_price", 0.0)),
                    float(snapshot_data.get("entry_cost_total", 0.0)),
                    float(snapshot_data["position_value"]),
                    float(snapshot_data["total_equity"]),
                    float(snapshot_data["daily_pnl_pct"]),
                    float(snapshot_data["drawdown_pct"]),
                    snapshot_data["source_mode"],
                    now,
                ),
            )
            return decision_id

    # --------------------------------------------------------------------------
    # CONSULTAS DE LECTURA Y RECUPERACIÓN
    # --------------------------------------------------------------------------
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 1;"
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_snapshots(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM portfolio_snapshots ORDER BY id ASC LIMIT ?;",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_orders(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM paper_orders ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
                results = []
                for r in rows:
                    d = dict(r)
                    for json_col in ("laya1_json", "qwen_json", "laya2_json"):
                        try:
                            d[json_col] = json.loads(d[json_col])
                        except json.JSONDecodeError:
                            pass
                    results.append(d)
                return results
            finally:
                conn.close()

    def list_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM error_logs ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_processed_candles(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM processed_candles;").fetchone()
                return int(row["cnt"]) if row else 0
            finally:
                conn.close()
