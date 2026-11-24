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

import hashlib
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

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    entry_cost_total REAL NOT NULL,
                    last_price REAL NOT NULL,
                    highest_price REAL NOT NULL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    trailing_stop_pct REAL NOT NULL DEFAULT 0.035,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    regime TEXT NOT NULL,
                    rsi_at_entry REAL NOT NULL,
                    vol_at_entry REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
                )
                dec_cols = {
                    str(r["name"])
                    for r in conn.execute("PRAGMA table_info(decisions);").fetchall()
                }
                if "prev_hash" not in dec_cols:
                    conn.execute("ALTER TABLE decisions ADD COLUMN prev_hash TEXT NOT NULL DEFAULT '';")
                if "decision_hash" not in dec_cols:
                    conn.execute("ALTER TABLE decisions ADD COLUMN decision_hash TEXT NOT NULL DEFAULT '';")
                if "git_sha" not in dec_cols:
                    conn.execute("ALTER TABLE decisions ADD COLUMN git_sha TEXT NOT NULL DEFAULT '';")
                if "stop_loss_price" not in dec_cols:
                    conn.execute("ALTER TABLE decisions ADD COLUMN stop_loss_price REAL;")
                if "take_profit_price" not in dec_cols:
                    conn.execute("ALTER TABLE decisions ADD COLUMN take_profit_price REAL;")
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

    @staticmethod
    def _compute_decision_hash(
        prev_hash: str,
        symbol: str,
        interval: str,
        candle_close_utc: str,
        model_pred: float,
        final_action: str,
        risk_status: str,
    ) -> str:
        payload = (
            f"{prev_hash}|{symbol}|{interval}|{candle_close_utc}|"
            f"{float(model_pred):.8f}|{final_action}|{risk_status}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        from backend.app.core.config import get_app_version_info

        now = utc_now_iso()
        git_sha = str(get_app_version_info().get("git_sha", "local"))
        with self.atomic_tx() as conn:
            prev_row = conn.execute(
                "SELECT decision_hash FROM decisions ORDER BY id DESC LIMIT 1;"
            ).fetchone()
            prev_hash = (
                str(prev_row["decision_hash"])
                if (prev_row is not None and str(prev_row["decision_hash"] or "").strip())
                else "GENESIS"
            )
            decision_hash = self._compute_decision_hash(
                prev_hash=prev_hash,
                symbol=str(decision_data["symbol"]),
                interval=str(decision_data["interval"]),
                candle_close_utc=str(decision_data["candle_close_utc"]),
                model_pred=float(decision_data["model_pred"]),
                final_action=str(decision_data["final_action"]),
                risk_status=str(decision_data["risk_status"]),
            )

            declared_sl = decision_data.get("stop_loss_price")
            if declared_sl is None:
                declared_sl = snapshot_data.get("stop_loss_price")
            declared_tp = decision_data.get("take_profit_price")
            if declared_tp is None:
                declared_tp = snapshot_data.get("take_profit_price")

            cur_dec = conn.execute(
                """
                INSERT INTO decisions (
                    candle_id, symbol, interval, candle_close_utc, model_pred,
                    laya1_json, qwen_json, laya2_json, risk_status, risk_reason,
                    final_action, stop_loss_price, take_profit_price,
                    latency_ms, qwen_execution_mode, laya_execution_mode,
                    source_mode, prev_hash, decision_hash, git_sha, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    float(declared_sl) if declared_sl is not None else None,
                    float(declared_tp) if declared_tp is not None else None,
                    float(decision_data["latency_ms"]),
                    decision_data["qwen_execution_mode"],
                    decision_data["laya_execution_mode"],
                    decision_data["source_mode"],
                    prev_hash,
                    decision_hash,
                    git_sha,
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

            pos_qty = float(snapshot_data.get("position_qty", 0.0))
            sym = str(snapshot_data["symbol"])
            if pos_qty >= 1e-8:
                last_px = float(
                    snapshot_data.get("last_price")
                    or (float(snapshot_data["position_value"]) / pos_qty)
                )
                self.upsert_position(
                    {
                        "symbol": sym,
                        "quantity": pos_qty,
                        "avg_entry_price": float(snapshot_data.get("avg_entry_price", last_px)),
                        "entry_cost_total": float(snapshot_data.get("entry_cost_total", pos_qty * last_px)),
                        "last_price": last_px,
                        "highest_price": float(snapshot_data.get("highest_price", last_px) or last_px),
                        "stop_loss_price": snapshot_data.get("stop_loss_price"),
                        "take_profit_price": snapshot_data.get("take_profit_price"),
                        "trailing_stop_pct": float(snapshot_data.get("trailing_stop_pct", 0.035)),
                        "opened_at": snapshot_data.get("candle_time", now),
                        "updated_at": now,
                    },
                    conn=conn,
                )
            else:
                self.delete_position(sym, conn=conn)

            return decision_id

    # --------------------------------------------------------------------------
    # POSICIONES MULTI-TICKER Y MEMORIA EPISÓDICA (FINMEM)
    # --------------------------------------------------------------------------
    def upsert_position(
        self, pos_data: Dict[str, Any], conn: Optional[sqlite3.Connection] = None
    ) -> None:
        now = utc_now_iso()
        params = (
            str(pos_data["symbol"]),
            float(pos_data["quantity"]),
            float(pos_data["avg_entry_price"]),
            float(pos_data["entry_cost_total"]),
            float(pos_data["last_price"]),
            float(pos_data.get("highest_price", pos_data["last_price"])),
            pos_data.get("stop_loss_price"),
            pos_data.get("take_profit_price"),
            float(pos_data.get("trailing_stop_pct", 0.035)),
            str(pos_data.get("opened_at", now)),
            str(pos_data.get("updated_at", now)),
        )
        sql = """
            INSERT INTO positions (
                symbol, quantity, avg_entry_price, entry_cost_total,
                last_price, highest_price, stop_loss_price, take_profit_price,
                trailing_stop_pct, opened_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_entry_price = excluded.avg_entry_price,
                entry_cost_total = excluded.entry_cost_total,
                last_price = excluded.last_price,
                highest_price = excluded.highest_price,
                stop_loss_price = excluded.stop_loss_price,
                take_profit_price = excluded.take_profit_price,
                trailing_stop_pct = excluded.trailing_stop_pct,
                updated_at = excluded.updated_at;
        """
        if conn is not None:
            conn.execute(sql, params)
            return
        with self.atomic_tx() as tx:
            tx.execute(sql, params)

    def delete_position(
        self, symbol: str, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        if conn is not None:
            conn.execute("DELETE FROM positions WHERE symbol = ?;", (symbol,))
            return
        with self.atomic_tx() as tx:
            tx.execute("DELETE FROM positions WHERE symbol = ?;", (symbol,))

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM positions WHERE symbol = ?;", (symbol,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE quantity >= 1e-8 ORDER BY updated_at DESC;"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def add_episodic_memory(self, mem_data: Dict[str, Any]) -> int:
        now = utc_now_iso()
        with self.atomic_tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodic_memories (
                    symbol, interval, candle_time, entry_price, exit_price,
                    realized_pnl, return_pct, regime, rsi_at_entry, vol_at_entry,
                    outcome, lesson, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(mem_data["symbol"]),
                    str(mem_data.get("interval", "1d")),
                    str(mem_data["candle_time"]),
                    float(mem_data["entry_price"]),
                    float(mem_data["exit_price"]),
                    float(mem_data["realized_pnl"]),
                    float(mem_data["return_pct"]),
                    str(mem_data.get("regime", "unknown")),
                    float(mem_data.get("rsi_at_entry", 50.0)),
                    float(mem_data.get("vol_at_entry", 0.02)),
                    str(mem_data["outcome"]),
                    str(mem_data["lesson"])[:400],
                    now,
                ),
            )
            return int(cur.lastrowid)

    def list_episodic_memories(
        self, symbol: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if symbol:
                    rows = conn.execute(
                        "SELECT * FROM episodic_memories WHERE symbol = ? ORDER BY id DESC LIMIT ?;",
                        (symbol, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM episodic_memories ORDER BY id DESC LIMIT ?;",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_decision_by_id(self, decision_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM decisions WHERE id = ?;", (int(decision_id),)
                ).fetchone()
                if not row:
                    return None
                d = dict(row)
                for json_col in ("laya1_json", "qwen_json", "laya2_json"):
                    try:
                        d[json_col] = json.loads(d[json_col])
                    except (json.JSONDecodeError, TypeError):
                        pass
                return d
            finally:
                conn.close()

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

    @staticmethod
    def _decode_decision_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for json_col in ("laya1_json", "qwen_json", "laya2_json"):
            try:
                d[json_col] = json.loads(d[json_col])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def list_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
                return [self._decode_decision_row(r) for r in rows]
            finally:
                conn.close()

    def list_decisions_with_settlement(self, limit: int = 100) -> Dict[str, Any]:
        """
        Devuelve decisiones recientes enriquecidas con detalle de movimiento (`paper_orders`)
        y estado de liquidación FIFO (`Sin liquidar`, `Liquidado`, `Liquidación (Venta)`, etc.).
        """
        with self._lock:
            conn = self._connect()
            try:
                dec_rows = conn.execute(
                    "SELECT * FROM decisions ORDER BY id DESC LIMIT ?;",
                    (limit,),
                ).fetchall()
                order_rows = conn.execute(
                    "SELECT * FROM paper_orders ORDER BY id ASC;"
                ).fetchall()
                pos_rows = conn.execute(
                    "SELECT * FROM positions WHERE quantity >= 1e-8;"
                ).fetchall()
            finally:
                conn.close()

        open_positions: Dict[str, Dict[str, Any]] = {
            str(r["symbol"]): dict(r) for r in pos_rows
        }
        open_lots_by_sym: Dict[str, List[Dict[str, Any]]] = {}
        buy_lots: List[Dict[str, Any]] = []
        movement_by_decision: Dict[int, Dict[str, Any]] = {}

        for r in order_rows:
            o = dict(r)
            sym = str(o["symbol"])
            side = str(o["side"]).upper()
            dec_id = int(o["decision_id"])
            exec_px = float(o["exec_price"])
            qty = float(o["quantity"])
            fee = float(o["fee"])
            notional = float(o["notional"])
            exec_at = str(o["executed_at"])

            if side == "BUY":
                lot = {
                    "order_id": int(o["id"]),
                    "decision_id": dec_id,
                    "symbol": sym,
                    "side": "BUY",
                    "exec_price": exec_px,
                    "quantity": qty,
                    "rem_qty": qty,
                    "fee": fee,
                    "notional": notional,
                    "executed_at": exec_at,
                    "exit_price": None,
                    "liquidated_at": None,
                    "liquidated_by_order_id": None,
                    "realized_pnl": 0.0,
                }
                open_lots_by_sym.setdefault(sym, []).append(lot)
                buy_lots.append(lot)
            elif side == "SELL":
                sell_qty_rem = qty
                sell_realized = float(o["realized_pnl"]) if o["realized_pnl"] is not None else None
                lots = open_lots_by_sym.get(sym, [])
                while lots and sell_qty_rem > 1e-6:
                    lot = lots[0]
                    consumed = min(lot["rem_qty"], sell_qty_rem)
                    if sell_realized is not None and qty > 0:
                        slice_pnl = (consumed / qty) * sell_realized
                    else:
                        lot_fee_slice = lot["fee"] * (consumed / max(lot["quantity"], 1e-9))
                        slice_pnl = (exec_px - lot["exec_price"]) * consumed - lot_fee_slice
                    lot["rem_qty"] = max(0.0, lot["rem_qty"] - consumed)
                    lot["exit_price"] = exec_px
                    lot["liquidated_at"] = exec_at
                    lot["liquidated_by_order_id"] = int(o["id"])
                    lot["realized_pnl"] = float(lot["realized_pnl"]) + slice_pnl
                    sell_qty_rem -= consumed
                    if lot["rem_qty"] <= 1e-6:
                        lots.pop(0)

                movement_by_decision[dec_id] = {
                    "has_movement": True,
                    "order_id": int(o["id"]),
                    "side": "SELL",
                    "exec_price": round(exec_px, 4),
                    "quantity": round(qty, 6),
                    "notional": round(notional, 2),
                    "fee": round(fee, 4),
                    "executed_at": exec_at,
                    "settlement_status": "LIQUIDATED_EXIT",
                    "settlement_label": "Liquidación (Venta)",
                    "is_liquidated": True,
                    "liquidated_pct": 100.0,
                    "exit_price": round(exec_px, 4),
                    "liquidated_at": exec_at,
                    "realized_pnl": round(sell_realized if sell_realized is not None else 0.0, 2),
                    "unrealized_pnl": None,
                    "current_price": None,
                    "stop_loss_price": None,
                    "take_profit_price": None,
                }

        for lot in buy_lots:
            sym = lot["symbol"]
            orig_qty = max(lot["quantity"], 1e-9)
            rem_qty = lot["rem_qty"]
            pos = open_positions.get(sym)

            if rem_qty <= 1e-6 or pos is None:
                movement_by_decision[lot["decision_id"]] = {
                    "has_movement": True,
                    "order_id": lot["order_id"],
                    "side": "BUY",
                    "exec_price": round(lot["exec_price"], 4),
                    "quantity": round(lot["quantity"], 6),
                    "notional": round(lot["notional"], 2),
                    "fee": round(lot["fee"], 4),
                    "executed_at": lot["executed_at"],
                    "settlement_status": "LIQUIDATED",
                    "settlement_label": "Liquidado",
                    "is_liquidated": True,
                    "liquidated_pct": 100.0,
                    "exit_price": round(float(lot["exit_price"]), 4) if lot["exit_price"] is not None else None,
                    "liquidated_at": lot["liquidated_at"],
                    "realized_pnl": round(float(lot["realized_pnl"]), 2),
                    "unrealized_pnl": 0.0,
                    "current_price": None,
                    "stop_loss_price": None,
                    "take_profit_price": None,
                }
            else:
                sold_frac = max(0.0, min(1.0, (orig_qty - rem_qty) / orig_qty))
                is_partial = sold_frac > 1e-4
                cur_px = float(pos["last_price"])
                fee_rem = float(lot["fee"]) * (rem_qty / orig_qty)
                unrealized = (cur_px - float(lot["exec_price"])) * rem_qty - fee_rem
                movement_by_decision[lot["decision_id"]] = {
                    "has_movement": True,
                    "order_id": lot["order_id"],
                    "side": "BUY",
                    "exec_price": round(lot["exec_price"], 4),
                    "quantity": round(lot["quantity"], 6),
                    "notional": round(lot["notional"], 2),
                    "fee": round(lot["fee"], 4),
                    "executed_at": lot["executed_at"],
                    "settlement_status": "PARTIALLY_LIQUIDATED" if is_partial else "OPEN",
                    "settlement_label": f"Parcial ({sold_frac * 100:.0f}%)" if is_partial else "Sin liquidar",
                    "is_liquidated": False,
                    "liquidated_pct": round(sold_frac * 100.0, 1),
                    "exit_price": round(float(lot["exit_price"]), 4) if lot["exit_price"] is not None else None,
                    "liquidated_at": lot["liquidated_at"],
                    "realized_pnl": round(float(lot["realized_pnl"]), 2) if is_partial else None,
                    "unrealized_pnl": round(unrealized, 2),
                    "current_price": round(cur_px, 4),
                    "stop_loss_price": pos.get("stop_loss_price"),
                    "take_profit_price": pos.get("take_profit_price"),
                    "target_recalculated_at": pos.get("updated_at"),
                }

        items: List[Dict[str, Any]] = []
        total_movements = 0
        open_unliquidated_count = 0
        liquidated_count = 0

        for r in dec_rows:
            d = self._decode_decision_row(r)
            dec_id = int(d["id"])
            declared_sl = d.get("stop_loss_price")
            declared_tp = d.get("take_profit_price")
            mov = movement_by_decision.get(dec_id)
            if mov is not None:
                mov["declared_sl"] = declared_sl
                mov["declared_tp"] = declared_tp
                if mov.get("stop_loss_price") is None:
                    mov["stop_loss_price"] = declared_sl
                if mov.get("take_profit_price") is None:
                    mov["take_profit_price"] = declared_tp
                mov.setdefault("target_recalculated_at", None)
                total_movements += 1
                if mov["is_liquidated"]:
                    liquidated_count += 1
                else:
                    open_unliquidated_count += 1
            else:
                final_act = str(d.get("final_action", "HOLD")).upper()
                risk_st = str(d.get("risk_status", "APPROVED")).upper()
                is_blocked = (final_act in {"BUY", "SELL"}) or (risk_st != "APPROVED")
                mov = {
                    "has_movement": False,
                    "order_id": None,
                    "side": None,
                    "exec_price": None,
                    "quantity": None,
                    "notional": None,
                    "fee": None,
                    "executed_at": None,
                    "settlement_status": "REJECTED" if is_blocked else "NONE",
                    "settlement_label": "Bloqueado" if is_blocked else "Sin movimiento",
                    "is_liquidated": False,
                    "liquidated_pct": 0.0,
                    "exit_price": None,
                    "liquidated_at": None,
                    "realized_pnl": None,
                    "unrealized_pnl": None,
                    "current_price": None,
                    "declared_sl": declared_sl,
                    "declared_tp": declared_tp,
                    "stop_loss_price": declared_sl,
                    "take_profit_price": declared_tp,
                    "target_recalculated_at": None,
                }
            d["movement"] = mov
            items.append(d)

        return {
            "items": items,
            "summary": {
                "total_movements": total_movements,
                "open_unliquidated_count": open_unliquidated_count,
                "liquidated_count": liquidated_count,
            },
        }

    def list_audited_movements(self, limit: int = 100) -> Dict[str, Any]:
        """
        Devuelve el registro completo de auditoría para movimientos ejecutados (`paper_orders`),
        incluyendo TP/SL declarados y vigentes, PnL neto, regla Risk Gate y verificación SHA-256.
        """
        chain = self.verify_decision_audit_chain(limit=5000)
        settled = self.list_decisions_with_settlement(limit=500)
        audited: List[Dict[str, Any]] = []
        total_fees = 0.0
        total_realized = 0.0
        total_unrealized = 0.0
        open_cnt = 0
        liq_cnt = 0

        for d in settled["items"]:
            m = d.get("movement") or {}
            if not m.get("has_movement"):
                continue

            dec_hash = str(d.get("decision_hash") or "")
            prev_h = str(d.get("prev_hash") or "")
            if dec_hash:
                expected = self._compute_decision_hash(
                    prev_hash=prev_h,
                    symbol=str(d["symbol"]),
                    interval=str(d["interval"]),
                    candle_close_utc=str(d["candle_close_utc"]),
                    model_pred=float(d["model_pred"]),
                    final_action=str(d["final_action"]),
                    risk_status=str(d["risk_status"]),
                )
                hash_verified = expected == dec_hash
            else:
                hash_verified = False

            fee = float(m.get("fee") or 0.0)
            realized = m.get("realized_pnl")
            unrealized = m.get("unrealized_pnl")
            total_fees += fee
            if m.get("is_liquidated"):
                liq_cnt += 1
                if realized is not None and m.get("side") == "SELL":
                    total_realized += float(realized)
            else:
                open_cnt += 1
                if unrealized is not None:
                    total_unrealized += float(unrealized)

            audited.append(
                {
                    "order_id": m.get("order_id"),
                    "decision_id": d["id"],
                    "symbol": d["symbol"],
                    "interval": d["interval"],
                    "side": m.get("side"),
                    "quantity": m.get("quantity"),
                    "exec_price": m.get("exec_price"),
                    "notional": m.get("notional"),
                    "fee": round(fee, 4),
                    "executed_at": m.get("executed_at"),
                    "source_mode": d.get("source_mode"),
                    "risk_status": d.get("risk_status"),
                    "risk_reason": d.get("risk_reason"),
                    "settlement_status": m.get("settlement_status"),
                    "settlement_label": m.get("settlement_label"),
                    "is_liquidated": bool(m.get("is_liquidated")),
                    "current_price": m.get("current_price"),
                    "exit_price": m.get("exit_price"),
                    "liquidated_at": m.get("liquidated_at"),
                    "realized_pnl": realized,
                    "unrealized_pnl": unrealized,
                    "declared_tp": m.get("declared_tp"),
                    "declared_sl": m.get("declared_sl"),
                    "take_profit_price": m.get("take_profit_price"),
                    "stop_loss_price": m.get("stop_loss_price"),
                    "target_recalculated_at": m.get("target_recalculated_at"),
                    "decision_hash": dec_hash,
                    "prev_hash": prev_h,
                    "git_sha": d.get("git_sha") or "",
                    "hash_verified": hash_verified,
                }
            )
            if len(audited) >= limit:
                break

        return {
            "items": audited,
            "summary": {
                "total_movements": len(audited),
                "open_count": open_cnt,
                "liquidated_count": liq_cnt,
                "total_fees": round(total_fees, 2),
                "total_realized_pnl": round(total_realized, 2),
                "total_unrealized_pnl": round(total_unrealized, 2),
                "chain_valid": bool(chain.get("valid", True)),
                "head_hash": chain.get("head_hash", "GENESIS"),
            },
        }

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

    def verify_decision_audit_chain(self, limit: int = 1000) -> Dict[str, Any]:
        """Verifica la integridad criptográfica SHA-256 encadenada de `decisions`."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, symbol, interval, candle_close_utc, model_pred,
                           final_action, risk_status, prev_hash, decision_hash, git_sha
                    FROM decisions
                    WHERE decision_hash IS NOT NULL AND decision_hash != ''
                    ORDER BY id ASC
                    LIMIT ?;
                    """,
                    (int(limit),),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return {
                "valid": True,
                "checked_count": 0,
                "head_hash": "GENESIS",
                "broken_at_id": None,
                "git_sha": "",
            }

        prev_expected: Optional[str] = None
        head_hash = "GENESIS"
        last_git_sha = ""
        for idx, r in enumerate(rows):
            if idx > 0 and str(r["prev_hash"]) != prev_expected:
                return {
                    "valid": False,
                    "checked_count": idx,
                    "head_hash": head_hash,
                    "broken_at_id": int(r["id"]),
                    "git_sha": str(r["git_sha"] or ""),
                }
            expected_hash = self._compute_decision_hash(
                prev_hash=str(r["prev_hash"]),
                symbol=str(r["symbol"]),
                interval=str(r["interval"]),
                candle_close_utc=str(r["candle_close_utc"]),
                model_pred=float(r["model_pred"]),
                final_action=str(r["final_action"]),
                risk_status=str(r["risk_status"]),
            )
            if expected_hash != str(r["decision_hash"]):
                return {
                    "valid": False,
                    "checked_count": idx,
                    "head_hash": head_hash,
                    "broken_at_id": int(r["id"]),
                    "git_sha": str(r["git_sha"] or ""),
                }
            prev_expected = expected_hash
            head_hash = expected_hash
            last_git_sha = str(r["git_sha"] or "")

        return {
            "valid": True,
            "checked_count": len(rows),
            "head_hash": head_hash,
            "broken_at_id": None,
            "git_sha": last_git_sha,
        }

    def list_prediction_return_pairs(self, limit: int = 300) -> tuple[List[float], List[float]]:
        """Empareja `model_pred` con el retorno realizado siguiente dentro de `processed_candles`."""
        with self._lock:
            conn = self._connect()
            try:
                decs = conn.execute(
                    """
                    SELECT symbol, interval, candle_close_utc, model_pred
                    FROM decisions
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (int(limit),),
                ).fetchall()
                candles = conn.execute(
                    """
                    SELECT symbol, interval, close_time, close
                    FROM processed_candles
                    ORDER BY close_time ASC;
                    """
                ).fetchall()
            finally:
                conn.close()

        by_key: Dict[tuple[str, str], List[tuple[str, float]]] = {}
        for c in candles:
            by_key.setdefault((str(c["symbol"]), str(c["interval"])), []).append(
                (str(c["close_time"]), float(c["close"]))
            )

        preds: List[float] = []
        reals: List[float] = []
        for d in decs:
            seq = by_key.get((str(d["symbol"]), str(d["interval"])), [])
            t_cur = str(d["candle_close_utc"])
            for i in range(len(seq) - 1):
                if seq[i][0] == t_cur and seq[i][1] > 0.0:
                    preds.append(float(d["model_pred"]))
                    reals.append((seq[i + 1][1] - seq[i][1]) / seq[i][1])
                    break
        return preds, reals

    def reset_paper_portfolio(self, initial_capital: float) -> Dict[str, Any]:
        """Reinicia atómicamente cartera, órdenes, velas procesadas y picos heredados."""
        now = utc_now_iso()
        with self.atomic_tx() as conn:
            conn.execute("DELETE FROM paper_orders;")
            conn.execute("DELETE FROM decisions;")
            conn.execute("DELETE FROM processed_candles;")
            conn.execute("DELETE FROM portfolio_snapshots;")
            conn.execute("DELETE FROM positions;")
            conn.execute("DELETE FROM system_state WHERE key = 'daily_equity_date';")
            for k, v in (
                ("initial_capital", str(float(initial_capital))),
                ("peak_equity", str(float(initial_capital))),
                ("daily_start_equity", str(float(initial_capital))),
                ("system_status", "RUNNING"),
            ):
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at;
                    """,
                    (k, v, now),
                )
        return {"initial_capital": float(initial_capital), "system_status": "RUNNING", "reset_at_utc": now}
