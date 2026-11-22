"""
Back Office - Motor de Liquidación Paper Trading (0 Órdenes Reales).

Cumple estrictamente:
- Soporte de perfiles `agent_paper` (50,000 capital, 0.0% comisión) y
  `practice_backtest` (100,000 capital, 0.10% comisión).
- Respeto de saldo, tamaño mínimo de lote (`MIN_LOT_SIZE`), precisión decimal (`STEP_SIZE`),
  deslizamiento documentado (`SLIPPAGE_BPS = 5`), restricción Long-Only (sin short ni apalancamiento).
- Recuperación exacta desde SQLite tras reinicio abrupto del backend.
- STOP (`STOPPED`) bloquea nuevas órdenes pero NO liquida automáticamente la posición abierta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.back_office.database import TradingDatabase, utc_now_iso
from backend.app.core.config import Settings


class InsufficientFundsError(RuntimeError):
    """Error lanzado al intentar comprar sin saldo suficiente."""


class SystemStoppedError(RuntimeError):
    """Error lanzado cuando el Kill-Switch STOP bloquea una nueva orden."""


@dataclass
class PortfolioState:
    cash: float
    position_qty: float
    avg_entry_price: float
    entry_cost_total: float
    last_price: float
    total_equity: float
    peak_equity: float
    daily_start_equity: float
    drawdown_pct: float
    daily_pnl_pct: float
    active_symbol: str


def floor_to_step(value: float, step_size: float) -> float:
    if step_size <= 0:
        return value
    decimals = max(0, int(round(-math.log10(step_size))))
    factor = 10 ** decimals
    return math.floor((value * factor) + 1e-9) / factor


def compute_directional_accuracy(
    y_true: List[float], y_pred: List[float]
) -> float | str:
    """
    Precisión direccional excluyendo y_i == 0.
    Si el denominador es 0, retorna 'No definido' en lugar de ZeroDivisionError.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return "No definido"
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != 0.0]
    if len(pairs) == 0:
        return "No definido"
    hits = sum(1 for yt, yp in pairs if (yt > 0 and yp > 0) or (yt < 0 and yp < 0))
    return float(hits / len(pairs))


class PaperTradingEngine:
    """Simulador Paper Trading con persistencia SQLite y recuperación ante reinicios."""

    def __init__(self, settings: Settings, db: TradingDatabase) -> None:
        if settings.enable_real_trading:
            raise RuntimeError("CRITICAL: Real trading must remain disabled.")
        self.settings = settings
        self.db = db
        self.initial_capital = float(settings.profile_config.initial_capital)
        self.fee_pct = float(settings.profile_config.fee_pct)
        self.slippage_bps = float(settings.slippage_bps)
        self.min_lot_size = float(settings.min_lot_size)
        self.step_size = float(settings.step_size)

    def get_portfolio_state(self, current_price: Optional[float] = None) -> PortfolioState:
        """Recupera exactamente efectivo, posición y máximos desde SQLite WAL."""
        snap = self.db.get_latest_snapshot()
        active_symbol = self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol

        if snap is None:
            cash = self.initial_capital
            qty = 0.0
            avg_entry = 0.0
            entry_cost = 0.0
            last_px = float(current_price) if current_price else 0.0
            equity = cash
        else:
            cash = float(snap["cash"])
            qty = float(snap["position_qty"])
            avg_entry = float(snap.get("avg_entry_price", 0.0))
            entry_cost = float(snap.get("entry_cost_total", 0.0))
            active_symbol = str(snap["symbol"])
            if current_price is not None:
                last_px = float(current_price)
            elif qty > 0:
                last_px = float(snap["position_value"]) / qty
            else:
                last_px = avg_entry
            equity = cash + qty * last_px

        peak_stored = float(self.db.get_state("peak_equity", str(self.initial_capital)) or self.initial_capital)
        peak_equity = max(peak_stored, equity)

        daily_start = float(
            self.db.get_state("daily_start_equity", str(self.initial_capital))
            or self.initial_capital
        )
        drawdown_pct = ((peak_equity - equity) / peak_equity) if peak_equity > 0 else 0.0
        daily_pnl_pct = ((equity - daily_start) / daily_start) if daily_start > 0 else 0.0

        return PortfolioState(
            cash=cash,
            position_qty=qty,
            avg_entry_price=avg_entry,
            entry_cost_total=entry_cost,
            last_price=last_px,
            total_equity=equity,
            peak_equity=peak_equity,
            daily_start_equity=daily_start,
            drawdown_pct=max(0.0, drawdown_pct),
            daily_pnl_pct=daily_pnl_pct,
            active_symbol=active_symbol,
        )

    def simulate_execution(
        self,
        action: str,
        symbol: str,
        market_price: float,
        target_exposure_pct: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Prepara la ejecución Paper para una acción aprobada (BUY | HOLD | SELL)
        sin escribir todavía en BD (para registro atómico junto con la decisión).
        """
        system_status = self.db.get_state("system_status", "RUNNING")
        if system_status == "STOPPED" and action in {"BUY", "SELL"}:
            self.db.log_error(
                error_code="STOPPED_ORDER_BLOCKED",
                component="PaperTradingEngine",
                message=f"Orden {action} bloqueada porque el sistema está en estado STOPPED.",
                payload_excerpt=f"symbol={symbol}, action={action}",
            )
            raise SystemStoppedError(
                "El sistema está detenido (STOPPED); nuevas órdenes bloqueadas preservando posición abierta."
            )

        state = self.get_portfolio_state(current_price=market_price)
        slip_mult = self.slippage_bps / 10000.0
        now = utc_now_iso()

        if action == "HOLD":
            pos_val = state.position_qty * market_price
            eq = state.cash + pos_val
            peak = max(state.peak_equity, eq)
            dd = ((peak - eq) / peak) if peak > 0 else 0.0
            daily_pnl = (
                (eq - state.daily_start_equity) / state.daily_start_equity
                if state.daily_start_equity > 0
                else 0.0
            )
            return {
                "order": None,
                "snapshot": {
                    "candle_time": now,
                    "symbol": symbol,
                    "cash": state.cash,
                    "position_qty": state.position_qty,
                    "avg_entry_price": state.avg_entry_price,
                    "entry_cost_total": state.entry_cost_total,
                    "position_value": pos_val,
                    "total_equity": eq,
                    "daily_pnl_pct": daily_pnl,
                    "drawdown_pct": dd,
                },
                "peak_equity": peak,
            }

        if action == "BUY":
            if market_price <= 0:
                raise ValueError("El precio de mercado debe ser positivo.")
            exec_price = market_price * (1.0 + slip_mult)
            # Capital a asignar respetando exposición máxima
            alloc_pct = min(float(self.settings.max_exposure_pct), float(target_exposure_pct))
            budget = min(state.cash, state.total_equity * alloc_pct)
            raw_qty = budget / (exec_price * (1.0 + self.fee_pct))
            qty = floor_to_step(raw_qty, self.step_size)

            if qty < self.min_lot_size:
                self.db.log_error(
                    error_code="INSUFFICIENT_FUNDS_BUY",
                    component="PaperTradingEngine",
                    message=(
                        f"Saldo insuficiente para comprar lote mínimo ({self.min_lot_size}) de {symbol}. "
                        f"Cash={state.cash:.2f}, exec_price={exec_price:.2f}"
                    ),
                    payload_excerpt=f"cash={state.cash:.2f}, qty={qty}",
                )
                raise InsufficientFundsError(
                    f"Saldo insuficiente ({state.cash:.2f}) para comprar {symbol} >= lote mínimo {self.min_lot_size}."
                )

            notional = qty * exec_price
            fee = notional * self.fee_pct
            total_cost = notional + fee

            if total_cost > state.cash + 1e-6:
                self.db.log_error(
                    error_code="INSUFFICIENT_FUNDS_BUY",
                    component="PaperTradingEngine",
                    message=f"Costo total ({total_cost:.2f}) supera efectivo ({state.cash:.2f}).",
                    payload_excerpt=f"symbol={symbol}, cost={total_cost:.2f}",
                )
                raise InsufficientFundsError(
                    f"Saldo insuficiente: requerido={total_cost:.2f}, disponible={state.cash:.2f}."
                )

            new_cash = max(0.0, state.cash - total_cost)
            new_qty = state.position_qty + qty
            new_entry_cost = state.entry_cost_total + total_cost
            new_avg_entry = new_entry_cost / new_qty if new_qty > 0 else 0.0
            pos_val = new_qty * market_price
            eq = new_cash + pos_val
            peak = max(state.peak_equity, eq)
            dd = ((peak - eq) / peak) if peak > 0 else 0.0
            daily_pnl = (
                (eq - state.daily_start_equity) / state.daily_start_equity
                if state.daily_start_equity > 0
                else 0.0
            )

            return {
                "order": {
                    "side": "BUY",
                    "symbol": symbol,
                    "exec_price": exec_price,
                    "quantity": qty,
                    "fee": fee,
                    "slippage_bps": self.slippage_bps,
                    "notional": notional,
                    "realized_pnl": None,
                    "executed_at": now,
                },
                "snapshot": {
                    "candle_time": now,
                    "symbol": symbol,
                    "cash": new_cash,
                    "position_qty": new_qty,
                    "avg_entry_price": new_avg_entry,
                    "entry_cost_total": new_entry_cost,
                    "position_value": pos_val,
                    "total_equity": eq,
                    "daily_pnl_pct": daily_pnl,
                    "drawdown_pct": dd,
                },
                "peak_equity": peak,
            }

        if action == "SELL":
            if state.position_qty < self.min_lot_size:
                raise ValueError(
                    f"Operación SELL rechazada (Long-Only): no hay posición abierta en {symbol}."
                )
            exec_price = market_price * (1.0 - slip_mult)
            qty = floor_to_step(state.position_qty, self.step_size)
            notional = qty * exec_price
            fee = notional * self.fee_pct
            net_proceeds = notional - fee

            # Costo proporcional de entrada para calcular PnL realizado exacto del ciclo
            sell_fraction = min(1.0, qty / state.position_qty) if state.position_qty > 0 else 1.0
            entry_cost_closed = state.entry_cost_total * sell_fraction
            realized_pnl = net_proceeds - entry_cost_closed

            new_qty = max(0.0, state.position_qty - qty)
            if new_qty < self.min_lot_size:
                new_qty = 0.0
                new_entry_cost = 0.0
                new_avg_entry = 0.0
            else:
                new_entry_cost = max(0.0, state.entry_cost_total - entry_cost_closed)
                new_avg_entry = new_entry_cost / new_qty

            new_cash = state.cash + net_proceeds
            pos_val = new_qty * market_price
            eq = new_cash + pos_val
            peak = max(state.peak_equity, eq)
            dd = ((peak - eq) / peak) if peak > 0 else 0.0
            daily_pnl = (
                (eq - state.daily_start_equity) / state.daily_start_equity
                if state.daily_start_equity > 0
                else 0.0
            )

            return {
                "order": {
                    "side": "SELL",
                    "symbol": symbol,
                    "exec_price": exec_price,
                    "quantity": qty,
                    "fee": fee,
                    "slippage_bps": self.slippage_bps,
                    "notional": notional,
                    "realized_pnl": realized_pnl,
                    "executed_at": now,
                },
                "snapshot": {
                    "candle_time": now,
                    "symbol": symbol,
                    "cash": new_cash,
                    "position_qty": new_qty,
                    "avg_entry_price": new_avg_entry,
                    "entry_cost_total": new_entry_cost,
                    "position_value": pos_val,
                    "total_equity": eq,
                    "daily_pnl_pct": daily_pnl,
                    "drawdown_pct": dd,
                },
                "peak_equity": peak,
            }

        raise ValueError(f"Acción inválida: {action}")

    def compute_portfolio_metrics(self) -> Dict[str, Any]:
        """Calcula métricas auditables del portafolio en escala original."""
        snapshots = self.db.list_snapshots(limit=5000)
        orders = self.db.list_orders(limit=5000)

        closed_sells = [
            o for o in orders if o["side"] == "SELL" and o["realized_pnl"] is not None
        ]
        if len(closed_sells) == 0:
            win_rate: float | str = "No definido"
        else:
            wins = sum(1 for o in closed_sells if float(o["realized_pnl"]) > 0.0)
            win_rate = float(wins / len(closed_sells))

        if not snapshots:
            return {
                "initial_capital": self.initial_capital,
                "current_equity": self.initial_capital,
                "net_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": win_rate,
                "closed_trades": 0,
                "total_orders": len(orders),
            }

        equities = np.asarray([float(s["total_equity"]) for s in snapshots], dtype=float)
        peaks = np.maximum.accumulate(equities)
        drawdowns = (peaks - equities) / np.where(peaks > 0, peaks, 1.0)
        max_dd = float(np.max(drawdowns))
        current_eq = float(equities[-1])
        net_ret = (current_eq - self.initial_capital) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_equity": current_eq,
            "net_return_pct": float(net_ret),
            "max_drawdown_pct": max_dd,
            "win_rate": win_rate,
            "closed_trades": len(closed_sells),
            "total_orders": len(orders),
        }
