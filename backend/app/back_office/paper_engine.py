"""
Back Office - Motor de Liquidación Paper Trading (0 Órdenes Reales).

Cumple estrictamente:
- Portafolio Multi-Ticker simultáneo respaldado por tabla `positions` en SQLite.
- Dynamic Position Sizing (Volatility Targeting + Convicción DRL).
- Gestión de Stop-Loss ATR, Take-Profit ATR y Trailing Stop dinámico.
- Reset diario UTC real de `daily_start_equity`.
- Soporte de perfiles `agent_paper` y `practice_backtest`.
- Recuperación exacta desde SQLite tras reinicio abrupto del backend.
- STOP (`STOPPED`) bloquea nuevas órdenes pero NO liquida automáticamente las posiciones abiertas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    highest_price: float = 0.0
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_exposure_pct: float = 0.0


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
    """Simulador Paper Trading multi-ticker con persistencia SQLite y recuperación ante reinicios."""

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

    def _load_positions_map(self, snap: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        rows = self.db.list_positions()
        positions_map: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            sym = str(r["symbol"])
            qty = float(r["quantity"])
            if qty >= self.min_lot_size:
                positions_map[sym] = {
                    "symbol": sym,
                    "quantity": qty,
                    "avg_entry_price": float(r["avg_entry_price"]),
                    "entry_cost_total": float(r["entry_cost_total"]),
                    "last_price": float(r["last_price"]),
                    "highest_price": float(r.get("highest_price") or r["last_price"]),
                    "stop_loss_price": r.get("stop_loss_price"),
                    "take_profit_price": r.get("take_profit_price"),
                    "trailing_stop_pct": float(r.get("trailing_stop_pct", self.settings.trailing_stop_pct)),
                }

        # Compatibilidad hacia atrás si la BD tenía snapshot previo pero tabla positions vacía
        if not positions_map and snap is not None and float(snap.get("position_qty", 0.0)) >= self.min_lot_size:
            sym = str(snap["symbol"])
            qty = float(snap["position_qty"])
            avg_e = float(snap.get("avg_entry_price", 0.0))
            last_p = float(snap["position_value"]) / qty if qty > 0 else avg_e
            positions_map[sym] = {
                "symbol": sym,
                "quantity": qty,
                "avg_entry_price": avg_e,
                "entry_cost_total": float(snap.get("entry_cost_total", 0.0)),
                "last_price": last_p,
                "highest_price": max(avg_e, last_p),
                "stop_loss_price": round(avg_e * 0.96, 4) if avg_e > 0 else None,
                "take_profit_price": round(avg_e * 1.07, 4) if avg_e > 0 else None,
                "trailing_stop_pct": self.settings.trailing_stop_pct,
            }
        return positions_map

    def get_portfolio_state(
        self,
        current_price: Optional[float] = None,
        symbol: Optional[str] = None,
        candle_time: Optional[str] = None,
    ) -> PortfolioState:
        """Recupera exactamente efectivo, todas las posiciones abiertas y máximos desde SQLite WAL."""
        snap = self.db.get_latest_snapshot()
        active_ticker = self.db.get_state("active_ticker", self.settings.trading_symbol) or self.settings.trading_symbol
        target_symbol = symbol or (str(snap["symbol"]) if snap is not None else active_ticker)

        cash = self.initial_capital if snap is None else float(snap["cash"])
        positions_map = self._load_positions_map(snap)

        pos_target = positions_map.get(target_symbol)
        if pos_target is not None:
            qty = float(pos_target["quantity"])
            avg_entry = float(pos_target["avg_entry_price"])
            entry_cost = float(pos_target["entry_cost_total"])
            last_px = float(current_price) if current_price is not None else float(pos_target["last_price"])
            highest_price = max(float(pos_target.get("highest_price") or last_px), last_px)
            stop_loss_price = (
                float(pos_target["stop_loss_price"])
                if pos_target.get("stop_loss_price") is not None
                else None
            )
            take_profit_price = (
                float(pos_target["take_profit_price"])
                if pos_target.get("take_profit_price") is not None
                else None
            )
            pos_target["last_price"] = last_px
            pos_target["highest_price"] = highest_price
        else:
            qty = 0.0
            avg_entry = 0.0
            entry_cost = 0.0
            last_px = float(current_price) if current_price is not None else 0.0
            highest_price = 0.0
            stop_loss_price = None
            take_profit_price = None

        total_pos_value = 0.0
        for s, p in positions_map.items():
            px = last_px if (s == target_symbol and current_price is not None) else float(p["last_price"])
            total_pos_value += float(p["quantity"]) * px

        equity = cash + total_pos_value
        total_exposure_pct = (total_pos_value / equity) if equity > 0 else 0.0

        peak_stored = float(self.db.get_state("peak_equity", str(self.initial_capital)) or self.initial_capital)
        peak_equity = max(peak_stored, equity)

        # Reset diario UTC real de daily_start_equity al cambiar de fecha YYYY-MM-DD
        daily_start = float(
            self.db.get_state("daily_start_equity", str(self.initial_capital))
            or self.initial_capital
        )
        if candle_time and len(str(candle_time)) >= 10:
            new_date = str(candle_time)[:10]
            stored_date = self.db.get_state("daily_equity_date")
            if stored_date is None:
                self.db.set_state("daily_equity_date", new_date)
            elif new_date > stored_date:
                daily_start = equity
                self.db.set_state("daily_start_equity", str(equity))
                self.db.set_state("daily_equity_date", new_date)

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
            active_symbol=target_symbol,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            highest_price=highest_price,
            positions=positions_map,
            total_exposure_pct=max(0.0, total_exposure_pct),
        )

    def _compute_dynamic_alloc_pct(
        self,
        target_exposure_pct: Optional[float],
        conviction_score: Optional[float],
        vol_20: Optional[float],
    ) -> float:
        max_single = float(self.settings.max_single_position_pct)
        if target_exposure_pct is not None:
            return min(float(self.settings.max_exposure_pct), float(target_exposure_pct))
        if conviction_score is not None or vol_20 is not None:
            v = max(float(vol_20 if vol_20 is not None else 0.02), 0.005)
            vol_scale = float(np.clip(0.018 / v, 0.35, 1.0))
            conv_val = float(conviction_score if conviction_score is not None else 0.65)
            conv_scale = float(np.clip(conv_val, 0.35, 1.0))
            return float(np.clip(max_single * vol_scale * conv_scale * 1.35, 0.12, max_single))
        return min(float(self.settings.max_exposure_pct), max_single)

    def simulate_execution(
        self,
        action: str,
        symbol: str,
        market_price: float,
        target_exposure_pct: Optional[float] = None,
        atr_14: Optional[float] = None,
        conviction_score: Optional[float] = None,
        vol_20: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepara la ejecución Paper para una acción aprobada (BUY | HOLD | SELL)
        aislando la posición por `symbol` sin perder la valuación de otros activos abiertos.
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

        state = self.get_portfolio_state(current_price=market_price, symbol=symbol)
        other_positions_value = sum(
            float(p["quantity"]) * float(p["last_price"])
            for s, p in state.positions.items()
            if s != symbol
        )
        slip_mult = self.slippage_bps / 10000.0
        now = utc_now_iso()

        if action == "HOLD":
            pos_val = state.position_qty * market_price
            eq = state.cash + other_positions_value + pos_val
            peak = max(state.peak_equity, eq)
            dd = ((peak - eq) / peak) if peak > 0 else 0.0
            daily_pnl = (
                (eq - state.daily_start_equity) / state.daily_start_equity
                if state.daily_start_equity > 0
                else 0.0
            )
            highest_px = max(state.highest_price, market_price) if state.position_qty >= self.min_lot_size else 0.0
            sl_px = state.stop_loss_price
            if (
                state.position_qty >= self.min_lot_size
                and state.avg_entry_price > 0
                and highest_px > state.avg_entry_price * 1.01
            ):
                trail_sl = round(highest_px * (1.0 - self.settings.trailing_stop_pct), 4)
                sl_px = round(max(float(sl_px or 0.0), trail_sl), 4)

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
                    "last_price": market_price,
                    "highest_price": highest_px,
                    "stop_loss_price": sl_px,
                    "take_profit_price": state.take_profit_price,
                    "trailing_stop_pct": self.settings.trailing_stop_pct,
                },
                "peak_equity": peak,
            }

        if action == "BUY":
            if market_price <= 0:
                raise ValueError("El precio de mercado debe ser positivo.")
            exec_price = market_price * (1.0 + slip_mult)
            alloc_pct = self._compute_dynamic_alloc_pct(
                target_exposure_pct=target_exposure_pct,
                conviction_score=conviction_score,
                vol_20=vol_20,
            )
            remaining_portfolio_budget = max(
                0.0, state.total_equity * float(self.settings.max_exposure_pct) - other_positions_value
            )
            budget = min(state.cash, state.total_equity * alloc_pct, remaining_portfolio_budget)
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

            atr_val = float(atr_14) if atr_14 is not None and float(atr_14) > 0 else 0.0
            sl_px = (
                round(exec_price - self.settings.stop_loss_atr_mult * atr_val, 4)
                if atr_val > 0
                else round(exec_price * 0.96, 4)
            )
            tp_px = (
                round(exec_price + self.settings.take_profit_atr_mult * atr_val, 4)
                if atr_val > 0
                else round(exec_price * 1.07, 4)
            )
            highest_px = max(state.highest_price, exec_price, market_price)

            pos_val = new_qty * market_price
            eq = new_cash + other_positions_value + pos_val
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
                    "alloc_pct": round(alloc_pct, 4),
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
                    "last_price": market_price,
                    "highest_price": highest_px,
                    "stop_loss_price": sl_px,
                    "take_profit_price": tp_px,
                    "trailing_stop_pct": self.settings.trailing_stop_pct,
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

            sell_fraction = min(1.0, qty / state.position_qty) if state.position_qty > 0 else 1.0
            entry_cost_closed = state.entry_cost_total * sell_fraction
            realized_pnl = net_proceeds - entry_cost_closed

            new_qty = max(0.0, state.position_qty - qty)
            if new_qty < self.min_lot_size:
                new_qty = 0.0
                new_entry_cost = 0.0
                new_avg_entry = 0.0
                sl_px = None
                tp_px = None
                highest_px = 0.0
            else:
                new_entry_cost = max(0.0, state.entry_cost_total - entry_cost_closed)
                new_avg_entry = new_entry_cost / new_qty
                sl_px = state.stop_loss_price
                tp_px = state.take_profit_price
                highest_px = state.highest_price

            new_cash = state.cash + net_proceeds
            pos_val = new_qty * market_price
            eq = new_cash + other_positions_value + pos_val
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
                    "avg_entry_price": state.avg_entry_price,
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
                    "last_price": market_price,
                    "highest_price": highest_px,
                    "stop_loss_price": sl_px,
                    "take_profit_price": tp_px,
                    "trailing_stop_pct": self.settings.trailing_stop_pct,
                },
                "peak_equity": peak,
            }

        raise ValueError(f"Acción inválida: {action}")

    def compute_portfolio_metrics(self) -> Dict[str, Any]:
        """Calcula métricas institucionales auditables del portafolio en escala original."""
        snapshots = self.db.list_snapshots(limit=5000)
        orders = self.db.list_orders(limit=5000)
        open_positions = self.db.list_positions()

        closed_sells = [
            o for o in orders if o["side"] == "SELL" and o["realized_pnl"] is not None
        ]
        pnls = [float(o["realized_pnl"]) for o in closed_sells]
        if len(closed_sells) == 0:
            win_rate: float | str = "No definido"
            profit_factor = 0.0
            expectancy = 0.0
        else:
            wins = sum(1 for p in pnls if p > 0.0)
            win_rate = float(wins / len(pnls))
            gross_win = sum(p for p in pnls if p > 0.0)
            gross_loss = abs(sum(p for p in pnls if p < 0.0))
            profit_factor = round(gross_win / gross_loss, 3) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
            expectancy = round(float(np.mean(pnls)), 2)

        if not snapshots:
            return {
                "initial_capital": self.initial_capital,
                "current_equity": self.initial_capital,
                "net_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": win_rate,
                "closed_trades": 0,
                "total_orders": len(orders),
                "open_positions_count": len(open_positions),
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "exposure_pct": 0.0,
            }

        equities = np.asarray([float(s["total_equity"]) for s in snapshots], dtype=float)
        peaks = np.maximum.accumulate(equities)
        drawdowns = (peaks - equities) / np.where(peaks > 0, peaks, 1.0)
        max_dd = float(np.max(drawdowns))
        current_eq = float(equities[-1])
        net_ret = (current_eq - self.initial_capital) / self.initial_capital

        sharpe = 0.0
        sortino = 0.0
        if len(equities) >= 3:
            rets = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1.0)
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets))
            if std_r > 1e-8:
                sharpe = round((mean_r / std_r) * math.sqrt(252.0), 3)
            downside = rets[rets < 0]
            if len(downside) > 0:
                down_std = float(np.sqrt(np.mean(downside ** 2)))
                if down_std > 1e-8:
                    sortino = round((mean_r / down_std) * math.sqrt(252.0), 3)

        calmar = round(net_ret / max_dd, 3) if max_dd > 1e-6 else 0.0
        state_now = self.get_portfolio_state()

        return {
            "initial_capital": self.initial_capital,
            "current_equity": current_eq,
            "net_return_pct": float(net_ret),
            "max_drawdown_pct": max_dd,
            "win_rate": win_rate,
            "closed_trades": len(closed_sells),
            "total_orders": len(orders),
            "open_positions_count": len(open_positions),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "exposure_pct": round(state_now.total_exposure_pct * 100.0, 2),
        }
