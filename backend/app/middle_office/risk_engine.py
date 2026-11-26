"""
Middle Office - Motor Determinista de Riesgo (Hard Gate en Python Puro).

REGLA CRÍTICA DE GOBERNANZA:
Ningún modelo LLM (ni Qwen por API/Local ni Dual Laya local) puede modificar,
suavizar o evadir los límites cargados desde `config/risk_limits.yaml`.
Incluye protección determinista Stop-Loss ATR, Trailing Stop y Take-Profit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.paper_engine import PortfolioState, floor_to_step
from backend.app.core.config import Settings

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


_US_NYSE_HOLIDAYS = {
    # 2025-2027 principales feriados NYSE/NASDAQ (YYYY-MM-DD)
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def _to_us_eastern(utc_dt: datetime) -> datetime:
    dt_utc = utc_dt.astimezone(timezone.utc) if utc_dt.tzinfo else utc_dt.replace(tzinfo=timezone.utc)
    if ZoneInfo is not None:
        try:
            return dt_utc.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            pass
    year = dt_utc.year
    # 2do domingo de marzo 07:00 UTC -> inicio EDT (UTC-4)
    mar1 = datetime(year, 3, 1, 7, 0, tzinfo=timezone.utc)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    # 1er domingo de noviembre 06:00 UTC -> fin EDT (UTC-5)
    nov1 = datetime(year, 11, 1, 6, 0, tzinfo=timezone.utc)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    offset = -4 if dst_start <= dt_utc < dst_end else -5
    return dt_utc.astimezone(timezone(timedelta(hours=offset)))


def get_market_session_status(
    symbol: str, now_utc: Optional[datetime] = None
) -> Dict[str, Any]:
    """Determina si el mercado del símbolo está abierto para operar."""
    sym = str(symbol or "").upper()
    if sym in {"BTC-USD", "BTCUSDT"}:
        return {
            "symbol": sym,
            "is_open": True,
            "market": "CRYPTO_24_7",
            "et_time": _to_us_eastern(now_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M ET"),
            "session_label": "ABIERTO 24/7 (MEXC Spot)",
            "reason": "Mercado cripto operativo las 24 horas.",
        }

    et_dt = _to_us_eastern(now_utc or datetime.now(timezone.utc))
    et_str = et_dt.strftime("%Y-%m-%d %H:%M ET")
    date_str = et_dt.strftime("%Y-%m-%d")

    if et_dt.weekday() >= 5 or date_str in _US_NYSE_HOLIDAYS:
        return {
            "symbol": sym,
            "is_open": False,
            "market": "US_EQUITIES",
            "et_time": et_str,
            "session_label": f"CERRADO (Fin de semana/Feriado • {et_str})",
            "reason": f"Bolsa NYSE/NASDAQ cerrada ({et_str}).",
        }

    mins = et_dt.hour * 60 + et_dt.minute
    if 570 <= mins < 960:  # 09:30 <= t < 16:00 ET
        return {
            "symbol": sym,
            "is_open": True,
            "market": "US_EQUITIES",
            "et_time": et_str,
            "session_label": f"ABIERTO NYSE/NASDAQ ({et_str})",
            "reason": f"Sesión regular activa ({et_str}).",
        }

    phase = "Pre-Market" if mins < 570 else "After-Hours"
    return {
        "symbol": sym,
        "is_open": False,
        "market": "US_EQUITIES",
        "et_time": et_str,
        "session_label": f"CERRADO ({phase} • {et_str} | Horario 09:30–16:00 ET)",
        "reason": f"Mercado regular cerrado ({phase}, {et_str}). Horario operativo: 09:30–16:00 ET.",
    }


@dataclass(frozen=True)
class RiskVerdict:
    approved: bool
    status: str  # APPROVED | REJECTED
    final_action: str  # BUY | HOLD | SELL
    reason: str
    rule_triggered: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "status": self.status,
            "final_action": self.final_action,
            "reason": self.reason,
            "rule_triggered": self.rule_triggered,
        }


class DeterministicRiskEngine:
    """Compuerta dura determinista (NO LLM) de Middle Office."""

    def __init__(self, settings: Settings, db: TradingDatabase) -> None:
        self._settings = settings
        self._db = db

    def evaluate(
        self,
        proposed_action: str,
        symbol: str,
        market_price: float,
        portfolio: PortfolioState,
        system_status: str,
        laya2_approved: bool = True,
        laya2_reason: str = "",
        atr_14: Optional[float] = None,
        candle_low: Optional[float] = None,
        candle_high: Optional[float] = None,
        source_mode: str = "replay",
        now_utc: Optional[datetime] = None,
    ) -> RiskVerdict:
        """
        Valida todas las restricciones duras de riesgo antes de liquidar en Paper.
        Si rechaza un BUY/SELL, fuerza final_action='HOLD' (0 órdenes ejecutadas).
        Si una posición abierta activa Stop-Loss, Trailing Stop o Take-Profit,
        fuerza salida protectora SELL determinista incluso si el LLM propuso HOLD.
        """
        # 1. Kill-Switch Humano / Estado del Sistema
        if system_status == "STOPPED":
            if proposed_action in {"BUY", "SELL"}:
                self._db.log_error(
                    error_code="RISK_KILL_SWITCH_STOP",
                    component="DeterministicRiskEngine",
                    message=f"Orden {proposed_action} bloqueada por Kill-Switch STOP (posición preservada).",
                    payload_excerpt=f"symbol={symbol}, pos={portfolio.position_qty}",
                )
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason="Sistema detenido (STOPPED): nuevas órdenes bloqueadas preservando posición actual.",
                rule_triggered="KILL_SWITCH_STOPPED",
            )

        if system_status == "PAUSED":
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason="Sistema en pausa (PAUSED): no se ejecutan órdenes.",
                rule_triggered="SYSTEM_PAUSED",
            )

        # Símbolo en lista blanca oficial (30 PRIMARY_TICKERS + BTCUSDT)
        if symbol not in self._settings.allowed_symbols:
            self._db.log_error(
                error_code="RISK_SYMBOL_NOT_WHITELISTED",
                component="DeterministicRiskEngine",
                message=f"Símbolo no autorizado: {symbol}",
                payload_excerpt=f"symbol={symbol}",
            )
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=f"Símbolo {symbol} fuera de la lista blanca oficial.",
                rule_triggered="WHITELIST_SYMBOL",
            )

        # 1a. Horario de mercado (en modo 'live' bloquea acciones si NYSE/NASDAQ está cerrado)
        if source_mode == "live":
            mkt_status = get_market_session_status(symbol, now_utc=now_utc)
            if not mkt_status["is_open"]:
                if proposed_action in {"BUY", "SELL"}:
                    self._db.log_error(
                        error_code="RISK_MARKET_CLOSED",
                        component="DeterministicRiskEngine",
                        message=f"Operación {proposed_action} bloqueada en {symbol}: {mkt_status['session_label']}",
                        payload_excerpt=f"symbol={symbol}, et_time={mkt_status['et_time']}",
                    )
                    return RiskVerdict(
                        approved=False,
                        status="REJECTED",
                        final_action="HOLD",
                        reason=f"Mercado cerrado para {symbol}: {mkt_status['session_label']}.",
                        rule_triggered="MARKET_CLOSED",
                    )
                return RiskVerdict(
                    approved=True,
                    status="APPROVED",
                    final_action="HOLD",
                    reason=f"Mercado cerrado ({mkt_status['session_label']}): posición en espera sin operar.",
                    rule_triggered="MARKET_CLOSED",
                )

        # 1b. Salidas Protectoras Deterministas (Stop-Loss / Trailing Stop / Take-Profit)
        has_open_pos = portfolio.position_qty >= self._settings.min_lot_size
        use_candle_extremes = source_mode != "live"
        if has_open_pos and market_price > 0 and system_status == "RUNNING":
            sl = portfolio.stop_loss_price
            if sl is not None and (
                market_price <= sl
                or (use_candle_extremes and candle_low is not None and candle_low <= sl)
            ):
                return RiskVerdict(
                    approved=True,
                    status="APPROVED",
                    final_action="SELL",
                    reason=(
                        f"Stop-Loss determinista activado en {symbol}: "
                        f"precio {market_price:.2f} (min={(candle_low if use_candle_extremes and candle_low is not None else market_price):.2f}) <= SL {sl:.2f}."
                    ),
                    rule_triggered="STOP_LOSS_TRIGGERED",
                )

            if (
                portfolio.avg_entry_price > 0
                and portfolio.highest_price > portfolio.avg_entry_price * 1.01
            ):
                trail_px = portfolio.highest_price * (1.0 - self._settings.trailing_stop_pct)
                if market_price <= trail_px:
                    return RiskVerdict(
                        approved=True,
                        status="APPROVED",
                        final_action="SELL",
                        reason=(
                            f"Trailing Stop determinista activado en {symbol}: "
                            f"precio {market_price:.2f} <= {trail_px:.2f} "
                            f"(máximo={portfolio.highest_price:.2f})."
                        ),
                        rule_triggered="TRAILING_STOP_TRIGGERED",
                    )

            tp = portfolio.take_profit_price
            if tp is not None and (
                market_price >= tp
                or (use_candle_extremes and candle_high is not None and candle_high >= tp)
            ):
                return RiskVerdict(
                    approved=True,
                    status="APPROVED",
                    final_action="SELL",
                    reason=(
                        f"Take-Profit determinista activado en {symbol}: "
                        f"precio {market_price:.2f} (máx={(candle_high if use_candle_extremes and candle_high is not None else market_price):.2f}) >= TP {tp:.2f}."
                    ),
                    rule_triggered="TAKE_PROFIT_TRIGGERED",
                )

        # 2. Auditoría semántica de Laya #2
        if not laya2_approved:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=f"Veto previo de Laya #2 Local: {laya2_reason}",
                rule_triggered="LAYA2_SEMANTIC_VETO",
            )

        # 4. Acción válida
        if proposed_action not in {"BUY", "HOLD", "SELL"}:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=f"Acción no permitida: {proposed_action}",
                rule_triggered="INVALID_ACTION",
            )

        if proposed_action == "HOLD":
            return RiskVerdict(
                approved=True,
                status="APPROVED",
                final_action="HOLD",
                reason="Sin operación requerida (HOLD aprobado).",
                rule_triggered="NONE",
            )

        # 5. Precio válido
        if market_price <= 0:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason="Precio de mercado inválido (<= 0).",
                rule_triggered="INVALID_PRICE",
            )

        # 6. Long-Only: prohibido SELL sin posición y prohibido short-selling
        if proposed_action == "SELL":
            if portfolio.position_qty < self._settings.min_lot_size:
                self._db.log_error(
                    error_code="RISK_SHORT_SELL_BLOCKED",
                    component="DeterministicRiskEngine",
                    message=f"Intento de SELL sin posición abierta en {symbol} (Long-Only).",
                    payload_excerpt=f"pos_qty={portfolio.position_qty}",
                )
                return RiskVerdict(
                    approved=False,
                    status="REJECTED",
                    final_action="HOLD",
                    reason="Restricción Long-Only: prohibido vender sin posición abierta.",
                    rule_triggered="LONG_ONLY_NO_SHORT",
                )
            return RiskVerdict(
                approved=True,
                status="APPROVED",
                final_action="SELL",
                reason="Venta Long-Only aprobada dentro de límites deterministas.",
                rule_triggered="NONE",
            )

        # 7. Controles para BUY
        # 7a. Pérdida diaria máxima
        if portfolio.daily_pnl_pct <= -self._settings.max_daily_loss_pct:
            self._db.log_error(
                error_code="RISK_MAX_DAILY_LOSS",
                component="DeterministicRiskEngine",
                message=(
                    f"Límite de pérdida diaria alcanzado ({portfolio.daily_pnl_pct:.2%} "
                    f"<= -{self._settings.max_daily_loss_pct:.2%})."
                ),
                payload_excerpt=f"symbol={symbol}, equity={portfolio.total_equity:.2f}",
            )
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=(
                    f"Límite de pérdida diaria excedido ({portfolio.daily_pnl_pct:.2%}). "
                    "Nuevas compras bloqueadas."
                ),
                rule_triggered="MAX_DAILY_LOSS",
            )

        # 7b. Exposición máxima total del portafolio y por activo individual
        single_exposure = (
            (portfolio.position_qty * market_price) / portfolio.total_equity
            if portfolio.total_equity > 0
            else 0.0
        )
        total_exposure = max(float(portfolio.total_exposure_pct), single_exposure)
        if total_exposure >= self._settings.max_exposure_pct - 1e-6:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=(
                    f"Exposición total del portafolio ({total_exposure:.2%}) alcanzó el máximo "
                    f"permitido ({self._settings.max_exposure_pct:.2%})."
                ),
                rule_triggered="MAX_EXPOSURE",
            )

        if single_exposure >= self._settings.max_single_position_pct - 1e-6:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=(
                    f"Exposición en {symbol} ({single_exposure:.2%}) alcanzó el límite por activo "
                    f"({self._settings.max_single_position_pct:.2%})."
                ),
                rule_triggered="MAX_SINGLE_POSITION_EXPOSURE",
            )

        # 7c. Saldo suficiente para lote mínimo considerando slippage y comisión
        slip_mult = self._settings.slippage_bps / 10000.0
        fee_pct = self._settings.profile_config.fee_pct
        exec_price = market_price * (1.0 + slip_mult)
        max_affordable_qty = floor_to_step(
            portfolio.cash / (exec_price * (1.0 + fee_pct)),
            self._settings.step_size,
        )
        if max_affordable_qty < self._settings.min_lot_size or portfolio.cash <= 0.0:
            self._db.log_error(
                error_code="RISK_INSUFFICIENT_FUNDS",
                component="DeterministicRiskEngine",
                message=(
                    f"Saldo insuficiente ({portfolio.cash:.2f}) para lote mínimo "
                    f"{self._settings.min_lot_size} de {symbol} a precio {exec_price:.2f}."
                ),
                payload_excerpt=f"cash={portfolio.cash:.2f}, max_qty={max_affordable_qty}",
            )
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=(
                    f"Saldo insuficiente (${portfolio.cash:.2f}) para comprar lote mínimo "
                    f"de {symbol}."
                ),
                rule_triggered="INSUFFICIENT_FUNDS",
            )

        return RiskVerdict(
            approved=True,
            status="APPROVED",
            final_action="BUY",
            reason="Compra Long-Only aprobada: saldo, exposición y pérdida diaria en regla.",
            rule_triggered="NONE",
        )
