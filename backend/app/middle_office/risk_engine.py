"""
Middle Office - Motor Determinista de Riesgo (Hard Gate en Python Puro).

REGLA CRÍTICA DE GOBERNANZA:
Ningún modelo LLM (ni Qwen 14B por API externa ni Dual Laya local) puede modificar,
suavizar o evadir los límites cargados desde `config/risk_limits.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from backend.app.back_office.database import TradingDatabase
from backend.app.back_office.paper_engine import PortfolioState, floor_to_step
from backend.app.core.config import Settings


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
    ) -> RiskVerdict:
        """
        Valida todas las restricciones duras de riesgo antes de liquidar en Paper.
        Si rechaza un BUY/SELL, fuerza final_action='HOLD' (0 órdenes ejecutadas).
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

        # 2. Auditoría semántica de Laya #2
        if not laya2_approved:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=f"Veto previo de Laya #2 Local: {laya2_reason}",
                rule_triggered="LAYA2_SEMANTIC_VETO",
            )

        # 3. Símbolo en lista blanca oficial (30 PRIMARY_TICKERS + BTCUSDT)
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

        # 7b. Exposición máxima (sin apalancamiento)
        current_exposure = (
            (portfolio.position_qty * market_price) / portfolio.total_equity
            if portfolio.total_equity > 0
            else 0.0
        )
        if current_exposure >= self._settings.max_exposure_pct - 1e-6:
            return RiskVerdict(
                approved=False,
                status="REJECTED",
                final_action="HOLD",
                reason=(
                    f"Exposición actual ({current_exposure:.2%}) alcanzó el máximo "
                    f"permitido ({self._settings.max_exposure_pct:.2%})."
                ),
                rule_triggered="MAX_EXPOSURE",
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
