"""
Back Office - Private Auth Gateway (Seguridad y Control de Acceso).

Valida `X-API-Key` o sesión firmada de operador (cookie HttpOnly / X-Operator-Token)
antes de permitir:
1) Llamadas de control (/api/control/*) desde UI o clientes externos.
2) Invocaciones salientes a los agentes LLM:
   - [DISPOSITIVO LOCAL - OLLAMA /v1/systemone] Laya #1 y Laya #2
   - [API EXTERNA - NO LOCAL] Qwen 14B Orquestador

Cualquier intento sin clave o con clave inválida se registra en `error_logs`
(sin exponer jamás el secreto real) y responde HTTP 401/403.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from backend.app.back_office.database import TradingDatabase
from backend.app.core.config import Settings

SESSION_COOKIE_NAME = "trading_operator_session"


class GatewayAuthError(PermissionError):
    """Error lanzado cuando falla la autenticación del Private Auth Gateway."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class PrivateAuthGateway:
    """
    Middleware / compuerta de autenticación que protege los endpoints de control
    y media todas las invocaciones a Laya Local y Qwen API Externa.
    """

    def __init__(self, settings: Settings, db: TradingDatabase) -> None:
        self.settings = settings
        self.db = db

    @staticmethod
    def _redact_key(raw: Optional[str]) -> str:
        if not raw:
            return "<MISSING>"
        if len(raw) <= 4:
            return "***"
        return f"{raw[:2]}***{raw[-2:]}"

    def create_operator_session_token(self, ttl_seconds: int = 86400) -> str:
        """Emite un token firmado HMAC para cookie HttpOnly de la UI (sin exponer GATEWAY_API_KEY)."""
        exp = int(time.time()) + ttl_seconds
        payload = f"operator:{exp}"
        sig = hmac.new(
            self.settings.gateway_api_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}:{sig}"

    def verify_operator_session_token(self, token: Optional[str]) -> bool:
        if not token:
            return False
        parts = token.split(":")
        if len(parts) != 3 or parts[0] != "operator":
            return False
        try:
            exp = int(parts[1])
        except ValueError:
            return False
        if exp < int(time.time()):
            return False
        payload = f"{parts[0]}:{parts[1]}"
        expected_sig = hmac.new(
            self.settings.gateway_api_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(parts[2], expected_sig)

    def authenticate_operator_secret(self, credential: str) -> bool:
        """Valida la credencial ingresada por el usuario en el login de la UI."""
        if not credential:
            return False
        return hmac.compare_digest(credential, self.settings.admin_control_token) or hmac.compare_digest(
            credential, self.settings.gateway_api_key
        )

    def verify_api_key_or_session(
        self,
        x_api_key: Optional[str] = None,
        x_operator_token: Optional[str] = None,
        session_cookie: Optional[str] = None,
        endpoint: str = "internal",
    ) -> str:
        """
        Verifica credenciales del Gateway.
        - Si no se provee ninguna credencial -> HTTP 401
        - Si se provee credencial inválida -> HTTP 403
        En ambos casos deja registro en `error_logs` sin secretos.
        """
        if x_api_key is not None:
            if hmac.compare_digest(x_api_key, self.settings.gateway_api_key):
                return "api_key"
            self.db.log_error(
                error_code="AUTH_403_INVALID_KEY",
                component="PrivateAuthGateway",
                message=f"Clave X-API-Key rechazada en {endpoint}",
                payload_excerpt=f"key_preview={self._redact_key(x_api_key)}",
            )
            raise GatewayAuthError(
                status.HTTP_403_FORBIDDEN,
                "Invalid X-API-Key for Private Auth Gateway.",
            )

        if x_operator_token is not None:
            if self.authenticate_operator_secret(x_operator_token):
                return "operator_token"
            self.db.log_error(
                error_code="AUTH_403_INVALID_OPERATOR_TOKEN",
                component="PrivateAuthGateway",
                message=f"Token de operador rechazado en {endpoint}",
                payload_excerpt=f"token_preview={self._redact_key(x_operator_token)}",
            )
            raise GatewayAuthError(
                status.HTTP_403_FORBIDDEN,
                "Invalid operator credential.",
            )

        if session_cookie is not None:
            if self.verify_operator_session_token(session_cookie):
                return "operator_cookie"
            self.db.log_error(
                error_code="AUTH_403_INVALID_SESSION",
                component="PrivateAuthGateway",
                message=f"Cookie de sesión expirada o inválida en {endpoint}",
                payload_excerpt="cookie_present=True",
            )
            raise GatewayAuthError(
                status.HTTP_403_FORBIDDEN,
                "Invalid or expired operator session.",
            )

        self.db.log_error(
            error_code="AUTH_401_MISSING_CREDENTIALS",
            component="PrivateAuthGateway",
            message=f"Acceso sin credenciales al Gateway en {endpoint}",
            payload_excerpt="headers_missing=X-API-Key",
        )
        raise GatewayAuthError(
            status.HTTP_401_UNAUTHORIZED,
            "Missing X-API-Key or valid operator session.",
        )

    def authorize_llm_call(
        self,
        target_agent: str,
        execution_mode: str,
        gateway_key: Optional[str],
    ) -> None:
        """
        Compuerta obligatoria antes de invocar cualquier modelo de IA
        (Laya Local o Qwen API Externa).
        """
        if gateway_key is None or not str(gateway_key).strip():
            self.db.log_error(
                error_code="AUTH_401_LLM_GATEWAY",
                component="PrivateAuthGateway",
                message=f"Llamada bloqueada hacia {target_agent} ({execution_mode}): sin GATEWAY_API_KEY.",
                payload_excerpt=f"agent={target_agent}, mode={execution_mode}",
            )
            raise GatewayAuthError(
                status.HTTP_401_UNAUTHORIZED,
                f"Unauthorized LLM gateway access to {target_agent}.",
            )
        if not hmac.compare_digest(gateway_key, self.settings.gateway_api_key):
            self.db.log_error(
                error_code="AUTH_403_LLM_GATEWAY",
                component="PrivateAuthGateway",
                message=f"Llamada bloqueada hacia {target_agent} ({execution_mode}): clave inválida.",
                payload_excerpt=f"agent={target_agent}, key_preview={self._redact_key(gateway_key)}",
            )
            raise GatewayAuthError(
                status.HTTP_403_FORBIDDEN,
                f"Forbidden LLM gateway access to {target_agent}.",
            )


def require_gateway_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_operator_token: Optional[str] = Header(default=None, alias="X-Operator-Token"),
) -> str:
    """Dependencia FastAPI para proteger rutas /api/control/*."""
    gateway: PrivateAuthGateway = request.app.state.gateway
    session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    try:
        return gateway.verify_api_key_or_session(
            x_api_key=x_api_key,
            x_operator_token=x_operator_token,
            session_cookie=session_cookie,
            endpoint=str(request.url.path),
        )
    except GatewayAuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
