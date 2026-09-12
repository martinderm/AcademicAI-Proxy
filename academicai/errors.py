"""
AcademicAI Fehler-Klassen (kein LiteLLM)
"""
from typing import Any, Optional


class AcademicAIError(Exception):
    """Basis-Exception für alle AcademicAI-Fehler."""
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_code: str = "api_error",
        error_type: str = "api_error",
        param: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.param = param


class AuthenticationError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 401,
        error_code: str = "invalid_api_key",
        error_type: str = "authentication_error",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


class PermissionDeniedError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 403,
        error_code: str = "permission_denied",
        error_type: str = "permission_error",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


class QuotaExceededError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 429,
        error_code: str = "insufficient_quota",
        error_type: str = "insufficient_quota",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


class NotFoundError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 404,
        error_code: str = "model_not_found",
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


class BadRequestError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: str = "invalid_request_error",
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


class ServiceUnavailableError(AcademicAIError):
    def __init__(
        self,
        message: str,
        status_code: int = 503,
        error_code: str = "service_unavailable",
        error_type: str = "api_error",
        param: Optional[str] = None,
    ):
        super().__init__(message, status_code, error_code, error_type, param)


def map_error(status_code: int, response_body: Any) -> AcademicAIError:
    """
    Wandelt AcademicAI HTTP-Fehlercodes in passende Exceptions um.
    Extrahiert tiefe Fehlermeldungen aus meta.error falls vorhanden.
    """
    detailed_msg = None
    internal_code = None
    code_str = ""

    if isinstance(response_body, dict):
        # 1. Versuche tief geschachtelte meta.error Struktur
        meta = response_body.get("meta")
        if isinstance(meta, dict):
            meta_err = meta.get("error")
            if isinstance(meta_err, dict):
                detailed_msg = meta_err.get("message")
                internal_code = meta_err.get("internalErrorCode")
                if not code_str:
                    code_str = str(meta_err.get("code") or "")

        # 2. Versuche error Feld
        if not detailed_msg:
            err_field = response_body.get("error")
            if isinstance(err_field, dict):
                detailed_msg = err_field.get("message")
                if not code_str:
                    code_str = str(err_field.get("code") or "")
            elif isinstance(err_field, str):
                detailed_msg = err_field

        # 3. Versuche message / detail Feld
        if not detailed_msg:
            detailed_msg = response_body.get("message") or response_body.get("detail")

        if not code_str:
            code_str = str(response_body.get("code") or "")

    if not detailed_msg:
        detailed_msg = str(response_body) if response_body else "Unknown upstream error"

    # Spezifische Behandlung: Knowledge Base unavailable
    if code_str == "KB_UNAVAILABLE":
        return ServiceUnavailableError(
            f"AcademicAI Knowledge Base nicht verfügbar: {detailed_msg}",
            status_code=status_code or 503,
            error_code="kb_unavailable",
        )

    msg_lower = detailed_msg.lower()

    # Quota / Budget / Cost Limit Reached
    # Upstream: internalErrorCode 201, oder "Cost limit reached", oder 429
    if (
        internal_code == 201
        or "cost limit reached" in msg_lower
        or "cost limit" in msg_lower
        or "quota exceeded" in msg_lower
        or "insufficient_quota" in msg_lower
        or status_code == 429
        or code_str in ("INSUFFICIENT_QUOTA", "QUOTA_EXCEEDED", "COST_LIMIT_REACHED")
    ):
        return QuotaExceededError(
            f"AcademicAI Cost Limit Reached: {detailed_msg}",
            status_code=429,
            error_code="insufficient_quota",
            error_type="insufficient_quota",
        )

    mapping = {
        401: (AuthenticationError, "invalid_api_key", "authentication_error"),
        403: (PermissionDeniedError, "permission_denied", "permission_error"),
        404: (NotFoundError, "model_not_found", "invalid_request_error"),
        422: (BadRequestError, "invalid_request_error", "invalid_request_error"),
        503: (ServiceUnavailableError, "service_unavailable", "api_error"),
        504: (ServiceUnavailableError, "gateway_timeout", "api_error"),
    }

    if status_code in mapping:
        exc_cls, err_code, err_type = mapping[status_code]
        return exc_cls(
            f"AcademicAI API Fehler {status_code}: {detailed_msg}",
            status_code=status_code,
            error_code=err_code,
            error_type=err_type,
        )

    return AcademicAIError(
        f"AcademicAI API Fehler {status_code}: {detailed_msg}",
        status_code=status_code or 502,
        error_code="api_error",
        error_type="api_error",
    )
