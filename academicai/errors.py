"""
AcademicAI Fehler-Klassen (kein LiteLLM)
"""


class AcademicAIError(Exception):
    """Basis-Exception für alle AcademicAI-Fehler."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class AuthenticationError(AcademicAIError):
    pass


class PermissionDeniedError(AcademicAIError):
    pass


class NotFoundError(AcademicAIError):
    pass


class BadRequestError(AcademicAIError):
    pass


class ServiceUnavailableError(AcademicAIError):
    pass


def map_error(status_code: int, response_body: dict) -> AcademicAIError:
    """
    Wandelt AcademicAI HTTP-Fehlercodes in passende Exceptions um.
    """
    message = response_body.get("message") or response_body.get("error") or str(response_body)
    error_code = response_body.get("code", "")

    if error_code == "KB_UNAVAILABLE":
        return ServiceUnavailableError(
            f"AcademicAI Knowledge Base nicht verfügbar: {message}", status_code
        )

    mapping = {
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        422: BadRequestError,
    }

    exc_class = mapping.get(status_code, AcademicAIError)
    return exc_class(f"AcademicAI API Fehler {status_code}: {message}", status_code)
