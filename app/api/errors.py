from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas.errors import (
    ErrorContentResponse,
    ErrorResponse,
    ValidationIssueResponse,
)


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[ValidationIssueResponse] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorContentResponse(
            code=code,
            message=message,
            request_id=_get_request_id(request),
            details=details,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(exclude_none=True),
    )


async def api_error_handler(
    request: Request,
    error: ApiError,
) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=error.status_code,
        code=error.code,
        message=error.message,
    )


async def validation_error_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    details = [
        ValidationIssueResponse(
            field=".".join(str(part) for part in issue["loc"]),
            message=issue["msg"],
        )
        for issue in error.errors()
    ]
    return _error_response(
        request=request,
        status_code=422,
        code="validation_error",
        message="Запрос не соответствует контракту.",
        details=details,
    )


def register_error_handlers(application: FastAPI) -> None:
    application.add_exception_handler(ApiError, api_error_handler)
    application.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )