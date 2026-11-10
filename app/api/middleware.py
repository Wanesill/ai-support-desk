import logging
from time import monotonic
from uuid import uuid4

from fastapi import Request, Response

from app.observability.context import request_id_context

logger = logging.getLogger(__name__)


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


async def add_request_id(
    request: Request,
    call_next,
) -> Response:
    request_id = uuid4().hex
    request.state.request_id = request_id
    context_token = request_id_context.set(request_id)
    started_at = monotonic()

    try:
        response = await call_next(request)
    except Exception as error:
        logger.error(
            "http_request_failed",
            extra={
                "method": request.method,
                "route": _route_path(request),
                "error_type": type(error).__name__,
                "response_start_seconds": (
                    monotonic() - started_at
                ),
            },
        )
        raise
    else:
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_completed",
            extra={
                "method": request.method,
                "route": _route_path(request),
                "status_code": response.status_code,
                "response_start_seconds": (
                    monotonic() - started_at
                ),
            },
        )
        return response
    finally:
        request_id_context.reset(context_token)