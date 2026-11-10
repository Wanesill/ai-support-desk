import json
import logging
import sys
from datetime import UTC, datetime

from app.observability.context import request_id_context

LOG_FIELDS = (
    "request_id",
    "method",
    "route",
    "status_code",
    "response_start_seconds",
    "duration_seconds",
    "operation",
    "model",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost",
    "cost_currency",
    "response_id",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id is None:
            request_id = request_id_context.get()

        if request_id is not None:
            payload["request_id"] = request_id

        for field_name in LOG_FIELDS:
            if field_name == "request_id":
                continue

            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value

        return json.dumps(payload, ensure_ascii=False)


def configure_application_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    application_logger = logging.getLogger("app")
    application_logger.handlers.clear()
    application_logger.addHandler(handler)
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False