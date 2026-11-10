import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from app.llm.client import ToolCall
from app.tools.registry import (
    RegisteredTool,
    ToolAccess,
    ToolNotRegisteredError,
    ToolRegistry,
)


class ToolBusinessError(RuntimeError):
    def __init__(self, message: str, code: str = "business_error") -> None:
        super().__init__(message)
        self.code = code


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, audit: "ToolAuditRecord") -> None:
        super().__init__(message)
        self.audit = audit


class ToolNotAllowedError(ToolExecutionError):
    pass


class ToolArgumentsError(ToolExecutionError):
    pass


class ToolTimeoutError(ToolExecutionError):
    pass


class ToolTechnicalError(ToolExecutionError):
    pass


@dataclass(frozen=True)
class ToolConfirmationRequest:
    confirmation_id: str
    tool_name: str
    arguments_json: str


class ToolConfirmationRequiredError(ToolExecutionError):
    def __init__(
        self,
        message: str,
        audit: "ToolAuditRecord",
        request: ToolConfirmationRequest,
    ) -> None:
        super().__init__(message, audit)
        self.request = request


class ToolConfirmationNotFoundError(RuntimeError):
    pass


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    BUSINESS_ERROR = "business_error"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TECHNICAL_ERROR = "technical_error"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolAuditRecord:
    call_id: str
    tool_name: str
    started_at: str
    duration_seconds: float
    status: ToolExecutionStatus
    error_code: str | None


@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    audit: ToolAuditRecord


@dataclass(frozen=True)
class _PendingToolExecution:
    tool_call: ToolCall
    tool: RegisteredTool[Any, Any]
    arguments: BaseModel


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError("Количество workers должно быть больше нуля.")

        self._registry = registry
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tool-executor"
        )
        self._pending: dict[str, _PendingToolExecution] = {}
        self._pending_lock = Lock()

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()

        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotRegisteredError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.REJECTED,
                error_code="unknown_tool",
            )
            raise ToolNotAllowedError(
                "Модель запросила неизвестный инструмент.", audit
            ) from error

        try:
            arguments = tool.arguments_model.model_validate_json(tool_call.arguments)
        except ValidationError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.REJECTED,
                error_code="invalid_arguments",
            )
            raise ToolArgumentsError(
                "Аргументы инструмента не прошли проверку.", audit
            ) from error

        if tool.access is ToolAccess.WRITE:
            confirmation_id = uuid4().hex
            request = ToolConfirmationRequest(
                confirmation_id=confirmation_id,
                tool_name=tool.name,
                arguments_json=arguments.model_dump_json(),
            )
            pending = _PendingToolExecution(
                tool_call=tool_call,
                tool=tool,
                arguments=arguments,
            )

            with self._pending_lock:
                self._pending[confirmation_id] = pending

            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=(ToolExecutionStatus.CONFIRMATION_REQUIRED),
                error_code="confirmation_required",
            )
            raise ToolConfirmationRequiredError(
                "Действие требует подтверждения пользователя.",
                audit,
                request,
            )

        return self._execute_handler(
            tool_call=tool_call,
            tool=tool,
            arguments=arguments,
            started_at=started_at,
            started=started,
        )

    def confirm(
        self,
        confirmation_id: str,
    ) -> ToolExecutionResult:
        pending = self._take_pending(confirmation_id)
        return self._execute_handler(
            tool_call=pending.tool_call,
            tool=pending.tool,
            arguments=pending.arguments,
            started_at=datetime.now(UTC).isoformat(),
            started=perf_counter(),
        )

    def cancel(
        self,
        confirmation_id: str,
    ) -> ToolAuditRecord:
        pending = self._take_pending(confirmation_id)
        started = perf_counter()
        return self._build_audit(
            tool_call=pending.tool_call,
            started_at=datetime.now(UTC).isoformat(),
            started=started,
            status=ToolExecutionStatus.CANCELLED,
            error_code="user_cancelled",
        )

    def _take_pending(
        self,
        confirmation_id: str,
    ) -> _PendingToolExecution:
        with self._pending_lock:
            try:
                return self._pending.pop(confirmation_id)
            except KeyError as error:
                raise ToolConfirmationNotFoundError(
                    "Подготовленное действие не найдено."
                ) from error

    def _execute_handler(
        self,
        tool_call: ToolCall,
        tool: RegisteredTool[Any, Any],
        arguments: BaseModel,
        started_at: str,
        started: float,
    ) -> ToolExecutionResult:
        future = self._pool.submit(tool.handler, arguments)

        try:
            result = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.TIMEOUT,
                error_code="timeout",
            )
            raise ToolTimeoutError(
                "Инструмент не завершился вовремя.",
                audit,
            ) from error
        except ToolBusinessError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.BUSINESS_ERROR,
                error_code=error.code,
            )
            return ToolExecutionResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": error.code,
                            "message": str(error),
                        },
                    },
                    ensure_ascii=False,
                ),
                audit=audit,
            )
        except Exception as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=(ToolExecutionStatus.TECHNICAL_ERROR),
                error_code="technical_error",
            )
            raise ToolTechnicalError(
                "Инструмент завершился технической ошибкой.",
                audit,
            ) from error

        if not isinstance(result, BaseModel):
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=(ToolExecutionStatus.TECHNICAL_ERROR),
                error_code="invalid_result_type",
            )
            raise ToolTechnicalError(
                "Инструмент вернул результат неверного типа.",
                audit,
            )

        audit = self._build_audit(
            tool_call=tool_call,
            started_at=started_at,
            started=started,
            status=ToolExecutionStatus.SUCCESS,
            error_code=None,
        )
        return ToolExecutionResult(
            content=json.dumps(
                {
                    "ok": True,
                    "data": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            audit=audit,
        )

    @staticmethod
    def _build_audit(
        tool_call: ToolCall,
        started_at: str,
        started: float,
        status: ToolExecutionStatus,
        error_code: str | None,
    ) -> ToolAuditRecord:
        return ToolAuditRecord(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            started_at=started_at,
            duration_seconds=perf_counter() - started,
            status=status,
            error_code=error_code,
        )

    def close(self) -> None:
        with self._pending_lock:
            self._pending.clear()

        self._pool.shutdown(wait=True, cancel_futures=True)
