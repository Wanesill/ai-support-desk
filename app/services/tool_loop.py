from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal, Protocol

from app.llm.client import (
    LLMResult,
    LLMStreamCompleted,
    LLMStreamEvent,
    LLMTextDelta,
    Message,
    ToolCall,
    ToolDefinition,
)
from app.llm.errors import LLMClientError
from app.tools.execution import (
    ToolAuditRecord,
    ToolConfirmationNotFoundError,
    ToolConfirmationRequest,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutor,
)
from app.tools.registry import ToolRegistry


class ToolCallingClient(Protocol):
    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> LLMResult: ...
    def stream(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            tool_choice: Literal["auto", "none"] = "auto",
    ) -> Generator[LLMStreamEvent, None, None]:
        ...


class ToolLoopError(RuntimeError):
    pass


class ToolLoopProtocolError(ToolLoopError):
    pass


class ToolLoopLimitError(ToolLoopError):
    pass


class ToolLoopExecutionError(ToolLoopError):
    def __init__(self, message: str, audit: ToolAuditRecord) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_ids: tuple[str, ...]
    model_steps: int
    tool_calls_count: int
    tool_names: tuple[str, ...]
    audit_records: tuple[ToolAuditRecord, ...]
    confirmation: ToolConfirmationRequest | None


@dataclass(frozen=True)
class ToolLoopTextDelta:
    text: str


@dataclass(frozen=True)
class ToolLoopStreamCompleted:
    result: ToolLoopResult


ToolLoopStreamEvent = ToolLoopTextDelta | ToolLoopStreamCompleted


class BoundedToolLoop:
    def __init__(
        self,
        client: ToolCallingClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        max_steps: int = 3,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps должен быть больше нуля.")

        self._client = client
        self._registry = registry
        self._executor = executor
        self._max_steps = max_steps

    def confirm(
        self,
        confirmation_id: str,
    ) -> ToolExecutionResult:
        try:
            return self._executor.confirm(confirmation_id)
        except ToolConfirmationNotFoundError as error:
            raise ToolLoopError(str(error)) from error
        except ToolExecutionError as error:
            raise ToolLoopExecutionError(
                str(error),
                error.audit,
            ) from error

    def cancel(
        self,
        confirmation_id: str,
    ) -> ToolAuditRecord:
        try:
            return self._executor.cancel(confirmation_id)
        except ToolConfirmationNotFoundError as error:
            raise ToolLoopError(str(error)) from error

    def run(self, messages: list[Message]) -> ToolLoopResult:
        completed: ToolLoopStreamCompleted | None = None

        for event in self.stream(messages):
            if isinstance(event, ToolLoopStreamCompleted):
                completed = event

        if completed is None:
            raise ToolLoopProtocolError(
                "Поток завершился без итогового результата."
            )

        return completed.result

    def stream(
            self,
            messages: list[Message],
    ) -> Generator[ToolLoopStreamEvent, None, None]:
        conversation = list(messages)
        llm_results: list[LLMResult] = []
        tool_names: list[str] = []
        audit_records: list[ToolAuditRecord] = []

        for step in range(1, self._max_steps + 1):
            text_parts: list[str] = []
            completed: LLMStreamCompleted | None = None

            try:
                events = self._client.stream(
                    messages=conversation,
                    tools=self._registry.api_schemas(),
                    tool_choice="auto",
                )

                try:
                    for event in events:
                        if isinstance(event, LLMTextDelta):
                            text_parts.append(event.text)
                            yield ToolLoopTextDelta(text=event.text)
                        else:
                            completed = event
                finally:
                    events.close()
            except LLMClientError as error:
                raise ToolLoopError(
                    f"Не удалось обратиться к модели: {error}"
                ) from error

            if completed is None:
                raise ToolLoopProtocolError(
                    "Поток модели завершился без итогового события."
                )

            result = LLMResult(
                text="".join(text_parts) or None,
                refusal=completed.refusal,
                tool_calls=completed.tool_calls,
                model=completed.model,
                finish_reason=completed.finish_reason,
                prompt_tokens=completed.prompt_tokens,
                completion_tokens=completed.completion_tokens,
                total_tokens=completed.total_tokens,
                response_id=completed.response_id,
            )
            llm_results.append(result)

            if result.refusal is not None:
                raise ToolLoopProtocolError(
                    f"Модель отказалась отвечать: {result.refusal}"
                )

            if result.finish_reason == "stop":
                if result.tool_calls:
                    raise ToolLoopProtocolError(
                        "Текстовый ответ содержит tool calls."
                    )

                text = self._require_text(result)
                yield ToolLoopStreamCompleted(
                    result=self._build_result(
                        text=text,
                        llm_results=llm_results,
                        tool_names=tool_names,
                        audit_records=audit_records,
                        confirmation=None,
                    )
                )
                return

            if result.finish_reason != "tool_calls":
                raise ToolLoopProtocolError(
                    "Неожиданная причина завершения: "
                    f"{result.finish_reason}."
                )

            if len(result.tool_calls) != 1:
                raise ToolLoopProtocolError(
                    "Ожидался ровно один tool call на шаг."
                )

            if text_parts:
                raise ToolLoopProtocolError(
                    "Вызов инструмента неожиданно содержит текст."
                )

            if step == self._max_steps:
                raise ToolLoopLimitError(
                    "Модель запросила инструмент на последнем "
                    "разрешённом шаге."
                )

            tool_call = result.tool_calls[0]
            conversation.append(
                self._build_tool_call_message(result, tool_call)
            )

            try:
                execution = self._executor.execute(tool_call)
            except ToolConfirmationRequiredError as error:
                audit_records.append(error.audit)
                confirmation_text = self._build_confirmation_text(
                    error.request
                )
                yield ToolLoopTextDelta(text=confirmation_text)
                yield ToolLoopStreamCompleted(
                    result=self._build_result(
                        text=confirmation_text,
                        llm_results=llm_results,
                        tool_names=tool_names,
                        audit_records=audit_records,
                        confirmation=error.request,
                    )
                )
                return
            except ToolExecutionError as error:
                raise ToolLoopExecutionError(
                    str(error),
                    error.audit,
                ) from error

            tool_names.append(tool_call.name)
            audit_records.append(execution.audit)
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": execution.content,
                }
            )

        raise ToolLoopLimitError(
            "Цикл завершился без финального ответа."
        )

    @staticmethod
    def _build_tool_call_message(result: LLMResult, tool_call: ToolCall) -> Message:
        return {
            "role": "assistant",
            "content": result.text,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            ],
        }

    @staticmethod
    def _require_text(result: LLMResult) -> str:
        if result.text is None:
            raise ToolLoopProtocolError("Модель не вернула текст.")

        text = result.text.strip()
        if not text:
            raise ToolLoopProtocolError("Модель вернула пустой текст.")
        return text

    @staticmethod
    def _build_confirmation_text(
        request: ToolConfirmationRequest,
    ) -> str:
        return (
            "Подготовлено действие, которое изменяет данные.\n"
            f"Инструмент: {request.tool_name}\n"
            f"Аргументы: {request.arguments_json}\n"
            "Действие ожидает подтверждения пользователя."
        )

    @staticmethod
    def _build_result(
        text: str,
        llm_results: list[LLMResult],
        tool_names: list[str],
        audit_records: list[ToolAuditRecord],
        confirmation: ToolConfirmationRequest | None,
    ) -> ToolLoopResult:
        def sum_usage(values: list[int | None]) -> int | None:
            if any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        return ToolLoopResult(
            text=text,
            model=llm_results[-1].model,
            prompt_tokens=sum_usage([result.prompt_tokens for result in llm_results]),
            completion_tokens=sum_usage(
                [result.completion_tokens for result in llm_results]
            ),
            total_tokens=sum_usage([result.total_tokens for result in llm_results]),
            response_ids=tuple(result.response_id for result in llm_results),
            model_steps=len(llm_results),
            tool_calls_count=len(tool_names),
            tool_names=tuple(tool_names),
            audit_records=tuple(audit_records),
            confirmation=confirmation,
        )
