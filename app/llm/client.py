from collections.abc import Callable, Generator
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from app.config import Settings
from app.llm.errors import (
    LLMAccessError,
    LLMBalanceError,
    LLMClientError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
)
from app.llm.limits import LLMConcurrencyLimiter
from app.observability.llm import LLMObserver

Message = dict[str, Any]
JsonSchema = dict[str, Any]
ToolDefinition = dict[str, Any]
VisionContentPart = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResult:
    text: str | None
    refusal: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


@dataclass(frozen=True)
class LLMTextDelta:
    text: str


@dataclass(frozen=True)
class LLMStreamCompleted:
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


LLMStreamEvent = LLMTextDelta | LLMStreamCompleted


class LLMClient:
    def __init__(self, settings: Settings, observer: LLMObserver | None = None) -> None:
        self._settings = settings
        self._observer = observer
        self._limiter = LLMConcurrencyLimiter(
            max_concurrent_requests=(
                settings.llm_max_concurrent_requests
            ),
            queue_timeout_seconds=settings.llm_queue_timeout_seconds,
        )
        self._client = OpenAI(
            base_url=str(settings.base_url),
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
        )

    def generate(self, messages: list[Message]) -> LLMResult:
        return self._run_observed(
            "generate",
            lambda: self._generate_once(messages),
        )

    def generate_structured(
            self,
            messages: list[Message],
            schema_name: str,
            schema: JsonSchema,
    ) -> LLMResult:
        return self._run_observed(
            "generate_structured",
            lambda: self._generate_structured_once(
                messages=messages,
                schema_name=schema_name,
                schema=schema,
            ),
        )

    def generate_with_tools(
            self,
            messages: list[Message],
            tools: list[ToolDefinition],
            tool_choice: Literal["auto", "none"] = "auto",
    ) -> LLMResult:
        return self._run_observed(
            "generate_with_tools",
            lambda: self._generate_with_tools_once(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            ),
        )

    def _run_observed(
            self,
            operation: str,
            call: Callable[[], LLMResult],
    ) -> LLMResult:
        started_at = monotonic()

        try:
            with self._limiter.slot():
                result = call()
        except openai.APIError as error:
            converted_error = self._convert_error(error)
            self._record_failure(
                operation,
                type(converted_error).__name__,
                started_at,
            )
            raise converted_error from error
        except Exception as error:
            self._record_failure(
                operation,
                type(error).__name__,
                started_at,
            )
            raise

        self._record_success(operation, result, started_at)
        return result

    def _record_success(
            self,
            operation: str,
            result: LLMResult | LLMStreamCompleted,
            started_at: float,
    ) -> None:
        if self._observer is None:
            return

        self._observer.record_success(
            operation=operation,
            model=result.model,
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            response_id=result.response_id,
            duration_seconds=monotonic() - started_at,
        )

    def _record_failure(
            self,
            operation: str,
            error_type: str,
            started_at: float,
    ) -> None:
        if self._observer is None:
            return

        self._observer.record_failure(
            operation=operation,
            error_type=error_type,
            duration_seconds=monotonic() - started_at,
        )

    def generate_vision_structured(
            self,
            instructions: str,
            content: list[VisionContentPart],
            schema_name: str,
            schema: JsonSchema,
    ) -> LLMResult:
        return self._run_observed(
            "generate_vision_structured",
            lambda: self._generate_vision_structured_once(
                instructions=instructions,
                content=content,
                schema_name=schema_name,
                schema=schema,
            ),
        )

    def stream(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            tool_choice: Literal["auto", "none"] = "auto",
    ) -> Generator[LLMStreamEvent, None, None]:
        started_at = monotonic()
        completed: LLMStreamCompleted | None = None

        try:
            with self._limiter.slot():
                for event in self._stream_once(
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                ):
                    if isinstance(event, LLMStreamCompleted):
                        completed = event
                    yield event
        except openai.APIError as error:
            converted_error = self._convert_error(error)
            self._record_failure(
                "stream",
                type(converted_error).__name__,
                started_at,
            )
            raise converted_error from error
        except GeneratorExit:
            if completed is None:
                self._record_failure(
                    "stream",
                    "stream_closed",
                    started_at,
                )
            else:
                self._record_success(
                    "stream",
                    completed,
                    started_at,
                )
            raise
        except Exception as error:
            self._record_failure(
                "stream",
                type(error).__name__,
                started_at,
            )
            raise
        else:
            if completed is None:
                self._record_failure(
                    "stream",
                    "incomplete_stream",
                    started_at,
                )
            else:
                self._record_success(
                    "stream",
                    completed,
                    started_at,
                )

    @staticmethod
    def _convert_error(error: openai.APIError) -> LLMClientError:
        request_id = getattr(error, "request_id", None)

        if isinstance(error, openai.AuthenticationError):
            return LLMAccessError(
                "Провайдер отклонил API-ключ.",
                request_id=request_id,
            )
        if isinstance(error, openai.PermissionDeniedError):
            return LLMAccessError(
                "Недостаточно прав для обращения к модели.",
                request_id=request_id,
            )
        if isinstance(
            error,
            (
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
            ),
        ):
            return LLMRequestError(
                "Провайдер отклонил модель или параметры запроса.",
                request_id=request_id,
            )
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError(
                "Достигнуто ограничение частоты или квоты.",
                request_id=request_id,
            )
        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError("Провайдер не ответил за отведенное время.")
        if isinstance(error, openai.APIConnectionError):
            return LLMConnectionError("Не удалось соединиться с провайдером.")
        if isinstance(error, openai.APIResponseValidationError):
            return LLMProviderError("Провайдер вернул ответ неожиданного формата.")
        if isinstance(error, openai.APIStatusError):
            if error.status_code == 402:
                return LLMBalanceError(
                    "Недостаточно средств на балансе ProxyAPI.",
                    request_id=request_id,
                )
            return LLMProviderError(
                f"Провайдер вернул HTTP {error.status_code}.",
                request_id=request_id,
            )
        return LLMProviderError("SDK сообщил о неизвестной ошибке провайдера.")

    def _generate_once(self, messages: list[Message]) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
        )
        return self._to_result(response)

    def _generate_structured_once(
        self,
        messages: list[Message],
        schema_name: str,
        schema: JsonSchema,
    ) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        return self._to_result(response)

    def _generate_with_tools_once(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: Literal["auto", "none"],
    ) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=(self._settings.max_output_tokens),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
        )
        return self._to_result(response)

    def _generate_vision_structured_once(
        self,
        instructions: str,
        content: list[VisionContentPart],
        schema_name: str,
        schema: JsonSchema,
    ) -> LLMResult:
        response = self._client.responses.create(
            model=self._settings.model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )
        return self._responses_to_result(response)

    def _stream_once(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None,
            tool_choice: Literal["auto", "none"],
    ) -> Generator[LLMStreamEvent, None, None]:
        if tools is None:
            stream = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_completion_tokens=self._settings.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            stream = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                temperature=self._settings.temperature,
                max_completion_tokens=self._settings.max_output_tokens,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,
                stream=True,
                stream_options={"include_usage": True},
            )

        started_at = monotonic()
        model: str | None = None
        finish_reason: str | None = None
        response_id: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        refusal_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, str]] = {}

        try:
            for chunk in stream:
                if (
                        monotonic() - started_at
                        > self._settings.stream_timeout_seconds
                ):
                    raise LLMTimeoutError(
                        "Поток не завершился за отведённое время."
                    )

                model = chunk.model
                response_id = chunk.id

                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                text = choice.delta.content
                if text:
                    yield LLMTextDelta(text=text)

                if choice.delta.refusal:
                    refusal_parts.append(choice.delta.refusal)

                for tool_call in choice.delta.tool_calls or []:
                    parts = tool_call_parts.setdefault(
                        tool_call.index,
                        {"id": "", "name": "", "arguments": ""},
                    )

                    if tool_call.id:
                        parts["id"] += tool_call.id

                    if tool_call.function is not None:
                        if tool_call.function.name:
                            parts["name"] += tool_call.function.name
                        if tool_call.function.arguments:
                            parts["arguments"] += (
                                tool_call.function.arguments
                            )
        finally:
            stream.close()

        if model is None or response_id is None:
            raise LLMProviderError(
                "Провайдер завершил поток без метаданных."
            )

        tool_calls: list[ToolCall] = []
        for _, parts in sorted(tool_call_parts.items()):
            if not parts["id"] or not parts["name"]:
                raise LLMProviderError(
                    "Провайдер вернул незавершённый вызов инструмента."
                )

            tool_calls.append(
                ToolCall(
                    id=parts["id"],
                    name=parts["name"],
                    arguments=parts["arguments"],
                )
            )

        yield LLMStreamCompleted(
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_id=response_id,
            refusal="".join(refusal_parts) or None,
            tool_calls=tuple(tool_calls),
        )

    @staticmethod
    def _to_result(response: ChatCompletion) -> LLMResult:
        if not response.choices:
            raise LLMProviderError("Провайдер вернул ответ без вариантов продолжения.")

        choice = response.choices[0]
        usage = response.usage

        tool_calls = tuple(
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (choice.message.tool_calls or [])
        )

        return LLMResult(
            text=choice.message.content,
            refusal=choice.message.refusal,
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=(usage.prompt_tokens if usage is not None else None),
            completion_tokens=(usage.completion_tokens if usage is not None else None),
            total_tokens=(usage.total_tokens if usage is not None else None),
            response_id=response.id,
        )

    @staticmethod
    def _responses_to_result(response: Response) -> LLMResult:
        refusal: str | None = None

        for item in response.output:
            if item.type != "message":
                continue

            for part in item.content:
                if part.type == "refusal":
                    refusal = part.refusal
                    break

            if refusal is not None:
                break

        if response.status == "completed":
            finish_reason = "stop"
        elif response.status == "incomplete":
            reason = (
                response.incomplete_details.reason
                if response.incomplete_details is not None
                else None
            )
            finish_reason = "length" if reason == "max_output_tokens" else reason
        else:
            finish_reason = response.status

        usage = response.usage

        return LLMResult(
            text=response.output_text or None,
            refusal=refusal,
            tool_calls=(),
            model=str(response.model),
            finish_reason=finish_reason,
            prompt_tokens=(usage.input_tokens if usage is not None else None),
            completion_tokens=(usage.output_tokens if usage is not None else None),
            total_tokens=(usage.total_tokens if usage is not None else None),
            response_id=response.id,
        )

    def close(self) -> None:
        self._client.close()
