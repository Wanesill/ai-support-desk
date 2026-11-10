from collections.abc import Generator

from app.llm.client import Message
from app.services.tool_loop import (
    ToolLoopError,
    ToolLoopResult,
    ToolLoopStreamCompleted,
    ToolLoopStreamEvent,
    ToolLoopTextDelta,
)


class FakeToolLoop:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def stream(self, messages: list[Message]) -> Generator[ToolLoopStreamEvent, None, None]:
        self.calls.append(messages)
        parts = ("Ваш заказ ", "находится в пути.")

        for part in parts:
            yield ToolLoopTextDelta(text=part)

        yield ToolLoopStreamCompleted(
            result=ToolLoopResult(
                text="".join(parts),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                response_ids=("fake-response-id",),
                model_steps=1,
                tool_calls_count=0,
                tool_names=(),
                audit_records=(),
                confirmation=None,
            )
        )


class FailingAfterDeltaToolLoop:
    def stream(self, messages: list[Message]) -> Generator[ToolLoopStreamEvent, None, None]:
        yield ToolLoopTextDelta(text="Начало ответа")
        raise ToolLoopError("Искусственный сбой")


class UnusedSummaryService:
    def summarize(self, previous_summary: str | None, messages: list[Message]) -> str:
        raise AssertionError("Summary не должно вызываться")