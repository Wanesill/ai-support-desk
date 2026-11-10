import json

import pytest

from app.llm.client import JsonSchema, LLMResult, Message
from app.services.support_request_analyzer import (
    InvalidSupportRequestError,
    SupportRequestAnalyzer,
)


class FakeStructuredLLMClient:
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0
        self.last_messages: list[Message] | None = None

    def generate_structured(
        self,
        messages: list[Message],
        schema_name: str,
        schema: JsonSchema,
    ) -> LLMResult:
        self.calls += 1
        self.last_messages = messages
        return self._results.pop(0)


def build_llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        refusal=None,
        tool_calls=(),
        model="fake-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        response_id="fake-response-id",
    )


def test_analyzer_parses_structured_result() -> None:
    response_text = json.dumps(
        {
            "category": "technical_issue",
            "order_id": "8041",
            "product": "кофемашина",
            "problem_description": "не включается",
            "urgency": "normal",
            "missing_fields": [],
            "confidence": 0.95,
        },
        ensure_ascii=False,
    )
    fake_client = FakeStructuredLLMClient([build_llm_result(response_text)])
    analyzer = SupportRequestAnalyzer(fake_client)

    result = analyzer.analyze("Заказ 8041, кофемашина не включается.")

    assert result.analysis.order_id == "8041"
    assert result.analysis.category == "technical_issue"
    assert result.repair_attempts == 0
    assert fake_client.calls == 1
    assert fake_client.last_messages is not None
    assert "8041" in fake_client.last_messages[-1]["content"]


def test_analyzer_rejects_invalid_json() -> None:
    fake_client = FakeStructuredLLMClient([build_llm_result("not-json")])
    analyzer = SupportRequestAnalyzer(fake_client)

    with pytest.raises(InvalidSupportRequestError, match="не соответствует контракту"):
        analyzer.analyze("Заказ 8041 не работает.")

    assert fake_client.calls == 1


def test_analyzer_limits_repair_attempts() -> None:
    wrong_text = json.dumps(
        {
            "category": "technical_issue",
            "order_id": "9999",
            "product": None,
            "problem_description": "не включается",
            "urgency": "normal",
            "missing_fields": ["product"],
            "confidence": 0.7,
        },
        ensure_ascii=False,
    )
    fake_client = FakeStructuredLLMClient(
        [
            build_llm_result(wrong_text),
            build_llm_result(wrong_text),
        ]
    )
    analyzer = SupportRequestAnalyzer(fake_client)

    with pytest.raises(InvalidSupportRequestError, match="смысловую проверку"):
        analyzer.analyze("Заказ 8041 не включается.")

    assert fake_client.calls == 2