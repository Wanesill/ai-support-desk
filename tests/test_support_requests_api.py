from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.models.support_request import SupportRequestAnalysis
from app.services.support_request_analyzer import (
    SupportRequestAnalyzerError,
    SupportRequestResult,
)


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_support_request_returns_contract(
    client: TestClient,
    fake_container: SimpleNamespace,
) -> None:
    analyzer = Mock()
    analyzer.analyze.return_value = SupportRequestResult(
        analysis=SupportRequestAnalysis(
            category="technical_issue",
            order_id="8041",
            product="кофемашина",
            problem_description="не включается",
            urgency="normal",
            missing_fields=[],
            confidence=0.95,
        ),
        model="fake-model",
        prompt_id="support_request_analysis",
        prompt_version="1.0.0",
        elapsed_seconds=0.01,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        response_id="fake-response-id",
        repair_attempts=0,
    )
    fake_container.request_analyzer = analyzer

    response = client.post(
        "/support-requests/analyze", json={"text": "Заказ 8041 не включается."}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["order_id"] == "8041"
    assert body["analysis"]["category"] == "technical_issue"
    assert body["usage"]["total_tokens"] == 15
    assert body["usage"]["estimated_cost"] == 0.00006
    assert body["usage"]["cost_currency"] == "RUB"
    analyzer.analyze.assert_called_once_with("Заказ 8041 не включается.")


def test_analyze_support_request_rejects_empty_text(
    client: TestClient,
) -> None:
    response = client.post("/support-requests/analyze", json={"text": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_analyze_support_request_hides_service_error(
    client: TestClient,
    fake_container: SimpleNamespace,
) -> None:
    analyzer = Mock()
    analyzer.analyze.side_effect = SupportRequestAnalyzerError("Техническая ошибка ProxyAPI")
    fake_container.request_analyzer = analyzer

    response = client.post(
        "/support-requests/analyze", json={"text": "Кофемашина не включается."}
    )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "support_analysis_failed"
    assert body["error"]["message"] == "Не удалось проанализировать обращение."
    assert "ProxyAPI" not in response.text
    assert "test-key" not in response.text