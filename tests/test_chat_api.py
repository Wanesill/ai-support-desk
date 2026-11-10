from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.chat_history import SlidingWindowHistory, TokenCounter
from app.services.chat_service import ChatService
from tests.fakes import (
    FailingAfterDeltaToolLoop,
    UnusedSummaryService,
)


def test_chat_sessions_are_isolated(
    client: TestClient,
    fake_container: SimpleNamespace,
    chat_service: ChatService,
) -> None:
    fake_container.chat_service = chat_service

    first_id = client.post("/chat/sessions").json()["session_id"]
    second_id = client.post("/chat/sessions").json()["session_id"]

    response = client.post(
        f"/chat/sessions/{first_id}/messages",
        json={"message": "Где мой заказ?"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Ваш заказ находится в пути."

    first = client.get(f"/chat/sessions/{first_id}")
    second = client.get(f"/chat/sessions/{second_id}")

    assert first.json()["history_messages"] == 2
    assert second.json()["history_messages"] == 0


def test_clear_and_delete_chat_session(
    client: TestClient,
    fake_container: SimpleNamespace,
    chat_service: ChatService,
) -> None:
    fake_container.chat_service = chat_service
    session_id = client.post("/chat/sessions").json()["session_id"]
    client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"message": "Где мой заказ?"},
    )

    clear_response = client.delete(f"/chat/sessions/{session_id}/messages")
    assert clear_response.status_code == 204
    session = client.get(f"/chat/sessions/{session_id}")
    assert session.json()["history_messages"] == 0

    delete_response = client.delete(f"/chat/sessions/{session_id}")
    assert delete_response.status_code == 204
    assert client.get(f"/chat/sessions/{session_id}").status_code == 404


def test_stream_returns_deltas_and_completed_event(
    client: TestClient,
    fake_container: SimpleNamespace,
    chat_service: ChatService,
) -> None:
    fake_container.chat_service = chat_service
    session_id = client.post("/chat/sessions").json()["session_id"]

    with client.stream(
        "POST",
        f"/chat/sessions/{session_id}/messages/stream",
        json={"message": "Где мой заказ?"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )
    assert body.count("event: delta") == 2
    assert "Ваш заказ " in body
    assert "находится в пути." in body
    assert "event: completed" in body
    session = client.get(f"/chat/sessions/{session_id}")
    assert session.json()["history_messages"] == 2


def test_stream_error_does_not_save_history(
    client: TestClient,
    fake_container: SimpleNamespace,
) -> None:
    fake_container.chat_service = ChatService(
        tool_loop=FailingAfterDeltaToolLoop(),
        history_policy=SlidingWindowHistory(
            token_counter=TokenCounter(),
            input_token_budget=6_000,
        ),
        summary_service=UnusedSummaryService(),
    )
    session_id = client.post("/chat/sessions").json()["session_id"]

    with client.stream(
        "POST",
        f"/chat/sessions/{session_id}/messages/stream",
        json={"message": "Дайте длинный ответ."},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: delta" in body
    assert "event: error" in body
    assert "chat_stream_failed" in body
    session = client.get(f"/chat/sessions/{session_id}")
    assert session.json()["history_messages"] == 0