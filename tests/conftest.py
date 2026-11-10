from collections.abc import Iterator
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_container
from app.api.main import create_api
from app.chat_history import SlidingWindowHistory, TokenCounter
from app.llm.cost import LLMCostEstimator
from app.services.chat_service import ChatService
from tests.fakes import FakeToolLoop, UnusedSummaryService


@pytest.fixture
def fake_container() -> SimpleNamespace:
    return SimpleNamespace(
        cost_estimator=LLMCostEstimator(
            input_price_per_million=Decimal(2),
            output_price_per_million=Decimal(8),
            currency="RUB",
        )
    )


@pytest.fixture
def application() -> Iterator[FastAPI]:
    app = create_api()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(
    application: FastAPI,
    fake_container: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    application.dependency_overrides[get_container] = lambda: fake_container

    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def chat_service() -> ChatService:
    return ChatService(
        tool_loop=FakeToolLoop(),
        history_policy=SlidingWindowHistory(
            token_counter=TokenCounter(),
            input_token_budget=6_000,
        ),
        summary_service=UnusedSummaryService(),
    )