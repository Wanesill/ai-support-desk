from decimal import Decimal

import pytest

from app.llm.cost import LLMCostEstimator
from app.llm.errors import LLMCapacityError
from app.llm.limits import LLMConcurrencyLimiter


def test_cost_estimator_uses_separate_prices() -> None:
    estimator = LLMCostEstimator(
        input_price_per_million=Decimal(2),
        output_price_per_million=Decimal(8),
        currency="RUB",
    )

    result = estimator.estimate(
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
    )

    assert result is not None
    assert result.amount == Decimal("6.000000")
    assert result.currency == "RUB"


def test_limiter_rejects_call_when_slot_is_busy() -> None:
    limiter = LLMConcurrencyLimiter(
        max_concurrent_requests=1,
        queue_timeout_seconds=0.01,
    )

    with limiter.slot(), pytest.raises(LLMCapacityError), limiter.slot():
        pass