import logging

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
)

from app.llm.cost import LLMCostEstimator

logger = logging.getLogger(__name__)


class LLMObserver:
    def __init__(
            self,
            cost_estimator: LLMCostEstimator,
    ) -> None:
        self.registry = CollectorRegistry()
        self._cost_estimator = cost_estimator

        self._calls = Counter(
            "ai_support_llm_calls",
            "Количество завершённых обращений к модели.",
            ("operation", "status"),
            registry=self.registry,
        )
        self._tokens = Counter(
            "ai_support_llm_tokens",
            "Количество токенов, сообщённых провайдером.",
            ("operation", "direction"),
            registry=self.registry,
        )
        self._duration = Histogram(
            "ai_support_llm_duration_seconds",
            "Продолжительность обращения к модели.",
            ("operation",),
            registry=self.registry,
        )
        self._estimated_cost = Counter(
            "ai_support_llm_estimated_cost",
            "Накопленная приблизительная стоимость.",
            ("currency",),
            registry=self.registry,
        )

    def record_success(
            self,
            operation: str,
            model: str,
            finish_reason: str | None,
            prompt_tokens: int | None,
            completion_tokens: int | None,
            total_tokens: int | None,
            response_id: str,
            duration_seconds: float,
    ) -> None:
        self._calls.labels(
            operation=operation,
            status="success",
        ).inc()
        self._duration.labels(operation=operation).observe(
            duration_seconds
        )

        if prompt_tokens is not None:
            self._tokens.labels(
                operation=operation,
                direction="input",
            ).inc(prompt_tokens)

        if completion_tokens is not None:
            self._tokens.labels(
                operation=operation,
                direction="output",
            ).inc(completion_tokens)

        estimate = self._cost_estimator.estimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        estimated_cost: float | None = None
        cost_currency: str | None = None

        if estimate is not None:
            estimated_cost = float(estimate.amount)
            cost_currency = estimate.currency
            self._estimated_cost.labels(
                currency=estimate.currency
            ).inc(estimated_cost)

        logger.info(
            "llm_call_completed",
            extra={
                "operation": operation,
                "model": model,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "cost_currency": cost_currency,
                "response_id": response_id,
                "duration_seconds": duration_seconds,
            },
        )

    def record_failure(
            self,
            operation: str,
            error_type: str,
            duration_seconds: float,
    ) -> None:
        self._calls.labels(
            operation=operation,
            status="error",
        ).inc()
        self._duration.labels(operation=operation).observe(
            duration_seconds
        )
        logger.error(
            "llm_call_failed",
            extra={
                "operation": operation,
                "error_type": error_type,
                "duration_seconds": duration_seconds,
            },
        )