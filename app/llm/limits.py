from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore

from app.llm.errors import LLMCapacityError


class LLMConcurrencyLimiter:
    def __init__(
        self,
        max_concurrent_requests: int,
        queue_timeout_seconds: float,
    ) -> None:
        self._semaphore = BoundedSemaphore(
            max_concurrent_requests
        )
        self._queue_timeout_seconds = queue_timeout_seconds

    @contextmanager
    def slot(self) -> Iterator[None]:
        acquired = self._semaphore.acquire(
            timeout=self._queue_timeout_seconds
        )
        if not acquired:
            raise LLMCapacityError(
                "Все доступные места для обращения к модели заняты."
            )

        try:
            yield
        finally:
            self._semaphore.release()