from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

TOKENS_PER_MILLION = Decimal(1000000)
COST_PRECISION = Decimal("0.000001")


@dataclass(frozen=True)
class LLMCostEstimate:
    amount: Decimal
    currency: str


class LLMCostEstimator:
    def __init__(
        self,
        input_price_per_million: Decimal | None,
        output_price_per_million: Decimal | None,
        currency: str,
    ) -> None:
        self._input_price = input_price_per_million
        self._output_price = output_price_per_million
        self._currency = currency.upper()

    def estimate(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> LLMCostEstimate | None:
        if (
            prompt_tokens is None
            or completion_tokens is None
            or self._input_price is None
            or self._output_price is None
        ):
            return None

        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Количество токенов не может быть отрицательным.")

        input_cost = (
            Decimal(prompt_tokens) * self._input_price
            / TOKENS_PER_MILLION
        )
        output_cost = (
            Decimal(completion_tokens) * self._output_price
            / TOKENS_PER_MILLION
        )
        amount = (input_cost + output_cost).quantize(
            COST_PRECISION,
            rounding=ROUND_HALF_UP,
        )
        return LLMCostEstimate(
            amount=amount,
            currency=self._currency,
        )