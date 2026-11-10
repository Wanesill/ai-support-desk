from pydantic import BaseModel, Field

from app.llm.cost import LLMCostEstimator


class TokenUsageResponse(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    cost_currency: str | None = None


def build_token_usage_response(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_estimator: LLMCostEstimator,
) -> TokenUsageResponse:
    estimate = cost_estimator.estimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return TokenUsageResponse(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost=(
            float(estimate.amount)
            if estimate is not None
            else None
        ),
        cost_currency=(
            estimate.currency
            if estimate is not None
            else None
        ),
    )