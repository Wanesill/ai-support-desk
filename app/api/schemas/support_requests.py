from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.common import TokenUsageResponse
from app.models.support_request import (
    MissingField,
    SupportCategory,
    Urgency,
)


class SupportRequestAnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str = Field(min_length=1, max_length=10_000)


class SupportRequestAnalysisResponse(BaseModel):
    category: SupportCategory
    order_id: str | None
    product: str | None
    problem_description: str | None
    urgency: Urgency
    missing_fields: list[MissingField]
    confidence: float = Field(ge=0, le=1)


class SupportRequestAnalyzeResponse(BaseModel):
    analysis: SupportRequestAnalysisResponse
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float = Field(ge=0)
    usage: TokenUsageResponse
    repair_attempts: int = Field(ge=0)