from pydantic import BaseModel, Field

from app.api.schemas.common import TokenUsageResponse


class DocumentFactResponse(BaseModel):
    text: str
    reference: str


class DocumentAnalysisResponse(BaseModel):
    summary: str
    answer: str
    key_facts: list[DocumentFactResponse]
    limitations: list[str]
    requires_human_review: bool


class DocumentAnalyzeResponse(BaseModel):
    source_name: str
    source_kind: str
    warnings: list[str]
    analysis: DocumentAnalysisResponse
    model: str
    prompt_id: str
    prompt_version: str
    model_calls: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0)
    usage: TokenUsageResponse
    response_id: str