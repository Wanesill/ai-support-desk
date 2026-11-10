from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.common import TokenUsageResponse


class ChatReplyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    message: str = Field(min_length=1, max_length=10_000)


class ChatStreamDeltaResponse(BaseModel):
    text: str


class ChatSessionCreatedResponse(BaseModel):
    session_id: str


class ChatHistoryMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatSessionResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryMessageResponse]
    history_messages: int = Field(ge=0)
    has_summary: bool
    pending_confirmation: bool


class ChatReplyResponse(BaseModel):
    text: str
    session_id: str
    history_messages: int = Field(ge=0)
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float = Field(ge=0)
    usage: TokenUsageResponse
    tool_calls_count: int = Field(ge=0)
    tool_names: list[str]
    requires_confirmation: bool
    summary_updated: bool
    has_summary: bool


class ToolActionResponse(BaseModel):
    text: str
    tool_name: str
    status: str
    error_code: str | None