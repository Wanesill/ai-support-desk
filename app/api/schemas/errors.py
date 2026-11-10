from pydantic import BaseModel


class ValidationIssueResponse(BaseModel):
    field: str
    message: str


class ErrorContentResponse(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[ValidationIssueResponse] | None = None


class ErrorResponse(BaseModel):
    error: ErrorContentResponse