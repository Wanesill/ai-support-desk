from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    reference: str = Field(min_length=1, max_length=100)

    @field_validator("text", "reference")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Значение не должно быть пустым.")
        return normalized


class DocumentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2_000)
    answer: str = Field(min_length=1, max_length=4_000)
    key_facts: list[DocumentFact] = Field(max_length=10)
    limitations: list[str] = Field(max_length=10)
    requires_human_review: bool

    @field_validator("summary", "answer")
    @classmethod
    def normalize_main_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Текст не должен быть пустым.")
        return normalized

    @field_validator("limitations")
    @classmethod
    def normalize_limitations(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [" ".join(value.split()) for value in values]
        if any(not value for value in normalized):
            raise ValueError("Ограничение не должно быть пустой строкой.")
        return list(dict.fromkeys(normalized))
