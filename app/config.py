from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    llm_api_key: SecretStr = Field(
        validation_alias="LLM_API_KEY",
    )
    base_url: AnyHttpUrl = Field(
        default="https://api.proxyapi.ru/openai/v1",
        validation_alias="BASE_URL",
    )
    model: str = Field(
        default="gpt-4.1",
        min_length=1,
        validation_alias="MODEL",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        validation_alias="TEMPERATURE",
    )
    max_output_tokens: int = Field(
        default=80,
        gt=0,
        validation_alias="MAX_OUTPUT_TOKENS",
    )
    chat_input_token_budget: int = Field(
        default=6000,
        gt=0,
        validation_alias="CHAT_INPUT_TOKEN_BUDGET",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias="REQUEST_TIMEOUT_SECONDS",
    )
    stream_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        validation_alias="STREAM_TIMEOUT_SECONDS",
    )
    llm_max_concurrent_requests: int = Field(
        default=4,
        gt=0,
        validation_alias="LLM_MAX_CONCURRENT_REQUESTS",
    )
    llm_queue_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias="LLM_QUEUE_TIMEOUT_SECONDS",
    )
    llm_input_price_per_million: Decimal | None = Field(
        default=None,
        ge=0,
        validation_alias="LLM_INPUT_PRICE_PER_MILLION",
    )
    llm_output_price_per_million: Decimal | None = Field(
        default=None,
        ge=0,
        validation_alias="LLM_OUTPUT_PRICE_PER_MILLION",
    )
    llm_cost_currency: str = Field(
        default="RUB",
        min_length=3,
        max_length=3,
        validation_alias="LLM_COST_CURRENCY",
    )
    max_retries: int = Field(default=2, ge=0, le=5, validation_alias="MAX_RETRIES")
    tool_max_steps: int = Field(
        default=3, ge=1, le=5, validation_alias="TOOL_MAX_STEPS"
    )
    file_max_size_bytes: int = Field(
        default=10 * 1_024 * 1_024,
        gt=0,
        validation_alias="FILE_MAX_SIZE_BYTES",
    )
    app_env: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_llm_pricing(self) -> "Settings":
        input_price_is_set = (
                self.llm_input_price_per_million is not None
        )
        output_price_is_set = (
                self.llm_output_price_per_million is not None
        )

        if input_price_is_set != output_price_is_set:
            raise ValueError(
                "Нужно указать обе цены модели или не указывать ни одной."
            )

        return self
