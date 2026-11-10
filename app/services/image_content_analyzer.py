from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from app.files.images.preparation import (
    ImageDetail,
    PreparedImageBatch,
)
from app.llm.client import (
    JsonSchema,
    LLMClient,
    LLMResult,
    VisionContentPart,
)
from app.llm.errors import LLMClientError
from app.models.document_analysis import DocumentAnalysis
from app.prompts.image_analysis import IMAGE_ANALYSIS_PROMPT
from app.services.document_content_analyzer import DocumentAIResult

IMAGE_ANALYSIS_SCHEMA_NAME = "image_analysis"
DEVELOPER_INSTRUCTION = IMAGE_ANALYSIS_PROMPT.render()


class ImageContentAnalyzerError(RuntimeError):
    pass


class ImageAnalysisRefusalError(ImageContentAnalyzerError):
    pass


class IncompleteImageAnalysisError(ImageContentAnalyzerError):
    pass


class InvalidImageAnalysisError(ImageContentAnalyzerError):
    pass


@dataclass(frozen=True)
class ImageAnalysisPolicy:
    detail: ImageDetail = "high"


class ImageContentAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        policy: ImageAnalysisPolicy | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._policy = policy or ImageAnalysisPolicy()

    def analyze(
        self,
        question: str,
        batch: PreparedImageBatch,
    ) -> DocumentAIResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ImageContentAnalyzerError("Вопрос не должен быть пустым.")

        started_at = perf_counter()

        try:
            llm_result = self._llm_client.generate_vision_structured(
                instructions=DEVELOPER_INSTRUCTION,
                content=self._build_content(cleaned_question, batch),
                schema_name=IMAGE_ANALYSIS_SCHEMA_NAME,
                schema=self._build_schema(batch.references),
            )
        except LLMClientError as error:
            raise ImageContentAnalyzerError(
                f"Не удалось получить анализ изображения: {error}"
            ) from error

        analysis = self._parse_result(
            llm_result,
            available_references=batch.references,
        )

        return DocumentAIResult(
            analysis=analysis,
            source_name=", ".join(image.source_name for image in batch.images),
            source_kind=batch.images[0].source_kind,
            warnings=batch.warnings,
            model=llm_result.model,
            prompt_id=IMAGE_ANALYSIS_PROMPT.prompt_id,
            prompt_version=IMAGE_ANALYSIS_PROMPT.version,
            elapsed_seconds=perf_counter() - started_at,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
        )

    def _build_content(
        self,
        question: str,
        batch: PreparedImageBatch,
    ) -> list[VisionContentPart]:
        references = "\n".join(
            f"- {image.reference}: {image.source_name}" for image in batch.images
        )

        content: list[VisionContentPart] = [
            {
                "type": "input_text",
                "text": (
                    "Вопрос пользователя:\n"
                    "<question>\n"
                    f"{question}\n"
                    "</question>\n\n"
                    "Доступные изображения:\n"
                    f"{references}"
                ),
            }
        ]

        for image in batch.images:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Следующее изображение имеет ссылку " f"{image.reference}."
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image.data_url,
                    "detail": self._policy.detail,
                }
            )

        return content

    @staticmethod
    def _build_schema(
        available_references: tuple[str, ...],
    ) -> JsonSchema:
        schema = DocumentAnalysis.model_json_schema()
        fact_schema = schema["$defs"]["DocumentFact"]
        reference_schema = fact_schema["properties"]["reference"]
        reference_schema["enum"] = list(available_references)
        return schema

    @staticmethod
    def _parse_result(
        result: LLMResult,
        available_references: tuple[str, ...],
    ) -> DocumentAnalysis:
        if result.refusal is not None:
            raise ImageAnalysisRefusalError(result.refusal)

        if result.finish_reason == "length":
            raise IncompleteImageAnalysisError("Модели не хватило выходных токенов.")
        if result.finish_reason != "stop":
            raise InvalidImageAnalysisError(
                "Неожиданная причина завершения: " f"{result.finish_reason}."
            )
        if result.text is None:
            raise InvalidImageAnalysisError("Модель не вернула данные.")

        try:
            analysis = DocumentAnalysis.model_validate_json(result.text)
        except ValidationError as error:
            raise InvalidImageAnalysisError(
                "Ответ модели не соответствует контракту."
            ) from error

        allowed = set(available_references)
        unknown = sorted(
            {
                fact.reference
                for fact in analysis.key_facts
                if fact.reference not in allowed
            }
        )
        if unknown:
            raise InvalidImageAnalysisError(
                "Модель указала неизвестные источники: " + ", ".join(unknown)
            )

        return analysis
