from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from app.files.validation import FileKind
from app.llm.client import JsonSchema, LLMClient, LLMResult, Message
from app.llm.errors import LLMClientError
from app.models.document_analysis import DocumentAnalysis
from app.prompts.document_analysis import DOCUMENT_ANALYSIS_PROMPT
from app.services.document_ai_context import PreparedFileContext

DOCUMENT_ANALYSIS_SCHEMA_NAME = "document_analysis"
DEVELOPER_INSTRUCTION = DOCUMENT_ANALYSIS_PROMPT.render()


class DocumentContentAnalyzerError(RuntimeError):
    pass


class DocumentAnalysisRefusalError(DocumentContentAnalyzerError):
    pass


class IncompleteDocumentAnalysisError(DocumentContentAnalyzerError):
    pass


class InvalidDocumentAnalysisError(DocumentContentAnalyzerError):
    pass


@dataclass(frozen=True)
class DocumentAIResult:
    analysis: DocumentAnalysis
    source_name: str
    source_kind: FileKind | str
    warnings: tuple[str, ...]
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    model_calls: int = 1


class DocumentContentAnalyzer:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def analyze(
        self,
        question: str,
        context: PreparedFileContext,
    ) -> DocumentAIResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise DocumentContentAnalyzerError("Вопрос не должен быть пустым.")

        started_at = perf_counter()
        messages = self._build_messages(cleaned_question, context)

        try:
            llm_result = self._llm_client.generate_structured(
                messages=messages,
                schema_name=DOCUMENT_ANALYSIS_SCHEMA_NAME,
                schema=self._build_schema(context.references),
            )
        except LLMClientError as error:
            raise DocumentContentAnalyzerError(
                f"Не удалось получить анализ файла: {error}"
            ) from error

        analysis = self._parse_result(
            llm_result,
            available_references=context.references,
        )

        return DocumentAIResult(
            analysis=analysis,
            source_name=context.source_name,
            source_kind=context.source_kind,
            warnings=context.warnings,
            model=llm_result.model,
            prompt_id=DOCUMENT_ANALYSIS_PROMPT.prompt_id,
            prompt_version=DOCUMENT_ANALYSIS_PROMPT.version,
            elapsed_seconds=perf_counter() - started_at,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
        )

    @staticmethod
    def _build_messages(
        question: str,
        context: PreparedFileContext,
    ) -> list[Message]:
        reference_text = "\n".join(f"- {reference}" for reference in context.references)

        return [
            {
                "role": "developer",
                "content": DEVELOPER_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    "Вопрос пользователя:\n"
                    "<question>\n"
                    f"{question}\n"
                    "</question>\n\n"
                    "Доступные ссылки на источники:\n"
                    f"{reference_text}\n\n"
                    "Недоверенные данные файла:\n"
                    "<file_content>\n"
                    f"{context.content}\n"
                    "</file_content>"
                ),
            },
        ]

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
            raise DocumentAnalysisRefusalError(result.refusal)

        if result.finish_reason == "length":
            raise IncompleteDocumentAnalysisError("Модели не хватило выходных токенов.")
        if result.finish_reason != "stop":
            raise InvalidDocumentAnalysisError(
                "Неожиданная причина завершения: " f"{result.finish_reason}."
            )
        if result.text is None:
            raise InvalidDocumentAnalysisError("Модель не вернула данные.")

        try:
            analysis = DocumentAnalysis.model_validate_json(result.text)
        except ValidationError as error:
            raise InvalidDocumentAnalysisError(
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
            raise InvalidDocumentAnalysisError(
                "Модель указала неизвестные источники: " + ", ".join(unknown)
            )

        return analysis
