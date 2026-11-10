import json
from pathlib import Path

from pydantic import ValidationError

from app.application import build_document_ai_service
from app.config import Settings
from app.llm.client import LLMClient
from app.services.document_ai_service import (
    DocumentAIRequest,
    DocumentAIService,
    DocumentAIServiceError,
)
from app.services.document_content_analyzer import DocumentAIResult

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_FILES_ROOT = PROJECT_ROOT / "sample_files"


def print_result(result: DocumentAIResult) -> None:
    analysis = result.analysis
    print(f"Источник: {result.source_name}")
    print(f"Формат: {result.source_kind}")
    print(f"Краткое содержание: {analysis.summary}")
    print(f"Ответ: {analysis.answer}")
    print("Ключевые факты:")
    for fact in analysis.key_facts:
        print(f"- {fact.text} [{fact.reference}]")

    if analysis.limitations:
        print("Ограничения:")
        for limitation in analysis.limitations:
            print(f"- {limitation}")

    for warning in result.warnings:
        print(f"Предупреждение: {warning}")

    print("Требуется проверка человеком: " f"{analysis.requires_human_review}")
    print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
    print(f"Модель: {result.model}")
    print(f"Запросов к модели: {result.model_calls}")
    print(f"Токенов: {result.total_tokens}")
    print(f"Время: {result.elapsed_seconds:.2f} с")


def print_json_report(result: DocumentAIResult) -> None:
    report = {
        "source": {
            "name": result.source_name,
            "kind": str(result.source_kind),
            "warnings": list(result.warnings),
        },
        "analysis": result.analysis.model_dump(mode="json"),
        "generation": {
            "model": result.model,
            "prompt_id": result.prompt_id,
            "prompt_version": result.prompt_version,
            "model_calls": result.model_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "response_id": result.response_id,
        },
    }

    print("\nСтруктурированный JSON-отчёт")
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


def run_case(
    service: DocumentAIService,
    title: str,
    request: DocumentAIRequest,
) -> DocumentAIResult | None:
    print(f"\n{title}")

    try:
        result = service.analyze(request)
        print_result(result)
        return result
    except DocumentAIServiceError as error:
        print(f"Анализ не выполнен: {error}")
        return None


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    llm_client = LLMClient(settings)

    try:
        service = build_document_ai_service(
            llm_client=llm_client,
            allowed_root=SAMPLE_FILES_ROOT,
        )

        comparison_result = run_case(
            service,
            "Сравнение двух версий документа",
            DocumentAIRequest(
                source_path=(
                    SAMPLE_FILES_ROOT / "return_policy_v1.md"
                ),
                additional_document_paths=(
                    SAMPLE_FILES_ROOT / "return_policy_v2.md",
                ),
                question=(
                    "Сравни две версии правил возврата. "
                    "Перечисли только подтверждённые изменения "
                    "в сроках и необходимых сведениях."
                ),
                require_human_review=True,
            ),
        )

        if comparison_result is not None:
            print_json_report(comparison_result)
    finally:
        llm_client.close()


if __name__ == "__main__":
    main()
