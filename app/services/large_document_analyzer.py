from collections.abc import Iterable
from dataclasses import dataclass, replace
from time import perf_counter

from app.services.document_ai_context import (
    DocumentAIContextError,
    PreparedFileContext,
)
from app.services.document_chunking import PreparedDocumentBatch
from app.services.document_content_analyzer import (
    DocumentAIResult,
    DocumentContentAnalyzer,
)


@dataclass(frozen=True)
class LargeDocumentAnalysisPolicy:
    max_reduce_characters: int = 32_000

    def __post_init__(self) -> None:
        if self.max_reduce_characters <= 0:
            raise ValueError("max_reduce_characters должен быть больше нуля.")


class LargeDocumentAnalyzer:
    def __init__(
        self,
        content_analyzer: DocumentContentAnalyzer,
        policy: LargeDocumentAnalysisPolicy | None = None,
    ) -> None:
        self._content_analyzer = content_analyzer
        self._policy = policy or LargeDocumentAnalysisPolicy()

    def analyze(
        self,
        question: str,
        batch: PreparedDocumentBatch,
    ) -> DocumentAIResult:
        started_at = perf_counter()
        partial_results = tuple(
            self._content_analyzer.analyze(question, chunk) for chunk in batch.chunks
        )
        reduce_context = self._build_reduce_context(
            batch,
            partial_results,
        )
        final_result = self._content_analyzer.analyze(
            question,
            reduce_context,
        )

        all_results = (*partial_results, final_result)
        warnings = tuple(
            dict.fromkeys(
                (
                    *batch.warnings,
                    *(warning for result in all_results for warning in result.warnings),
                )
            )
        )

        return replace(
            final_result,
            source_name=batch.source_name,
            source_kind=batch.source_kind,
            warnings=warnings,
            elapsed_seconds=perf_counter() - started_at,
            prompt_tokens=self._sum_optional(
                result.prompt_tokens for result in all_results
            ),
            completion_tokens=self._sum_optional(
                result.completion_tokens for result in all_results
            ),
            total_tokens=self._sum_optional(
                result.total_tokens for result in all_results
            ),
            model_calls=len(all_results),
        )

    def _build_reduce_context(
        self,
        batch: PreparedDocumentBatch,
        results: tuple[DocumentAIResult, ...],
    ) -> PreparedFileContext:
        blocks = [
            self._render_partial(index, result)
            for index, result in enumerate(results, start=1)
        ]
        content = "\n\n".join(blocks)

        if len(content) > self._policy.max_reduce_characters:
            raise DocumentAIContextError(
                "Промежуточные результаты слишком велики " "для объединения."
            )

        return PreparedFileContext(
            source_name=batch.source_name,
            source_kind=batch.source_kind,
            content=content,
            references=batch.references,
            warnings=batch.warnings,
        )

    @staticmethod
    def _render_partial(
        index: int,
        result: DocumentAIResult,
    ) -> str:
        analysis = result.analysis
        fact_lines = (
            "\n".join(
                f"- {fact.text} [{fact.reference}]" for fact in analysis.key_facts
            )
            or "- нет"
        )
        limitation_lines = (
            "\n".join(f"- {limitation}" for limitation in analysis.limitations)
            or "- нет"
        )

        return (
            f"[partial:{index}]\n"
            f"Краткое содержание: {analysis.summary}\n"
            f"Ответ по части: {analysis.answer}\n"
            f"Ключевые факты:\n{fact_lines}\n"
            f"Ограничения:\n{limitation_lines}\n"
            "Требуется проверка человеком: "
            f"{analysis.requires_human_review}"
        )

    @staticmethod
    def _sum_optional(
        values: Iterable[int | None],
    ) -> int | None:
        known = tuple(value for value in values if isinstance(value, int))
        return sum(known) if known else None
