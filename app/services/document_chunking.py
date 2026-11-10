from dataclasses import dataclass, replace

from app.files.documents.extraction import (
    DocumentPart,
    ExtractedDocument,
)
from app.files.validation import FileKind
from app.services.document_ai_context import (
    DocumentAIContextError,
    PreparedFileContext,
)


class DocumentChunkingError(DocumentAIContextError):
    pass


class DocumentSelectionError(DocumentChunkingError):
    pass


class TooManyDocumentChunksError(DocumentChunkingError):
    pass


@dataclass(frozen=True)
class DocumentChunkingPolicy:
    max_chunk_characters: int = 12_000
    max_chunks: int = 8

    def __post_init__(self) -> None:
        if self.max_chunk_characters <= 0:
            raise ValueError("max_chunk_characters должен быть больше нуля.")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks должен быть больше нуля.")


@dataclass(frozen=True)
class PreparedDocumentBatch:
    source_name: str
    source_kind: FileKind
    chunks: tuple[PreparedFileContext, ...]
    references: tuple[str, ...]
    warnings: tuple[str, ...]


class DocumentChunkPreparer:
    def __init__(
        self,
        policy: DocumentChunkingPolicy | None = None,
    ) -> None:
        self._policy = policy or DocumentChunkingPolicy()

    def select(
        self,
        document: ExtractedDocument,
        references: tuple[str, ...] | None,
    ) -> ExtractedDocument:
        if references is None:
            return document

        requested = tuple(dict.fromkeys(value.strip() for value in references))
        if not requested or any(not value for value in requested):
            raise DocumentSelectionError(
                "Список ссылок документа не должен быть пустым."
            )

        available = {part.reference for part in document.parts}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise DocumentSelectionError(
                "В документе нет указанных частей: " + ", ".join(unknown)
            )

        requested_set = set(requested)
        selected = tuple(
            part for part in document.parts if part.reference in requested_set
        )
        text = "\n\n".join(self._render_part_text(part) for part in selected).strip()
        warnings = document.warnings
        if len(selected) < len(document.parts):
            warnings += ("Для анализа выбраны не все части документа.",)

        return replace(
            document,
            text=text,
            parts=selected,
            metadata=replace(
                document.metadata,
                part_count=len(selected),
                character_count=len(text),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def prepare(
        self,
        document: ExtractedDocument,
    ) -> PreparedDocumentBatch:
        blocks: list[tuple[str, tuple[str, ...]]] = []

        for part in document.parts:
            blocks.extend(self._split_part(part))

        chunks: list[PreparedFileContext] = []
        current_blocks: list[str] = []
        current_references: list[str] = []

        def flush_chunk() -> None:
            if not current_blocks:
                return

            chunks.append(
                PreparedFileContext(
                    source_name=document.metadata.original_name,
                    source_kind=document.metadata.kind,
                    content="\n\n".join(current_blocks),
                    references=tuple(dict.fromkeys(current_references)),
                    warnings=(),
                )
            )
            current_blocks.clear()
            current_references.clear()

        for block, references in blocks:
            candidate = "\n\n".join((*current_blocks, block))
            if current_blocks and len(candidate) > self._policy.max_chunk_characters:
                flush_chunk()

            current_blocks.append(block)
            current_references.extend(references)

        flush_chunk()

        if not chunks:
            raise DocumentChunkingError("Не удалось подготовить части документа.")
        if len(chunks) > self._policy.max_chunks:
            raise TooManyDocumentChunksError(
                "Документ требует слишком много запросов к модели."
            )

        references = tuple(
            dict.fromkeys(
                reference for chunk in chunks for reference in chunk.references
            )
        )
        warnings = document.warnings + (
            f"Документ обработан по частям: {len(chunks)}.",
        )

        return PreparedDocumentBatch(
            source_name=document.metadata.original_name,
            source_kind=document.metadata.kind,
            chunks=tuple(chunks),
            references=references,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _split_part(
        self,
        part: DocumentPart,
    ) -> list[tuple[str, tuple[str, ...]]]:
        prefix = f"[source:{part.reference}]\n"
        text = self._render_part_text(part)
        available = self._policy.max_chunk_characters - len(prefix)
        if available <= 0:
            raise DocumentChunkingError(
                "Лимит части слишком мал для ссылки на источник."
            )

        pieces = self._split_text(text, available)
        return [(prefix + piece, (part.reference,)) for piece in pieces]

    @staticmethod
    def _render_part_text(part: DocumentPart) -> str:
        values: list[str] = []
        if part.section_title is not None:
            values.append(part.section_title)
        if part.text:
            values.append(part.text)
        return "\n".join(values).strip()

    @staticmethod
    def _split_text(text: str, limit: int) -> list[str]:
        pieces: list[str] = []
        remaining = text.strip()

        while len(remaining) > limit:
            cut = remaining.rfind("\n\n", 0, limit + 1)
            if cut < limit // 2:
                cut = remaining.rfind("\n", 0, limit + 1)
            if cut < limit // 2:
                cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit

            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()

        if remaining:
            pieces.append(remaining)

        return pieces
