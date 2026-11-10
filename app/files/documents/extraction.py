import unicodedata
from dataclasses import dataclass
from typing import Protocol

from app.files.validation import FileKind, ValidatedFile


class DocumentExtractionError(RuntimeError):
    pass


class UnsupportedDocumentFormatError(DocumentExtractionError):
    pass


class DocumentReadError(DocumentExtractionError):
    pass


class EmptyDocumentError(DocumentExtractionError):
    pass


class EncryptedDocumentError(DocumentExtractionError):
    pass


class DocumentLimitError(DocumentExtractionError):
    pass


@dataclass(frozen=True)
class DocumentPart:
    reference: str
    text: str
    page_number: int | None = None
    section_title: str | None = None
    heading_level: int | None = None


@dataclass(frozen=True)
class DocumentMetadata:
    original_name: str
    kind: FileKind
    media_type: str
    size_bytes: int
    part_count: int
    page_count: int | None
    character_count: int


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    parts: tuple[DocumentPart, ...]
    metadata: DocumentMetadata
    warnings: tuple[str, ...]


class DocumentFormatExtractor(Protocol):
    @property
    def supported_kinds(self) -> frozenset[FileKind]: ...

    def extract(self, file: ValidatedFile) -> ExtractedDocument: ...


def normalize_extracted_text(value: str) -> str:
    normalized = value.replace("\ufeff", "")
    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")

    visible_characters = (
        character
        for character in normalized
        if character in {"\n", "\t"}
        or not unicodedata.category(character).startswith("C")
    )
    normalized = "".join(visible_characters)

    result_lines: list[str] = []
    previous_line_was_empty = False

    for line in normalized.split("\n"):
        cleaned_line = line.rstrip()
        line_is_empty = not cleaned_line.strip()

        if line_is_empty and previous_line_was_empty:
            continue

        result_lines.append("" if line_is_empty else cleaned_line)
        previous_line_was_empty = line_is_empty

    return "\n".join(result_lines).strip()


def build_extracted_document(
    file: ValidatedFile,
    parts: tuple[DocumentPart, ...],
    warnings: tuple[str, ...] = (),
    page_count: int | None = None,
) -> ExtractedDocument:
    useful_parts = tuple(part for part in parts if part.text or part.section_title)

    rendered_parts: list[str] = []

    for part in useful_parts:
        values: list[str] = []
        if part.section_title is not None:
            values.append(part.section_title)
        if part.text:
            values.append(part.text)
        rendered_parts.append("\n\n".join(values))

    full_text = "\n\n".join(rendered_parts).strip()

    if not full_text:
        raise EmptyDocumentError("В документе не найден пригодный текст.")

    metadata = DocumentMetadata(
        original_name=file.original_name,
        kind=file.kind,
        media_type=file.media_type,
        size_bytes=file.size_bytes,
        part_count=len(useful_parts),
        page_count=page_count,
        character_count=len(full_text),
    )

    return ExtractedDocument(
        text=full_text,
        parts=useful_parts,
        metadata=metadata,
        warnings=tuple(dict.fromkeys(warnings)),
    )


class DocumentExtractor:
    def __init__(
        self,
        extractors: tuple[DocumentFormatExtractor, ...],
    ) -> None:
        if not extractors:
            raise ValueError("Нужен хотя бы один экстрактор формата.")

        registry: dict[FileKind, DocumentFormatExtractor] = {}

        for extractor in extractors:
            for kind in extractor.supported_kinds:
                if kind in registry:
                    raise ValueError(f"Для формата {kind} уже есть экстрактор.")
                registry[kind] = extractor

        self._registry = registry

    def extract(self, file: ValidatedFile) -> ExtractedDocument:
        extractor = self._registry.get(file.kind)
        if extractor is None:
            raise UnsupportedDocumentFormatError(
                f"Извлечение из формата {file.kind} " "пока не поддерживается."
            )

        return extractor.extract(file)
