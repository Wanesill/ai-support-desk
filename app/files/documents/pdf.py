from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.files.documents.extraction import (
    DocumentLimitError,
    DocumentPart,
    DocumentReadError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ExtractedDocument,
    UnsupportedDocumentFormatError,
    build_extracted_document,
    normalize_extracted_text,
)
from app.files.validation import FileKind, ValidatedFile


@dataclass(frozen=True)
class PdfExtractionPolicy:
    max_pages: int = 200
    max_page_content_bytes: int = 5 * 1_024 * 1_024

    def __post_init__(self) -> None:
        if self.max_pages <= 0:
            raise ValueError("max_pages должен быть больше нуля.")
        if self.max_page_content_bytes <= 0:
            raise ValueError("max_page_content_bytes должен быть больше нуля.")


class PdfDocumentExtractor:
    supported_kinds = frozenset({FileKind.PDF})

    def __init__(
        self,
        policy: PdfExtractionPolicy | None = None,
    ) -> None:
        self._policy = policy or PdfExtractionPolicy()

    def extract(self, file: ValidatedFile) -> ExtractedDocument:
        if file.kind not in self.supported_kinds:
            raise UnsupportedDocumentFormatError(
                f"PdfDocumentExtractor не поддерживает {file.kind}."
            )

        reader = self._open_reader(file)

        try:
            if reader.is_encrypted:
                raise EncryptedDocumentError(
                    "Зашифрованные PDF пока не поддерживаются."
                )

            page_count = len(reader.pages)
            if page_count > self._policy.max_pages:
                raise DocumentLimitError("PDF содержит слишком много страниц.")

            parts: list[DocumentPart] = []
            warnings: list[str] = []

            for index, page in enumerate(reader.pages):
                page_number = index + 1
                text = self._extract_page(page, page_number)

                if not text:
                    warnings.append(
                        f"Страница {page_number}: " "текстовый слой не найден."
                    )
                    continue

                parts.append(
                    DocumentPart(
                        reference=f"page:{page_number}",
                        text=text,
                        page_number=page_number,
                    )
                )

            if not parts:
                raise EmptyDocumentError(
                    "В PDF не найден текстовый слой. " "Вероятно, документ требует OCR."
                )

            return build_extracted_document(
                file=file,
                parts=tuple(parts),
                warnings=tuple(warnings),
                page_count=page_count,
            )
        finally:
            reader.close()

    @staticmethod
    def _open_reader(file: ValidatedFile) -> PdfReader:
        try:
            return PdfReader(file.path, strict=False)
        except (OSError, PdfReadError) as error:
            raise DocumentReadError("Не удалось открыть PDF.") from error

    def _extract_page(
        self,
        page: Any,
        page_number: int,
    ) -> str:
        try:
            contents = page.get_contents()
            if contents is not None:
                content_size = len(contents.get_data())
                if content_size > self._policy.max_page_content_bytes:
                    raise DocumentLimitError(
                        f"Страница {page_number} превышает "
                        "лимит распакованного содержимого."
                    )

            raw_text = page.extract_text() or ""
        except DocumentLimitError:
            raise
        except (
            OSError,
            PdfReadError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise DocumentReadError(
                f"Не удалось извлечь страницу {page_number}."
            ) from error

        return normalize_extracted_text(raw_text)
