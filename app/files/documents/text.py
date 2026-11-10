import re

from app.files.documents.extraction import (
    DocumentPart,
    DocumentReadError,
    ExtractedDocument,
    UnsupportedDocumentFormatError,
    build_extracted_document,
    normalize_extracted_text,
)
from app.files.validation import FileKind, ValidatedFile


class TextDocumentExtractor:
    supported_kinds = frozenset({FileKind.TEXT, FileKind.MARKDOWN})

    _heading_pattern = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")

    def extract(self, file: ValidatedFile) -> ExtractedDocument:
        if file.kind not in self.supported_kinds:
            raise UnsupportedDocumentFormatError(
                f"TextDocumentExtractor не поддерживает {file.kind}."
            )

        try:
            raw_text = file.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise DocumentReadError(
                "Не удалось прочитать текстовый документ."
            ) from error

        text = normalize_extracted_text(raw_text)

        if file.kind == FileKind.TEXT:
            parts = (
                DocumentPart(
                    reference="document",
                    text=text,
                ),
            )
        else:
            parts = self._split_markdown(text)

        return build_extracted_document(file, parts)

    def _split_markdown(
        self,
        text: str,
    ) -> tuple[DocumentPart, ...]:
        parts: list[DocumentPart] = []
        body_lines: list[str] = []
        current_reference = "document"
        current_title: str | None = None
        current_level: int | None = None
        section_number = 0

        def flush_part() -> None:
            body = normalize_extracted_text("\n".join(body_lines))
            if not body and current_title is None:
                return

            parts.append(
                DocumentPart(
                    reference=current_reference,
                    text=body,
                    section_title=current_title,
                    heading_level=current_level,
                )
            )
            body_lines.clear()

        for line in text.split("\n"):
            match = self._heading_pattern.match(line)
            if match is None:
                body_lines.append(line)
                continue

            flush_part()
            section_number += 1
            current_reference = f"section:{section_number}"
            current_level = len(match.group(1))
            current_title = match.group(2).strip()

        flush_part()
        return tuple(parts)
