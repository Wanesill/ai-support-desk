import re
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocumentObject
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.files.documents.extraction import (
    DocumentPart,
    DocumentReadError,
    EmptyDocumentError,
    ExtractedDocument,
    UnsupportedDocumentFormatError,
    build_extracted_document,
    normalize_extracted_text,
)
from app.files.validation import FileKind, ValidatedFile


class DocxDocumentExtractor:
    supported_kinds = frozenset({FileKind.DOCX})

    _heading_pattern = re.compile(r"^Heading ([1-9])$")

    def extract(self, file: ValidatedFile) -> ExtractedDocument:
        if file.kind not in self.supported_kinds:
            raise UnsupportedDocumentFormatError(
                f"DocxDocumentExtractor не поддерживает {file.kind}."
            )

        try:
            document = Document(file.path)
            parts, table_found = self._extract_parts(document)
        except EmptyDocumentError:
            raise
        except (
            OSError,
            BadZipFile,
            PackageNotFoundError,
            KeyError,
            ValueError,
        ) as error:
            raise DocumentReadError("Не удалось прочитать DOCX.") from error

        warnings = [
            "Для DOCX номера страниц не определяются: " "сохранены ссылки на разделы."
        ]
        if table_found:
            warnings.append("Таблицы DOCX преобразованы в обычный текст.")

        return build_extracted_document(
            file=file,
            parts=parts,
            warnings=tuple(warnings),
        )

    def _extract_parts(
        self,
        document: DocumentObject,
    ) -> tuple[tuple[DocumentPart, ...], bool]:
        parts: list[DocumentPart] = []
        body_blocks: list[str] = []
        current_reference = "document"
        current_title: str | None = None
        current_level: int | None = None
        section_number = 0
        table_found = False

        def flush_part() -> None:
            body = normalize_extracted_text("\n\n".join(body_blocks))
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
            body_blocks.clear()

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = normalize_extracted_text(block.text)
                if not text:
                    continue

                heading_level = self._get_heading_level(block)
                if heading_level is None:
                    body_blocks.append(text)
                    continue

                flush_part()
                section_number += 1
                current_reference = f"section:{section_number}"
                current_title = text
                current_level = heading_level
                continue

            if isinstance(block, Table):
                table_text = self._table_to_text(block)
                if table_text:
                    body_blocks.append(table_text)
                    table_found = True

        flush_part()

        if not parts:
            raise EmptyDocumentError("В DOCX не найден пригодный текст.")

        return tuple(parts), table_found

    def _get_heading_level(
        self,
        paragraph: Paragraph,
    ) -> int | None:
        style = paragraph.style
        if style is None:
            return None

        match = self._heading_pattern.fullmatch(style.name)
        if match is None:
            return None

        return int(match.group(1))

    @staticmethod
    def _table_to_text(table: Table) -> str:
        result_rows: list[str] = []

        for row in table.rows:
            cells = []
            for cell in row.cells:
                cell_text = normalize_extracted_text(cell.text)
                cells.append(cell_text.replace("\n", " / "))

            if any(cells):
                result_rows.append(" | ".join(cells))

        return "\n".join(result_rows)
