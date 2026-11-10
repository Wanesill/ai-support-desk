from zipfile import BadZipFile

import pandas as pd

from app.files.tables.extraction import (
    ExtractedTable,
    TableExtractionError,
    TableExtractionPolicy,
    TableReadError,
    TableSheetError,
    UnsupportedTableFormatError,
    build_extracted_table,
)
from app.files.validation import FileKind, ValidatedFile


class ExcelTableExtractor:
    supported_kinds = frozenset({FileKind.XLSX})

    def __init__(self, policy: TableExtractionPolicy) -> None:
        self._policy = policy

    def extract(
        self,
        file: ValidatedFile,
        sheet_name: str | None = None,
    ) -> ExtractedTable:
        if file.kind not in self.supported_kinds:
            raise UnsupportedTableFormatError(
                f"ExcelTableExtractor не поддерживает {file.kind}."
            )

        try:
            with pd.ExcelFile(
                file.path,
                engine="openpyxl",
            ) as workbook:
                available_sheets = tuple(workbook.sheet_names)
                if not available_sheets:
                    raise TableSheetError("В книге Excel не найдено листов.")

                selected_sheet = sheet_name or available_sheets[0]
                if selected_sheet not in available_sheets:
                    raise TableSheetError(f"Лист {selected_sheet!r} не найден.")

                frame = workbook.parse(
                    sheet_name=selected_sheet,
                    nrows=self._policy.max_rows + 1,
                )
        except TableExtractionError:
            raise
        except (
            OSError,
            BadZipFile,
            ImportError,
            KeyError,
            ValueError,
        ) as error:
            raise TableReadError("Не удалось прочитать XLSX.") from error

        warnings = ()
        if sheet_name is None:
            warnings = (f"Автоматически выбран первый лист: " f"{selected_sheet}.",)

        return build_extracted_table(
            file=file,
            frame=frame,
            policy=self._policy,
            sheet_name=selected_sheet,
            available_sheets=available_sheets,
            warnings=warnings,
        )
