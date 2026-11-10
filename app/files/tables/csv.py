import pandas as pd

from app.files.tables.extraction import (
    ExtractedTable,
    TableExtractionPolicy,
    TableReadError,
    TableSheetError,
    UnsupportedTableFormatError,
    build_extracted_table,
)
from app.files.validation import FileKind, ValidatedFile


class CsvTableExtractor:
    supported_kinds = frozenset({FileKind.CSV})

    def __init__(self, policy: TableExtractionPolicy) -> None:
        self._policy = policy

    def extract(
        self,
        file: ValidatedFile,
        sheet_name: str | None = None,
    ) -> ExtractedTable:
        if file.kind not in self.supported_kinds:
            raise UnsupportedTableFormatError(
                f"CsvTableExtractor не поддерживает {file.kind}."
            )

        if sheet_name is not None:
            raise TableSheetError("CSV не содержит листов.")

        try:
            frame = pd.read_csv(
                file.path,
                encoding="utf-8",
                sep=None,
                engine="python",
                nrows=self._policy.max_rows + 1,
                index_col=False,
                keep_default_na=True,
                on_bad_lines="error",
            )
        except (
            OSError,
            UnicodeError,
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as error:
            raise TableReadError("Не удалось прочитать CSV.") from error

        return build_extracted_table(
            file=file,
            frame=frame,
            policy=self._policy,
        )
