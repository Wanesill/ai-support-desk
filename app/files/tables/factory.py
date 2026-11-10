from app.files.tables.csv import CsvTableExtractor
from app.files.tables.excel import ExcelTableExtractor
from app.files.tables.extraction import (
    TableExtractionPolicy,
    TableExtractor,
)


def build_default_table_extractor(
    policy: TableExtractionPolicy | None = None,
) -> TableExtractor:
    extraction_policy = policy or TableExtractionPolicy()

    return TableExtractor(
        extractors=(
            CsvTableExtractor(extraction_policy),
            ExcelTableExtractor(extraction_policy),
        )
    )
