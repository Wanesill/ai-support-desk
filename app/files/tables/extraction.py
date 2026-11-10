from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from app.files.validation import FileKind, ValidatedFile


class TableExtractionError(RuntimeError):
    pass


class UnsupportedTableFormatError(TableExtractionError):
    pass


class TableReadError(TableExtractionError):
    pass


class EmptyTableError(TableExtractionError):
    pass


class TableLimitError(TableExtractionError):
    pass


class TableSheetError(TableExtractionError):
    pass


@dataclass(frozen=True)
class TableExtractionPolicy:
    max_rows: int = 5_000
    max_columns: int = 100

    def __post_init__(self) -> None:
        if self.max_rows <= 0:
            raise ValueError("max_rows должен быть больше нуля.")
        if self.max_columns <= 0:
            raise ValueError("max_columns должен быть больше нуля.")


@dataclass(frozen=True)
class TableMetadata:
    original_name: str
    kind: FileKind
    size_bytes: int
    sheet_name: str | None
    available_sheets: tuple[str, ...]
    loaded_row_count: int
    column_count: int
    truncated: bool


@dataclass
class ExtractedTable:
    frame: pd.DataFrame
    metadata: TableMetadata
    warnings: tuple[str, ...]


class TableFormatExtractor(Protocol):
    @property
    def supported_kinds(self) -> frozenset[FileKind]: ...

    def extract(
        self,
        file: ValidatedFile,
        sheet_name: str | None = None,
    ) -> ExtractedTable: ...


def normalize_table_frame(
    frame: pd.DataFrame,
    policy: TableExtractionPolicy,
) -> tuple[pd.DataFrame, bool, tuple[str, ...]]:
    normalized = frame.copy()
    normalized = normalized.dropna(how="all")
    normalized = normalized.dropna(
        axis="columns",
        how="all",
    )

    if normalized.empty:
        raise EmptyTableError("В таблице не найдено строк с данными.")

    warnings: list[str] = []
    column_names: list[str] = []

    for index, value in enumerate(normalized.columns):
        name = str(value).strip()
        if not name or name.startswith("Unnamed:"):
            name = f"column_{index + 1}"
            warnings.append("Пустым заголовкам колонок присвоены имена.")
        column_names.append(name)

    if pd.Index(column_names).duplicated().any():
        raise TableReadError("После нормализации найдены одинаковые колонки.")

    normalized.columns = column_names

    if len(normalized.columns) > policy.max_columns:
        raise TableLimitError("Таблица содержит слишком много колонок.")

    truncated = len(normalized) > policy.max_rows
    if truncated:
        normalized = normalized.head(policy.max_rows).copy()
        warnings.append("Количество строк ограничено политикой извлечения.")

    normalized = normalized.reset_index(drop=True)
    return normalized, truncated, tuple(warnings)


def build_extracted_table(
    file: ValidatedFile,
    frame: pd.DataFrame,
    policy: TableExtractionPolicy,
    sheet_name: str | None = None,
    available_sheets: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> ExtractedTable:
    normalized, truncated, normalization_warnings = normalize_table_frame(frame, policy)
    all_warnings = warnings + normalization_warnings

    metadata = TableMetadata(
        original_name=file.original_name,
        kind=file.kind,
        size_bytes=file.size_bytes,
        sheet_name=sheet_name,
        available_sheets=available_sheets,
        loaded_row_count=len(normalized),
        column_count=len(normalized.columns),
        truncated=truncated,
    )

    return ExtractedTable(
        frame=normalized,
        metadata=metadata,
        warnings=tuple(dict.fromkeys(all_warnings)),
    )


class TableExtractor:
    def __init__(
        self,
        extractors: tuple[TableFormatExtractor, ...],
    ) -> None:
        if not extractors:
            raise ValueError("Нужен хотя бы один табличный экстрактор.")

        registry: dict[FileKind, TableFormatExtractor] = {}

        for extractor in extractors:
            for kind in extractor.supported_kinds:
                if kind in registry:
                    raise ValueError(f"Для формата {kind} уже есть экстрактор.")
                registry[kind] = extractor

        self._registry = registry

    def extract(
        self,
        file: ValidatedFile,
        sheet_name: str | None = None,
    ) -> ExtractedTable:
        extractor = self._registry.get(file.kind)
        if extractor is None:
            raise UnsupportedTableFormatError(
                f"Табличное извлечение для {file.kind} " "не поддерживается."
            )

        return extractor.extract(file, sheet_name=sheet_name)
