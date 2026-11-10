from dataclasses import dataclass
from typing import Literal

import pandas as pd

from app.files.tables.extraction import ExtractedTable

AggregationOperation = Literal[
    "count",
    "sum",
    "mean",
    "min",
    "max",
]


class TableAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    data_type: str
    missing_count: int
    unique_count: int


@dataclass(frozen=True)
class TableProfile:
    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]


@dataclass(frozen=True)
class PreparedTableContext:
    text: str
    columns: tuple[str, ...]
    included_rows: int
    omitted_rows: int
    source_truncated: bool


@dataclass(frozen=True)
class TableContextPolicy:
    max_preview_rows: int = 20
    max_characters: int = 12_000

    def __post_init__(self) -> None:
        if self.max_preview_rows <= 0:
            raise ValueError("max_preview_rows должен быть больше нуля.")
        if self.max_characters <= 0:
            raise ValueError("max_characters должен быть больше нуля.")


class TableAnalyzer:
    def __init__(
        self,
        context_policy: TableContextPolicy | None = None,
    ) -> None:
        self._context_policy = context_policy or TableContextPolicy()

    @staticmethod
    def profile(table: ExtractedTable) -> TableProfile:
        column_profiles: list[ColumnProfile] = []

        for name in table.frame.columns:
            series = table.frame[name]
            column_profiles.append(
                ColumnProfile(
                    name=name,
                    data_type=str(series.dtype),
                    missing_count=int(series.isna().sum()),
                    unique_count=int(series.nunique(dropna=True)),
                )
            )

        return TableProfile(
            row_count=len(table.frame),
            column_count=len(table.frame.columns),
            columns=tuple(column_profiles),
        )

    def prepare_context(
        self,
        table: ExtractedTable,
        columns: tuple[str, ...] | None = None,
    ) -> PreparedTableContext:
        selected_columns = tuple(table.frame.columns) if columns is None else columns

        if not selected_columns:
            raise TableAnalysisError("Нужно выбрать хотя бы одну колонку.")

        self._require_columns(table.frame, selected_columns)

        preview = (
            table.frame.loc[:, list(selected_columns)]
            .head(self._context_policy.max_preview_rows)
            .copy()
        )
        text = preview.to_csv(index=False)

        if len(text) > self._context_policy.max_characters:
            raise TableAnalysisError(
                "Подготовленное представление слишком большое. "
                "Выберите меньше колонок."
            )

        return PreparedTableContext(
            text=text,
            columns=selected_columns,
            included_rows=len(preview),
            omitted_rows=max(
                len(table.frame) - len(preview),
                0,
            ),
            source_truncated=table.metadata.truncated,
        )

    def aggregate(
        self,
        table: ExtractedTable,
        group_by: str,
        operation: AggregationOperation,
        value_column: str | None = None,
    ) -> pd.DataFrame:
        allowed_operations = {
            "count",
            "sum",
            "mean",
            "min",
            "max",
        }
        if operation not in allowed_operations:
            raise TableAnalysisError(f"Операция {operation!r} не поддерживается.")

        self._require_columns(table.frame, (group_by,))
        frame = table.frame.copy()

        if operation == "count":
            return (
                frame.groupby(
                    group_by,
                    dropna=False,
                    sort=True,
                )
                .size()
                .reset_index(name="count")
            )

        if value_column is None:
            raise TableAnalysisError("Для числовой операции нужна колонка значений.")

        self._require_columns(table.frame, (value_column,))
        frame[value_column] = pd.to_numeric(
            frame[value_column],
            errors="coerce",
        )
        frame = frame.dropna(subset=[value_column])

        if frame.empty:
            raise TableAnalysisError(f"В колонке {value_column!r} нет чисел.")

        result = (
            frame.groupby(
                group_by,
                dropna=False,
                sort=True,
            )[value_column]
            .agg(operation)
            .reset_index()
        )
        result_name = f"{operation}_{value_column}"
        return result.rename(columns={value_column: result_name})

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> None:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise TableAnalysisError(
                "В таблице отсутствуют колонки: " + ", ".join(missing)
            )
