from dataclasses import dataclass

from app.files.documents.extraction import ExtractedDocument
from app.files.tables.analysis import (
    AggregationOperation,
    TableAnalyzer,
)
from app.files.tables.extraction import ExtractedTable
from app.files.validation import FileKind


class DocumentAIContextError(RuntimeError):
    pass


class DocumentContextTooLargeError(DocumentAIContextError):
    pass


@dataclass(frozen=True)
class PreparedFileContext:
    source_name: str
    source_kind: FileKind | str
    content: str
    references: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DocumentContextPolicy:
    max_characters: int = 24_000
    max_documents: int = 2

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError(
                "max_characters должен быть больше нуля."
            )
        if self.max_documents < 2:
            raise ValueError(
                "max_documents должен быть не меньше двух."
            )


class DocumentContextPreparer:
    def __init__(
        self,
        policy: DocumentContextPolicy | None = None,
    ) -> None:
        self._policy = policy or DocumentContextPolicy()

    def prepare(
        self,
        document: ExtractedDocument,
    ) -> PreparedFileContext:
        blocks: list[str] = []
        references: list[str] = []

        for part in document.parts:
            values: list[str] = []
            if part.section_title is not None:
                values.append(part.section_title)
            if part.text:
                values.append(part.text)

            if not values:
                continue

            blocks.append(f"[source:{part.reference}]\n" + "\n".join(values))
            references.append(part.reference)

        content = "\n\n".join(blocks).strip()
        if not content:
            raise DocumentAIContextError("Не удалось подготовить текст документа.")

        if len(content) > self._policy.max_characters:
            raise DocumentContextTooLargeError(
                "Документ слишком большой для обработки целиком."
            )

        return PreparedFileContext(
            source_name=document.metadata.original_name,
            source_kind=document.metadata.kind,
            content=content,
            references=tuple(dict.fromkeys(references)),
            warnings=document.warnings,
        )

    def prepare_many(
        self,
        documents: tuple[ExtractedDocument, ...],
    ) -> PreparedFileContext:
        if len(documents) < 2:
            raise DocumentAIContextError(
                "Для сравнения нужны минимум два документа."
            )
        if len(documents) > self._policy.max_documents:
            raise DocumentAIContextError(
                "Передано слишком много документов."
            )

        blocks: list[str] = []
        references: list[str] = []
        source_names: list[str] = []
        warnings: list[str] = []

        for document_index, document in enumerate(
            documents,
            start=1,
        ):
            source_name = " ".join(
                document.metadata.original_name.split()
            )
            source_names.append(source_name)
            warnings.extend(document.warnings)
            blocks.append(
                f"[document:{document_index}]\n"
                f"Имя файла: {source_name}"
            )

            for part in document.parts:
                values: list[str] = []
                if part.section_title is not None:
                    values.append(part.section_title)
                if part.text:
                    values.append(part.text)

                if not values:
                    continue

                reference = (
                    f"document:{document_index}/{part.reference}"
                )
                blocks.append(
                    f"[source:{reference}]\n"
                    + "\n".join(values)
                )
                references.append(reference)

        content = "\n\n".join(blocks).strip()
        if not content:
            raise DocumentAIContextError(
                "Не удалось подготовить документы."
            )
        if len(content) > self._policy.max_characters:
            raise DocumentContextTooLargeError(
                "Документы слишком большие для сравнения целиком."
            )

        return PreparedFileContext(
            source_name=", ".join(source_names),
            source_kind="multiple_documents",
            content=content,
            references=tuple(dict.fromkeys(references)),
            warnings=tuple(dict.fromkeys(warnings)),
        )


@dataclass(frozen=True)
class TableAnalysisPlan:
    columns: tuple[str, ...]
    group_by: str
    operation: AggregationOperation
    value_column: str | None = None

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("Нужно выбрать хотя бы одну колонку.")
        if any(not name.strip() for name in self.columns):
            raise ValueError("Названия колонок не должны быть пустыми.")
        if self.group_by not in self.columns:
            raise ValueError("Колонка группировки должна входить в columns.")

        if self.operation == "count":
            if self.value_column is not None:
                raise ValueError("Для count колонка значений не нужна.")
        elif self.value_column is None:
            raise ValueError("Для числовой операции нужна value_column.")
        elif self.value_column not in self.columns:
            raise ValueError("value_column должна входить в columns.")


@dataclass(frozen=True)
class TableAIContextPolicy:
    max_result_rows: int = 100
    max_characters: int = 16_000

    def __post_init__(self) -> None:
        if self.max_result_rows <= 0:
            raise ValueError("max_result_rows должен быть больше нуля.")
        if self.max_characters <= 0:
            raise ValueError("max_characters должен быть больше нуля.")


class TableContextPreparer:
    def __init__(
        self,
        analyzer: TableAnalyzer,
        policy: TableAIContextPolicy | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._policy = policy or TableAIContextPolicy()

    def prepare(
        self,
        table: ExtractedTable,
        plan: TableAnalysisPlan,
    ) -> PreparedFileContext:
        preview = self._analyzer.prepare_context(
            table,
            columns=plan.columns,
        )
        calculation = self._analyzer.aggregate(
            table,
            group_by=plan.group_by,
            operation=plan.operation,
            value_column=plan.value_column,
        )

        if len(calculation) > self._policy.max_result_rows:
            raise DocumentAIContextError(
                "Результат вычисления содержит слишком много строк."
            )

        reference = (
            f"sheet:{table.metadata.sheet_name}"
            if table.metadata.sheet_name is not None
            else "table"
        )
        calculation_text = calculation.to_csv(index=False)
        content = (
            f"[source:{reference}]\n"
            f"Выбранные колонки: {', '.join(plan.columns)}\n"
            "Фрагмент таблицы:\n"
            f"{preview.text}\n"
            f"Проверенный результат операции {plan.operation}:\n"
            f"{calculation_text}"
        ).strip()

        if len(content) > self._policy.max_characters:
            raise DocumentAIContextError("Контекст таблицы слишком большой.")

        warnings = list(table.warnings)
        if preview.omitted_rows > 0:
            warnings.append("В контекст вошли не все загруженные строки.")
        if preview.source_truncated:
            warnings.append("Исходная таблица была сокращена при извлечении.")

        return PreparedFileContext(
            source_name=table.metadata.original_name,
            source_kind=table.metadata.kind,
            content=content,
            references=(reference,),
            warnings=tuple(dict.fromkeys(warnings)),
        )
