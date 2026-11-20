from dataclasses import dataclass, replace
from pathlib import Path

from app.files.documents.extraction import (
    DocumentExtractionError,
    DocumentExtractor,
)
from app.files.images.preparation import (
    IMAGE_KINDS,
    ImageInputPreparer,
    ImagePreparationError,
    PreparedImageBatch,
)
from app.files.tables.analysis import TableAnalysisError
from app.files.tables.extraction import (
    TableExtractionError,
    TableExtractor,
)
from app.files.temporary_storage import (
    TemporaryFileStorage,
    TemporaryFileStorageError,
)
from app.files.validation import (
    FileKind,
    FileValidationError,
    FileValidationPolicy,
    FileValidator,
    ValidatedFile,
)
from app.services.document_ai_context import (
    DocumentAIContextError,
    DocumentContextPreparer,
    DocumentContextTooLargeError,
    PreparedFileContext,
    TableAnalysisPlan,
    TableContextPreparer,
)
from app.services.document_chunking import (
    DocumentChunkPreparer,
    PreparedDocumentBatch,
)
from app.services.document_content_analyzer import (
    DocumentAIResult,
    DocumentContentAnalyzer,
    DocumentContentAnalyzerError,
)
from app.services.image_content_analyzer import (
    ImageContentAnalyzer,
    ImageContentAnalyzerError,
)
from app.services.large_document_analyzer import LargeDocumentAnalyzer

DOCUMENT_KINDS = frozenset(
    {
        FileKind.TEXT,
        FileKind.MARKDOWN,
        FileKind.PDF,
        FileKind.DOCX,
    }
)
TABLE_KINDS = frozenset({FileKind.CSV, FileKind.XLSX})


class DocumentAIServiceError(RuntimeError):
    pass


class DocumentAIInputError(DocumentAIServiceError):
    pass


class DocumentAIPreparationError(DocumentAIServiceError):
    pass


class DocumentAIModelError(DocumentAIServiceError):
    pass


@dataclass(frozen=True)
class DocumentAIRequest:
    source_path: Path
    question: str
    declared_media_type: str | None = None
    sheet_name: str | None = None
    table_plan: TableAnalysisPlan | None = None
    document_references: tuple[str, ...] | None = None
    additional_document_paths: tuple[Path, ...] = ()
    additional_image_paths: tuple[Path, ...] = ()
    require_human_review: bool = False


class DocumentAIService:
    def __init__(
        self,
        file_policy: FileValidationPolicy,
        validator: FileValidator,
        storage: TemporaryFileStorage,
        document_extractor: DocumentExtractor,
        table_extractor: TableExtractor,
        document_context_preparer: DocumentContextPreparer,
        table_context_preparer: TableContextPreparer,
        content_analyzer: DocumentContentAnalyzer,
        document_chunk_preparer: DocumentChunkPreparer,
        large_document_analyzer: LargeDocumentAnalyzer,
        image_preparer: ImageInputPreparer,
        image_analyzer: ImageContentAnalyzer,
    ) -> None:
        self._file_policy = file_policy
        self._validator = validator
        self._storage = storage
        self._document_extractor = document_extractor
        self._table_extractor = table_extractor
        self._document_context_preparer = document_context_preparer
        self._table_context_preparer = table_context_preparer
        self._content_analyzer = content_analyzer
        self._document_chunk_preparer = document_chunk_preparer
        self._large_document_analyzer = large_document_analyzer
        self._image_preparer = image_preparer
        self._image_analyzer = image_analyzer

    def analyze(
        self,
        request: DocumentAIRequest,
    ) -> DocumentAIResult:
        question = request.question.strip()
        if not question:
            raise DocumentAIInputError("Вопрос не должен быть пустым.")

        try:
            context = self._prepare_from_file(request)
        except DocumentAIServiceError:
            raise
        except (
            FileValidationError,
            TemporaryFileStorageError,
        ) as error:
            raise DocumentAIInputError(str(error)) from error
        except (
            DocumentExtractionError,
            TableExtractionError,
            TableAnalysisError,
            DocumentAIContextError,
            ImagePreparationError,
        ) as error:
            raise DocumentAIPreparationError(str(error)) from error

        try:
            if isinstance(context, PreparedImageBatch):
                result = self._image_analyzer.analyze(
                    question,
                    context,
                )
            elif isinstance(context, PreparedDocumentBatch):
                result = self._large_document_analyzer.analyze(
                    question,
                    context,
                )
            else:
                result = self._content_analyzer.analyze(
                    question,
                    context,
                )
        except DocumentAIContextError as error:
            raise DocumentAIPreparationError(str(error)) from error
        except (
            DocumentContentAnalyzerError,
            ImageContentAnalyzerError,
        ) as error:
            raise DocumentAIModelError(str(error)) from error

        return self._apply_review_policy(
            result,
            required=request.require_human_review,
        )

    @staticmethod
    def _apply_review_policy(
        result: DocumentAIResult,
        required: bool,
    ) -> DocumentAIResult:
        if not required or result.analysis.requires_human_review:
            return result

        analysis = result.analysis.model_copy(
            update={"requires_human_review": True}
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *result.warnings,
                    "Проверка человеком обязательна по правилу сценария.",
                )
            )
        )

        return replace(
            result,
            analysis=analysis,
            warnings=warnings,
        )

    def _prepare_from_file(
        self,
        request: DocumentAIRequest,
    ) -> (
        PreparedFileContext
        | PreparedDocumentBatch
        | PreparedImageBatch
    ):
        validated = self._validator.validate(
            request.source_path,
            declared_media_type=request.declared_media_type,
        )

        if validated.kind in IMAGE_KINDS:
            if request.additional_document_paths:
                raise DocumentAIInputError(
                    "Документы нельзя добавить к изображению."
                )
            return self._prepare_images(validated, request)

        if request.additional_image_paths:
            raise DocumentAIInputError(
                "Дополнительные изображения можно передать "
                "только вместе с основным изображением."
            )

        if request.additional_document_paths:
            if validated.kind not in DOCUMENT_KINDS:
                raise DocumentAIInputError(
                    "Сравнивать можно только текстовые документы."
                )
            return self._prepare_documents(validated, request)

        with self._storage.stage(validated) as temporary_path:
            staged_file = self._validate_staged_file(
                validated,
                temporary_path,
            )
            return self._prepare_context(staged_file, request)

    def _prepare_documents(
        self,
        first_file: ValidatedFile,
        request: DocumentAIRequest,
    ) -> PreparedFileContext:
        if request.sheet_name is not None:
            raise DocumentAIInputError(
                "Для сравнения документов нельзя выбирать лист."
            )
        if request.table_plan is not None:
            raise DocumentAIInputError(
                "Для сравнения документов не нужен табличный план."
            )
        if request.document_references is not None:
            raise DocumentAIInputError(
                "В этом сценарии документы сравниваются целиком."
            )

        validated_files = [first_file]
        known_paths = {first_file.path}

        for path in request.additional_document_paths:
            additional = self._validator.validate(path)
            if additional.kind not in DOCUMENT_KINDS:
                raise DocumentAIInputError(
                    "Все дополнительные файлы должны быть "
                    "документами TXT, Markdown, PDF или DOCX."
                )
            if additional.path in known_paths:
                raise DocumentAIInputError(
                    "Один документ нельзя сравнивать с самим собой."
                )

            known_paths.add(additional.path)
            validated_files.append(additional)

        documents = []

        for validated in validated_files:
            with self._storage.stage(validated) as temporary_path:
                staged_file = self._validate_staged_file(
                    validated,
                    temporary_path,
                )
                documents.append(
                    self._document_extractor.extract(staged_file)
                )

        return self._document_context_preparer.prepare_many(
            tuple(documents)
        )

    def _validate_staged_file(
        self,
        original: ValidatedFile,
        temporary_path: Path,
    ) -> ValidatedFile:
        staged_policy = replace(
            self._file_policy,
            allowed_root=temporary_path.parent,
        )
        staged_validator = FileValidator(staged_policy)
        staged_file = staged_validator.validate(
            temporary_path,
            declared_media_type=original.media_type,
        )
        return replace(
            staged_file,
            original_name=original.original_name,
        )

    def _prepare_images(
        self,
        first_file: ValidatedFile,
        request: DocumentAIRequest,
    ) -> PreparedImageBatch:
        if request.sheet_name is not None:
            raise DocumentAIInputError("Для изображения нельзя выбирать лист.")
        if request.table_plan is not None:
            raise DocumentAIInputError("Для изображения не нужен табличный план.")
        if request.document_references is not None:
            raise DocumentAIInputError(
                "Для изображения нельзя выбирать части документа."
            )

        validated_files = [first_file]

        for path in request.additional_image_paths:
            additional = self._validator.validate(path)
            if additional.kind not in IMAGE_KINDS:
                raise DocumentAIInputError(
                    "Все дополнительные файлы должны быть "
                    "изображениями PNG или JPEG."
                )
            validated_files.append(additional)

        prepared_images = []

        for index, validated in enumerate(validated_files, start=1):
            with self._storage.stage(validated) as temporary_path:
                staged_file = self._validate_staged_file(
                    validated,
                    temporary_path,
                )
                prepared_images.append(
                    self._image_preparer.prepare(
                        staged_file,
                        reference=f"image:{index}",
                    )
                )

        return self._image_preparer.build_batch(tuple(prepared_images))

    def _prepare_context(
        self,
        file: ValidatedFile,
        request: DocumentAIRequest,
    ) -> PreparedFileContext | PreparedDocumentBatch:
        if file.kind in DOCUMENT_KINDS:
            if request.sheet_name is not None:
                raise DocumentAIInputError("Для документа нельзя выбирать лист.")
            if request.table_plan is not None:
                raise DocumentAIInputError("Для документа не нужен табличный план.")

            document = self._document_extractor.extract(file)
            document = self._document_chunk_preparer.select(
                document,
                request.document_references,
            )

            try:
                return self._document_context_preparer.prepare(document)
            except DocumentContextTooLargeError:
                return self._document_chunk_preparer.prepare(document)

        if file.kind in TABLE_KINDS:
            if request.document_references is not None:
                raise DocumentAIInputError(
                    "Для таблицы нельзя выбирать части документа."
                )
            if request.table_plan is None:
                raise DocumentAIInputError("Для таблицы нужен TableAnalysisPlan.")

            table = self._table_extractor.extract(
                file,
                sheet_name=request.sheet_name,
            )
            return self._table_context_preparer.prepare(
                table,
                request.table_plan,
            )

        raise DocumentAIPreparationError(
            f"AI-анализ формата {file.kind} пока не поддерживается."
        )
