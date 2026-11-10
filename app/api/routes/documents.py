from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    status,
)

from app.api.dependencies import get_container
from app.api.errors import ApiError
from app.api.schemas.common import build_token_usage_response
from app.api.schemas.documents import (
    DocumentAnalysisResponse,
    DocumentAnalyzeResponse,
    DocumentFactResponse,
)
from app.api.schemas.errors import ErrorResponse
from app.application import ApplicationContainer
from app.files.upload_storage import (
    UploadedFileEmptyError,
    UploadedFileNameError,
    UploadedFileSizeError,
    UploadedFileStorageError,
)
from app.services.document_ai_service import (
    DocumentAIInputError,
    DocumentAIModelError,
    DocumentAIPreparationError,
    DocumentAIRequest,
)

router = APIRouter(tags=["documents"])


@router.post(
    "/files/analyze",
    responses={
        413: {
            "model": ErrorResponse,
            "description": "Файл превышает лимит.",
        },
        422: {
            "model": ErrorResponse,
            "description": "Файл или параметры не прошли проверку.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Временный файл не подготовлен.",
        },
        502: {
            "model": ErrorResponse,
            "description": "AI-анализ не выполнен.",
        },
    },
)
def analyze_file(
    file: Annotated[UploadFile, File()],
    question: Annotated[
        str,
        Form(min_length=1, max_length=4_000),
    ],
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
    require_human_review: Annotated[bool, Form()] = False,
) -> DocumentAnalyzeResponse:
    try:
        with container.upload_storage.stage(
            source=file.file,
            original_name=file.filename,
        ) as temporary_path:
            result = container.document_ai_service.analyze(
                DocumentAIRequest(
                    source_path=temporary_path,
                    question=question,
                    declared_media_type=file.content_type,
                    require_human_review=require_human_review,
                )
            )
    except UploadedFileSizeError as error:
        raise ApiError(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            code="file_too_large",
            message=str(error),
        ) from error
    except (
        UploadedFileEmptyError,
        UploadedFileNameError,
        DocumentAIInputError,
        DocumentAIPreparationError,
    ) as error:
        raise ApiError(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_file",
            message=str(error),
        ) from error
    except UploadedFileStorageError as error:
        raise ApiError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="file_storage_failed",
            message="Не удалось подготовить файл.",
        ) from error
    except DocumentAIModelError as error:
        raise ApiError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="document_analysis_failed",
            message="Не удалось проанализировать файл.",
        ) from error

    return DocumentAnalyzeResponse(
        source_name=result.source_name,
        source_kind=str(result.source_kind),
        warnings=list(result.warnings),
        analysis=DocumentAnalysisResponse(
            summary=result.analysis.summary,
            answer=result.analysis.answer,
            key_facts=[
                DocumentFactResponse(
                    text=fact.text,
                    reference=fact.reference,
                )
                for fact in result.analysis.key_facts
            ],
            limitations=result.analysis.limitations,
            requires_human_review=(
                result.analysis.requires_human_review
            ),
        ),
        model=result.model,
        prompt_id=result.prompt_id,
        prompt_version=result.prompt_version,
        model_calls=result.model_calls,
        elapsed_seconds=result.elapsed_seconds,
        usage=build_token_usage_response(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_estimator=container.cost_estimator,
        ),
        response_id=result.response_id,
    )