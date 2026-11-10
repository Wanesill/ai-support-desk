from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_container
from app.api.errors import ApiError
from app.api.schemas.common import build_token_usage_response
from app.api.schemas.errors import ErrorResponse
from app.api.schemas.support_requests import (
    SupportRequestAnalysisResponse,
    SupportRequestAnalyzeRequest,
    SupportRequestAnalyzeResponse,
)
from app.application import ApplicationContainer
from app.services.support_request_analyzer import SupportRequestAnalyzerError

router = APIRouter(tags=["support requests"])


@router.post(
    "/support-requests/analyze",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Запрос не прошёл проверку.",
        },
        502: {
            "model": ErrorResponse,
            "description": "AI-анализ не выполнен.",
        },
    },
)
def analyze_support_request(
    payload: SupportRequestAnalyzeRequest,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> SupportRequestAnalyzeResponse:
    """Анализирует обращение клиента."""
    try:
        result = container.request_analyzer.analyze(payload.text)
    except SupportRequestAnalyzerError as error:
        raise ApiError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="support_analysis_failed",
            message="Не удалось проанализировать обращение.",
        ) from error

    return SupportRequestAnalyzeResponse(
        analysis=SupportRequestAnalysisResponse(
            category=result.analysis.category,
            order_id=result.analysis.order_id,
            product=result.analysis.product,
            problem_description=result.analysis.problem_description,
            urgency=result.analysis.urgency,
            missing_fields=result.analysis.missing_fields,
            confidence=result.analysis.confidence,
        ),
        model=result.model,
        prompt_id=result.prompt_id,
        prompt_version=result.prompt_version,
        elapsed_seconds=result.elapsed_seconds,
        usage=build_token_usage_response(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_estimator=container.cost_estimator,
        ),
        repair_attempts=result.repair_attempts,
    )