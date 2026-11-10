from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.dependencies import get_container
from app.application import ApplicationContainer

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
def read_metrics(
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> Response:
    content = generate_latest(
        container.llm_observer.registry
    )
    return Response(
        content=content,
        media_type=CONTENT_TYPE_LATEST,
    )