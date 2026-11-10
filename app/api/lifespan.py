from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.application import build_application
from app.config import Settings


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    settings = Settings()
    container = build_application(settings)
    application.state.container = container

    try:
        yield
    finally:
        container.close()