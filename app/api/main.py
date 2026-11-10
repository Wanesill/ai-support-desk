from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.lifespan import lifespan
from app.api.middleware import add_request_id
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.support_requests import router as support_requests_router
from app.observability.logging import configure_application_logging


def create_api() -> FastAPI:
    application = FastAPI(
        title="AI Support Desk API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.middleware("http")(add_request_id)
    register_error_handlers(application)
    application.include_router(health_router)
    application.include_router(support_requests_router)
    application.include_router(chat_router)
    application.include_router(documents_router)
    application.include_router(metrics_router)
    return application


configure_application_logging()
app = create_api()