from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.dependencies import get_container
from app.api.errors import ApiError
from app.api.schemas.chat import (
    ChatHistoryMessageResponse,
    ChatReplyRequest,
    ChatReplyResponse,
    ChatSessionCreatedResponse,
    ChatSessionResponse,
    ChatStreamDeltaResponse,
    ToolActionResponse,
)
from app.api.schemas.common import build_token_usage_response
from app.api.schemas.errors import (
    ErrorContentResponse,
    ErrorResponse,
)
from app.api.sse import encode_sse
from app.application import ApplicationContainer
from app.services.chat_service import (
    ChatServiceError,
    ChatSessionConflictError,
    ChatSessionNotFoundError,
    ChatStreamCompleted,
    ChatTextDelta,
)

router = APIRouter(tags=["chat"])


@router.post(
    "/chat/sessions",
    status_code=status.HTTP_201_CREATED,
)
def create_chat_session(
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> ChatSessionCreatedResponse:
    session_id = container.chat_service.create_session()
    return ChatSessionCreatedResponse(
        session_id=session_id
    )


@router.get(
    "/chat/sessions/{session_id}",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
    },
)
def get_chat_session(
    session_id: str,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> ChatSessionResponse:
    try:
        snapshot = container.chat_service.get_session(
            session_id
        )
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error

    return ChatSessionResponse(
        session_id=snapshot.session_id,
        messages=[
            ChatHistoryMessageResponse(
                role=message.role,
                content=message.content,
            )
            for message in snapshot.messages
        ],
        history_messages=snapshot.history_messages,
        has_summary=snapshot.has_summary,
        pending_confirmation=(
            snapshot.pending_confirmation
        ),
    )


@router.post(
    "/chat/sessions/{session_id}/messages",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "Состояние сессии не позволяет "
                "отправить сообщение."
            ),
        },
        422: {
            "model": ErrorResponse,
            "description": "Запрос не прошёл проверку.",
        },
        502: {
            "model": ErrorResponse,
            "description": "Ответ чата не получен.",
        },
    },
)
def reply_to_chat(
    session_id: str,
    payload: ChatReplyRequest,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> ChatReplyResponse:
    """Добавляет сообщение в существующую сессию."""
    try:
        result = container.chat_service.reply(
            session_id=session_id,
            user_text=payload.message,
        )
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error
    except ChatSessionConflictError as error:
        raise ApiError(
            status_code=status.HTTP_409_CONFLICT,
            code="chat_session_conflict",
            message=str(error),
        ) from error
    except ChatServiceError as error:
        raise ApiError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="chat_reply_failed",
            message="Не удалось получить ответ чата.",
        ) from error

    return ChatReplyResponse(
        text=result.text,
        session_id=result.session_id,
        history_messages=result.history_messages,
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
        tool_calls_count=result.tool_calls_count,
        tool_names=list(result.tool_names),
        requires_confirmation=(
            result.requires_confirmation
        ),
        summary_updated=result.summary_updated,
        has_summary=result.has_summary,
    )


@router.post(
    "/chat/sessions/{session_id}/messages/stream",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "Состояние сессии не позволяет "
                "отправить сообщение."
            ),
        },
        422: {
            "model": ErrorResponse,
            "description": "Запрос не прошёл проверку.",
        },
    },
)
def stream_chat_reply(
        session_id: str,
        payload: ChatReplyRequest,
        request: Request,
        container: Annotated[
            ApplicationContainer,
            Depends(get_container),
        ],
) -> StreamingResponse:
    try:
        session = container.chat_service.get_session(session_id)
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error

    if session.pending_confirmation:
        raise ApiError(
            status_code=status.HTTP_409_CONFLICT,
            code="chat_session_conflict",
            message=(
                "Сначала подтвердите или отмените "
                "подготовленное действие."
            ),
        )

    events = _stream_chat_events(
        request=request,
        session_id=session_id,
        message=payload.message,
        container=container,
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_chat_events(
        request: Request,
        session_id: str,
        message: str,
        container: ApplicationContainer,
) -> AsyncGenerator[str, None]:
    stream = container.chat_service.stream_reply(
        session_id=session_id,
        user_text=message,
    )

    try:
        async for event in iterate_in_threadpool(stream):
            if await request.is_disconnected():
                break
            if isinstance(event, ChatTextDelta):
                yield encode_sse(
                    "delta",
                    ChatStreamDeltaResponse(text=event.text),
                )
                continue

            if isinstance(event, ChatStreamCompleted):
                result = event.result
                yield encode_sse(
                    "completed",
                    ChatReplyResponse(
                        text=result.text,
                        session_id=result.session_id,
                        history_messages=result.history_messages,
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
                        tool_calls_count=result.tool_calls_count,
                        tool_names=list(result.tool_names),
                        requires_confirmation=(
                            result.requires_confirmation
                        ),
                        summary_updated=result.summary_updated,
                        has_summary=result.has_summary,
                    ),
                )

    except ChatSessionNotFoundError:
        yield _stream_error(
            request=request,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        )
    except ChatSessionConflictError as error:
        yield _stream_error(
            request=request,
            code="chat_session_conflict",
            message=str(error),
        )
    except ChatServiceError:
        yield _stream_error(
            request=request,
            code="chat_stream_failed",
            message="Не удалось завершить потоковый ответ.",
        )
    finally:
        stream.close()


def _stream_error(
        request: Request,
        code: str,
        message: str,
) -> str:
    payload = ErrorResponse(
        error=ErrorContentResponse(
            code=code,
            message=message,
            request_id=getattr(
                request.state,
                "request_id",
                "unknown",
            ),
        )
    )
    return encode_sse("error", payload)


@router.delete(
    "/chat/sessions/{session_id}/messages",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
    },
)
def clear_chat_session(
    session_id: str,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> Response:
    try:
        container.chat_service.clear_session(session_id)
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.delete(
    "/chat/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
    },
)
def delete_chat_session(
    session_id: str,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> Response:
    try:
        container.chat_service.delete_session(session_id)
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.post(
    "/chat/sessions/{session_id}/confirm",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "В сессии нет действия "
                "для подтверждения."
            ),
        },
        502: {
            "model": ErrorResponse,
            "description": "Действие не выполнено.",
        },
    },
)
def confirm_chat_action(
    session_id: str,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> ToolActionResponse:
    try:
        result = container.chat_service.confirm_pending(
            session_id
        )
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error
    except ChatSessionConflictError as error:
        raise ApiError(
            status_code=status.HTTP_409_CONFLICT,
            code="chat_session_conflict",
            message=str(error),
        ) from error
    except ChatServiceError as error:
        raise ApiError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="tool_confirmation_failed",
            message=(
                "Не удалось выполнить "
                "подтверждённое действие."
            ),
        ) from error

    return ToolActionResponse(
        text=result.text,
        tool_name=result.audit.tool_name,
        status=result.audit.status,
        error_code=result.audit.error_code,
    )


@router.post(
    "/chat/sessions/{session_id}/cancel",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Чат-сессия не найдена.",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "В сессии нет действия для отмены."
            ),
        },
        502: {
            "model": ErrorResponse,
            "description": "Действие не отменено.",
        },
    },
)
def cancel_chat_action(
    session_id: str,
    container: Annotated[
        ApplicationContainer,
        Depends(get_container),
    ],
) -> ToolActionResponse:
    try:
        result = container.chat_service.cancel_pending(
            session_id
        )
    except ChatSessionNotFoundError as error:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="chat_session_not_found",
            message="Чат-сессия не найдена.",
        ) from error
    except ChatSessionConflictError as error:
        raise ApiError(
            status_code=status.HTTP_409_CONFLICT,
            code="chat_session_conflict",
            message=str(error),
        ) from error
    except ChatServiceError as error:
        raise ApiError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            code="tool_cancellation_failed",
            message=(
                "Не удалось отменить "
                "подготовленное действие."
            ),
        ) from error

    return ToolActionResponse(
        text=result.text,
        tool_name=result.audit.tool_name,
        status=result.audit.status,
        error_code=result.audit.error_code,
    )