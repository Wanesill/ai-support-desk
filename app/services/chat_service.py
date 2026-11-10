import json
from collections.abc import Generator
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from app.chat_history import (
    ChatHistoryError,
    SlidingWindowHistory,
)
from app.llm.client import (
    Message,
)
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT
from app.prompts.template import PromptTemplate
from app.services.conversation_summary_service import (
    ConversationSummaryError,
    ConversationSummaryService,
)
from app.services.tool_loop import (
    BoundedToolLoop,
    ToolLoopError,
    ToolLoopStreamCompleted,
    ToolLoopTextDelta,
)
from app.tools.execution import ToolAuditRecord


class ChatServiceError(RuntimeError):
    pass


class ChatSessionNotFoundError(ChatServiceError):
    pass


class ChatSessionConflictError(ChatServiceError):
    pass


@dataclass(frozen=True)
class ChatReply:
    text: str
    session_id: str
    history_messages: int
    context_history_messages: int
    removed_history_messages: int
    estimated_prompt_tokens: int
    summary_updated: bool
    has_summary: bool
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    response_ids: tuple[str, ...]
    tool_calls_count: int
    tool_names: tuple[str, ...]
    requires_confirmation: bool


@dataclass(frozen=True)
class ToolActionReply:
    text: str
    audit: ToolAuditRecord


@dataclass(frozen=True)
class ChatHistoryMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatSessionSnapshot:
    session_id: str
    messages: tuple[ChatHistoryMessage, ...]
    history_messages: int
    has_summary: bool
    pending_confirmation: bool


@dataclass
class ChatSession:
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None
    lock: Lock = field(default_factory=Lock)
    pending_confirmation_id: str | None = None


@dataclass(frozen=True)
class ChatTextDelta:
    text: str


@dataclass(frozen=True)
class ChatStreamCompleted:
    result: ChatReply


ChatStreamEvent = ChatTextDelta | ChatStreamCompleted


class ChatService:
    def __init__(
        self,
        tool_loop: BoundedToolLoop,
        history_policy: SlidingWindowHistory,
        summary_service: ConversationSummaryService,
        prompt: PromptTemplate = SUPPORT_CHAT_PROMPT,
    ) -> None:
        self._tool_loop = tool_loop
        self._history_policy = history_policy
        self._summary_service = summary_service
        self._prompt = prompt
        self._chat_instruction = prompt.render()
        self._sessions: dict[str, ChatSession] = {}
        self._sessions_lock = Lock()

    def create_session(self) -> str:
        session_id = uuid4().hex

        with self._sessions_lock:
            self._sessions[session_id] = ChatSession()

        return session_id

    def get_session(
            self,
            session_id: str,
    ) -> ChatSessionSnapshot:
        session_key = self._normalize_session_id(session_id)
        session = self._require_session(session_key)

        with session.lock:
            return self._build_session_snapshot(
                session_key,
                session,
            )

    def reply(self, session_id: str, user_text: str) -> ChatReply:
        result: ChatReply | None = None

        for event in self.stream_reply(session_id, user_text):
            if isinstance(event, ChatStreamCompleted):
                result = event.result

        if result is None:
            raise ChatServiceError("Поток завершился без итогового результата.")

        return result

    def stream_reply(
        self,
        session_id: str,
        user_text: str,
    ) -> Generator[ChatStreamEvent, None, None]:
        session_key = self._normalize_session_id(session_id)

        text = user_text.strip()
        if not text:
            raise ChatServiceError("Сообщение не должно быть пустым.")

        session = self._require_session(session_key)

        with session.lock:
            if session.pending_confirmation_id is not None:
                raise ChatSessionConflictError(
                    "Сначала подтвердите или отмените "
                    "подготовленное действие."
                )

            developer_message = self._build_developer_message()
            user_message = self._build_user_message(text)
            summary_message = self._build_summary_message(session.summary)
            context_messages = [summary_message] if summary_message is not None else []

            try:
                prepared = self._history_policy.prepare(
                    developer_message=developer_message,
                    history=session.messages,
                    user_message=user_message,
                    context_messages=context_messages,
                )
            except ChatHistoryError as error:
                raise ChatServiceError(
                    f"Не удалось подготовить историю: {error}"
                ) from error

            started_at = perf_counter()
            new_summary = session.summary
            summary_updated = False

            if prepared.removed_history_messages > 0:
                try:
                    new_summary = self._summary_service.summarize(
                        previous_summary=session.summary,
                        messages=session.messages,
                    )
                except ConversationSummaryError as error:
                    raise ChatServiceError(
                        f"Не удалось сократить историю: {error}"
                    ) from error

                summary_updated = True
                summary_message = self._build_summary_message(new_summary)
                if summary_message is None:
                    raise ChatServiceError("Не удалось подготовить новое summary.")

                try:
                    prepared = self._history_policy.prepare(
                        developer_message=developer_message,
                        history=[],
                        user_message=user_message,
                        context_messages=[summary_message],
                    )
                except ChatHistoryError as error:
                    raise ChatServiceError(
                        "Новое summary не помещается " f"в бюджет: {error}"
                    ) from error

            try:
                loop_completed: ToolLoopStreamCompleted | None = None

                for event in self._tool_loop.stream(prepared.messages):
                    if isinstance(event, ToolLoopTextDelta):
                        yield ChatTextDelta(text=event.text)
                    else:
                        loop_completed = event
            except ToolLoopError as error:
                raise ChatServiceError(
                    f"Не удалось получить ответ: {error}"
                ) from error

            if loop_completed is None:
                raise ChatServiceError(
                    "Поток завершился без итогового результата."
                )

            loop_result = loop_completed.result
            elapsed_seconds = perf_counter() - started_at
            reply_text = self._validate_reply(loop_result.text)

            if loop_result.confirmation is not None:
                session.pending_confirmation_id = (
                    loop_result.confirmation.confirmation_id
                )

            assistant_message: Message = {
                "role": "assistant",
                "content": reply_text,
            }

            if summary_updated:
                session.summary = new_summary
                session.messages = [user_message, assistant_message]
            else:
                session.messages.extend([user_message, assistant_message])

            history_messages = len(session.messages)

            result = ChatReply(
                text=reply_text,
                session_id=session_key,
                history_messages=history_messages,
                context_history_messages=prepared.context_history_messages,
                removed_history_messages=prepared.removed_history_messages,
                estimated_prompt_tokens=prepared.estimated_tokens,
                model=loop_result.model,
                prompt_id=self._prompt.prompt_id,
                prompt_version=self._prompt.version,
                elapsed_seconds=elapsed_seconds,
                prompt_tokens=loop_result.prompt_tokens,
                completion_tokens=loop_result.completion_tokens,
                total_tokens=loop_result.total_tokens,
                response_id=loop_result.response_ids[-1],
                response_ids=loop_result.response_ids,
                tool_calls_count=loop_result.tool_calls_count,
                tool_names=loop_result.tool_names,
                summary_updated=summary_updated,
                has_summary=session.summary is not None,
                requires_confirmation=(loop_result.confirmation is not None),
            )

        yield ChatStreamCompleted(result=result)

    def confirm_pending(
        self,
        session_id: str,
    ) -> ToolActionReply:
        session_key = self._normalize_session_id(session_id)
        session = self._require_session(session_key)

        with session.lock:
            confirmation_id = session.pending_confirmation_id
            if confirmation_id is None:
                raise ChatSessionConflictError(
                    "В этой сессии нет действия для подтверждения."
                )

            try:
                execution = self._tool_loop.confirm(confirmation_id)
            except ToolLoopError as error:
                raise ChatServiceError(
                    f"Не удалось выполнить действие: {error}"
                ) from error
            finally:
                session.pending_confirmation_id = None

            payload = json.loads(execution.content)
            text = "Результат подтвержденного действия:\n" + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            session.messages.extend(
                [
                    {
                        "role": "user",
                        "content": "Подтверждаю ранее подготовленное " "действие.",
                    },
                    {
                        "role": "assistant",
                        "content": text,
                    },
                ]
            )
            return ToolActionReply(
                text=text,
                audit=execution.audit,
            )

    def cancel_pending(
        self,
        session_id: str,
    ) -> ToolActionReply:
        session_key = self._normalize_session_id(session_id)
        session = self._require_session(session_key)

        with session.lock:
            confirmation_id = session.pending_confirmation_id
            if confirmation_id is None:
                raise ChatSessionConflictError(
                    "В этой сессии нет действия для отмены."
                )

            try:
                audit = self._tool_loop.cancel(confirmation_id)
            except ToolLoopError as error:
                raise ChatServiceError(
                    f"Не удалось отменить действие: {error}"
                ) from error
            finally:
                session.pending_confirmation_id = None

            text = "Подготовленное действие отменено."
            session.messages.extend(
                [
                    {
                        "role": "user",
                        "content": "Отменяю ранее подготовленное " "действие.",
                    },
                    {"role": "assistant", "content": text},
                ]
            )
            return ToolActionReply(text=text, audit=audit)

    def clear_session(self, session_id: str) -> None:
        session_key = self._normalize_session_id(session_id)
        session = self._require_session(session_key)

        with session.lock:
            confirmation_id = session.pending_confirmation_id
            session.pending_confirmation_id = None
            session.messages.clear()
            session.summary = None

        if confirmation_id is not None:
            try:
                self._tool_loop.cancel(confirmation_id)
            except ToolLoopError:
                pass

    def delete_session(self, session_id: str) -> None:
        session_key = self._normalize_session_id(session_id)

        with self._sessions_lock:
            session = self._sessions.pop(session_key, None)

        if session is None:
            raise ChatSessionNotFoundError(
                "Чат-сессия не найдена."
            )

        with session.lock:
            confirmation_id = session.pending_confirmation_id
            session.pending_confirmation_id = None
            session.messages.clear()
            session.summary = None

        if confirmation_id is not None:
            try:
                self._tool_loop.cancel(confirmation_id)
            except ToolLoopError:
                pass

    def _require_session(
            self,
            session_id: str,
    ) -> ChatSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise ChatSessionNotFoundError(
                "Чат-сессия не найдена."
            )

        return session

    @classmethod
    def _build_session_snapshot(
            cls,
            session_id: str,
            session: ChatSession,
    ) -> ChatSessionSnapshot:
        messages: list[ChatHistoryMessage] = []

        for message in session.messages:
            role = message["role"]
            content = message.get("content")

            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str):
                continue

            messages.append(
                ChatHistoryMessage(
                    role=role,
                    content=cls._to_public_content(
                        role,
                        content,
                    ),
                )
            )

        return ChatSessionSnapshot(
            session_id=session_id,
            messages=tuple(messages),
            history_messages=len(messages),
            has_summary=session.summary is not None,
            pending_confirmation=(
                    session.pending_confirmation_id is not None
            ),
        )

    @staticmethod
    def _to_public_content(
            role: Literal["user", "assistant"],
            content: str,
    ) -> str:
        prefix = "<customer_message>\n"
        suffix = "\n</customer_message>"

        if (
                role == "user"
                and content.startswith(prefix)
                and content.endswith(suffix)
        ):
            return content[len(prefix):-len(suffix)]

        return content

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        session_key = session_id.strip()
        if not session_key:
            raise ChatServiceError("Session ID не должен быть пустым.")
        return session_key

    def _build_developer_message(self) -> Message:
        return {
            "role": "developer",
            "content": self._chat_instruction,
        }

    @staticmethod
    def _build_user_message(user_text: str) -> Message:
        return {
            "role": "user",
            "content": f"<customer_message>\n{user_text}\n</customer_message>",
        }

    @staticmethod
    def _validate_reply(reply: str | None) -> str:
        if reply is None:
            raise ChatServiceError("Модель не вернула текст ответа.")

        cleaned_reply = reply.strip()
        if not cleaned_reply:
            raise ChatServiceError("Модель вернула пустой ответ.")

        return cleaned_reply

    @staticmethod
    def _build_summary_message(
        summary: str | None,
    ) -> Message | None:
        if summary is None:
            return None

        return {
            "role": "user",
            "content": f"<conversation_summary>\n{summary}\n</conversation_summary>",
        }
