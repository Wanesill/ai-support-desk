from enum import StrEnum
from threading import Lock
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.tools.execution import ToolBusinessError
from app.tools.order_status import ORDER_DATA
from app.tools.registry import RegisteredTool, ToolAccess

CREATE_RETURN_REQUEST_TOOL_NAME = "create_return_request"


class ReturnReason(StrEnum):
    DEFECTIVE = "defective"
    DAMAGED = "damaged"
    OTHER = "other"


class CreateReturnRequestArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        pattern=r"^\d{4}$",
        description="Номер заказа из четырех цифр.",
    )
    reason: ReturnReason
    comment: str | None = Field(
        max_length=300,
        description=("Краткое пояснение клиента или null, " "если пояснения нет."),
    )

    @field_validator("comment")
    @classmethod
    def normalize_comment(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())
        return normalized or None


class ReturnRequestStatus(StrEnum):
    CREATED = "created"


class CreateReturnRequestResult(BaseModel):
    request_id: str
    order_id: str
    status: ReturnRequestStatus
    created_now: bool
    message: str


_RETURN_REQUESTS_BY_ORDER: dict[str, CreateReturnRequestResult] = {}
_RETURN_REQUESTS_LOCK = Lock()


def create_return_request(
    arguments: CreateReturnRequestArguments,
) -> CreateReturnRequestResult:
    if arguments.order_id not in ORDER_DATA:
        raise ToolBusinessError(
            "Заказ с таким номером не найден.",
            code="order_not_found",
        )

    with _RETURN_REQUESTS_LOCK:
        existing = _RETURN_REQUESTS_BY_ORDER.get(arguments.order_id)
        if existing is not None:
            return existing.model_copy(
                update={
                    "created_now": False,
                    "message": ("Заявка на возврат уже существует."),
                }
            )

        result = CreateReturnRequestResult(
            request_id=f"return_{uuid4().hex[:12]}",
            order_id=arguments.order_id,
            status=ReturnRequestStatus.CREATED,
            created_now=True,
            message="Заявка на возврат создана.",
        )
        _RETURN_REQUESTS_BY_ORDER[arguments.order_id] = result
        return result


RETURN_REQUEST_TOOL = RegisteredTool[
    CreateReturnRequestArguments,
    CreateReturnRequestResult,
](
    name=CREATE_RETURN_REQUEST_TOOL_NAME,
    description=(
        "Подготавливает создание заявки на возврат "
        "для существующего заказа. Используй только "
        "если пользователь явно просит оформить возврат "
        "и указал номер заказа и причину."
    ),
    arguments_model=CreateReturnRequestArguments,
    handler=create_return_request,
    access=ToolAccess.WRITE,
    timeout_seconds=2.0,
)
