from pydantic import BaseModel


def encode_sse(event: str, payload: BaseModel) -> str:
    data = payload.model_dump_json(exclude_none=True)
    return f"event: {event}\ndata: {data}\n\n"