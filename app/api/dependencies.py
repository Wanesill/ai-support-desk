from fastapi import Request

from app.application import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, ApplicationContainer):
        raise TypeError(
            "Контейнер приложения не инициализирован."
        )
    return container