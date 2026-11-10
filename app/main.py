from pydantic import ValidationError

from app.application import build_application
from app.config import Settings
from app.console import run_console


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    application = build_application(settings)

    try:
        run_console(
            application.chat_service,
            application.request_analyzer,
            app_env=application.settings.app_env,
            configured_model=application.settings.model,
        )
    finally:
        application.close()


if __name__ == "__main__":
    main()
