from app.prompts.image_analysis.v1_0_0 import (
    IMAGE_ANALYSIS_PROMPT_V1_0_0,
)
from app.prompts.template import PromptTemplate, PromptTemplateError

ACTIVE_IMAGE_ANALYSIS_VERSION = "1.0.0"

_PROMPTS: dict[str, PromptTemplate] = {
    IMAGE_ANALYSIS_PROMPT_V1_0_0.version: (IMAGE_ANALYSIS_PROMPT_V1_0_0),
}


def get_image_analysis_prompt(
    version: str | None = None,
) -> PromptTemplate:
    selected = ACTIVE_IMAGE_ANALYSIS_VERSION if version is None else version.strip()

    try:
        return _PROMPTS[selected]
    except KeyError as error:
        available = ", ".join(_PROMPTS)
        raise PromptTemplateError(
            f"Неизвестная версия: {selected}. " f"Доступны: {available}."
        ) from error


def list_image_analysis_versions() -> tuple[str, ...]:
    return tuple(_PROMPTS)
