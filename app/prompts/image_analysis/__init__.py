from app.prompts.image_analysis.registry import (
    ACTIVE_IMAGE_ANALYSIS_VERSION,
    get_image_analysis_prompt,
    list_image_analysis_versions,
)

IMAGE_ANALYSIS_PROMPT = get_image_analysis_prompt()

__all__ = [
    "ACTIVE_IMAGE_ANALYSIS_VERSION",
    "IMAGE_ANALYSIS_PROMPT",
    "get_image_analysis_prompt",
    "list_image_analysis_versions",
]
