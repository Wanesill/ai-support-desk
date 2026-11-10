from app.prompts.document_analysis.registry import (
    ACTIVE_DOCUMENT_ANALYSIS_VERSION,
    get_document_analysis_prompt,
    list_document_analysis_versions,
)

DOCUMENT_ANALYSIS_PROMPT = get_document_analysis_prompt()

__all__ = [
    "ACTIVE_DOCUMENT_ANALYSIS_VERSION",
    "DOCUMENT_ANALYSIS_PROMPT",
    "get_document_analysis_prompt",
    "list_document_analysis_versions",
]
