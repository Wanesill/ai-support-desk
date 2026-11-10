from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from app.chat_history import SlidingWindowHistory, TokenCounter
from app.config import Settings
from app.files.documents.factory import (
    build_default_document_extractor,
)
from app.files.images.preparation import (
    ImageInputPreparer,
    ImagePreparationPolicy,
)
from app.files.tables.analysis import TableAnalyzer
from app.files.tables.extraction import TableExtractionPolicy
from app.files.tables.factory import (
    build_default_table_extractor,
)
from app.files.temporary_storage import TemporaryFileStorage
from app.files.upload_storage import UploadedFileStorage
from app.files.validation import FileValidationPolicy, FileValidator
from app.llm.client import LLMClient
from app.llm.cost import LLMCostEstimator
from app.observability.llm import LLMObserver
from app.services.chat_service import ChatService
from app.services.conversation_summary_service import (
    ConversationSummaryService,
)
from app.services.document_ai_context import (
    DocumentContextPreparer,
    TableContextPreparer,
)
from app.services.document_ai_service import DocumentAIService
from app.services.document_chunking import (
    DocumentChunkingPolicy,
    DocumentChunkPreparer,
)
from app.services.document_content_analyzer import (
    DocumentContentAnalyzer,
)
from app.services.image_content_analyzer import ImageContentAnalyzer
from app.services.large_document_analyzer import (
    LargeDocumentAnalysisPolicy,
    LargeDocumentAnalyzer,
)
from app.services.support_request_analyzer import (
    SupportRequestAnalyzer,
)
from app.services.tool_loop import BoundedToolLoop
from app.tools.default_registry import build_default_tool_registry
from app.tools.execution import ToolExecutor


@dataclass
class ApplicationContainer:
    settings: Settings
    chat_service: ChatService
    request_analyzer: SupportRequestAnalyzer
    document_ai_service: DocumentAIService
    upload_storage: UploadedFileStorage
    llm_client: LLMClient
    cost_estimator: LLMCostEstimator
    llm_observer: LLMObserver
    tool_executor: ToolExecutor
    upload_runtime: TemporaryDirectory[str]

    def close(self) -> None:
        try:
            self.tool_executor.close()
        finally:
            try:
                self.llm_client.close()
            finally:
                self.upload_runtime.cleanup()


def build_document_ai_service(
        llm_client: LLMClient,
        allowed_root: Path,
        max_size_bytes: int = 10 * 1_024 * 1_024,
) -> DocumentAIService:
    file_policy = FileValidationPolicy(
        allowed_root=allowed_root,
        max_size_bytes=max_size_bytes,
    )
    validator = FileValidator(file_policy)
    storage = TemporaryFileStorage(max_size_bytes=file_policy.max_size_bytes)

    document_extractor = build_default_document_extractor()

    table_policy = TableExtractionPolicy(
        max_rows=5_000,
        max_columns=100,
    )
    table_extractor = build_default_table_extractor(table_policy)
    table_analyzer = TableAnalyzer()
    content_analyzer = DocumentContentAnalyzer(llm_client)
    document_chunk_preparer = DocumentChunkPreparer(
        DocumentChunkingPolicy(
            max_chunk_characters=12_000,
            max_chunks=8,
        )
    )
    large_document_analyzer = LargeDocumentAnalyzer(
        content_analyzer=content_analyzer,
        policy=LargeDocumentAnalysisPolicy(
            max_reduce_characters=32_000,
        ),
    )
    image_preparer = ImageInputPreparer(
        ImagePreparationPolicy(
            max_images=4,
            max_pixels=25_000_000,
            max_side=2_048,
        )
    )
    image_analyzer = ImageContentAnalyzer(llm_client)

    return DocumentAIService(
        file_policy=file_policy,
        validator=validator,
        storage=storage,
        document_extractor=document_extractor,
        table_extractor=table_extractor,
        document_context_preparer=DocumentContextPreparer(),
        table_context_preparer=TableContextPreparer(table_analyzer),
        content_analyzer=content_analyzer,
        document_chunk_preparer=document_chunk_preparer,
        large_document_analyzer=large_document_analyzer,
        image_preparer=image_preparer,
        image_analyzer=image_analyzer,
    )


def build_application(settings: Settings) -> ApplicationContainer:
    cost_estimator = LLMCostEstimator(
        input_price_per_million=(
            settings.llm_input_price_per_million
        ),
        output_price_per_million=(
            settings.llm_output_price_per_million
        ),
        currency=settings.llm_cost_currency,
    )
    llm_observer = LLMObserver(cost_estimator)
    llm_client = LLMClient(
        settings,
        observer=llm_observer,
    )

    upload_runtime = TemporaryDirectory(
        prefix="ai-support-api-"
    )
    upload_root = Path(upload_runtime.name)

    upload_storage = UploadedFileStorage(
        root=upload_root,
        max_size_bytes=settings.file_max_size_bytes,
    )
    document_ai_service = build_document_ai_service(
        llm_client=llm_client,
        allowed_root=upload_root,
        max_size_bytes=settings.file_max_size_bytes,
    )

    tool_registry = build_default_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    tool_loop = BoundedToolLoop(
        client=llm_client,
        registry=tool_registry,
        executor=tool_executor,
        max_steps=settings.tool_max_steps,
    )

    token_counter = TokenCounter()
    history_policy = SlidingWindowHistory(
        token_counter=token_counter,
        input_token_budget=settings.chat_input_token_budget,
    )
    summary_service = ConversationSummaryService(llm_client)

    chat_service = ChatService(
        tool_loop=tool_loop,
        history_policy=history_policy,
        summary_service=summary_service,
    )
    request_analyzer = SupportRequestAnalyzer(llm_client)

    return ApplicationContainer(
        settings=settings,
        chat_service=chat_service,
        request_analyzer=request_analyzer,
        document_ai_service=document_ai_service,
        upload_storage=upload_storage,
        llm_client=llm_client,
        cost_estimator=cost_estimator,
        llm_observer=llm_observer,
        tool_executor=tool_executor,
        upload_runtime=upload_runtime,
    )
