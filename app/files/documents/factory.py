from app.files.documents.docx import DocxDocumentExtractor
from app.files.documents.extraction import DocumentExtractor
from app.files.documents.pdf import (
    PdfDocumentExtractor,
    PdfExtractionPolicy,
)
from app.files.documents.text import TextDocumentExtractor


def build_default_document_extractor(
    pdf_policy: PdfExtractionPolicy | None = None,
) -> DocumentExtractor:
    return DocumentExtractor(
        extractors=(
            TextDocumentExtractor(),
            PdfDocumentExtractor(policy=pdf_policy),
            DocxDocumentExtractor(),
        )
    )
