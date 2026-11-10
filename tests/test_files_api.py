from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.files.upload_storage import UploadedFileStorage
from app.files.validation import FileKind
from app.models.document_analysis import (
    DocumentAnalysis,
    DocumentFact,
)
from app.services.document_ai_service import (
    DocumentAIInputError,
    DocumentAIRequest,
)
from app.services.document_content_analyzer import DocumentAIResult


class FakeDocumentAIService:
    def __init__(self) -> None:
        self.received_request: DocumentAIRequest | None = None
        self.received_text: str | None = None

    def analyze(self, request: DocumentAIRequest) -> DocumentAIResult:
        self.received_request = request
        self.received_text = request.source_path.read_text(encoding="utf-8")

        return DocumentAIResult(
            analysis=DocumentAnalysis(
                summary="Краткое резюме.",
                answer="Ответ найден в документе.",
                key_facts=[
                    DocumentFact(
                        text="Факт из документа.",
                        reference="document",
                    )
                ],
                limitations=[],
                requires_human_review=False,
            ),
            source_name=request.source_path.name,
            source_kind=FileKind.TEXT,
            warnings=(),
            model="fake-model",
            prompt_id="document_analysis",
            prompt_version="1.0.0",
            elapsed_seconds=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            response_id="fake-response-id",
        )


def test_file_upload_returns_analysis_and_removes_file(
    client: TestClient,
    fake_container: SimpleNamespace,
    tmp_path: Path,
) -> None:
    service = FakeDocumentAIService()
    fake_container.upload_storage = UploadedFileStorage(root=tmp_path, max_size_bytes=1_000)
    fake_container.document_ai_service = service

    response = client.post(
        "/files/analyze",
        files={
            "file": (
                "request.txt",
                b"Order 8041 is damaged.",
                "text/plain",
            )
        },
        data={"question": "Что случилось?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_name"] == "request.txt"
    assert body["source_kind"] == "text"
    assert body["analysis"]["answer"] == "Ответ найден в документе."
    assert service.received_text == "Order 8041 is damaged."
    assert service.received_request is not None
    assert service.received_request.declared_media_type == "text/plain"
    assert not service.received_request.source_path.exists()


def test_large_file_returns_413_before_analysis(
    client: TestClient,
    fake_container: SimpleNamespace,
    tmp_path: Path,
) -> None:
    service = Mock()
    fake_container.upload_storage = UploadedFileStorage(root=tmp_path, max_size_bytes=3)
    fake_container.document_ai_service = service

    response = client.post(
        "/files/analyze",
        files={"file": ("large.txt", b"1234", "text/plain")},
        data={"question": "Что в файле?"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"
    service.analyze.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_invalid_file_returns_422(
    client: TestClient,
    fake_container: SimpleNamespace,
    tmp_path: Path,
) -> None:
    service = Mock()
    service.analyze.side_effect = DocumentAIInputError("Формат .json не поддерживается.")
    fake_container.upload_storage = UploadedFileStorage(root=tmp_path, max_size_bytes=1_000)
    fake_container.document_ai_service = service

    response = client.post(
        "/files/analyze",
        files={
            "file": ("data.json", b"{}", "application/json")
        },
        data={"question": "Что в файле?"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_file"
    service.analyze.assert_called_once()