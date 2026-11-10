from io import BytesIO
from pathlib import Path

import pytest

from app.files.upload_storage import (
    UploadedFileEmptyError,
    UploadedFileSizeError,
    UploadedFileStorage,
)
from app.files.validation import (
    FileTypeError,
    FileValidationPolicy,
    FileValidator,
)


def test_storage_creates_and_removes_file(tmp_path: Path) -> None:
    storage = UploadedFileStorage(root=tmp_path, max_size_bytes=100)
    source = BytesIO(b"hello")

    with storage.stage(source, "message.txt") as path:
        assert path.exists()
        assert path.name == "message.txt"
        assert path.read_bytes() == b"hello"

    assert not path.exists()


def test_storage_rejects_empty_file(tmp_path: Path) -> None:
    storage = UploadedFileStorage(root=tmp_path, max_size_bytes=100)

    with pytest.raises(UploadedFileEmptyError, match="Пустой файл"):
        with storage.stage(BytesIO(b""), "empty.txt"):
            pass


def test_storage_rejects_large_file_and_cleans_directory(tmp_path: Path) -> None:
    storage = UploadedFileStorage(root=tmp_path, max_size_bytes=3)

    with pytest.raises(UploadedFileSizeError, match="превышает"):
        with storage.stage(BytesIO(b"1234"), "large.txt"):
            pass

    assert list(tmp_path.iterdir()) == []


def test_storage_keeps_only_file_name(tmp_path: Path) -> None:
    storage = UploadedFileStorage(root=tmp_path, max_size_bytes=100)

    with storage.stage(BytesIO(b"text"), "../../report.txt") as path:
        assert path.name == "report.txt"
        assert path.parent.parent == tmp_path


def test_validator_rejects_unknown_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "data.json"
    file_path.write_text("{}", encoding="utf-8")
    validator = FileValidator(FileValidationPolicy(allowed_root=tmp_path))

    with pytest.raises(FileTypeError, match="не поддерживается"):
        validator.validate(file_path)
