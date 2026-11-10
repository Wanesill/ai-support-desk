from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from app.files.validation import FileSizeError, ValidatedFile


class TemporaryFileStorageError(RuntimeError):
    pass


class TemporaryFileStorage:
    def __init__(self, max_size_bytes: int) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes должен быть больше нуля.")
        self._max_size_bytes = max_size_bytes

    @contextmanager
    def stage(self, file: ValidatedFile) -> Iterator[Path]:
        try:
            temporary_context = TemporaryDirectory(prefix="ai-support-desk-")
        except OSError as error:
            raise TemporaryFileStorageError(
                "Не удалось создать временный каталог."
            ) from error

        with temporary_context as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_path = temporary_root / (f"{uuid4().hex}{file.extension}")

            copied_bytes = 0

            try:
                with file.path.open("rb") as source:
                    with temporary_path.open("xb") as target:
                        while chunk := source.read(64 * 1_024):
                            copied_bytes += len(chunk)
                            if copied_bytes > self._max_size_bytes:
                                raise FileSizeError("Файл изменился и превысил лимит.")
                            target.write(chunk)
            except FileSizeError:
                raise
            except OSError as error:
                raise TemporaryFileStorageError(
                    "Не удалось подготовить временную копию."
                ) from error

            yield temporary_path
