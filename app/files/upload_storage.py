from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO


class UploadedFileStorageError(RuntimeError):
    pass


class UploadedFileNameError(UploadedFileStorageError):
    pass


class UploadedFileEmptyError(UploadedFileStorageError):
    pass


class UploadedFileSizeError(UploadedFileStorageError):
    pass


class UploadedFileStorage:
    def __init__(
        self,
        root: Path,
        max_size_bytes: int,
    ) -> None:
        if max_size_bytes <= 0:
            raise ValueError(
                "max_size_bytes должен быть больше нуля."
            )

        try:
            resolved_root = root.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                "Каталог загрузок недоступен."
            ) from error

        if not resolved_root.is_dir():
            raise ValueError(
                "Каталог загрузок должен быть каталогом."
            )

        self._root = resolved_root
        self._max_size_bytes = max_size_bytes

    @contextmanager
    def stage(
            self,
            source: BinaryIO,
            original_name: str | None,
    ) -> Iterator[Path]:
        safe_name = self._safe_name(original_name)

        try:
            temporary_context = TemporaryDirectory(
                prefix="ai-support-upload-",
                dir=self._root,
            )
        except OSError as error:
            raise UploadedFileStorageError(
                "Не удалось создать каталог загрузки."
            ) from error

        with temporary_context as temporary_directory:
            temporary_path = Path(temporary_directory) / safe_name
            copied_bytes = 0

            try:
                source.seek(0)
                with temporary_path.open("xb") as target:
                    while chunk := source.read(64 * 1_024):
                        copied_bytes += len(chunk)
                        if copied_bytes > self._max_size_bytes:
                            raise UploadedFileSizeError(
                                "Файл превышает разрешённый размер."
                            )
                        target.write(chunk)
            except UploadedFileSizeError:
                raise
            except OSError as error:
                raise UploadedFileStorageError(
                    "Не удалось сохранить загруженный файл."
                ) from error

            if copied_bytes == 0:
                raise UploadedFileEmptyError(
                    "Пустой файл нельзя обработать."
                )

            yield temporary_path

    @staticmethod
    def _safe_name(original_name: str | None) -> str:
        normalized = (original_name or "").strip().replace("\\", "/")
        safe_name = normalized.rsplit("/", maxsplit=1)[-1]

        if safe_name in {"", ".", ".."} or "\x00" in safe_name:
            raise UploadedFileNameError(
                "Клиент не передал корректное имя файла."
            )

        return safe_name