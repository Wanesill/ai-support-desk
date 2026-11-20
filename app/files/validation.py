import codecs
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


class FileKind(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PNG = "png"
    JPEG = "jpeg"


@dataclass(frozen=True)
class FileRule:
    kind: FileKind
    media_types: tuple[str, ...]


FILE_RULES: dict[str, FileRule] = {
    ".txt": FileRule(FileKind.TEXT, ("text/plain",)),
    ".md": FileRule(
        FileKind.MARKDOWN,
        ("text/markdown", "text/plain"),
    ),
    ".csv": FileRule(
        FileKind.CSV,
        ("text/csv", "application/csv", "text/plain"),
    ),
    ".pdf": FileRule(FileKind.PDF, ("application/pdf",)),
    ".docx": FileRule(
        FileKind.DOCX,
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    ),
    ".xlsx": FileRule(
        FileKind.XLSX,
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
    ),
    ".png": FileRule(FileKind.PNG, ("image/png",)),
    ".jpg": FileRule(FileKind.JPEG, ("image/jpeg",)),
    ".jpeg": FileRule(FileKind.JPEG, ("image/jpeg",)),
}


@dataclass(frozen=True)
class FileValidationPolicy:
    allowed_root: Path
    max_size_bytes: int = 10 * 1024 * 1024
    max_archive_entries: int = 2_000
    max_archive_uncompressed_bytes: int = 50 * 1024 * 1024
    max_compression_ratio: float = 200.0

    def __post_init__(self) -> None:
        if self.max_size_bytes <= 0:
            raise ValueError("max_size_bytes должен быть больше нуля.")
        if self.max_archive_entries <= 0:
            raise ValueError("max_archive_entries должен быть больше нуля.")
        if self.max_archive_uncompressed_bytes <= 0:
            raise ValueError("Лимит распакованного размера должен быть положительным.")
        if self.max_compression_ratio <= 1:
            raise ValueError("max_compression_ratio должен быть больше единицы.")


@dataclass(frozen=True)
class ValidatedFile:
    path: Path
    original_name: str
    extension: str
    kind: FileKind
    media_type: str
    size_bytes: int


class FileValidationError(RuntimeError):
    pass


class FilePathError(FileValidationError):
    pass


class FileTypeError(FileValidationError):
    pass


class FileSizeError(FileValidationError):
    pass


class FileContentError(FileValidationError):
    pass


class FileArchiveError(FileValidationError):
    pass


class FileValidator:
    def __init__(self, policy: FileValidationPolicy) -> None:
        self._policy = policy

        try:
            allowed_root = policy.allowed_root.expanduser().resolve(strict=True)
        except OSError as error:
            raise ValueError(
                "Разрешённый каталог не существует или недоступен."
            ) from error

        if not allowed_root.is_dir():
            raise ValueError("allowed_root должен быть каталогом.")

        self._allowed_root = allowed_root

    def validate(
        self,
        file_path: str | Path,
        declared_media_type: str | None = None,
    ) -> ValidatedFile:
        path = self._resolve_path(file_path)
        extension, rule = self._resolve_rule(path)
        self._validate_declared_media_type(
            declared_media_type,
            rule,
        )
        size_bytes = self._validate_size(path)
        self._validate_content(path, rule.kind)

        return ValidatedFile(
            path=path,
            original_name=path.name,
            extension=extension,
            kind=rule.kind,
            media_type=rule.media_types[0],
            size_bytes=size_bytes,
        )

    def _resolve_path(self, file_path: str | Path) -> Path:
        candidate = Path(file_path).expanduser()

        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise FilePathError("Файл не существует или недоступен.") from error

        try:
            resolved.relative_to(self._allowed_root)
        except ValueError as error:
            raise FilePathError("Файл находится вне разрешённого каталога.") from error

        if not resolved.is_file():
            raise FilePathError("Ожидался обычный файл.")

        return resolved

    @staticmethod
    def _resolve_rule(path: Path) -> tuple[str, FileRule]:
        extension = path.suffix.lower()
        rule = FILE_RULES.get(extension)

        if rule is None:
            raise FileTypeError(
                f"Формат {extension or 'без расширения'} не поддерживается."
            )

        return extension, rule

    @staticmethod
    def _validate_declared_media_type(
        declared_media_type: str | None,
        rule: FileRule,
    ) -> None:
        if declared_media_type is None:
            return

        normalized = declared_media_type.split(";", maxsplit=1)[0]
        normalized = normalized.strip().lower()

        if normalized not in rule.media_types:
            raise FileTypeError("Заявленный MIME type не соответствует расширению.")

    def _validate_size(self, path: Path) -> int:
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            raise FilePathError("Не удалось получить размер файла.") from error

        if size_bytes == 0:
            raise FileSizeError("Пустой файл нельзя обработать.")

        if size_bytes > self._policy.max_size_bytes:
            raise FileSizeError("Файл превышает разрешённый размер.")

        return size_bytes

    def _validate_content(self, path: Path, kind: FileKind) -> None:
        if kind in {
            FileKind.TEXT,
            FileKind.MARKDOWN,
            FileKind.CSV,
        }:
            self._validate_utf8_text(path)
            return

        try:
            with path.open("rb") as file:
                header = file.read(4_096)
        except OSError as error:
            raise FilePathError("Не удалось прочитать начало файла.") from error

        if kind is FileKind.PDF:
            if b"%PDF-" not in header[:1_024]:
                raise FileContentError("Сигнатура PDF не найдена.")
            return

        if kind is FileKind.PNG:
            if not header.startswith(b"\x89PNG\r\n\x1a\n"):
                raise FileContentError("Сигнатура PNG не найдена.")
            return

        if kind is FileKind.JPEG:
            if not header.startswith(b"\xff\xd8\xff"):
                raise FileContentError("Сигнатура JPEG не найдена.")
            return

        if kind in {FileKind.DOCX, FileKind.XLSX}:
            if not header.startswith(b"PK"):
                raise FileContentError("Файл Office не является ZIP-контейнером.")
            self._validate_office_archive(path, kind)
            return

        raise FileTypeError("Для формата не настроена проверка.")

    @staticmethod
    def _validate_utf8_text(path: Path) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")()

        try:
            with path.open("rb") as file:
                while chunk := file.read(64 * 1_024):
                    if b"\x00" in chunk:
                        raise FileContentError("Текстовый файл содержит нулевые байты.")
                    decoder.decode(chunk)

                decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise FileContentError(
                "Текстовый файл должен быть сохранён в UTF-8."
            ) from error
        except OSError as error:
            raise FilePathError("Не удалось прочитать текстовый файл.") from error

    def _validate_office_archive(
        self,
        path: Path,
        kind: FileKind,
    ) -> None:
        required_entry = {
            FileKind.DOCX: "word/document.xml",
            FileKind.XLSX: "xl/workbook.xml",
        }[kind]

        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()

                if len(entries) > self._policy.max_archive_entries:
                    raise FileArchiveError("В архиве слишком много элементов.")

                names = {entry.filename for entry in entries}
                if required_entry not in names:
                    raise FileArchiveError(
                        "Структура Office-документа не подтверждена."
                    )

                total_uncompressed = 0

                for entry in entries:
                    entry_path = PurePosixPath(entry.filename)
                    if entry_path.is_absolute() or ".." in entry_path.parts:
                        raise FileArchiveError("Архив содержит небезопасный путь.")

                    if entry.flag_bits & 0x1:
                        raise FileArchiveError(
                            "Зашифрованные архивы не поддерживаются."
                        )

                    total_uncompressed += entry.file_size
                    if total_uncompressed > self._policy.max_archive_uncompressed_bytes:
                        raise FileArchiveError(
                            "Распакованное содержимое слишком большое."
                        )

                    if entry.file_size == 0:
                        continue

                    if entry.compress_size == 0:
                        raise FileArchiveError(
                            "Обнаружен подозрительный коэффициент сжатия."
                        )

                    compression_ratio = entry.file_size / entry.compress_size
                    if compression_ratio > self._policy.max_compression_ratio:
                        raise FileArchiveError("Коэффициент сжатия превышает лимит.")
        except BadZipFile as error:
            raise FileArchiveError(
                "Файл Office содержит повреждённый ZIP-архив."
            ) from error
        except OSError as error:
            raise FilePathError("Не удалось прочитать архив документа.") from error
