from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from typing import Literal
from warnings import catch_warnings, simplefilter

from PIL import Image, ImageOps, UnidentifiedImageError

from app.files.validation import FileKind, ValidatedFile

IMAGE_KINDS = frozenset({FileKind.PNG, FileKind.JPEG})


class ImagePreparationError(RuntimeError):
    pass


class UnsupportedImageFormatError(ImagePreparationError):
    pass


class ImageQualityError(ImagePreparationError):
    pass


@dataclass(frozen=True)
class ImagePreparationPolicy:
    max_images: int = 4
    max_pixels: int = 25_000_000
    max_side: int = 2_048
    min_side: int = 128
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if self.max_images <= 0:
            raise ValueError("max_images должен быть больше нуля.")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels должен быть больше нуля.")
        if self.max_side <= 0 or self.min_side <= 0:
            raise ValueError("Размеры изображения должны быть положительными.")
        if self.min_side > self.max_side:
            raise ValueError("min_side не должен превышать max_side.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality должен находиться от 1 до 100.")


@dataclass(frozen=True)
class PreparedImage:
    reference: str
    source_name: str
    source_kind: FileKind
    media_type: str
    data_url: str
    size_bytes: int
    original_width: int
    original_height: int
    width: int
    height: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PreparedImageBatch:
    images: tuple[PreparedImage, ...]
    references: tuple[str, ...]
    warnings: tuple[str, ...]


ImageDetail = Literal["low", "high", "auto"]


class ImageInputPreparer:
    def __init__(
        self,
        policy: ImagePreparationPolicy | None = None,
    ) -> None:
        self._policy = policy or ImagePreparationPolicy()

    def prepare(
        self,
        file: ValidatedFile,
        reference: str,
    ) -> PreparedImage:
        if file.kind not in IMAGE_KINDS:
            raise UnsupportedImageFormatError(
                f"Подготовка изображения не поддерживает {file.kind}."
            )
        if not reference.strip():
            raise ImagePreparationError("Ссылка изображения не должна быть пустой.")

        warnings: list[str] = []

        try:
            with catch_warnings():
                simplefilter("error", Image.DecompressionBombWarning)

                with Image.open(file.path) as source:
                    expected_format = "PNG" if file.kind is FileKind.PNG else "JPEG"
                    if source.format != expected_format:
                        raise UnsupportedImageFormatError(
                            "Содержимое изображения не соответствует формату."
                        )

                    original_width, original_height = source.size
                    if original_width * original_height > self._policy.max_pixels:
                        raise ImageQualityError(
                            "Изображение содержит слишком много пикселей."
                        )

                    source.load()
                    normalized = ImageOps.exif_transpose(source)

                    if min(normalized.size) < self._policy.min_side:
                        warnings.append(
                            "Низкое разрешение может ухудшить распознавание."
                        )

                    if max(normalized.size) > self._policy.max_side:
                        normalized.thumbnail(
                            (self._policy.max_side, self._policy.max_side),
                            Image.Resampling.LANCZOS,
                        )
                        warnings.append("Изображение уменьшено перед анализом.")

                    output = BytesIO()
                    if file.kind is FileKind.JPEG:
                        normalized.convert("RGB").save(
                            output,
                            format="JPEG",
                            quality=self._policy.jpeg_quality,
                            optimize=True,
                        )
                        media_type = "image/jpeg"
                    else:
                        if normalized.mode not in {"RGB", "RGBA"}:
                            normalized = normalized.convert("RGBA")
                        normalized.save(output, format="PNG", optimize=True)
                        media_type = "image/png"

                    width, height = normalized.size
                    image_bytes = output.getvalue()
        except UnsupportedImageFormatError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise ImageQualityError(
                "Pillow отклонил подозрительно большое изображение."
            ) from error
        except (UnidentifiedImageError, OSError) as error:
            raise ImagePreparationError("Не удалось прочитать изображение.") from error

        encoded = b64encode(image_bytes).decode("ascii")

        return PreparedImage(
            reference=reference.strip(),
            source_name=file.original_name,
            source_kind=file.kind,
            media_type=media_type,
            data_url=f"data:{media_type};base64,{encoded}",
            size_bytes=len(image_bytes),
            original_width=original_width,
            original_height=original_height,
            width=width,
            height=height,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def build_batch(
        self,
        images: tuple[PreparedImage, ...],
    ) -> PreparedImageBatch:
        if not images:
            raise ImagePreparationError("Нужно подготовить хотя бы одно изображение.")
        if len(images) > self._policy.max_images:
            raise ImagePreparationError("Передано слишком много изображений.")

        references = tuple(image.reference for image in images)
        if len(set(references)) != len(references):
            raise ImagePreparationError("Ссылки изображений не должны повторяться.")

        warnings = tuple(
            dict.fromkeys(warning for image in images for warning in image.warnings)
        )

        return PreparedImageBatch(
            images=images,
            references=references,
            warnings=warnings,
        )
