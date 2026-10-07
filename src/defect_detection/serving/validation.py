"""Upload validation: everything an image must pass before it reaches the model.

Order (cheapest and most dangerous checks first):

1. size is already capped while streaming by BodySizeLimitMiddleware (413);
2. magic bytes decide the format; ``Content-Type`` and filenames are ignored (415);
3. Pillow opens the file restricted to that one format; the pixel count from the header is
   checked against ``MAX_IMAGE_PIXELS`` before any pixel is decoded (decompression bombs,
   400); ``verify()`` checks the container structure;
4. the image is re-opened (``verify()`` leaves it unusable), multi-frame/animated files are
   rejected, dimensions and aspect ratio are checked, then the pixels are fully decoded,
   which catches truncated files.

Uploads stay in memory (never written to disk) and their client-supplied filenames are never
used for anything.
"""

import io
import warnings

from PIL import Image, UnidentifiedImageError

from defect_detection.serving.errors import InvalidImageError, UnsupportedMediaTypeError
from defect_detection.serving.settings import Settings

# Pillow format name -> leading signature bytes.
_SIGNATURES: tuple[tuple[str, bytes], ...] = (
    ("JPEG", b"\xff\xd8\xff"),
    ("PNG", b"\x89PNG\r\n\x1a\n"),
    ("BMP", b"BM"),
)


def sniff_format(data: bytes) -> str | None:
    """Image format from the file's first bytes, or None if it is not an accepted type."""
    for name, signature in _SIGNATURES:
        if data.startswith(signature):
            return name
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def configure_pillow(settings: Settings) -> None:
    """Process-wide decompression-bomb cap (Pillow reads it on every open)."""
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    """Validate and decode one upload; raises InvalidImageError / UnsupportedMediaTypeError."""
    fmt = sniff_format(data)
    if fmt is None:
        raise UnsupportedMediaTypeError("Only JPEG, PNG, BMP and WebP images are accepted.")
    try:
        with warnings.catch_warnings():
            # Between 1x and 2x MAX_IMAGE_PIXELS Pillow only warns; treat that as a bomb too.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[fmt]) as probe:
                # open() reads only the header: refuse oversized images before any decoding,
                # independent of Pillow's process-wide MAX_IMAGE_PIXELS.
                width, height = probe.size
                if width * height > settings.max_image_pixels:
                    raise InvalidImageError("Image has too many pixels.")
                probe.verify()
            image = Image.open(io.BytesIO(data), formats=[fmt])
            if getattr(image, "n_frames", 1) > 1:
                raise InvalidImageError("Multi-frame or animated images are not accepted.")
            _check_dimensions(image.size, settings)
            image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("Image has too many pixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise InvalidImageError("File could not be decoded as an image.") from exc
    return image


def _check_dimensions(size: tuple[int, int], settings: Settings) -> None:
    width, height = size
    if min(width, height) < settings.min_image_side:
        raise InvalidImageError(f"Image sides must be at least {settings.min_image_side} px.")
    if max(width, height) > settings.max_image_side:
        raise InvalidImageError(f"Image sides must be at most {settings.max_image_side} px.")
    if max(width, height) / min(width, height) > settings.max_aspect_ratio:
        raise InvalidImageError(
            f"Aspect ratio must be at most {settings.max_aspect_ratio}:1 (images are resized "
            "to a square)."
        )
