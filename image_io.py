"""Central image format definitions and robust loading for Herbivora."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_BASE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jfif",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".gif",
    }
)
_HEIF_EXTENSIONS = frozenset({".heic", ".heif"})

# Formats where OpenCV decode is unreliable; prefer Pillow directly.
_PILOW_FIRST_EXTENSIONS = frozenset(
    {".tif", ".tiff", ".webp", ".gif", ".heic", ".heif", ".jpe", ".jfif"}
)

_heif_registered = False


def register_optional_plugins() -> None:
    """Register pillow-heif for HEIC/HEIF when the package is installed."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _heif_registered = True
    except ImportError:
        pass


register_optional_plugins()

VALID_IMAGE_EXTENSIONS: frozenset[str] = (
    _BASE_EXTENSIONS | _HEIF_EXTENSIONS
    if _heif_registered
    else _BASE_EXTENSIONS
)


def sniff_image_format(path: Path | str) -> str | None:
    """Detect image format from file header (ignores extension)."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        return None
    if len(head) < 4:
        return None
    if head[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if head[:2] == b"BM":
        return "BMP"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return "HEIF"
    return None


def is_image_path(path: Path | str) -> bool:
    """True if extension or file content indicates a supported photo."""
    path = Path(path)
    if path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
        return True
    sniffed = sniff_image_format(path)
    if sniffed is None:
        return False
    if sniffed == "HEIF":
        return _heif_registered
    return sniffed in {"JPEG", "PNG", "GIF", "TIFF", "BMP", "WEBP"}


def open_pil(path: Path | str) -> Image.Image:
    """Open a photo as RGB PIL Image with EXIF orientation and GIF frame 0."""
    path = Path(path)
    with Image.open(path) as pil:
        if getattr(pil, "is_animated", False):
            pil.seek(0)
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("I", "I;16", "F", "RGBA", "LA", "P", "CMYK"):
            return pil.convert("RGB")
        return pil.convert("RGB")


def _load_bgr_via_pillow(path: Path) -> np.ndarray | None:
    try:
        rgb = load_rgb(path)
    except OSError:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_bgr(path: Path | str) -> np.ndarray | None:
    """Load a photo as HxWx3 uint8 BGR array (Pillow for TIFF/WebP/GIF/HEIC)."""
    path = Path(path)
    ext = path.suffix.lower()
    sniffed = sniff_image_format(path)
    use_pillow = (
        ext in _PILOW_FIRST_EXTENSIONS
        or sniffed in {"TIFF", "WEBP", "GIF", "HEIF"}
        or (sniffed is not None and ext not in VALID_IMAGE_EXTENSIONS)
    )
    if not use_pillow:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
    return _load_bgr_via_pillow(path)


def supported_formats_label() -> str:
    """Short label listing accepted photo extensions for the GUI."""
    names = sorted(ext.lstrip(".").upper() for ext in VALID_IMAGE_EXTENSIONS)
    return "Supported formats: " + ", ".join(names)


def load_rgb(path: Path | str) -> np.ndarray:
    """Load a photo as HxWx3 uint8 RGB array."""
    return np.asarray(open_pil(path))


def format_error_hint(path: Path | str) -> str:
    """Human-readable hint when a photo cannot be opened."""
    path = Path(path)
    ext = path.suffix.lower()
    sniffed = sniff_image_format(path)
    if sniffed == "HEIF" and not _heif_registered:
        return (
            f"Could not open {path.name}: HEIC/HEIF requires pillow-heif "
            "(pip install pillow-heif)."
        )
    if sniffed and ext not in VALID_IMAGE_EXTENSIONS:
        return (
            f"Could not open {path.name}: detected {sniffed} content "
            f"but extension '{ext or '(none)'}' may be wrong."
        )
    return f"Could not open {Path(path).name}: unsupported or corrupt image."
