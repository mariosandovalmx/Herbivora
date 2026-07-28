"""Route conventions for the HerbivoR project."""

from __future__ import annotations

import sys
from pathlib import Path
import re


def app_root() -> Path:
    """Writable project root (next to the exe when frozen; repo root otherwise)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    """Read-only resources root (PyInstaller extract dir, or same as app_root)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return app_root()


REPO_ROOT = app_root()

from image_io import VALID_IMAGE_EXTENSIONS as VALID_EXT, is_image_path  # noqa: E402


def _natural_key(p: Path) -> list:
    """Sort key that orders embedded numbers numerically (IMG_2 before IMG_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)]


# ── Unified model folder (weights downloaded via download_models.py) ─────────
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MOBILESAM = MODELS_DIR / "mobile_sam.pt"
DEFAULT_UNET_SHAPE_MODEL = MODELS_DIR / "best_unet_shape.pth"
DEFAULT_UNET_SHAPE_CONFIG = bundle_root() / "contour" / "configs" / "config_train_f2lsm.yaml"
DEFAULT_DAMAGE_MODEL = MODELS_DIR / "best_model.pth"
# Alias used by Contour leaf-weights args
DEFAULT_LEAF_MODEL = DEFAULT_UNET_SHAPE_MODEL

SEGMENT_METHOD_CHOICES = ("birefnet_mobilesam", "intact", "interactive_mobilesam")
SEGMENT_METHOD_LABELS = {
    "birefnet_mobilesam": "A. BiRefNet + MobileSAM [RECOMMENDED]",
    "intact": "B. Otsu + LAB [FAST]",
    "interactive_mobilesam": "C. Interactive segmentation",
}

# Fixed canvas / inference size for Segmentation → Contour → Analysis.
PIPELINE_RESOLUTION = 1024

BIREFNET_DEFAULTS: dict[str, bool | float | int | str] = {
    "birefnet_known_diameter_mm": 6.0,
    "birefnet_hybrid_mode": "birefnet_primary",
    "birefnet_seg_resolution": PIPELINE_RESOLUTION,
    "birefnet_output_size": PIPELINE_RESOLUTION,
    "birefnet_agreement_threshold": 0.85,
}

# Legacy FastSAM tuning defaults (kept for Segment tab advanced panel compatibility).
FASTSAM_PETRI_DEFAULTS: dict[str, bool | float | int | None] = {
    "segmentation_method": "birefnet_mobilesam",
    "fastsam_max_leaves": 2,
    "fastsam_output_size": PIPELINE_RESOLUTION,
    "fastsam_min_overlap": 0.16,
    "fastsam_dark_ratio_threshold": 0.50,
    "fastsam_dark_value_threshold": 62,
    "fastsam_component_margin_px": 4,
    "fastsam_strict_crop": True,
    "fastsam_conf": 0.35,
    "fastsam_imgsz": PIPELINE_RESOLUTION,
    "fastsam_prior_dilate_px": 35,
    "fastsam_max_area_ratio": 0.75,
    "stressed_leaves": True,
    "remove_blue": True,
    "reject_dark_artifacts": True,
}

CONTOUR_METHOD_CHOICES = ("recon_unet_shape",)
CONTOUR_METHOD_LABELS = {
    "recon_unet_shape": "A. UNET Shape (512px Mask-to-Mask, F2LSM)",
}

VALID_ROI_MODES = ("filled", "mask", "closed", "hull")
LEAF_REFINE_CHOICES = ("complete", "bridged", "raw")


def auto_detect_models() -> dict[str, Path | None]:
    """Scan models/ for the three weights used by the HerbivoR GUI.

    Roles
    -----
    mobilesam   — Segmentation (BiRefNet + MobileSAM / Interactive)
    unet_shape  — Contour / ROI (UNET Shape)
    damage      — Analysis (Damage U-Net)
    """
    mobilesam = DEFAULT_MOBILESAM if DEFAULT_MOBILESAM.is_file() else None
    unet_shape = DEFAULT_UNET_SHAPE_MODEL if DEFAULT_UNET_SHAPE_MODEL.is_file() else None
    damage = DEFAULT_DAMAGE_MODEL if DEFAULT_DAMAGE_MODEL.is_file() else None
    return {
        "mobilesam": mobilesam or DEFAULT_MOBILESAM,
        "unet_shape": unet_shape or DEFAULT_UNET_SHAPE_MODEL,
        "damage": damage or DEFAULT_DAMAGE_MODEL,
    }


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and is_image_path(p)),
        key=_natural_key,
    )


def is_preextracted_leaf(path: Path) -> bool:
    stem = path.stem.lower()
    if "_white_bg" in stem:
        return True
    if "_leaf_" in stem and stem.endswith("_white_bg"):
        return True
    return False


def work_dir(output_root: Path) -> Path:
    return output_root / ".temp_fastsam"


def segmentation_dir(output_root: Path) -> Path:
    return output_root / "segmentation"


def white_bg_dir(output_root: Path) -> Path:
    return segmentation_dir(output_root) / "white_bg"


def masks_dir(output_root: Path) -> Path:
    return segmentation_dir(output_root) / "masks"


def leaf_roi_preview_dir(output_root: Path) -> Path:
    return output_root / "leaf_roi_preview"


def analyzed_dir(output_root: Path) -> Path:
    return output_root / "analyzed"


def analyzed_stem_from_jpg(analyzed_jpg: Path) -> str:
    """`foo_white_bg_analyzed.jpg` -> `foo_white_bg`."""
    stem = analyzed_jpg.stem
    if stem.endswith("_analyzed"):
        return stem[: -len("_analyzed")]
    return stem


def analyzed_damage_mask_path(analyzed_jpg: Path) -> Path:
    return analyzed_jpg.with_name(f"{analyzed_stem_from_jpg(analyzed_jpg)}_damage_mask.png")


def analyzed_leaf_roi_path(analyzed_jpg: Path) -> Path:
    return analyzed_jpg.with_name(f"{analyzed_stem_from_jpg(analyzed_jpg)}_leaf_roi.png")


def analyzed_meta_path(analyzed_jpg: Path) -> Path:
    return analyzed_jpg.with_name(f"{analyzed_stem_from_jpg(analyzed_jpg)}_meta.json")


def unlink_analyzed_artifacts(analyzed_dir_path: Path, stem_variant: str) -> None:
    """Remove analyzed JPG and editable sidecars for one stem variant."""
    for suffix in ("_analyzed.jpg", "_damage_mask.png", "_leaf_roi.png", "_meta.json"):
        (analyzed_dir_path / f"{stem_variant}{suffix}").unlink(missing_ok=True)


def count_white_bg_leaves(output_root: Path) -> int:
    wb = white_bg_dir(output_root)
    if not wb.is_dir():
        return 0
    return sum(1 for p in wb.iterdir() if p.is_file() and is_image_path(p))


def count_masks(output_root: Path) -> int:
    md = masks_dir(output_root)
    if not md.is_dir():
        return 0
    return sum(1 for p in md.glob("*_mask.png"))


def canonical_leaf_id(stem: str) -> str:
    """Extract the base leaf ID from a white_bg image stem."""
    nested = re.match(r"^(.+)_white_bg_leaf_\d+_white_bg$", stem, re.IGNORECASE)
    if nested:
        return nested.group(1)
    if stem.lower().endswith("_white_bg"):
        return stem[: -len("_white_bg")]
    return stem


def mask_path_for_white_bg(white_bg_image: Path, output_root: Path) -> Path:
    """ROI mask path associated with a *_white_bg image."""
    leaf_id = canonical_leaf_id(white_bg_image.stem)
    seg = segmentation_dir(output_root)
    try:
        rel = white_bg_image.relative_to(seg)
        if "white_bg" in rel.parts:
            idx = rel.parts.index("white_bg")
            species_parts = rel.parts[:idx]
            return seg.joinpath(*species_parts, "masks", f"{leaf_id}_mask.png")
    except ValueError:
        pass
    return masks_dir(output_root) / f"{leaf_id}_mask.png"


def work_white_bg_copy(output_root: Path, filename: str) -> Path:
    return work_dir(output_root) / "white_bg" / filename


def stem_from_leaf_overlay(overlay_path: Path) -> str:
    """`foo_white_bg_leaf_overlay.jpg` -> `foo_white_bg`."""
    stem = overlay_path.stem
    if stem.endswith("_leaf_overlay"):
        return stem[: -len("_leaf_overlay")]
    return stem


def leaf_preview_mask_path(overlay_path: Path, output_root: Path) -> Path:
    stem = stem_from_leaf_overlay(overlay_path)
    return leaf_roi_preview_dir(output_root) / "masks" / f"{stem}_leaf_mask.png"


def white_bg_path_for_stem(stem: str, output_root: Path) -> Path | None:
    """Resolve white_bg image for a leaf stem (prefer PNG/JPEG over exotic formats)."""
    wb = white_bg_dir(output_root)
    preferred = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".gif",
        ".jpe",
        ".jfif",
        ".heic",
        ".heif",
    )
    seen: set[str] = set()
    for ext in (*preferred, *sorted(VALID_EXT)):
        if ext in seen:
            continue
        seen.add(ext)
        candidate = wb / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    extensionless = wb / stem
    if extensionless.is_file() and is_image_path(extensionless):
        return extensionless
    return None
