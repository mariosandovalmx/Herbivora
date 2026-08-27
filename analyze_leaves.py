"""
analyze_leaves.py — U-Net on individual leaves (FastSAM or whitebg_masks output).

Reads white_bg + mask pairs from a folder and saves visualizations + CSV.

ROI Modes (--roi-mode):

    filled   -> DEFAULT ImageJ-friendly: area = silhouette + filled internal holes;
                damage = U-Net + white holes surrounded by tissue (no hull or outer edge).
                One ROI per connected fragment if the mask has multiple pieces.
    mask     -> Silhouette mask with internal holes only (no adding holes to damage).
    closed   -> Morphological edge closing (legacy).
    hull     -> Convex hull + marginal edge damage (legacy, may overestimate).
    lama     -> LaMa inpainting to reconstruct leaf margins (advanced).

Usage:
    python analyze_leaves.py --segmentation-dir output_segmentation_test --roi-mode filled
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 on Windows (pythonw.exe defaults to cp1252 which rejects ² → ×)
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision.transforms import functional as TF

from model import build_model
from image_io import VALID_IMAGE_EXTENSIONS, load_rgb  # noqa: E402

_LEAF_DIR = Path(__file__).resolve().parent / "leaf_contour"
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_LEAF_DIR))
sys.path.insert(0, str(_REPO_ROOT))
try:
    from mask_smooth import smooth_mask_contour  # noqa: E402
except ImportError:

    def smooth_mask_contour(mask, epsilon_factor=0.0018):  # type: ignore
        return mask.astype(bool) if hasattr(mask, "astype") else mask

import json as _json_mod

_WHITE_BG_STRIP = re.compile(r"_white_bg(?:_leaf_\d+(?:_white_bg)?)?$", re.IGNORECASE)


def load_scale_json(path) -> dict:
    try:
        return _json_mod.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, _json_mod.JSONDecodeError):
        return {}


def lookup_scale_factor(scale_data: dict, image_name: str) -> float | None:
    """Return cm2/px2 for image_name, trying bare stem and _white_bg-stripped variants."""
    if not scale_data:
        return None
    # 1. Exact match
    v = scale_data.get(image_name)
    if v is not None and not isinstance(v, dict):
        return float(v)
    # 2. Bare stem  e.g. "IMG.png" -> "IMG"
    stem = Path(image_name).stem
    v = scale_data.get(stem)
    if v is not None and not isinstance(v, dict):
        return float(v)
    # 3. Strip _white_bg suffix and try bare base + extensions
    base = _WHITE_BG_STRIP.sub("", stem)
    if base != stem:
        v = scale_data.get(base)
        if v is not None and not isinstance(v, dict):
            return float(v)
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
            v = scale_data.get(base + ext)
            if v is not None and not isinstance(v, dict):
                return float(v)
    # 4. Multi-leaf: photo1_leaf_2 → try photo1 (shared scale from parent photo)
    leaf_m = re.match(r"^(.+)_leaf_\d+$", base if base != stem else stem, re.IGNORECASE)
    if leaf_m:
        parent = leaf_m.group(1)
        v = scale_data.get(parent)
        if v is not None and not isinstance(v, dict):
            return float(v)
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
            v = scale_data.get(parent + ext)
            if v is not None and not isinstance(v, dict):
                return float(v)
    # 5. Stem with extensions
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
        v = scale_data.get(stem + ext)
        if v is not None and not isinstance(v, dict):
            return float(v)
    return None


NESTED_DUP_RE = re.compile(r"_white_bg_leaf_\d+_white_bg$", re.IGNORECASE)
DEFAULT_MAX_LEAF_AREA_RATIO = 0.62
DEFAULT_ROI_MODE = "filled"
VALID_ROI_MODES = ("mask", "closed", "hull", "filled", "lama", "reconstruction")
CLOSE_KERNEL_DIVISOR = 12.0  # kernel_radius = sqrt(leaf_area) / DIVISOR
MIN_FRAGMENT_AREA_RATIO = 0.002
# Internal holes (filled - tissue) count as damage; not open bites on the outer edge.
DEFAULT_FILL_MARGINAL = True
DEFAULT_WHITE_HOLE_BRIGHTNESS = 235
DEFAULT_WHITE_HOLE_MIN_AREA = 3
DEFAULT_WHITE_HOLE_EDGE_BAND = 2
DEFAULT_WHITE_HOLE_ADAPTIVE = True
DEFAULT_WHITE_HOLE_AUTO_FLOOR = 175
DEFAULT_WHITE_HOLE_AUTO_CEILING = 250
DEFAULT_SUPERFICIAL_DAMAGE = True
DEFAULT_SUPERFICIAL_MIN_AREA = 20
DEFAULT_SUPERFICIAL_VS_RATIO = 1.1
DEFAULT_SUPERFICIAL_VS_MULT = 2.0
DEFAULT_SUPERFICIAL_MAX_SAT = 110
DEFAULT_FRASS_ZONE_DILATE = 5
DEFAULT_FRASS_DARK_THRESHOLD = 75
DEFAULT_FRASS_LOCAL_CONTRAST = 16.0
DEFAULT_FRASS_MIN_AREA = 4
DEFAULT_FRASS_MAX_AREA = 1200
DEFAULT_FRASS_NEUTRAL_CHROMA = 55
DEFAULT_FRASS_DAMAGE_DILATE = 5
DEFAULT_FRASS_IN_DAMAGE_CONTRAST = 14.0
DEFAULT_FRASS_IN_DAMAGE_GRAY_MAX = 108
DEFAULT_FRASS_IN_DAMAGE_MAX_AREA = 900
DEFAULT_DAMAGE_OVERLAY_ALPHA = 0.45
# Reported / stored herbivory percentage precision (GUI, CSV, overlay title, meta.json)
DAMAGE_PCT_DECIMALS = 3
DEFAULT_EDGE_ARTIFACT_FILTER = True


def round_damage_pct(pct: float) -> float:
    return round(float(pct), DAMAGE_PCT_DECIMALS)


def format_damage_pct(pct: float) -> str:
    return f"{float(pct):.{DAMAGE_PCT_DECIMALS}f}"


# Target thin contour ghosts only — keep real holes / notches / interior damage.
DEFAULT_EDGE_MIN_INWARD_PX = 3.5
EDGE_MIN_AREA_FLOOR = 80
EDGE_MIN_AREA_RATIO = 0.0015
EDGE_HARD_BAND_PX = 1.5  # anti-alias strip only (do not carve into real damage)
EDGE_ABS_MIN_COMPONENT_PX = 12  # drop isolated 1–few px speckles on the rim
EDGE_RIM_FRAC_DROP = 0.85  # drop only when almost the whole blob sits on the rim
DEFAULT_SCALE_AREA_CM2 = 0.2827  # area of a 6.0mm-diameter (0.6cm) blue reference dot
DEFAULT_RECONSTRUCTION_MODEL = str(_REPO_ROOT / "leaf_reconstruction" / "models" / "unet_shape_completion.pt")

UNET_PATH = str(_REPO_ROOT / "best_model.pth")
ENCODER = "resnet34"
NUM_CLASSES = 4

UNET_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEGMENTATION_DIR = "output_segmentation"
DEFAULT_OUT_DIR = "unet2_analyzed"

CLASS_NAMES = {0: "Background", 1: "Damage", 2: "Frass", 3: "Undamage"}
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [220, 50, 50],
    2: [50, 150, 50],
    3: [50, 100, 220],
}


def load_unet(unet_path: str):
    path = Path(unet_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find U-Net model: {path.resolve()}\n"
            "Please place best_model.pth in the project root or specify its path via --unet-path PATH.pth"
        )
    model = build_model(encoder_name=ENCODER, num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(str(path), map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def predict_unet(unet, orig_rgb: np.ndarray, unet_size: int) -> np.ndarray:
    w0, h0 = orig_rgb.shape[1], orig_rgb.shape[0]
    resized = Image.fromarray(orig_rgb).resize((unet_size, unet_size), Image.BILINEAR)
    tensor = TF.normalize(
        TF.to_tensor(resized),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = unet(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return np.array(Image.fromarray(pred).resize((w0, h0), Image.NEAREST))


def mask_to_rgb(pred_mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[pred_mask == cls] = color
    return rgb


def mask_to_damage_rgb(pred_mask: np.ndarray) -> np.ndarray:
    """RGB overlay with Damaged / Undamaged classes only (for analysis figures)."""
    rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    rgb[pred_mask == 1] = CLASS_COLORS[1]
    rgb[pred_mask == 3] = CLASS_COLORS[3]
    return rgb


def cleanup_predictions_outside_leaf(pred_mask: np.ndarray, leaf_roi: np.ndarray) -> np.ndarray:
    """Forces background (0) on pixels outside the ROI; classes 1/2/3 only count inside."""
    cleaned = pred_mask.copy()
    outside = ~leaf_roi.astype(bool)
    cleaned[outside] = 0
    return cleaned


def filter_boundary_damage_artifacts(
    pred_mask: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    enabled: bool = True,
    min_component_area: int | None = None,
    min_inward_px: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    Remove thin U-Net class-1 *contour ghosts* without eating real herbivory.

    Drops only:
      - tiny speckles
      - hairline / elongated strips hugging the outer contour
      - rim-dominated blobs (almost all pixels in the outermost ~1.5 px)
      - shallow exterior-touching outlines (max depth < min_inward)

    Real interior holes, notches, and compact damage patches are kept intact.
    A light hard-band strip (EDGE_HARD_BAND_PX) removes only anti-alias fringe.
    """
    if not enabled:
        return pred_mask, 0

    tissue = tissue_mask.astype(bool)
    if int(tissue.sum()) < 50:
        return pred_mask, 0

    filled = fill_roi_holes(tissue)
    damage_bool = (pred_mask == 1) & filled
    if not np.any(damage_bool):
        return pred_mask, 0

    requested = DEFAULT_EDGE_MIN_INWARD_PX if min_inward_px is None else float(min_inward_px)
    # Cap overly aggressive caller/config values so real damage is not erased.
    inward = float(np.clip(requested, 2.5, 4.5))
    hard = float(EDGE_HARD_BAND_PX)
    try:
        from gap_detector import classify_morphology

        if classify_morphology(tissue.astype(np.uint8) * 255) == "serrated":
            inward += 0.5
    except Exception:
        pass

    dist_in = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    exterior = ~filled
    exterior_touch = cv2.dilate(exterior.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)

    keep = np.zeros_like(damage_bool, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        damage_bool.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < EDGE_ABS_MIN_COMPONENT_PX or not np.any(comp):
            continue

        depths = dist_in[comp]
        max_d = float(depths.max())
        rim_frac = float((depths < hard + 0.5).mean())
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        thin_side = min(bw, bh)
        long_side = max(bw, bh)
        touches_out = bool(np.any(comp & exterior_touch))

        # Hairline contour strokes (false red outline)
        if thin_side <= 2 and long_side >= 15 and touches_out:
            continue
        # Almost entirely on the outermost fringe
        if rim_frac >= EDGE_RIM_FRAC_DROP and max_d < inward:
            continue
        # Shallow outline that only hugs the exterior
        if touches_out and max_d < inward and rim_frac >= 0.5:
            continue

        keep |= comp

    # Light anti-alias fringe only — do not carve EDGE_HARD_BAND out of kept blobs
    # beyond pixels that belong to dropped components.
    hard_band = damage_bool & (dist_in < hard) & ~keep
    # Also strip the 1.5 px fringe from kept components (visual anti-alias) but
    # leave the rest of each kept blob intact.
    fringe = keep & (dist_in < hard)
    drop = (damage_bool & ~keep) | hard_band | fringe

    filtered_px = int(drop.sum())
    if filtered_px == 0:
        return pred_mask, 0

    out = pred_mask.copy()
    out[drop] = 3
    return out, filtered_px


def strip_perimeter_damage_bool(
    damage: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    band_px: float | None = None,
) -> np.ndarray:
    """Remove only hairline contour ghosts from a boolean damage mask.

    Preserves interior white holes, fenestrations, and open notches. Does **not**
    erode legitimate damage away from the leaf edge.
    """
    dmg = damage.astype(bool)
    tissue = tissue_mask.astype(bool)
    if not np.any(dmg) or not np.any(tissue):
        return dmg

    filled = fill_roi_holes(tissue)
    hard = float(EDGE_HARD_BAND_PX if band_px is None else band_px)
    dist_in = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    exterior = ~filled
    exterior_touch = cv2.dilate(exterior.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)

    drop = np.zeros_like(dmg, dtype=bool)
    work = dmg & filled
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        work.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not np.any(comp):
            continue
        depths = dist_in[comp]
        max_d = float(depths.max())
        rim_frac = float((depths < hard + 0.5).mean())
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        thin_side = min(bw, bh)
        long_side = max(bw, bh)
        touches_out = bool(np.any(comp & exterior_touch))

        # Only kill hairline / rim-ghost components
        if area < EDGE_ABS_MIN_COMPONENT_PX and touches_out and max_d < 3.0:
            drop |= comp
            continue
        if thin_side <= 2 and long_side >= 15 and touches_out and max_d < 4.0:
            drop |= comp
            continue
        if rim_frac >= EDGE_RIM_FRAC_DROP and max_d < 3.5 and touches_out:
            drop |= comp
            continue

    # Tiny anti-alias fringe on remaining mask (1.5 px) — optional light clean
    fringe = work & (dist_in < hard) & exterior_touch
    out = work & ~drop & ~fringe
    return out


def canonical_leaf_id(stem: str) -> str:
    """Stable ID to match image/mask pairs and deduplicate nested copies."""
    nested = re.match(r"^(.+)_white_bg_leaf_\d+_white_bg$", stem, re.IGNORECASE)
    if nested:
        return nested.group(1)
    if stem.lower().endswith("_white_bg"):
        return stem[: -len("_white_bg")]
    return stem


def refine_leaf_roi_from_image(
    leaf_roi: np.ndarray,
    orig_rgb: np.ndarray,
    max_area_ratio: float = DEFAULT_MAX_LEAF_AREA_RATIO,
 ) -> tuple[np.ndarray, bool]:
    """If the saved mask is too large, re-estimate it directly from the image."""
    area_ratio = float(leaf_roi.sum()) / float(leaf_roi.size)
    if area_ratio <= max_area_ratio:
        return leaf_roi, False
    try:
        from whitebg_masks import leaf_mask_from_rgb

        mask_u8, _ = leaf_mask_from_rgb(orig_rgb, max_leaf_area_ratio=max_area_ratio)
        refined = mask_u8 > 127
        if refined.sum() < 100:
            return leaf_roi, False
        if float(refined.sum()) / float(refined.size) < area_ratio:
            return refined, True
    except Exception:
        pass
    return leaf_roi, False


def fill_roi_holes(leaf_roi: np.ndarray) -> np.ndarray:
    """Fills internal holes in the ROI in case the saved mask has them."""
    return ndimage.binary_fill_holes(leaf_roi).astype(bool)


def morphological_close_mask(leaf_roi: np.ndarray, kernel_radius: int) -> np.ndarray:
    """Morphological closing to bridge edge bites with the main silhouette."""
    if kernel_radius < 1 or not leaf_roi.any():
        return leaf_roi.copy()
    k = 2 * int(kernel_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(
        (leaf_roi.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=1
    )
    return closed > 127


def convex_hull_mask(leaf_roi: np.ndarray) -> np.ndarray:
    """Convex hull of the leaf silhouette, filled as a binary mask."""
    if not leaf_roi.any():
        return leaf_roi.copy()
    contours, _ = cv2.findContours(
        (leaf_roi.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return leaf_roi.copy()
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return leaf_roi.copy()
    hull = cv2.convexHull(contour)
    h, w = leaf_roi.shape
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(out, [hull], 1)
    return out.astype(bool)


# LaMa inpainter — loaded lazily when roi_mode == "lama"
_lama_inpainter = None


def _get_lama_inpainter():
    """Lazy-load the LaMa inpainter (singleton)."""
    global _lama_inpainter
    if _lama_inpainter is None:
        try:
            from lama_inpainting.inpaint_pipeline import get_inpainter
            _lama_inpainter = get_inpainter(device=DEVICE)
        except ImportError as e:
            raise ImportError(
                "LaMa inpainting requires the 'simple-lama-inpainting' package.\n"
                "Install with: pip install simple-lama-inpainting\n"
                f"Error: {e}"
            )
    return _lama_inpainter


# Lazy-loaded shape-completion model for reconstruction mode
_reconstruction_model = None
_reconstruction_model_path = None


def _get_reconstruction_model(model_path: str | None = None):
    """Lazy-load the shape-completion U-Net (singleton)."""
    global _reconstruction_model, _reconstruction_model_path
    path = model_path or DEFAULT_RECONSTRUCTION_MODEL
    if _reconstruction_model is not None and _reconstruction_model_path == path:
        return _reconstruction_model
    if not Path(path).is_file():
        return None
    try:
        from leaf_reconstruction.src.unet_inference import load_shape_completion_model
        _reconstruction_model = load_shape_completion_model(path, device=DEVICE)
        _reconstruction_model_path = path
        return _reconstruction_model
    except Exception as e:
        print(f"    WARNING: Could not load reconstruction model: {e}")
        return None


def expand_roi(
    leaf_roi: np.ndarray,
    mode: str,
    orig_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Returns the adjusted ROI based on mode ('filled', 'mask', 'closed', 'hull', 'lama', or 'reconstruction')."""
    base = fill_roi_holes(leaf_roi)
    if mode in ("mask", "filled"):
        return base
    leaf_area = int(base.sum())
    if leaf_area < 50:
        return base
    if mode in ("reconstruction", "lama"):
        print(f"    WARNING: ROI mode '{mode}' is deprecated during analysis. "
              "Please use the Contour tab to reconstruct leaves. Falling back to 'filled' mode.")
        return base
    radius = max(8, int(round((leaf_area ** 0.5) / CLOSE_KERNEL_DIVISOR)))
    closed = morphological_close_mask(base, radius)
    if mode == "closed":
        return closed
    if mode == "hull":
        hull = convex_hull_mask(base)
        # Hull covers all edge bites; we close gaps just in case.
        return fill_roi_holes(hull | closed | base)
    raise ValueError(f"Invalid roi_mode: {mode!r}")


def internal_holes_mask(tissue_mask: np.ndarray, leaf_roi: np.ndarray) -> np.ndarray:
    """White holes surrounded by tissue (not open bites on the outer edge)."""
    filled = fill_roi_holes(tissue_mask)
    return filled & ~tissue_mask & leaf_roi


def _foliage_mask_rgb(orig_rgb: np.ndarray) -> np.ndarray:
    """Green-tissue mask from RGB (HSV); False where background or white holes show through."""
    try:
        _seg_dir = _REPO_ROOT / "segmentation"
        if str(_seg_dir) not in sys.path:
            sys.path.insert(0, str(_seg_dir))
        from whitebg_masks import foliage_mask_hsv

        bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        return foliage_mask_hsv(bgr) > 0
    except Exception:
        hsv = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2HSV)
        return (hsv[:, :, 1] >= 28) & (hsv[:, :, 2] >= 25) & (hsv[:, :, 2] <= 250)


def detect_enclosed_white_holes(
    orig_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    leaf_roi: np.ndarray,
    *,
    threshold: int = DEFAULT_WHITE_HOLE_BRIGHTNESS,
    min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    edge_band_px: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    adaptive: bool = DEFAULT_WHITE_HOLE_ADAPTIVE,
) -> np.ndarray:
    """
    Detect bright non-foliage regions inside the leaf ROI (RGB herbivory holes).

    When ``adaptive`` is True, Brightness is chosen automatically per image
    (tissue-relative seed + elbow sweep) to catch more real holes without
    flooding the ROI with noise.
    """
    if threshold <= 0 and not adaptive:
        return np.zeros_like(leaf_roi, dtype=bool)

    roi = leaf_roi.astype(bool)
    tissue = tissue_mask.astype(bool)
    if not np.any(roi):
        return np.zeros_like(leaf_roi, dtype=bool)

    thresh = int(threshold)
    if adaptive:
        thresh = estimate_auto_white_hole_threshold(
            orig_rgb,
            tissue,
            roi,
            min_area=min_area,
            edge_band_px=edge_band_px,
            floor=DEFAULT_WHITE_HOLE_AUTO_FLOOR,
            ceiling=DEFAULT_WHITE_HOLE_AUTO_CEILING,
            manual_hint=threshold if threshold > 0 else None,
        )
        if thresh <= 0:
            return np.zeros_like(leaf_roi, dtype=bool)

    return _white_holes_at_threshold(
        orig_rgb,
        tissue,
        roi,
        threshold=thresh,
        min_area=min_area,
        edge_band_px=edge_band_px,
    )


def estimate_auto_white_hole_threshold(
    orig_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    leaf_roi: np.ndarray,
    *,
    min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    edge_band_px: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    floor: int = DEFAULT_WHITE_HOLE_AUTO_FLOOR,
    ceiling: int = DEFAULT_WHITE_HOLE_AUTO_CEILING,
    manual_hint: int | None = None,
) -> int:
    """
    Pick a per-image Brightness that recovers enclosed white holes.

    Strategy:
      1) Tissue-relative seed from foliage brightness inside the Contour ROI.
      2) Sweep thresholds high→low; stop at noise explosion (area jump / ROI fraction).
      3) Keep the threshold with the best score before that break.
    """
    roi = leaf_roi.astype(bool)
    tissue = tissue_mask.astype(bool)
    if not np.any(roi):
        return int(manual_hint or DEFAULT_WHITE_HOLE_BRIGHTNESS)

    foliage = _foliage_mask_rgb(orig_rgb)
    fol = foliage & roi
    floor_i = int(max(1, floor))
    ceiling_i = int(min(255, max(floor_i, ceiling)))

    seed = int(manual_hint) if manual_hint and manual_hint > 0 else 220
    if int(fol.sum()) > 100:
        fol_min = np.min(orig_rgb[fol].astype(np.float32), axis=1)
        t_p90 = float(np.percentile(fol_min, 90))
        # Holes should be clearly brighter than typical foliage.
        seed = int(np.clip(t_p90 + 20.0, floor_i, ceiling_i))

    # Also consider Otsu on non-foliage ROI pixels when available.
    non_fol = roi & ~foliage
    if int(non_fol.sum()) > 200:
        gray = np.min(orig_rgb.astype(np.float32), axis=2)
        sample = gray[non_fol]
        # Otsu on uint8 sample
        hist_img = np.clip(sample, 0, 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(hist_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu_t > 0:
            seed = int(np.clip(0.5 * seed + 0.5 * float(otsu_t), floor_i, ceiling_i))

    thresholds = list(range(ceiling_i, floor_i - 1, -5))
    if seed not in thresholds:
        thresholds.append(seed)
        thresholds = sorted(set(thresholds), reverse=True)

    roi_area = max(1, int(roi.sum()))
    best_t = seed
    best_score = -1.0
    prev_area = 0
    for t in thresholds:
        holes = _white_holes_at_threshold(
            orig_rgb,
            tissue,
            roi,
            threshold=t,
            min_area=min_area,
            edge_band_px=edge_band_px,
        )
        area = int(holes.sum())
        frac = area / float(roi_area)
        growth = area / float(max(prev_area, 1))

        # Noise explosion or covering too much of the leaf → stop.
        if prev_area > 200 and growth > 2.5 and area > 800:
            break
        if frac > 0.40:
            break

        # Prefer more recovered holes; soft-penalize large fractions.
        score = float(area) * (1.0 - max(0.0, frac - 0.12) * 2.5)
        # Slight preference for thresholds near the tissue-relative seed.
        score -= 0.15 * abs(t - seed) * max(area, 1) / 5000.0
        if score >= best_score:
            best_score = score
            best_t = t
        prev_area = area

    # Do not wander too far below the tissue seed (protect against underexposed noise).
    best_t = int(np.clip(best_t, max(floor_i, seed - 45), ceiling_i))
    return best_t


def _white_holes_at_threshold(
    orig_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    leaf_roi: np.ndarray,
    *,
    threshold: int,
    min_area: int,
    edge_band_px: int,
) -> np.ndarray:
    """Core white-hole detector at a fixed Brightness threshold."""
    roi = leaf_roi.astype(bool)
    tissue = tissue_mask.astype(bool)
    if threshold <= 0 or not np.any(roi):
        return np.zeros_like(roi, dtype=bool)

    foliage = _foliage_mask_rgb(orig_rgb)
    thresh = int(threshold)
    r, g, b = orig_rgb[:, :, 0], orig_rgb[:, :, 1], orig_rgb[:, :, 2]
    channel_white = (r >= thresh) & (g >= thresh) & (b >= thresh)
    candidates = roi & ~foliage & channel_white

    if edge_band_px > 0 and np.any(tissue):
        dist_source = fill_roi_holes(tissue) if tissue.shape == roi.shape else roi
        dist_in = cv2.distanceTransform(dist_source.astype(np.uint8), cv2.DIST_L2, 5)
        candidates &= dist_in > float(edge_band_px)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_u8 = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    foliage_dilated = cv2.dilate(foliage.astype(np.uint8), kernel, iterations=2) > 0

    out = np.zeros_like(roi, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = labels == i
        eroded = cv2.erode(comp.astype(np.uint8), kernel, iterations=1)
        comp_border = comp & ~eroded.astype(bool)
        if not np.any(comp_border) or foliage_dilated[comp_border].mean() >= 0.45:
            out |= comp
    return out


def _healthy_tissue_reference_mask(
    orig_rgb: np.ndarray,
    base_roi: np.ndarray,
) -> np.ndarray:
    """Dark-green reference pixels used to calibrate superficial-damage thresholds."""
    roi = base_roi.astype(bool)
    if not np.any(roi):
        return np.zeros_like(roi, dtype=bool)

    hsv = cv2.cvtColor(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_h = (h >= 18) & (h <= 100)
    healthy = roi & green_h & (s >= 100) & (v <= 130)
    if int(healthy.sum()) < 500:
        healthy = roi & green_h & (s >= 70) & (v <= 150)
    if int(healthy.sum()) < 100 and int(roi.sum()) > 0:
        sat_thr = float(np.percentile(s[roi], 85))
        healthy = roi & (s >= sat_thr) & (v <= 150)
    return healthy


def trim_contour_halo_from_tissue(
    tissue_mask: np.ndarray,
    orig_rgb: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Remove exterior Contour halo pixels that extend onto white paper."""
    tissue = tissue_mask.astype(bool)
    if not np.any(tissue):
        return tissue, 0

    foliage = _foliage_mask_rgb(orig_rgb)
    filled = fill_roi_holes(tissue)
    exterior_touch = cv2.dilate(
        (~filled).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    halo = tissue & ~foliage & exterior_touch
    removed = int(halo.sum())
    if removed == 0:
        return tissue, 0
    return tissue & ~halo, removed


def _rim_nonfoliage_ghost_mask(
    tissue_mask: np.ndarray,
    orig_rgb: np.ndarray,
    edge_band_px: float,
) -> np.ndarray:
    """Non-foliage anti-alias strip hugging the outer lamina contour."""
    tissue = tissue_mask.astype(bool)
    if not np.any(tissue) or edge_band_px <= 0:
        return np.zeros_like(tissue, dtype=bool)

    foliage = _foliage_mask_rgb(orig_rgb)
    filled = fill_roi_holes(tissue)
    dist_in = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    band = max(1.0, float(edge_band_px))
    return (~foliage) & filled & (dist_in <= band)


def partition_frass_by_context(
    frass_roi: np.ndarray,
    pred_mask: np.ndarray,
    leaf_roi: np.ndarray,
    damage_context: np.ndarray,
    orig_rgb: np.ndarray | None = None,
    *,
    zone_dilate_px: int = DEFAULT_FRASS_ZONE_DILATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign U-Net frass pixels to damage or undamaged tissue by spatial context.

    Frass sitting on damaged lamina is merged into the headline damage mask;
    frass on healthy tissue is merged into the undamaged tally so it is not
    excluded from the ROI partition.
    """
    frass = frass_roi.astype(bool)
    if not np.any(frass):
        return np.zeros_like(frass, dtype=bool), np.zeros_like(frass, dtype=bool)

    roi = leaf_roi.astype(bool)
    damage_ctx = damage_context.astype(bool)
    undamage = (pred_mask == 3) & roi

    k = max(3, 2 * int(zone_dilate_px) + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dmg_zone = cv2.dilate(damage_ctx.astype(np.uint8), ker, iterations=1) > 0
    und_zone = cv2.dilate(undamage.astype(np.uint8), ker, iterations=1) > 0

    frass_on_damage = frass & dmg_zone
    frass_on_undamage = frass & und_zone & ~dmg_zone
    remaining = frass & ~frass_on_damage & ~frass_on_undamage

    if np.any(remaining) and orig_rgb is not None:
        foliage = _foliage_mask_rgb(orig_rgb)
        hsv = cv2.cvtColor(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        green_healthy = foliage & (s >= 70) & (v <= 170)
        frass_on_undamage |= remaining & green_healthy
        frass_on_damage |= remaining & ~green_healthy
    elif np.any(remaining):
        frass_on_damage |= remaining

    return frass_on_damage, frass_on_undamage


def detect_frass_rgb(
    orig_rgb: np.ndarray,
    leaf_roi: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    *,
    dark_threshold: int = DEFAULT_FRASS_DARK_THRESHOLD,
    local_contrast: float = DEFAULT_FRASS_LOCAL_CONTRAST,
    min_area: int = DEFAULT_FRASS_MIN_AREA,
    max_area: int = DEFAULT_FRASS_MAX_AREA,
    neutral_chroma: int = DEFAULT_FRASS_NEUTRAL_CHROMA,
) -> np.ndarray:
    """
    Detect dark frass pellets from RGB (local contrast + compact dark blobs).

    The damage U-Net often labels frass pixels as undamaged tissue; this helper
    recovers small neutral dark spots on the lamina.
    """
    roi = leaf_roi.astype(bool)
    if not np.any(roi):
        return np.zeros_like(roi, dtype=bool)

    tissue = tissue_mask.astype(bool) if tissue_mask is not None else roi
    bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (13, 13))
    contrast_mask = (local_mean - gray) >= float(local_contrast)
    dark_mask = gray <= float(dark_threshold)

    r, g, b = orig_rgb[:, :, 0], orig_rgb[:, :, 1], orig_rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    neutral = (mx.astype(np.int16) - mn.astype(np.int16)) <= int(neutral_chroma)
    white = (r >= 250) & (g >= 250) & (b >= 250)
    candidates = roi & tissue & contrast_mask & dark_mask & neutral & ~white

    out = np.zeros_like(roi, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if int(min_area) <= area <= int(max_area):
            out |= labels == i
    return out


def detect_frass_in_damage_zone(
    orig_rgb: np.ndarray,
    damage_context: np.ndarray,
    leaf_roi: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    *,
    local_contrast: float = DEFAULT_FRASS_IN_DAMAGE_CONTRAST,
    gray_max: int = DEFAULT_FRASS_IN_DAMAGE_GRAY_MAX,
    min_area: int = DEFAULT_FRASS_MIN_AREA,
    max_area: int = DEFAULT_FRASS_IN_DAMAGE_MAX_AREA,
) -> np.ndarray:
    """
    Detect brown/maroon frass pellets sitting on scraped or damaged lamina.

    On pale damaged tissue, frass often lacks global contrast; here we require
    only a modest local darkening relative to the surrounding damage patch.
    """
    ctx = damage_context.astype(bool)
    roi = leaf_roi.astype(bool)
    if not np.any(ctx) or orig_rgb is None:
        return np.zeros_like(roi, dtype=bool)

    tissue = tissue_mask.astype(bool) if tissue_mask is not None else roi
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    zone = cv2.dilate(ctx.astype(np.uint8), ker, iterations=1) > 0
    zone &= roi & tissue

    bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (11, 11))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h >= 25) & (h <= 90) & (s >= 55)
    r, g, b = orig_rgb[:, :, 0], orig_rgb[:, :, 1], orig_rgb[:, :, 2]
    white = (r >= 250) & (g >= 250) & (b >= 250)

    candidates = (
        zone
        & ~white
        & ~green
        & ((local_mean - gray) >= float(local_contrast))
        & (gray <= float(gray_max))
    )

    out = np.zeros_like(roi, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (int(min_area) <= area <= int(max_area)):
            continue
        if not np.any(comp & ctx):
            continue
        out |= comp
    return out


def detect_superficial_damage(
    orig_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    leaf_roi: np.ndarray,
    *,
    min_area: int = DEFAULT_SUPERFICIAL_MIN_AREA,
    min_vs_ratio: float = DEFAULT_SUPERFICIAL_VS_RATIO,
    vs_mult: float = DEFAULT_SUPERFICIAL_VS_MULT,
    max_sat: int = DEFAULT_SUPERFICIAL_MAX_SAT,
    edge_band_px: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
) -> tuple[np.ndarray, dict]:
    """
    Detect pale scraped tissue inside the leaf ROI.

    Scraped areas keep a thin tissue layer (not pure white holes). They are
    brighter and less saturated than healthy lamina, which shows up as a higher
    value/saturation (V/S) ratio in HSV.
    """
    roi = leaf_roi.astype(bool)
    if not np.any(roi):
        return np.zeros_like(roi, dtype=bool), {}

    hsv = cv2.cvtColor(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    _h, s, v = hsv[:, :, 0], hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    r, g, b = orig_rgb[:, :, 0], orig_rgb[:, :, 1], orig_rgb[:, :, 2]
    white = (r >= 250) & (g >= 250) & (b >= 250)
    base = roi & ~white

    healthy = _healthy_tissue_reference_mask(orig_rgb, base)
    meta: dict = {
        "healthy_ref_px": int(healthy.sum()),
        "vs_threshold": float(min_vs_ratio),
        "healthy_vs_median": None,
    }
    if int(healthy.sum()) < 50:
        return np.zeros_like(roi, dtype=bool), meta

    ratio = v / np.maximum(s, 1.0)
    ref_vs = float(np.median(ratio[healthy]))
    meta["healthy_vs_median"] = ref_vs
    vs_thr = max(float(min_vs_ratio), ref_vs * float(vs_mult))
    meta["vs_threshold"] = vs_thr

    superficial = (
        base
        & (ratio >= vs_thr)
        & (s >= 12)
        & (s <= int(max_sat))
        & (v >= 55)
        & (v <= 245)
    )

    if edge_band_px > 0 and np.any(tissue_mask):
        dist_source = fill_roi_holes(tissue_mask.astype(bool))
        dist_in = cv2.distanceTransform(dist_source.astype(np.uint8), cv2.DIST_L2, 5)
        superficial &= dist_in > float(edge_band_px)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_u8 = cv2.morphologyEx(superficial.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    out = np.zeros_like(roi, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out |= labels == i
    meta["raw_px"] = int(out.sum())
    return out, meta


def white_holes_inside_roi(
    orig_rgb: np.ndarray,
    leaf_roi: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    threshold: int = DEFAULT_WHITE_HOLE_BRIGHTNESS,
    min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    edge_band_px: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    adaptive: bool = DEFAULT_WHITE_HOLE_ADAPTIVE,
) -> np.ndarray:
    """Legacy wrapper; prefer detect_enclosed_white_holes when tissue_mask is available."""
    tissue = tissue_mask if tissue_mask is not None else leaf_roi
    return detect_enclosed_white_holes(
        orig_rgb,
        tissue,
        leaf_roi,
        threshold=threshold,
        min_area=min_area,
        edge_band_px=edge_band_px,
        adaptive=adaptive,
    )


def compute_marginal_damage_roi(
    leaf_roi: np.ndarray,
    tissue_mask: np.ndarray,
    pred_mask: np.ndarray,
    fill_marginal: bool,
    roi_mode: str = DEFAULT_ROI_MODE,
    orig_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute marginal (edge) damage that the U-Net missed.

    For roi_mode "filled": detects internal holes only (enclosed white gaps).
    For other modes: uses the unified GapDetector to identify genuine edge gaps,
    filtered by petiole/apex exclusion, multi-scale curvature, and whiteness
    verification — replacing the old raw hull-minus-mask approach.
    """
    if not fill_marginal:
        return np.zeros_like(leaf_roi, dtype=bool)

    if roi_mode in ("filled",):
        internal = internal_holes_mask(tissue_mask, leaf_roi)
        if not np.any(internal):
            return np.zeros_like(leaf_roi, dtype=bool)
        return internal & (pred_mask != 1)

    # Use GapDetector for edge gaps in hull / closed / other modes
    tissue_u8 = (tissue_mask.astype(np.uint8)) * 255
    try:
        sys.path.insert(0, str(_LEAF_DIR))
        from gap_detector import GapDetector
        det = GapDetector(tissue_u8, rgb_image=orig_rgb)
        result = det.detect(threshold=0.50, min_depth_px=10.0)
        gap_detected = result.damage_mask.astype(bool)
    except Exception:
        # Graceful degradation to legacy approach if GapDetector unavailable
        gap_detected = (leaf_roi & ~tissue_mask)

    return gap_detected & (pred_mask != 1) & leaf_roi


def compute_leaf_damage_metrics(
    pred_mask: np.ndarray,
    leaf_roi: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    raw_mask: np.ndarray | None = None,
    fill_marginal: bool = DEFAULT_FILL_MARGINAL,
    roi_mode: str = DEFAULT_ROI_MODE,
    orig_rgb: np.ndarray | None = None,
    white_threshold: int = DEFAULT_WHITE_HOLE_BRIGHTNESS,
    white_hole_min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    white_hole_edge_band: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    white_hole_adaptive: bool = DEFAULT_WHITE_HOLE_ADAPTIVE,
    superficial_damage: bool = DEFAULT_SUPERFICIAL_DAMAGE,
    superficial_min_area: int = DEFAULT_SUPERFICIAL_MIN_AREA,
) -> dict:
    """
    % damage = (U-Net class 1 + optional internal holes + white holes) / ROI area.
    tissue_mask = visible silhouette; leaf_roi = area with internal holes filled.
    """
    leaf_area_px = int(leaf_roi.sum())
    tissue = tissue_mask if tissue_mask is not None else leaf_roi

    visible_damage_px = int(((pred_mask == 1) & leaf_roi).sum())

    internal_mask = internal_holes_mask(tissue, leaf_roi)
    internal_holes_px = int(internal_mask.sum())

    if orig_rgb is not None and (white_threshold > 0 or white_hole_adaptive):
        used_thresh = int(white_threshold)
        if white_hole_adaptive:
            used_thresh = estimate_auto_white_hole_threshold(
                orig_rgb,
                tissue,
                leaf_roi,
                min_area=white_hole_min_area,
                edge_band_px=white_hole_edge_band,
                manual_hint=white_threshold if white_threshold > 0 else None,
            )
        rgb_holes_raw = _white_holes_at_threshold(
            orig_rgb,
            tissue,
            leaf_roi,
            threshold=used_thresh,
            min_area=white_hole_min_area,
            edge_band_px=white_hole_edge_band,
        )
        rgb_holes = rgb_holes_raw & (pred_mask != 1)
    else:
        used_thresh = int(white_threshold)
        rgb_holes = np.zeros_like(leaf_roi, dtype=bool)
    rgb_holes_px = int(rgb_holes.sum())

    internal_extra = internal_mask & (pred_mask != 1)
    white_holes = strip_perimeter_damage_bool(internal_extra | rgb_holes, tissue)
    white_holes_px = int(white_holes.sum())

    superficial_meta: dict = {}
    rim_ghost = np.zeros_like(leaf_roi, dtype=bool)
    if orig_rgb is not None:
        rim_ghost = _rim_nonfoliage_ghost_mask(tissue, orig_rgb, white_hole_edge_band)

    if superficial_damage and orig_rgb is not None:
        superficial_raw, superficial_meta = detect_superficial_damage(
            orig_rgb,
            tissue,
            leaf_roi,
            min_area=superficial_min_area,
            edge_band_px=white_hole_edge_band,
        )
        superficial_raw = superficial_raw & (pred_mask != 1) & ~white_holes & ~rim_ghost
        superficial = strip_perimeter_damage_bool(superficial_raw, tissue)
    else:
        superficial = np.zeros_like(leaf_roi, dtype=bool)
    superficial_px = int(superficial.sum())

    marginal_roi = compute_marginal_damage_roi(
        leaf_roi, tissue, pred_mask, fill_marginal=fill_marginal, roi_mode=roi_mode,
        orig_rgb=orig_rgb,
    )
    marginal_roi = strip_perimeter_damage_bool(marginal_roi, tissue)
    marginal_damage_px = int(marginal_roi.sum())

    damage_context = strip_perimeter_damage_bool(
        (((pred_mask == 1) & leaf_roi) & ~rim_ghost) | white_holes | superficial,
        tissue,
    )

    rgb_frass_global = (
        detect_frass_rgb(orig_rgb, leaf_roi, tissue)
        if orig_rgb is not None
        else np.zeros_like(leaf_roi, dtype=bool)
    )
    rgb_frass_in_damage = (
        detect_frass_in_damage_zone(orig_rgb, damage_context, leaf_roi, tissue)
        if orig_rgb is not None
        else np.zeros_like(leaf_roi, dtype=bool)
    )
    rgb_frass = rgb_frass_global | rgb_frass_in_damage
    frass_roi = ((pred_mask == 2) & leaf_roi) | rgb_frass

    frass_on_damage, frass_on_undamage = partition_frass_by_context(
        frass_roi,
        pred_mask,
        leaf_roi,
        damage_context,
        orig_rgb=orig_rgb,
    )
    # Any frass pellet inside the current damage zone fills the red mask gap.
    if np.any(frass_roi) and np.any(damage_context):
        k = max(
            3,
            2 * int(DEFAULT_FRASS_DAMAGE_DILATE) + 1,
        )
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dmg_zone = cv2.dilate(damage_context.astype(np.uint8), ker, iterations=1) > 0
        frass_inside_damage = frass_roi & dmg_zone
        frass_on_damage = frass_on_damage | frass_inside_damage | rgb_frass_in_damage
        frass_on_undamage = frass_on_undamage & ~frass_on_damage

    if roi_mode in ("filled", "lama") and fill_marginal:
        damage_union = strip_perimeter_damage_bool(
            damage_context | frass_on_damage,
            tissue,
        )
    else:
        damage_union = strip_perimeter_damage_bool(
            damage_context | frass_on_damage | marginal_roi,
            tissue,
        )
    damage_px = int(damage_union.sum())

    frass_px = int(frass_roi.sum())
    frass_rgb_px = int(rgb_frass.sum())
    frass_rgb_in_damage_px = int(rgb_frass_in_damage.sum())
    frass_unet_px = int(((pred_mask == 2) & leaf_roi).sum())
    frass_on_damage_px = int(frass_on_damage.sum())
    frass_on_undamage_px = int(frass_on_undamage.sum())
    undamage_base = (pred_mask == 3) & leaf_roi
    undamage_px = int((undamage_base | frass_on_undamage).sum())
    ignored_px = max(0, leaf_area_px - damage_px - undamage_px)

    healthy_px = leaf_area_px - damage_px

    damage_pct = (damage_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0
    visible_damage_pct = (
        (visible_damage_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0
    )
    marginal_damage_pct = (
        (marginal_damage_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0
    )
    undamage_pct = (undamage_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0

    tissue_area_px = int(tissue.sum())
    raw_mask_area_px = int(raw_mask.sum()) if raw_mask is not None else tissue_area_px
    roi_expansion_px = leaf_area_px - tissue_area_px
    white_holes_pct = (white_holes_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0

    return {
        "leaf_area_px": leaf_area_px,
        "tissue_area_px": tissue_area_px,
        "damage_px": damage_px,
        "damage_pct": damage_pct,
        "visible_damage_px": visible_damage_px,
        "visible_damage_pct": visible_damage_pct,
        "marginal_damage_px": marginal_damage_px,
        "marginal_damage_pct": marginal_damage_pct,
        "marginal_roi": marginal_roi,
        "white_holes_px": white_holes_px,
        "white_holes_pct": white_holes_pct,
        "white_holes_roi": white_holes,
        "internal_holes_px": internal_holes_px,
        "rgb_holes_px": rgb_holes_px,
        "white_hole_threshold_used": used_thresh,
        "superficial_px": superficial_px,
        "superficial_pct": (superficial_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0,
        "superficial_roi": superficial,
        "superficial_meta": superficial_meta,
        "frass_px": frass_px,
        "frass_pct": (frass_px / leaf_area_px * 100.0) if leaf_area_px > 0 else 0.0,
        "frass_rgb_px": frass_rgb_px,
        "frass_rgb_in_damage_px": frass_rgb_in_damage_px,
        "frass_unet_px": frass_unet_px,
        "frass_on_damage_px": frass_on_damage_px,
        "frass_on_undamage_px": frass_on_undamage_px,
        "frass_on_damage_roi": frass_on_damage,
        "frass_on_undamage_roi": frass_on_undamage,
        "frass_rgb_roi": rgb_frass,
        "frass_in_damage_roi": rgb_frass_in_damage,
        "damage_roi": damage_union,
        "undamage_px": undamage_px,
        "undamage_pct": undamage_pct,
        "healthy_px": healthy_px,
        "ignored_px": ignored_px,
        "raw_mask_area_px": raw_mask_area_px,
        "roi_expansion_px": roi_expansion_px,
        "tissue_mask": tissue,
    }


def collect_leaf_pairs(segmentation_root: str) -> list[dict]:
    """Image + mask pairs; deduplicates by canonical_leaf_id (prefers shorter filename)."""
    img_dir = os.path.join(segmentation_root, "white_bg")
    mask_dir = os.path.join(segmentation_root, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return []

    mask_map: dict[str, str] = {}
    for fname in os.listdir(mask_dir):
        stem = os.path.splitext(fname)[0]
        if stem.endswith("_mask"):
            mask_map[canonical_leaf_id(stem[: -len("_mask")])] = os.path.join(mask_dir, fname)

    candidates: dict[str, dict] = {}
    skipped_nested = 0

    for fname in os.listdir(img_dir):
        if Path(fname).suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        stem = os.path.splitext(fname)[0]
        if NESTED_DUP_RE.search(stem):
            skipped_nested += 1
            continue

        leaf_id = canonical_leaf_id(stem)
        mask_path = mask_map.get(leaf_id)
        if mask_path is None:
            print(f"  WARNING: No mask found for '{fname}' (id='{leaf_id}')")
            continue

        entry = {
            "img_path": os.path.join(img_dir, fname),
            "mask_path": mask_path,
            "image_name": fname,
            "mask_name": os.path.basename(mask_path),
            "leaf_id": leaf_id,
        }
        prev = candidates.get(leaf_id)
        if prev is None or len(fname) < len(prev["image_name"]):
            if prev is not None:
                print(f"  DEDUPE: {prev['image_name']} -> {fname}")
            candidates[leaf_id] = entry
        else:
            print(f"  DEDUPE: omitted {fname} (already exists {prev['image_name']})")

    if skipped_nested:
        print(f"  Skipped {skipped_nested} nested files (*_white_bg_leaf_XX_white_bg)")
    return list(candidates.values())


def split_tissue_fragments(
    tissue_mask: np.ndarray,
    min_area_ratio: float = MIN_FRAGMENT_AREA_RATIO,
) -> list[np.ndarray]:
    """A boolean mask per connected component (leaf fragment)."""
    h, w = tissue_mask.shape[:2]
    min_area = max(80, int(min_area_ratio * h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        tissue_mask.astype(np.uint8), connectivity=8
    )
    fragments: list[np.ndarray] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            fragments.append(labels == i)
    fragments.sort(key=lambda m: int(m.sum()), reverse=True)
    return fragments


def build_editable_damage_mask(
    pred_mask: np.ndarray,
    leaf_roi: np.ndarray,
    metrics: dict,
) -> np.ndarray:
    """Boolean mask of all pixels counted as damage (matches metrics damage_px)."""
    damage_roi = metrics.get("damage_roi")
    if damage_roi is not None:
        return damage_roi.astype(bool)

    visible_damage = (pred_mask == 1) & leaf_roi
    white_holes_roi = metrics.get("white_holes_roi", np.zeros_like(leaf_roi, dtype=bool))
    superficial_roi = metrics.get("superficial_roi", np.zeros_like(leaf_roi, dtype=bool))
    frass_on_damage = metrics.get("frass_on_damage_roi", np.zeros_like(leaf_roi, dtype=bool))
    marginal = metrics.get("marginal_roi", np.zeros_like(leaf_roi, dtype=bool))
    combined = visible_damage | white_holes_roi | superficial_roi | frass_on_damage | marginal
    tissue = metrics.get("tissue_mask")
    if tissue is None:
        tissue = leaf_roi
    return strip_perimeter_damage_bool(combined, tissue)


def compose_damage_rgb(
    orig_rgb: np.ndarray,
    damage_mask: np.ndarray,
    leaf_roi: np.ndarray,
    *,
    overlay_alpha: float = DEFAULT_DAMAGE_OVERLAY_ALPHA,
) -> np.ndarray:
    """RGB preview: leaf on white background; damage shown as semi-transparent red."""
    roi_vis = orig_rgb.copy().astype(np.float32)
    damage = damage_mask.astype(bool)
    roi_vis[~leaf_roi] = 255.0
    if np.any(damage):
        alpha = float(np.clip(overlay_alpha, 0.05, 1.0))
        red = np.array([220.0, 50.0, 50.0], dtype=np.float32)
        roi_vis[damage] = (1.0 - alpha) * roi_vis[damage] + alpha * red
    return np.clip(roi_vis, 0, 255).astype(np.uint8)


def connected_damage_component(
    damage_mask: np.ndarray,
    seed_xy: tuple[int, int],
) -> np.ndarray:
    """Boolean mask of the connected damage blob containing seed_xy, or empty."""
    x, y = int(seed_xy[0]), int(seed_xy[1])
    h, w = damage_mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or not damage_mask[y, x]:
        return np.zeros_like(damage_mask, dtype=bool)
    dmg_u8 = damage_mask.astype(np.uint8)
    n, labels = cv2.connectedComponents(dmg_u8, connectivity=8)
    label = labels[y, x]
    if label == 0:
        return np.zeros_like(damage_mask, dtype=bool)
    return labels == label


def flood_select_region(
    rgb: np.ndarray,
    seed_xy: tuple[int, int],
    leaf_roi: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Color-similar contiguous region inside leaf_roi (Lab flood fill)."""
    x, y = int(seed_xy[0]), int(seed_xy[1])
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or not leaf_roi[y, x]:
        return np.zeros((h, w), dtype=bool)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # OpenCV floodFill needs a mask 2px larger
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    tol = max(1, int(tolerance))
    lo = (tol, tol, tol)
    hi = (tol, tol, tol)
    work = lab.copy()
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(work, flood_mask, (x, y), 0, lo, hi, flags)
    region = flood_mask[1:-1, 1:-1] > 0
    return region & leaf_roi


def hybrid_select_damage_region(
    rgb: np.ndarray,
    seed_xy: tuple[int, int],
    leaf_roi: np.ndarray,
    tolerance: int,
    *,
    damage_mask: np.ndarray | None = None,
    mobilesam_model=None,
    max_leaf_frac: float = 0.35,
) -> np.ndarray:
    """Select a damage fragment: MobileSAM first, Lab flood fill as fallback.

    SAM is discarded when unavailable, empty after ROI clip, or when the
    selected area exceeds ``max_leaf_frac`` of the leaf (over-segmentation).
    Returned region is inside ``leaf_roi`` and excludes existing damage.
    """
    leaf = leaf_roi.astype(bool)
    dmg = (
        damage_mask.astype(bool)
        if damage_mask is not None
        else np.zeros(leaf.shape, dtype=bool)
    )

    def _flood_fallback() -> np.ndarray:
        region = flood_select_region(rgb, seed_xy, leaf, tolerance)
        return region & ~dmg

    if mobilesam_model is None:
        return _flood_fallback()

    x, y = int(seed_xy[0]), int(seed_xy[1])
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or not leaf[y, x]:
        return np.zeros((h, w), dtype=bool)

    try:
        biref_dir = str(_REPO_ROOT / "segmentation" / "birefnet_mobilesam")
        if biref_dir not in sys.path:
            sys.path.insert(0, biref_dir)
        from utils.segmentation_utils import run_mobilesam_point

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sam_mask = run_mobilesam_point(bgr, mobilesam_model, point=(x, y))
    except Exception:
        return _flood_fallback()

    region = sam_mask.astype(bool) & leaf & ~dmg
    leaf_area = int(leaf.sum())
    if not region.any() or leaf_area == 0:
        return _flood_fallback()
    if float(region.sum()) / float(leaf_area) > float(max_leaf_frac):
        return _flood_fallback()
    return region


def save_damage_preview_pil(
    orig_rgb: np.ndarray,
    damage_mask: np.ndarray,
    leaf_roi: np.ndarray,
    image_name: str,
    damage_pct: float,
    damage_px: int,
    out_path: str,
    scale_cm2_per_px: float | None = None,
) -> None:
    """Lightweight analyzed JPG for GUI edits (no matplotlib)."""
    roi_vis = compose_damage_rgb(orig_rgb, damage_mask, leaf_roi)
    img = Image.fromarray(roi_vis)
    title = damage_title(image_name, damage_pct, damage_px, scale_cm2_per_px)
    # Reserve space for title text above the image
    title_h = 36 if "\n" in title else 22
    canvas = Image.new("RGB", (img.width, img.height + title_h), (255, 255, 255))
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arialbd.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
    lines = title.split("\n")
    y_off = 4
    for i, line in enumerate(lines):
        f = font_bold if i == 0 else font
        draw.text((8, y_off), line, fill=(0, 0, 0), font=f)
        y_off += 14
    canvas.paste(img, (0, title_h))
    canvas.save(out_path, quality=90)


def damage_title(
    image_name: str,
    damage_pct: float,
    damage_px: int,
    scale_cm2_per_px: float | None = None,
) -> str:
    if scale_cm2_per_px is not None:
        damage_cm2 = damage_px * scale_cm2_per_px
        return f"{image_name}\nDamage: {format_damage_pct(damage_pct)}%  |  {damage_cm2:.2f} cm²"
    return f"{image_name}  |  Damage: {format_damage_pct(damage_pct)}%"


def save_visualization(
    orig_rgb,
    pred_mask,
    leaf_roi,
    raw_mask,
    metrics: dict,
    image_name: str,
    out_path: str,
    roi_mode: str,
    scale_cm2_per_px: float | None = None,
    damage_mask: np.ndarray | None = None,
) -> None:
    """Save analyzed preview at the same pixel size as the leaf (no matplotlib)."""
    damage_pct = metrics["damage_pct"]
    damage_px = metrics["damage_px"]
    if damage_mask is None:
        damage_mask = build_editable_damage_mask(pred_mask, leaf_roi, metrics)

    save_damage_preview_pil(
        orig_rgb,
        damage_mask,
        leaf_roi,
        image_name,
        damage_pct,
        damage_px,
        out_path,
        scale_cm2_per_px=scale_cm2_per_px,
    )


def save_damage_visualization_from_mask(
    orig_rgb: np.ndarray,
    damage_mask: np.ndarray,
    leaf_roi: np.ndarray,
    image_name: str,
    damage_pct: float,
    damage_px: int,
    out_path: str,
    scale_cm2_per_px: float | None = None,
) -> None:
    """Save analyzed JPG from an edited damage mask (no leaf contour)."""
    save_damage_preview_pil(
        orig_rgb,
        damage_mask,
        leaf_roi,
        image_name,
        damage_pct,
        damage_px,
        out_path,
        scale_cm2_per_px=scale_cm2_per_px,
    )


def save_damage_sidecars(
    out_dir: str,
    out_stem: str,
    damage_mask: np.ndarray,
    leaf_roi: np.ndarray,
    image_name: str,
    metrics: dict,
    scale_cm2_per_px: float | None,
) -> None:
    """Persist editable damage mask + ROI + metadata for the GUI editor."""
    dmg_path = os.path.join(out_dir, f"{out_stem}_damage_mask.png")
    roi_path = os.path.join(out_dir, f"{out_stem}_leaf_roi.png")
    meta_path = os.path.join(out_dir, f"{out_stem}_meta.json")
    cv2.imwrite(dmg_path, (damage_mask.astype(np.uint8) * 255))
    cv2.imwrite(roi_path, (leaf_roi.astype(np.uint8) * 255))
    meta = {
        "image_name": image_name,
        "leaf_area_px": int(metrics["leaf_area_px"]),
        "damage_px": int(metrics["damage_px"]),
        "damage_pct": round_damage_pct(metrics["damage_pct"]),
        "scale_cm2_per_px": scale_cm2_per_px,
    }
    Path(meta_path).write_text(_json_mod.dumps(meta, indent=2), encoding="utf-8")


def process_leaf_pair(
    img_path: str,
    mask_path: str,
    unet,
    unet_size: int,
    out_dir: str,
    roi_mode: str,
    fill_marginal: bool = DEFAULT_FILL_MARGINAL,
    draw_hull_line: bool = False,
    scale_data: dict | None = None,
    scale_area_cm2: float = DEFAULT_SCALE_AREA_CM2,
    white_hole_brightness: int = DEFAULT_WHITE_HOLE_BRIGHTNESS,
    white_hole_min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    white_hole_edge_band: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    white_hole_adaptive: bool = DEFAULT_WHITE_HOLE_ADAPTIVE,
    superficial_damage: bool = DEFAULT_SUPERFICIAL_DAMAGE,
    superficial_min_area: int = DEFAULT_SUPERFICIAL_MIN_AREA,
    edge_artifact_filter: bool = DEFAULT_EDGE_ARTIFACT_FILTER,
    edge_min_area: int | None = None,
    edge_min_inward_px: float = DEFAULT_EDGE_MIN_INWARD_PX,
) -> list[dict]:
    image_name = os.path.basename(img_path)
    print(f"\n[{image_name}]")

    # Resolve scale factor for this image (cm² per pixel), if available
    scale_cm2_per_px: float | None = None
    if scale_data:
        factor = lookup_scale_factor(scale_data, image_name)
        if factor is not None:
            scale_cm2_per_px = factor  # already cm²/px from scale_reference.json
            print(f"  Scale: {scale_cm2_per_px:.6f} cm2/px  (known area = {scale_area_cm2} cm2)")
        else:
            print("  Scale: blue dot not detected for this image - reporting % only")

    orig_rgb = load_rgb(img_path)
    orig_h, orig_w = orig_rgb.shape[:2]

    # NOTE: scale_cm2_per_px from scale_reference.json is already expressed in
    # white_bg-pixel space (gui/pipeline.py's run_scale_detection() applies the
    # BiRefNet crop/letterbox scale_factor before writing it out). No further
    # resolution correction is needed here.

    leaf_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if leaf_mask_raw is None:
        print(f"  ERROR: Could not read mask '{mask_path}'")
        return []
    if leaf_mask_raw.shape[:2] != (orig_h, orig_w):
        leaf_mask_raw = cv2.resize(
            leaf_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    tissue_full = leaf_mask_raw > 127
    tissue_full, mask_refined = refine_leaf_roi_from_image(tissue_full, orig_rgb)
    tissue_full, halo_trim_px = trim_contour_halo_from_tissue(tissue_full, orig_rgb)
    if halo_trim_px:
        print(
            f"  Contour halo trim: removed {halo_trim_px:,} px "
            f"outside visible lamina"
        )
    if int(tissue_full.sum()) == 0:
        print("  WARNING: Empty leaf mask, skipping.")
        return []

    if roi_mode in ("filled", "lama"):
        fragments = split_tissue_fragments(tissue_full)
        if not fragments:
            fragments = [tissue_full]
    else:
        fragments = [tissue_full]

    use_marginal = fill_marginal and roi_mode in ("hull", "closed", "filled", "lama")

    unet_input_rgb = orig_rgb.copy()
    # Restrict UNet input to the leaf contour from Tab 3: pixels outside are forced
    # to pure white so the model cannot hallucinate damage in background regions.
    if roi_mode in ("hull", "closed"):
        unet_input_mask = convex_hull_mask(tissue_full)
    else:
        unet_input_mask = fill_roi_holes(tissue_full)
    unet_input_rgb[~unet_input_mask] = 255

    if draw_hull_line:
        hull_mask = convex_hull_mask(tissue_full)
        contours, _ = cv2.findContours(
            (hull_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.polylines(unet_input_rgb, contours, True, (0, 255, 0), 2)

    pred_mask = predict_unet(unet, unet_input_rgb, unet_size)
    # Hard-clip predictions to the contour from Tab 3 immediately after inference.
    # This is the authoritative constraint — nothing outside the user-selected leaf
    # area can be classified as damage, regardless of downstream ROI mode.
    pred_mask = cleanup_predictions_outside_leaf(pred_mask, unet_input_mask)
    pred_mask, edge_artifact_px = filter_boundary_damage_artifacts(
        pred_mask,
        tissue_full,
        enabled=edge_artifact_filter,
        min_component_area=edge_min_area,
        min_inward_px=edge_min_inward_px,
    )
    if edge_artifact_px:
        tissue_px = max(1, int(tissue_full.sum()))
        print(
            f"  Edge artifact filter: removed {edge_artifact_px:,} px "
            f"({edge_artifact_px / tissue_px * 100:.3f}% of tissue)"
        )

    combined_roi = np.zeros_like(tissue_full, dtype=bool)
    combined_tissue = np.zeros_like(tissue_full, dtype=bool)
    combined_marginal = np.zeros_like(tissue_full, dtype=bool)
    combined_white_holes = np.zeros_like(tissue_full, dtype=bool)
    combined_superficial = np.zeros_like(tissue_full, dtype=bool)
    combined_frass_on_damage = np.zeros_like(tissue_full, dtype=bool)
    agg_damage = agg_area = agg_visible = agg_marginal = agg_white_holes = agg_superficial = 0

    rows: list[dict] = []
    n_frag = len(fragments)

    for frag_i, tissue_mask in enumerate(fragments, start=1):
        raw_mask = fill_roi_holes(tissue_mask.copy())
        leaf_roi = expand_roi(tissue_mask, mode=roi_mode, orig_rgb=orig_rgb)
        pred_clip = cleanup_predictions_outside_leaf(pred_mask, leaf_roi)
        metrics = compute_leaf_damage_metrics(
            pred_clip,
            leaf_roi,
            tissue_mask=tissue_mask,
            raw_mask=raw_mask,
            fill_marginal=use_marginal,
            roi_mode=roi_mode,
            orig_rgb=orig_rgb,
            white_threshold=white_hole_brightness,
            white_hole_min_area=white_hole_min_area,
            white_hole_edge_band=white_hole_edge_band,
            white_hole_adaptive=white_hole_adaptive,
            superficial_damage=superficial_damage,
            superficial_min_area=superficial_min_area,
        )

        rh = metrics.get("rgb_holes_px", 0)
        ih = metrics.get("internal_holes_px", 0)
        sp = metrics.get("superficial_px", 0)
        fp = metrics.get("frass_px", 0)
        fpd = metrics.get("frass_on_damage_px", 0)
        fpu = metrics.get("frass_on_undamage_px", 0)
        used_t = metrics.get("white_hole_threshold_used")
        if rh or ih or sp or fp or used_t is not None:
            auto_tag = f"  auto_brightness={used_t}" if white_hole_adaptive else f"  brightness={used_t}"
            frass_tag = ""
            if fp:
                frass_tag = f"  |  frass {fp:,} px (→ damage {fpd:,}, undamage {fpu:,})"
            print(
                f"  Hole detection: U-Net {metrics['visible_damage_px']:,} px  |  "
                f"RGB holes {rh:,} px  |  mask holes {ih:,} px  |  "
                f"superficial {sp:,} px{frass_tag}{auto_tag}"
            )

        combined_roi |= leaf_roi
        combined_tissue |= tissue_mask
        combined_marginal |= metrics.get("marginal_roi", np.zeros_like(leaf_roi))
        combined_white_holes |= metrics.get("white_holes_roi", np.zeros_like(leaf_roi))
        combined_superficial |= metrics.get("superficial_roi", np.zeros_like(leaf_roi))
        combined_frass_on_damage |= metrics.get("frass_on_damage_roi", np.zeros_like(leaf_roi))
        agg_damage += metrics["damage_px"]
        agg_area += metrics["leaf_area_px"]
        agg_visible += metrics["visible_damage_px"]
        agg_marginal += metrics["marginal_damage_px"]
        agg_white_holes += metrics.get("white_holes_px", 0)
        agg_superficial += metrics.get("superficial_px", 0)

        frag_tag = f" frag {frag_i}/{n_frag}" if n_frag > 1 else ""
        if scale_cm2_per_px:
            leaf_area_cm2 = metrics["leaf_area_px"] * scale_cm2_per_px
            damage_cm2 = metrics["damage_px"] * scale_cm2_per_px
            print(
                f"  {image_name}{frag_tag}: Leaf area {leaf_area_cm2:.2f} cm²  |  "
                f"Damage area {damage_cm2:.4f} cm²  ({format_damage_pct(metrics['damage_pct'])}%)"
            )
        else:
            print(
                f"  {image_name}{frag_tag}: Leaf area {metrics['leaf_area_px']:,} px  |  "
                f"Damage area {metrics['damage_px']:,} px  ({format_damage_pct(metrics['damage_pct'])}%)"
            )

        rows.append(
            {
                "image_name": image_name,
                "leaf_area_px": metrics["leaf_area_px"],
                "leaf_area_cm2": round(metrics["leaf_area_px"] * scale_cm2_per_px, 4) if scale_cm2_per_px else "",
                "damage_px": metrics["damage_px"],
                "damage_pct": round_damage_pct(metrics["damage_pct"]),
                "damage_cm2": round(metrics["damage_px"] * scale_cm2_per_px, 4) if scale_cm2_per_px else "",
            }
        )

    note = " [re-estimated mask]" if mask_refined else ""
    if n_frag > 1:
        print(f"  => {n_frag} fragments; weighted image damage: {format_damage_pct(agg_damage / max(1, agg_area) * 100)}%{note}")

    agg_pct = agg_damage / max(1, agg_area) * 100.0
    viz_metrics = compute_leaf_damage_metrics(
        cleanup_predictions_outside_leaf(pred_mask, combined_roi),
        combined_roi,
        tissue_mask=combined_tissue,
        raw_mask=fill_roi_holes(combined_tissue),
        fill_marginal=use_marginal,
        roi_mode=roi_mode,
        orig_rgb=orig_rgb,
        white_threshold=white_hole_brightness,
        white_hole_min_area=white_hole_min_area,
        white_hole_edge_band=white_hole_edge_band,
        white_hole_adaptive=white_hole_adaptive,
        superficial_damage=superficial_damage,
        superficial_min_area=superficial_min_area,
    )
    viz_metrics["damage_px"] = agg_damage
    viz_metrics["damage_pct"] = agg_pct
    viz_metrics["visible_damage_px"] = agg_visible
    viz_metrics["marginal_damage_px"] = agg_marginal
    viz_metrics["marginal_roi"] = combined_marginal
    viz_metrics["white_holes_roi"] = combined_white_holes
    viz_metrics["white_holes_px"] = agg_white_holes
    viz_metrics["superficial_roi"] = combined_superficial
    viz_metrics["superficial_px"] = agg_superficial
    viz_metrics["frass_on_damage_roi"] = combined_frass_on_damage
    viz_metrics["damage_roi"] = strip_perimeter_damage_bool(
        (((pred_mask == 1) & combined_roi) | combined_white_holes | combined_superficial | combined_frass_on_damage | combined_marginal),
        combined_tissue,
    )
    viz_metrics["tissue_mask"] = combined_tissue

    out_stem = os.path.splitext(image_name)[0]
    out_path = os.path.join(out_dir, f"{out_stem}_analyzed.jpg")
    pred_clip = cleanup_predictions_outside_leaf(pred_mask, combined_roi)
    editable_damage = build_editable_damage_mask(pred_clip, combined_roi, viz_metrics)
    save_visualization(
        orig_rgb,
        pred_clip,
        combined_roi,
        fill_roi_holes(combined_tissue),
        viz_metrics,
        image_name,
        out_path,
        roi_mode,
        scale_cm2_per_px=scale_cm2_per_px,
        damage_mask=editable_damage,
    )
    save_damage_sidecars(
        out_dir,
        out_stem,
        editable_damage,
        combined_roi,
        image_name,
        viz_metrics,
        scale_cm2_per_px,
    )
    print(f"  => Saved: {out_path}")

    return rows


def extract_photo_number(image_name: str) -> int | None:
    m = re.match(r"^(\d+)", image_name)
    return int(m.group(1)) if m else None


def run(
    segmentation_dir: str,
    out_dir: str,
    unet_path: str,
    unet_size: int,
    roi_mode: str = DEFAULT_ROI_MODE,
    fill_marginal: bool = DEFAULT_FILL_MARGINAL,
    draw_hull_line: bool = False,
    scale_file: str | None = None,
    scale_area_cm2: float = DEFAULT_SCALE_AREA_CM2,
    white_hole_brightness: int = DEFAULT_WHITE_HOLE_BRIGHTNESS,
    white_hole_min_area: int = DEFAULT_WHITE_HOLE_MIN_AREA,
    white_hole_edge_band: int = DEFAULT_WHITE_HOLE_EDGE_BAND,
    white_hole_adaptive: bool = DEFAULT_WHITE_HOLE_ADAPTIVE,
    superficial_damage: bool = DEFAULT_SUPERFICIAL_DAMAGE,
    superficial_min_area: int = DEFAULT_SUPERFICIAL_MIN_AREA,
    edge_artifact_filter: bool = DEFAULT_EDGE_ARTIFACT_FILTER,
    edge_min_area: int | None = None,
    edge_min_inward_px: float = DEFAULT_EDGE_MIN_INWARD_PX,
) -> None:
    if roi_mode not in VALID_ROI_MODES:
        raise ValueError(f"roi_mode must be one of {VALID_ROI_MODES}, received {roi_mode!r}")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    print("=" * 65)
    print("  U-Net Analysis - individual leaves")
    print(f"  Segmentation input : {segmentation_dir}")
    print(f"  Output             : {out_dir}/")
    print(f"  Device             : {DEVICE}  |  Model: {unet_path}  |  UNet: {unet_size}")
    print(f"  ROI mode           : {roi_mode}")
    if roi_mode == "filled":
        marg_desc = (
            "internal holes as damage"
            if fill_marginal
            else "U-Net only (no internal holes)"
        )
    else:
        marg_desc = (
            "hull/closed border"
            if fill_marginal and roi_mode in ("hull", "closed")
            else "NO"
        )
    print(f"  Extra damage fill     : {marg_desc}")
    print(f"  Edge artifact filter  : {'ON' if edge_artifact_filter else 'OFF'}")
    print(
        f"  White hole detection  : "
        f"{'AUTO brightness (per image)' if white_hole_adaptive else f'brightness>={white_hole_brightness}'}  "
        f"min_area={white_hole_min_area}  edge_band={white_hole_edge_band}px"
    )
    print(
        f"  Superficial damage   : "
        f"{'ON' if superficial_damage else 'OFF'}"
        f"{f'  min_area={superficial_min_area}' if superficial_damage else ''}"
    )
    print(f"  Fragments             : one ROI per connected component")
    print(f"  Scale reference    : {scale_file if scale_file else 'None (% only)'}")
    print("=" * 65)

    # Load scale reference JSON if provided
    scale_data: dict | None = None
    if scale_file and Path(scale_file).is_file():
        scale_data = load_scale_json(Path(scale_file))
        print(f"  Scale data loaded: {len(scale_data)} entries")
    elif scale_file:
        print(f"  WARNING: scale file not found: {scale_file} - reporting % only")

    pairs = collect_leaf_pairs(segmentation_dir)
    if not pairs:
        print(
            "ERROR: No image-mask pairs found in white_bg/ and masks/.\n"
            "First run: python whitebg_masks.py --input <folder>"
        )
        return

    print(f"\nLeaves found: {len(pairs)}")
    unet = load_unet(unet_path)
    print("  U-Net loaded.\n")

    all_rows = []
    for p in pairs:
        rows = process_leaf_pair(
            p["img_path"],
            p["mask_path"],
            unet,
            unet_size,
            out_dir,
            roi_mode=roi_mode,
            fill_marginal=fill_marginal,
            draw_hull_line=draw_hull_line,
            scale_data=scale_data,
            scale_area_cm2=scale_area_cm2,
            white_hole_brightness=white_hole_brightness,
            white_hole_min_area=white_hole_min_area,
            white_hole_edge_band=white_hole_edge_band,
            white_hole_adaptive=white_hole_adaptive,
            superficial_damage=superficial_damage,
            superficial_min_area=superficial_min_area,
            edge_artifact_filter=edge_artifact_filter,
            edge_min_area=edge_min_area,
            edge_min_inward_px=edge_min_inward_px,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo results generated.")
        return

    by_image = defaultdict(lambda: {
        "total_damage_px": 0, "total_area_px": 0,
        "total_damage_cm2": 0.0, "total_area_cm2": 0.0, "has_cm2": False,
    })
    for row in all_rows:
        totals = by_image[row["image_name"]]
        totals["total_damage_px"] += row["damage_px"]
        totals["total_area_px"] += row["leaf_area_px"]
        if row["leaf_area_cm2"] != "" and row["damage_cm2"] != "":
            totals["total_damage_cm2"] += row["damage_cm2"]
            totals["total_area_cm2"] += row["leaf_area_cm2"]
            totals["has_cm2"] = True

    # Whether the cm² case applies (GUI selected cm² mode and scale was resolved)
    has_cm2 = any(t["has_cm2"] for t in by_image.values())
    leaf_area_col = "leaf_area_cm2" if has_cm2 else "leaf_area_px"
    damage_col = "damage_cm2" if has_cm2 else "damage_pct"

    summary_rows = []
    for key, totals in by_image.items():
        area_px = totals["total_area_px"]
        damage_pct = round_damage_pct(totals["total_damage_px"] / area_px * 100) if area_px > 0 else 0.0
        if totals["has_cm2"]:
            leaf_area_value = round(totals["total_area_cm2"], 4)
            damage_value = round(totals["total_damage_cm2"], 4)
        else:
            leaf_area_value = totals["total_area_px"]
            damage_value = damage_pct
        summary_rows.append({
            "photo_number": extract_photo_number(key),
            "image_name": key,
            leaf_area_col: leaf_area_value,
            damage_col: damage_value,
        })
    summary_rows.sort(
        key=lambda r: (
            r["photo_number"] if r["photo_number"] is not None else float("inf"),
            r["image_name"],
        )
    )

    csv_fields = ["image_name", leaf_area_col, damage_col]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in csv_fields} for row in summary_rows)

    print(f"\nCSV output  : {csv_path}")
    print(f"Total images: {len(summary_rows)}")
    print(f"\nDone. Results saved in: {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U-Net on individual segmented leaves (without convex hull).",
    )
    parser.add_argument(
        "--segmentation-dir",
        default=DEFAULT_SEGMENTATION_DIR,
        help=f"Folder containing white_bg/ and masks/ (default: {DEFAULT_SEGMENTATION_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--unet-path",
        default=UNET_PATH,
        help=f"U-Net checkpoint (default: {UNET_PATH})",
    )
    parser.add_argument(
        "--unet-size",
        type=int,
        default=UNET_SIZE,
        help=f"U-Net input resolution (default: {UNET_SIZE})",
    )
    parser.add_argument(
        "--roi-mode",
        default=DEFAULT_ROI_MODE,
        choices=VALID_ROI_MODES,
        help=(
            "ROI Mode: 'filled' (default, ImageJ-friendly), 'mask', 'closed', 'hull', 'lama'."
        ),
    )
    parser.add_argument(
        "--no-fill-marginal",
        action="store_true",
        help=(
            "filled/lama: do not add internal holes to damage. "
            "hull/closed: do not add edge-silhouette gaps to damage."
        ),
    )
    parser.add_argument(
        "--draw-hull-line",
        action="store_true",
        help="Draws a green temporal hull contour before passing image to U-Net.",
    )
    parser.add_argument(
        "--scale-file",
        type=str,
        default=None,
        help="Path to scale_reference.json produced by scale_detect.py. Enables cm² output.",
    )
    parser.add_argument(
        "--scale-area-cm2",
        type=float,
        default=DEFAULT_SCALE_AREA_CM2,
        help=f"Known physical area of the blue dot reference in cm² (default: {DEFAULT_SCALE_AREA_CM2}).",
    )
    parser.add_argument(
        "--white-hole-brightness",
        type=int,
        default=DEFAULT_WHITE_HOLE_BRIGHTNESS,
        help=(
            f"Manual per-channel brightness (0-255) for white holes when auto is off "
            f"(default: {DEFAULT_WHITE_HOLE_BRIGHTNESS}). Ignored as a hard cutoff when "
            f"auto brightness is enabled (used only as a soft hint). Use 0 to disable "
            f"when auto is off."
        ),
    )
    parser.add_argument(
        "--white-hole-min-area",
        type=int,
        default=DEFAULT_WHITE_HOLE_MIN_AREA,
        help=f"Minimum connected area (px) for RGB white holes (default: {DEFAULT_WHITE_HOLE_MIN_AREA}).",
    )
    parser.add_argument(
        "--white-hole-edge-band",
        type=int,
        default=DEFAULT_WHITE_HOLE_EDGE_BAND,
        help=(
            f"Exclude white-hole candidates within this many px of the tissue boundary "
            f"(default: {DEFAULT_WHITE_HOLE_EDGE_BAND})."
        ),
    )
    parser.add_argument(
        "--no-white-hole-adaptive",
        action="store_true",
        help="Disable per-image AUTO brightness (use --white-hole-brightness as fixed threshold).",
    )
    parser.add_argument(
        "--no-superficial-damage",
        action="store_true",
        help="Disable pale scraped-tissue detection (superficial herbivory).",
    )
    parser.add_argument(
        "--superficial-min-area",
        type=int,
        default=DEFAULT_SUPERFICIAL_MIN_AREA,
        help=f"Minimum connected area (px) for superficial damage (default: {DEFAULT_SUPERFICIAL_MIN_AREA}).",
    )
    parser.add_argument(
        "--no-edge-artifact-filter",
        action="store_true",
        help=(
            "Disable peripheral U-Net damage artifact filter "
            "(thin false positives along the leaf contour)."
        ),
    )
    parser.add_argument(
        "--edge-min-area",
        type=int,
        default=None,
        help="Override minimum damage component area (px) for the edge artifact filter.",
    )
    parser.add_argument(
        "--edge-min-inward-px",
        type=float,
        default=DEFAULT_EDGE_MIN_INWARD_PX,
        help=(
            f"Minimum inward depth (px) from the contour to keep small edge damage "
            f"(default: {DEFAULT_EDGE_MIN_INWARD_PX})."
        ),
    )
    args = parser.parse_args()
    run(
        segmentation_dir=args.segmentation_dir,
        out_dir=args.out_dir,
        unet_path=args.unet_path,
        unet_size=args.unet_size,
        roi_mode=args.roi_mode,
        fill_marginal=not args.no_fill_marginal,
        draw_hull_line=args.draw_hull_line,
        scale_file=args.scale_file,
        scale_area_cm2=args.scale_area_cm2,
        white_hole_brightness=args.white_hole_brightness,
        white_hole_min_area=args.white_hole_min_area,
        white_hole_edge_band=args.white_hole_edge_band,
        white_hole_adaptive=not args.no_white_hole_adaptive,
        superficial_damage=not args.no_superficial_damage,
        superficial_min_area=args.superficial_min_area,
        edge_artifact_filter=not args.no_edge_artifact_filter,
        edge_min_area=args.edge_min_area,
        edge_min_inward_px=args.edge_min_inward_px,
    )


if __name__ == "__main__":
    main()
