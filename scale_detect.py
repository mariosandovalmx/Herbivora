"""Scale calibration via blue reference dot detection in scene images.

Provides the functions used by analyze_leaves.py:
    load_scale_json, lookup_scale_factor, write_scale_json, scan_folder

JSON format: {image_name_or_stem: cm2_per_px2, ..., "_orig_dims": {name: [w, h]}}
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from image_io import VALID_IMAGE_EXTENSIONS as _VALID_EXT, load_bgr
_WHITE_BG_RE = re.compile(r"_white_bg(?:_leaf_\d+(?:_white_bg)?)?$", re.IGNORECASE)

_BIREFNET_UTILS = Path(__file__).resolve().parent / "segmentation" / "birefnet_mobilesam"
if str(_BIREFNET_UTILS) not in sys.path:
    sys.path.insert(0, str(_BIREFNET_UTILS))

_DETECT_CFG = {
    "circle": {
        "hsv_lower": [100, 80, 40],
        "hsv_upper": [130, 255, 255],
        "min_circularity": 0.65,
        "min_area_px": 100,
    }
}


def _detect_blue_circle_bgr(bgr: np.ndarray) -> dict | None:
    """Detect the blue/dark reference dot. Returns {'area_px', 'diameter_px'} or None.

    Delegates to circle_utils.detect_blue_circle's multi-method cascade
    (HSV bands + Hough + LAB + dark-blob) so this fallback path catches
    the same dark/near-black dots the primary BiRefNet pipeline does.
    """
    from utils.circle_utils import detect_blue_circle

    result = detect_blue_circle(bgr, _DETECT_CFG)
    if not result.get("found"):
        return None
    return {"area_px": result["area_px"], "diameter_px": result["diameter_px"]}


def scan_folder(
    input_dir: str | Path, known_cm2: float = 0.2827
) -> dict[str, float | None]:
    """Scan scene images for blue reference dot.

    Returns {filename: cm2_per_px2 | None}.
    """
    results: dict[str, float | None] = {}
    for p in sorted(Path(input_dir).iterdir()):
        if p.suffix.lower() not in _VALID_EXT:
            continue
        bgr = load_bgr(p)
        if bgr is None:
            results[p.name] = None
            continue
        info = _detect_blue_circle_bgr(bgr)
        if info is None:
            results[p.name] = None
        else:
            circle_area_px = math.pi * (info["diameter_px"] / 2) ** 2
            results[p.name] = known_cm2 / circle_area_px
    return results


def write_scale_json(results: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_scale_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def lookup_scale_factor(scale_data: dict, image_name: str) -> float | None:
    """Return cm²/px² for image_name, trying bare stem and _white_bg-stripped variants.

    Keys in scale_data may be bare stems (from BiRefNet metadata) or
    full filenames with extension (from scan_folder).
    """
    if not scale_data:
        return None

    # 1. Exact filename match
    v = scale_data.get(image_name)
    if v is not None:
        return float(v)

    # 2. Bare stem (e.g. "IMG_001.png" → "IMG_001")
    stem = Path(image_name).stem
    v = scale_data.get(stem)
    if v is not None:
        return float(v)

    # 3. Strip _white_bg suffix and try bare base + extensions
    base = _WHITE_BG_RE.sub("", stem)
    if base != stem:
        v = scale_data.get(base)
        if v is not None:
            return float(v)
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
            v = scale_data.get(base + ext)
            if v is not None:
                return float(v)

    # 4. Stem (no _white_bg stripped) with extensions
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"):
        v = scale_data.get(stem + ext)
        if v is not None:
            return float(v)

    return None
