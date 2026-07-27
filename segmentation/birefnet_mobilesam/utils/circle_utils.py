"""Blue circle detection and scale calibration utilities.

All color-based detection runs on a gray-world white-balance-normalized
copy of the image, neutralizing unusual lighting/camera color casts before
any thresholding (see `_gray_world_normalize`).

Five complementary methods tried in cascade and scored:
  1. HSV multi-range  — handles exposure and white-balance variation
  2. Hough Circle Transform — robust to imperfect circularity, runs on
     the fused blueness map (see method 5) rather than a single fixed mask
  3. LAB b* channel   — perceptually uniform blue, exposure-independent
  4. Dark blob + hue verification — handles near-black/navy dots whose
     value (brightness) is too low for any HSV band's V floor, but whose
     hue is still verifiably blue once isolated by darkness alone.
  5. Fused blueness map + adaptive Otsu threshold — combines LAB b*, YCbCr
     Cb, HSV hue/saturation, and an RGB blue-dominance chromaticity ratio
     into one continuous per-pixel score, then thresholds it per-image
     with Otsu instead of a fixed cutoff. Generalizes methods 1/3/4 into a
     single adaptive signal that isn't tied to thresholds tuned on one
     dataset.

The best-scoring candidate across all methods is returned. A `low_confidence`
flag is set when the winning score falls below `circle.min_confidence_score`,
so the GUI can prompt for manual correction instead of silently propagating
a bad detection into the scale calibration.
"""

from __future__ import annotations

import math
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV candidate ranges (tried in order; first match wins only if score > 0)
# Each entry: (lower_hsv, upper_hsv)
# ---------------------------------------------------------------------------
_HSV_BANDS: list[tuple[list[int], list[int]]] = [
    ([100, 80, 40],  [130, 255, 255]),   # standard well-lit blue
    ([95,  50, 25],  [135, 255, 255]),   # wider hue, lower saturation
    ([100, 30, 15],  [130, 220, 200]),   # dark / underexposed
    ([85,  35, 50],  [145, 255, 255]),   # very wide: unusual white-balance
]


def _gray_world_normalize(image_bgr: np.ndarray) -> np.ndarray:
    """Gray-world white-balance correction.

    Scales each color channel so its mean matches the overall gray mean,
    neutralizing color casts from unusual lighting or camera white-balance
    settings before any color-based thresholding runs. Geometry (shape,
    pixel coordinates) is unchanged — only color values shift.
    """
    img = image_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mean_b, mean_g, mean_r = float(b.mean()), float(g.mean()), float(r.mean())
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    eps = 1e-6
    b = np.clip(b * (mean_gray / (mean_b + eps)), 0, 255)
    g = np.clip(g * (mean_gray / (mean_g + eps)), 0, 255)
    r = np.clip(r * (mean_gray / (mean_r + eps)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def _blueness_map(image_bgr: np.ndarray) -> np.ndarray:
    """Fuse four brightness/exposure-decoupled blue indicators into one
    continuous 0-255 per-pixel score, instead of relying on a single
    hand-tuned hue/sat/value cutoff:

      - LAB b* (inverted)             — exposure-independent
      - YCbCr Cb                      — independent blue chrominance signal
      - HSV hue-proximity-to-blue * saturation — classic hue/sat signal
      - RGB blue-dominance chromaticity ratio  — brightness-independent,
        the principle behind the dark-blob method generalized into a
        continuous score

    A per-image adaptive threshold (Otsu) on this map handles dots whose
    color falls outside any single hard-coded band.
    """
    img = image_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b_score = np.clip((128.0 - lab[:, :, 2]) / 60.0, 0.0, 1.0)

    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cb_score = np.clip((ycc[:, :, 2] - 128.0) / 60.0, 0.0, 1.0)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1] / 255.0
    hue_dist = np.minimum(np.abs(hue - 115.0), 180.0 - np.abs(hue - 115.0))
    hue_score = np.clip(1.0 - hue_dist / 35.0, 0.0, 1.0) * sat

    total = r + g + b + 1e-6
    chroma_score = np.clip((b - np.maximum(r, g)) / total * 4.0, 0.0, 1.0)

    fused = (lab_b_score + cb_score + hue_score + chroma_score) / 4.0
    return (fused * 255.0).astype(np.uint8)


def _blue_score(image_bgr: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Score [0-1]: fraction of pixels inside the circle that are visually blue.

    Combines blue-hue proportion with mean saturation so that washed-out or
    non-blue regions score low even when they are circular.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
    pixels = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(pixels) == 0:
        return 0.0
    hue, sat = pixels[:, 0].astype(np.float32), pixels[:, 1].astype(np.float32)
    in_blue = ((hue >= 85) & (hue <= 145)).mean()
    mean_sat = sat.mean() / 255.0
    return float(in_blue * mean_sat)


def _contour_candidate(
    contour,
    image_bgr: np.ndarray,
    min_circ: float,
    min_area: float,
    method: str,
) -> dict | None:
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    perim = cv2.arcLength(contour, True)
    if perim == 0:
        return None
    circ = 4 * math.pi * area / (perim ** 2)
    if circ < min_circ:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    score = _blue_score(image_bgr, cx, cy, r) * circ
    return {
        "found": True,
        "center_px": (float(cx), float(cy)),
        "diameter_px": float(2 * r),
        "area_px": float(area),
        "circularity": float(circ),
        "bbox": tuple(int(v) for v in cv2.boundingRect(contour)),
        "method": method,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Method 1 — HSV multi-range
# ---------------------------------------------------------------------------

def _detect_hsv(image_bgr: np.ndarray, cfg: dict) -> dict | None:
    c_cfg = cfg["circle"]
    min_circ = float(c_cfg.get("min_circularity", 0.65))
    min_area = float(c_cfg.get("min_area_px", 50))

    close_k = np.ones((7, 7), np.uint8)
    open_k  = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Primary range from config + fallback bands
    primary = (
        np.array(c_cfg["hsv_lower"], dtype=np.uint8),
        np.array(c_cfg["hsv_upper"], dtype=np.uint8),
    )
    bands = [primary] + [
        (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        for lo, hi in _HSV_BANDS[1:]
    ]

    best: dict | None = None

    for lower, upper in bands:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            cand = _contour_candidate(c, image_bgr, min_circ, min_area, "hsv")
            if cand and (best is None or cand["score"] > best["score"]):
                best = cand

    return best


# ---------------------------------------------------------------------------
# Method 2 — Hough Circle Transform on the fused blueness map
# ---------------------------------------------------------------------------

def _detect_hough(image_bgr: np.ndarray, cfg: dict) -> dict | None:
    c_cfg = cfg["circle"]
    min_area = float(c_cfg.get("min_area_px", 50))
    h, w = image_bgr.shape[:2]

    # Continuous blueness intensity — Hough handles shape, color score
    # downstream filters quality; using the fused map instead of a single
    # fixed-range mask lets this method see the same wide color coverage
    # as _detect_blueness.
    bmap = _blueness_map(image_bgr)
    blurred = cv2.GaussianBlur(bmap, (9, 9), 2)

    min_r = max(4, int(math.sqrt(min_area / math.pi)))
    max_r = min(w, h) // 3

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, min(w, h) // 5),
        param1=60,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    best: dict | None = None
    min_score = 0.20  # must be reasonably blue to count

    for cx, cy, r in circles[0]:
        score = _blue_score(image_bgr, cx, cy, r)
        if score < min_score:
            continue
        # Compute circularity proxy from Hough (it finds round objects by design)
        area_px = math.pi * r ** 2
        bx, by = int(cx - r), int(cy - r)
        bw, bh = int(2 * r), int(2 * r)
        cand = {
            "found": True,
            "center_px": (float(cx), float(cy)),
            "diameter_px": float(2 * r),
            "area_px": float(area_px),
            "circularity": 0.95,
            "bbox": (max(0, bx), max(0, by), bw, bh),
            "method": "hough",
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


# ---------------------------------------------------------------------------
# Method 3 — LAB b* channel (negative b* = blue, robust to luminance)
# ---------------------------------------------------------------------------

def _detect_lab(image_bgr: np.ndarray, cfg: dict) -> dict | None:
    c_cfg = cfg["circle"]
    min_circ = float(c_cfg.get("min_circularity", 0.65))
    min_area = float(c_cfg.get("min_area_px", 50))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    b_chan = lab[:, :, 2]  # 0-255; 128=neutral, <128=blue, >128=yellow

    # Adaptive threshold: pixels clearly below neutral
    mean_b = float(b_chan.mean())
    thresh = min(115, mean_b - 15)
    blue_mask = np.where(b_chan.astype(np.float32) < thresh, 255, 0).astype(np.uint8)

    close_k = np.ones((7, 7), np.uint8)
    open_k  = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_k)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  open_k)

    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: dict | None = None
    for c in cnts:
        cand = _contour_candidate(c, image_bgr, min_circ, min_area, "lab")
        if cand and (best is None or cand["score"] > best["score"]):
            best = cand

    return best


# ---------------------------------------------------------------------------
# Method 4 — Dark blob + hue verification (near-black / navy dots)
#
# Some reference dots are so dark (low V) that no HSV band can isolate them
# without either missing most of the dot (V floor too high) or pulling in
# unrelated bright-but-saturated background. Isolating by darkness alone
# first, then verifying blue hue *inside* the isolated blob, sidesteps the
# V-floor problem entirely.
# ---------------------------------------------------------------------------

def _detect_dark_blob(image_bgr: np.ndarray, cfg: dict) -> dict | None:
    c_cfg = cfg["circle"]
    min_circ = float(c_cfg.get("min_circularity", 0.65))
    min_area = float(c_cfg.get("min_area_px", 50))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Background (leaf scene) is predominantly bright; the dot is a small
    # dark fraction of the frame, so a high percentile is a safe background
    # brightness estimate even when the dot itself is sizeable.
    bg_level = float(np.percentile(gray, 90))
    thresh = max(40.0, bg_level * 0.6)
    dark_mask = (gray < thresh).astype(np.uint8) * 255

    close_k = np.ones((9, 9), np.uint8)
    open_k = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_k)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_k)

    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: dict | None = None
    for c in cnts:
        cand = _contour_candidate(c, image_bgr, min_circ, min_area, "dark_blob")
        if cand and (best is None or cand["score"] > best["score"]):
            best = cand

    return best


# ---------------------------------------------------------------------------
# Method 5 — Fused blueness map + adaptive Otsu threshold
#
# Generalizes methods 1/3/4: instead of more hard-coded bands, fuse several
# brightness-decoupled blue indicators into one continuous score and let
# Otsu pick the cut per-image. Adapts to each photo's own exposure/color
# balance instead of relying on thresholds tuned against one dataset.
# ---------------------------------------------------------------------------

def _detect_blueness(image_bgr: np.ndarray, cfg: dict) -> dict | None:
    c_cfg = cfg["circle"]
    min_circ = float(c_cfg.get("min_circularity", 0.65))
    min_area = float(c_cfg.get("min_area_px", 50))

    bmap = _blueness_map(image_bgr)
    _, mask = cv2.threshold(bmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = np.ones((9, 9), np.uint8)
    open_k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: dict | None = None
    for c in cnts:
        cand = _contour_candidate(c, image_bgr, min_circ, min_area, "blueness")
        if cand and (best is None or cand["score"] > best["score"]):
            best = cand

    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_blue_circle(image_bgr: np.ndarray, cfg: dict) -> dict:
    """Detect the blue reference dot using a cascade of five methods.

    Each method returns its best candidate; the one with the highest
    combined blue-score × circularity is returned.

    Keys in the returned dict:
        found, center_px, diameter_px, area_px, circularity, bbox,
        method, score, low_confidence, mm_per_px, mm2_per_px2
    """
    # Downscale very large images for faster detection (> 2 MP)
    h, w = image_bgr.shape[:2]
    scale_down = 1.0
    if max(h, w) > 2000:
        scale_down = 2000 / max(h, w)
        small = cv2.resize(image_bgr, (int(w * scale_down), int(h * scale_down)))
    else:
        small = image_bgr

    # Neutralize unusual white balance / lighting color casts before any
    # color-based thresholding, so detection isn't tied to one dataset's
    # specific color profile.
    normalized = _gray_world_normalize(small)

    candidates: list[dict] = []
    for fn in (_detect_hsv, _detect_hough, _detect_lab, _detect_dark_blob, _detect_blueness):
        result = fn(normalized, cfg)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return {"found": False, "low_confidence": True}

    best = max(candidates, key=lambda c: c["score"])

    min_conf = float(cfg["circle"].get("min_confidence_score", 0.45))
    best["low_confidence"] = best["score"] < min_conf

    # Scale coordinates back to original resolution
    if scale_down != 1.0:
        inv = 1.0 / scale_down
        cx, cy = best["center_px"]
        best["center_px"] = (cx * inv, cy * inv)
        best["diameter_px"] *= inv
        best["area_px"] *= inv * inv
        bx, by, bw, bh = best["bbox"]
        best["bbox"] = (
            int(bx * inv), int(by * inv),
            int(bw * inv), int(bh * inv),
        )

    # Compute scale calibration from known physical diameter
    d_mm = cfg["circle"].get("known_diameter_mm")
    if d_mm:
        best["mm_per_px"] = d_mm / best["diameter_px"]
        known_area_mm2 = math.pi * (d_mm / 2) ** 2
        best["mm2_per_px2"] = known_area_mm2 / best["area_px"]
    else:
        best["mm_per_px"] = None
        best["mm2_per_px2"] = None

    return best


def crop_circle(
    image_bgr: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 5
) -> np.ndarray:
    x, y, bw, bh = bbox
    H, W = image_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + bw + pad)
    y2 = min(H, y + bh + pad)
    return image_bgr[y1:y2, x1:x2].copy()


def compute_circle_scale(
    diameter_px: float, known_diameter_mm: float
) -> tuple[float, float]:
    """Return (mm_per_px, mm2_per_px2)."""
    mm_per_px = known_diameter_mm / diameter_px
    known_area_mm2 = math.pi * (known_diameter_mm / 2) ** 2
    circle_area_px = math.pi * (diameter_px / 2) ** 2
    mm2_per_px2 = known_area_mm2 / circle_area_px
    return mm_per_px, mm2_per_px2
