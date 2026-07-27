"""Morphology-aware post-refinement for UNET shape completion masks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contour.inference.gap_detector import classify_morphology
from leaf_contour.contour_bridge import bridge_exterior_gaps

REFINE_MORPHOLOGIES = frozenset({"serrated", "lobed"})


def _paper_like_pixels(
    bgr: np.ndarray,
    white_thresh: int = 240,
) -> np.ndarray:
    """Pixels that look like white background (not tissue)."""
    from contour.inference.bg_normalize import estimate_paper_color_bgr

    paper = estimate_paper_color_bgr(bgr)
    diff = np.linalg.norm(bgr.astype(np.float32) - paper.reshape(1, 1, 3), axis=2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    near_paper = diff < 38.0
    low_sat_bright = (sat < 50) & (val > 195)
    white_rgb = (
        (bgr[:, :, 0] >= white_thresh)
        & (bgr[:, :, 1] >= white_thresh)
        & (bgr[:, :, 2] >= white_thresh)
    )
    return near_paper | low_sat_bright | white_rgb


def _clip_added_paper(
    mask: np.ndarray,
    partial: np.ndarray,
    bgr: np.ndarray,
    white_thresh: int = 240,
) -> np.ndarray:
    """Remove newly added pixels that look like white background."""
    out = mask.astype(bool)
    partial_bool = partial > 0
    added = out & ~partial_bool
    if not added.any():
        return out

    paper = _paper_like_pixels(bgr, white_thresh=white_thresh)
    out[added & paper] = False
    return out


def refine_unet_mask(
    mask: np.ndarray,
    partial: np.ndarray,
    bgr: np.ndarray | None = None,
    *,
    morphology: str | None = None,
    bridge_max_growth: float = 0.08,
    clip_color: bool = True,
    white_thresh: int = 240,
) -> tuple[np.ndarray, str]:
    """
    Refine UNET output for serrated/lobed leaves only.

    Smooth and elliptic leaves pass through unchanged. Never applies contour
    smoothing (approxPolyDP) on serrated shapes.

    Returns:
        (refined_mask_u8, morphology_used)
    """
    partial_u8 = (partial > 0).astype(np.uint8) * 255
    morph = morphology or classify_morphology(partial_u8)

    if morph not in REFINE_MORPHOLOGIES:
        return (mask > 0).astype(np.uint8) * 255, morph

    base = (mask > 0).astype(bool)
    refined, _bridge = bridge_exterior_gaps(base, max_area_growth=bridge_max_growth)

    if bgr is not None and clip_color:
        refined = _clip_added_paper(
            refined.astype(np.uint8) * 255,
            partial_u8,
            bgr,
            white_thresh=white_thresh,
        )

    partial_bool = partial_u8 > 0
    refined = refined.astype(bool) | partial_bool
    return refined.astype(np.uint8) * 255, morph
