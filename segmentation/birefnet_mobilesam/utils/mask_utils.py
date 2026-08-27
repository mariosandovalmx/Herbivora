"""Binary mask manipulation utilities."""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask keeping only the largest connected component."""
    mask_uint8 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if n <= 1:
        return mask.copy()
    # label 0 is background — pick the largest foreground component
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(bool)


def component_at_point(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Return the connected component containing (x, y)."""
    H, W = mask.shape[:2]
    px = int(max(0, min(W - 1, x)))
    py = int(max(0, min(H - 1, y)))
    mask_uint8 = mask.astype(np.uint8) * 255
    n, labels, _, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if n <= 1:
        return mask.copy()
    label = int(labels[py, px])
    if label == 0:
        return largest_component(mask)
    return (labels == label).astype(bool)


def largest_component_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Return (cx, cy) of the largest connected component."""
    comp = largest_component(mask)
    ys, xs = np.where(comp)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a boolean mask."""
    return binary_fill_holes(mask).astype(bool)


def remove_region(mask: np.ndarray,
                  bbox: tuple[int, int, int, int],
                  expand: int = 5) -> np.ndarray:
    """Zero-out a bounding-box region (with optional expansion) from the mask."""
    x, y, w, h = bbox
    H, W = mask.shape[:2]
    x1 = max(0, x - expand)
    y1 = max(0, y - expand)
    x2 = min(W, x + w + expand)
    y2 = min(H, y + h + expand)
    out = mask.copy()
    out[y1:y2, x1:x2] = False
    return out


def box_region_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    margin_frac: float = 0.25,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Boolean mask (True inside) for a box expanded by margin_frac of its larger side.

    Used to hard-constrain segmentation to the neighborhood of a detector's
    (e.g. YOLO) bounding box, clipped to image bounds. Returns the mask and
    the expanded (x1, y1, x2, y2) box.
    """
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    margin = int(max(x2 - x1, y2 - y1) * margin_frac)
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(W, x2 + margin)
    ey2 = min(H, y2 + margin)
    mask = np.zeros((H, W), dtype=bool)
    mask[ey1:ey2, ex1:ex2] = True
    return mask, (ex1, ey1, ex2, ey2)


def dilate_mask(mask: np.ndarray, k: int = 15) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float((a & b).sum())
    union = float((a | b).sum())
    return intersection / union if union > 0 else 0.0


def refine_boundary(coarse: np.ndarray, fine_edges: np.ndarray,
                    band: int = 20) -> np.ndarray:
    """Replace the boundary band of coarse mask with fine_edges within that band."""
    eroded = cv2.erode(coarse.astype(np.uint8), np.ones((band, band), np.uint8))
    interior = eroded.astype(bool)
    boundary_band = coarse & ~interior
    # In the boundary band, defer to fine_edges; interior stays from coarse
    return interior | (boundary_band & fine_edges)
