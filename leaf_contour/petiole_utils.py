"""
Shared petiole and apex exclusion utilities for gap detection.

Both structures create false positives in gap detectors:
  - Petiole: a narrow attachment at the bottom creates a deep convexity defect
    that mimics a large herbivory bite.
  - Apex: the tapered leaf tip produces high curvature values similar to a bite.

All functions are stateless and operate on binary masks (uint8 or bool).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Petiole detection
# ---------------------------------------------------------------------------

def detect_petiole_zone(
    leaf_mask: np.ndarray,
    neck_fraction: float = 0.30,
    search_bottom_fraction: float = 0.50,
) -> np.ndarray:
    """
    Detect the petiole attachment zone and return an exclusion mask.

    The petiole neck is the first row in the bottom portion of the leaf
    whose horizontal width drops below neck_fraction × max_leaf_width.
    Everything from that row to the bottom edge is marked as petiole zone.

    The search is restricted to the bottom half (search_bottom_fraction) so
    the leaf apex (also narrow) is never misidentified as a petiole.

    Parameters
    ----------
    leaf_mask              : uint8 or bool (H, W), non-zero = leaf
    neck_fraction          : width threshold relative to max leaf width (default 0.30)
    search_bottom_fraction : fraction of image height to search from the bottom (default 0.50)

    Returns
    -------
    exclusion_mask : uint8 (H, W), 255 where petiole zone, 0 elsewhere.
                     All-zero if no petiole is detected.
    """
    h, w = leaf_mask.shape[:2]
    excl = np.zeros((h, w), dtype=np.uint8)

    if not np.any(leaf_mask > 0):
        return excl

    row_widths = (leaf_mask > 0).sum(axis=1).astype(float)
    max_w = row_widths.max()
    if max_w < 1:
        return excl

    search_start = int(h * (1.0 - search_bottom_fraction))
    bottom_widths = row_widths[search_start:]
    narrow = np.where(bottom_widths < neck_fraction * max_w)[0]

    if len(narrow) == 0:
        return excl

    petiole_row = search_start + int(narrow.min())
    excl[petiole_row:, :] = 255
    return excl


def filter_damage_by_petiole(
    damage_mask: np.ndarray,
    leaf_mask: np.ndarray,
    neck_fraction: float = 0.30,
) -> np.ndarray:
    """
    Zero-out any damage detection that falls within the petiole zone.

    Convenience wrapper around detect_petiole_zone.
    """
    if not np.any(damage_mask > 0):
        return damage_mask
    excl = detect_petiole_zone(leaf_mask, neck_fraction=neck_fraction)
    result = damage_mask.copy()
    result[excl > 0] = 0
    return result


# ---------------------------------------------------------------------------
# Apex detection
# ---------------------------------------------------------------------------

def detect_apex_zone(
    leaf_mask: np.ndarray,
    arc_fraction: float = 0.06,
    neck_fraction: float = 0.30,
) -> np.ndarray:
    """
    Detect the leaf apex (tip) zone and return an exclusion mask.

    The apex is the contour point most distant from the petiole midpoint.
    An arc of arc_fraction × contour_length is excluded around that point
    to suppress the high-curvature false positive at the tapered tip.

    Parameters
    ----------
    leaf_mask     : uint8 or bool (H, W), non-zero = leaf
    arc_fraction  : fraction of total contour length to exclude around apex (default 0.06)
    neck_fraction : passed to detect_petiole_zone to locate petiole reference row

    Returns
    -------
    exclusion_mask : uint8 (H, W), 255 where apex zone, 0 elsewhere.
                     All-zero if contour cannot be found.
    """
    h, w = leaf_mask.shape[:2]
    excl = np.zeros((h, w), dtype=np.uint8)

    mask_u8 = (leaf_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return excl

    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.squeeze(axis=1)  # (N, 2)
    if len(pts) < 10:
        return excl

    # Reference point: midpoint of the petiole row (bottom of the leaf)
    petiole_excl = detect_petiole_zone(leaf_mask, neck_fraction=neck_fraction)
    petiole_rows = np.where(petiole_excl[:, 0] > 0)[0]
    if len(petiole_rows) > 0:
        ref_row = int(petiole_rows.min())
        # Find the leaf pixel at the top of the petiole zone
        leaf_cols = np.where(leaf_mask[ref_row, :] > 0)[0]
        if len(leaf_cols) > 0:
            ref_pt = np.array([float(leaf_cols.mean()), float(ref_row)])
        else:
            ref_pt = np.array([w / 2.0, float(ref_row)])
    else:
        # No petiole found — use the bottom-centre of the bounding box as reference
        ys, xs = np.where(leaf_mask > 0)
        ref_pt = np.array([float(xs.mean()), float(ys.max())])

    # Find the contour point farthest from the reference (= apex)
    dists = np.linalg.norm(pts.astype(float) - ref_pt, axis=1)
    apex_idx = int(np.argmax(dists))

    # Exclude an arc of arc_fraction around the apex index
    n = len(pts)
    half_arc = max(1, int(round(n * arc_fraction / 2.0)))

    indices = np.arange(apex_idx - half_arc, apex_idx + half_arc + 1) % n
    apex_pts = pts[indices].reshape(-1, 1, 2).astype(np.int32)

    # Draw the arc region as a small dilated polygon
    temp = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(temp, [apex_pts], isClosed=False, color=255, thickness=1)
    radius = max(5, int(round(n * arc_fraction * 0.5)))
    radius = min(radius, 40)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    excl = cv2.dilate(temp, k, iterations=1)
    return excl


def detect_exclusion_zones(
    leaf_mask: np.ndarray,
    neck_fraction: float = 0.30,
    apex_arc_fraction: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute both petiole and apex exclusion masks in one call.

    Returns
    -------
    petiole_excl : uint8 (H, W) — petiole zone (255 = exclude)
    apex_excl    : uint8 (H, W) — apex zone    (255 = exclude)
    """
    petiole_excl = detect_petiole_zone(leaf_mask, neck_fraction=neck_fraction)
    apex_excl = detect_apex_zone(
        leaf_mask,
        arc_fraction=apex_arc_fraction,
        neck_fraction=neck_fraction,
    )
    return petiole_excl, apex_excl


def combined_exclusion_mask(
    leaf_mask: np.ndarray,
    neck_fraction: float = 0.30,
    apex_arc_fraction: float = 0.06,
) -> np.ndarray:
    """
    Return a single mask combining petiole + apex exclusion zones.

    Pixels at 255 should be ignored by all gap detectors.
    """
    p, a = detect_exclusion_zones(leaf_mask, neck_fraction, apex_arc_fraction)
    return cv2.bitwise_or(p, a)
