"""Connected-component leaf instances from a BiRefNet saliency mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LeafInstance:
    """One leaf candidate from a connected component."""

    leaf_index: int  # 1-based after sorting
    label: int
    area_px: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (inclusive-exclusive style for SAM)
    centroid: tuple[int, int]  # (cx, cy)
    mask: np.ndarray  # bool (H, W)


def extract_leaf_instances(
    mask: np.ndarray,
    *,
    min_area_px: int = 5000,
    max_leaves: int | None = None,
) -> list[LeafInstance]:
    """Split a binary mask into leaf instances via 8-connected components.

    Filters by ``min_area_px``, then sorts left→right, top→bottom by centroid
    so numbering is stable for photos with leaves laid out in a row/column.

    If ``max_leaves`` is set, keeps the largest N components by area *before*
    spatial sorting (so small noise is dropped first when capping count).
    """
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return []

    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int, int, int], np.ndarray]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        cx = int(round(float(centroids[lab, 0])))
        cy = int(round(float(centroids[lab, 1])))
        comp = labels == lab
        candidates.append((lab, area, (cx, cy), (x, y, x + w, y + h), comp))

    if not candidates:
        return []

    if max_leaves is not None and max_leaves > 0 and len(candidates) > max_leaves:
        candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = candidates[:max_leaves]

    # Stable spatial order: left→right, then top→bottom
    candidates.sort(key=lambda c: (c[2][0], c[2][1]))

    out: list[LeafInstance] = []
    for i, (lab, area, centroid, bbox, comp) in enumerate(candidates, start=1):
        out.append(
            LeafInstance(
                leaf_index=i,
                label=lab,
                area_px=area,
                bbox=bbox,
                centroid=centroid,
                mask=comp,
            )
        )
    return out
