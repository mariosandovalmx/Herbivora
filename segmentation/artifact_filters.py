"""
Shared artifact filters for leaf masks (blue scale dot, ink markings).

Used by whitebg_masks.py (white_bg crops).
"""

from __future__ import annotations

import cv2
import numpy as np

BLUE_HSV_LOW = np.array([90, 20, 20], dtype=np.uint8)
BLUE_HSV_HIGH = np.array([135, 255, 255], dtype=np.uint8)
BLUE_COMPONENT_RATIO = 0.20


def blue_mask_hsv(hsv: np.ndarray) -> np.ndarray:
    """Binary mask of blue / blue-violet (markers, scale reference dot). OpenCV H: 0-179."""
    return cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)


def remove_blue_from_mask(mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Remove blue reference dots and ink from a binary leaf mask.

    Drops connected components that are mostly blue, then subtracts all blue pixels.
    """
    if cv2.countNonZero(mask) == 0:
        return mask

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue = blue_mask_hsv(hsv)
    out = mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.uint8) * 255
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area == 0:
            continue
        blue_overlap = cv2.countNonZero(cv2.bitwise_and(comp_mask, blue))
        if blue_overlap / float(area) > BLUE_COMPONENT_RATIO:
            out[labels == label_id] = 0

    return cv2.bitwise_and(out, cv2.bitwise_not(blue))
