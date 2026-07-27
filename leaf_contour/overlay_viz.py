"""Leaf-mask overlay visualization (no PyTorch dependency)."""

from __future__ import annotations

import cv2
import numpy as np

# Green transparency: lower = leaf tissue more visible under the mask.
GREEN_OVERLAY_ALPHA = 0.22
GREEN_BGR = np.array([0, 170, 0], dtype=np.float64)
# Partial/raw hint: soft orange in BGR (was incorrectly RGB-ordered → strong blue).
PARTIAL_BGR = np.array([0, 140, 255], dtype=np.float64)
PARTIAL_ALPHA = 0.28
CONTOUR_BGR = np.array([0, 90, 0], dtype=np.float64)
CONTOUR_LINE_THICKNESS = 1
CONTOUR_ALPHA = 0.40


def overlay_leaf(
    bgr: np.ndarray,
    leaf_bool: np.ndarray,
    raw_bool: np.ndarray | None = None,
    *,
    green_alpha: float = GREEN_OVERLAY_ALPHA,
) -> np.ndarray:
    vis = bgr.copy()
    # Masks must be HxW; HxWx1 (or HxWx3) boolean indexing crashes on BGR images.
    leaf_bool = np.asarray(leaf_bool)
    if leaf_bool.ndim > 2:
        leaf_bool = np.squeeze(leaf_bool)
    if leaf_bool.ndim > 2:
        leaf_bool = leaf_bool[..., 0]
    leaf_bool = leaf_bool.astype(bool, copy=False)

    if raw_bool is not None:
        raw_bool = np.asarray(raw_bool)
        if raw_bool.ndim > 2:
            raw_bool = np.squeeze(raw_bool)
        if raw_bool.ndim > 2:
            raw_bool = raw_bool[..., 0]
        raw_bool = raw_bool.astype(bool, copy=False)

    if raw_bool is not None and np.any(raw_bool):
        pa = float(np.clip(PARTIAL_ALPHA, 0.05, 0.95))
        vis[raw_bool] = (
            (1.0 - pa) * vis[raw_bool].astype(np.float64) + pa * PARTIAL_BGR
        ).astype(np.uint8)
    if np.any(leaf_bool):
        a = float(np.clip(green_alpha, 0.05, 0.95))
        vis[leaf_bool] = (
            (1.0 - a) * vis[leaf_bool].astype(np.float64) + a * GREEN_BGR
        ).astype(np.uint8)
        cnts, _ = cv2.findContours(
            (leaf_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            contour_mask = np.zeros(leaf_bool.shape, dtype=np.uint8)
            cv2.drawContours(
                contour_mask, cnts, -1, 255, CONTOUR_LINE_THICKNESS
            )
            contour_pixels = contour_mask > 0
            if np.any(contour_pixels):
                ca = float(np.clip(CONTOUR_ALPHA, 0.05, 0.95))
                vis[contour_pixels] = (
                    (1.0 - ca) * vis[contour_pixels].astype(np.float64) + ca * CONTOUR_BGR
                ).astype(np.uint8)
    return vis
