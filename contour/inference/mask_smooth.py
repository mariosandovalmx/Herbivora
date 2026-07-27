"""Suavizado de contorno de mascaras (sin dependencias circulares)."""

from __future__ import annotations

import cv2
import numpy as np


def smooth_mask_contour(mask: np.ndarray, epsilon_factor: float = 0.0018) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 50:
        return mask.astype(bool)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    if peri < 20:
        return mask.astype(bool)
    eps = max(1.5, float(epsilon_factor) * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return mask.astype(bool)
    out = np.zeros_like(m)
    cv2.fillPoly(out, [approx], 255)
    return out > 127
