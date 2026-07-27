"""Background normalization for herbarium images (cream/gray paper → uniform white).

Used by whitebg_masks.py and predict_leaf_roi.py for a uniform white background.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_paper_color_bgr(bgr: np.ndarray, border_frac: float = 0.08) -> np.ndarray:
    """Mean paper color sampled from an image border (BGR float)."""
    h, w = bgr.shape[:2]
    bw = max(4, int(min(h, w) * border_frac))
    strips = [
        bgr[:bw, :].reshape(-1, 3),
        bgr[-bw:, :].reshape(-1, 3),
        bgr[:, :bw].reshape(-1, 3),
        bgr[:, -bw:].reshape(-1, 3),
    ]
    border_px = np.vstack(strips).astype(np.float32)
    return np.median(border_px, axis=0)


def normalize_white_background_bgr(
    bgr: np.ndarray,
    paper_bgr: np.ndarray | None = None,
    distance_thresh: float = 42.0,
    bright_thresh: int = 200,
    sat_thresh: int = 55,
    white_level: int = 255,
    feather: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """
    Empuja fondo hacia blanco puro preservando tejido foliar (verde/saturado).

    Reglas (en HSV):
      - Pixels con S baja y V alta -> fondo candidato
      - Pixels cercanos al color de papel estimado en BGR -> fondo candidato
      - No modificar pixels con S alta (hoja)
    """
    out = bgr.copy()
    h, w = bgr.shape[:2]
    if paper_bgr is None:
        paper_bgr = estimate_paper_color_bgr(bgr)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    paper = paper_bgr.reshape(1, 1, 3).astype(np.float32)
    diff = np.linalg.norm(bgr.astype(np.float32) - paper, axis=2)
    near_paper = diff < distance_thresh

    low_sat_bright = (sat < sat_thresh) & (val > bright_thresh)
    bg_candidate = near_paper | low_sat_bright

    # Proteger verdes (tejido)
    green_mask = cv2.inRange(hsv, (25, 28, 20), (95, 255, 255))
    bg_candidate &= green_mask == 0

    # Suavizar transicion en borde fondo/hoja.
    # Usa boxFilter (O(n), independiente del kernel) en lugar de GaussianBlur (O(n*k)).
    bg_u8 = bg_candidate.astype(np.uint8) * 255
    if feather > 0:
        k = max(3, int(min(h, w) * feather) | 1)
        bg_u8 = cv2.boxFilter(bg_u8, -1, (k, k), normalize=True)
    alpha = bg_u8.astype(np.float32) / 255.0

    # Blend vectorizado (sin loop sobre canales)
    alpha3 = alpha[:, :, np.newaxis]
    blended = out.astype(np.float32) * (1.0 - alpha3) + white_level * alpha3
    out = np.clip(blended, 0, 255).astype(np.uint8)

    meta = {
        "paper_bgr": [int(x) for x in paper_bgr],
        "bg_fraction": round(float(bg_candidate.mean()), 4),
    }
    return out, meta
