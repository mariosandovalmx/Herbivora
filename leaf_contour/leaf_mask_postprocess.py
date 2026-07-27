"""Refine Leaf-UNet masks: fill internal holes (damage), close bites,
clip to color silhouette (non-uniform background), and remove false islands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_LEAF_DIR = Path(__file__).resolve().parent
_ROOT = _LEAF_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from whitebg_masks import leaf_mask_from_white_bg  # noqa: E402
from bg_normalize import estimate_paper_color_bgr  # noqa: E402

CLOSE_KERNEL_DIVISOR = 12.0


def largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best)


def remove_small_components(mask: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    """Elimina islas de mascara desconectadas (falsos positivos en fondo gris)."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    min_area = max(80, int(min_area_ratio * h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label 0
    valid = np.where(areas >= min_area)[0] + 1  # back to 1-indexed labels
    if valid.size == 0:
        return largest_component(mask)
    return np.isin(labels, valid)


def paper_like_pixels(
    bgr: np.ndarray,
    white_thresh: int = 240,
    sat_thresh: int = 50,
    bright_thresh: int = 195,
    paper_dist: float = 38.0,
) -> np.ndarray:
    """Pixeles que parecen papel/fondo (no tejido)."""
    paper = estimate_paper_color_bgr(bgr)
    diff = np.linalg.norm(bgr.astype(np.float32) - paper.reshape(1, 1, 3), axis=2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    near_paper = diff < paper_dist
    low_sat_bright = (sat < sat_thresh) & (val > bright_thresh)
    white_rgb = (
        (bgr[:, :, 0] >= white_thresh)
        & (bgr[:, :, 1] >= white_thresh)
        & (bgr[:, :, 2] >= white_thresh)
    )
    return near_paper | low_sat_bright | white_rgb


def clip_exterior_paper_halo(
    mask: np.ndarray,
    paper: np.ndarray,
    max_depth_px: int = 12,
) -> np.ndarray:
    """
    Quita pixeles tipo papel solo cerca del borde exterior de la mascara (halos),
    sin abrir huecos internos de daño ya rellenados.
    """
    if not mask.any() or max_depth_px < 1:
        return mask.astype(bool)
    m = mask.astype(bool)
    dist_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    halo = m & paper & (dist_in < max_depth_px)
    return m & ~halo


def color_silhouette_mask(
    bgr: np.ndarray,
    white_thresh: int = 248,
    fill_holes: bool = True,
) -> np.ndarray:
    """Silueta por umbral de fondo (misma logica que pseudo-etiquetas)."""
    mask_u8, _meta = leaf_mask_from_white_bg(
        bgr,
        white_thresh=white_thresh,
        min_area_ratio=0.0003,
        max_leaf_area_ratio=0.92,
        require_foliage=False,
        fill_holes=fill_holes,
    )
    return mask_u8 > 127


def color_guided_clip(
    bgr: np.ndarray,
    mask: np.ndarray,
    dilate_color_px: int = 6,
    white_thresh: int = 248,
    fill_holes: bool = True,
) -> np.ndarray:
    """La mascara U-Net no puede extenderse mas alla de la silueta por color (+margen)."""
    color_m = color_silhouette_mask(bgr, white_thresh=white_thresh, fill_holes=fill_holes)
    if dilate_color_px > 0:
        color_m = morph_close(color_m, dilate_color_px)
    return mask.astype(bool) & color_m


def _maybe_fill_holes(mask: np.ndarray, fill_internal_holes: bool) -> np.ndarray:
    return fill_holes(mask) if fill_internal_holes else mask.astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes using OpenCV contour fill (5-10x faster than scipy)."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def morph_open(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or not mask.any():
        return mask.copy()
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    return out > 127


def morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or not mask.any():
        return mask.copy()
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out > 127


def morph_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or not mask.any():
        return mask.copy()
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1)
    return out > 127


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask.copy()
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return mask.copy()
    hull = cv2.convexHull(cnt)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(out, [hull], 1)
    return out.astype(bool)


def close_radius_for_area(leaf_area: int, divisor: float = CLOSE_KERNEL_DIVISOR) -> int:
    if leaf_area < 50:
        return 0
    return max(8, int(round((leaf_area**0.5) / divisor)))


def refine_leaf_mask(
    pred: np.ndarray,
    mode: str = "complete",
    close_divisor: float = CLOSE_KERNEL_DIVISOR,
    erode_px: int = 1,
    open_px: int = 2,
    bgr: np.ndarray | None = None,
    clip_color: bool = True,
    clip_paper_halo: bool = True,
    min_component_ratio: float = 0.002,
    white_thresh: int = 248,
    paper_halo_depth: int = 10,
    smooth_contour: bool = True,
    smooth_epsilon: float = 0.0018,
    bridge_max_growth: float = 0.6,
    fill_internal_holes: bool | None = None,
    dilate_color_px: int = 6,
) -> np.ndarray:
    """
    mode:
      raw        - solo componente principal
      mask       - + relleno de huecos internos (agujeros de daño)
      closed     - + cierre morfologico (bordes mordidos / lamina faltante)
      hull       - envolvente convexa (maximo relleno de margen)
      complete   - huecos + cierre adaptativo + erode (recomendado en test)
      bridged    - cierra muescas abiertas del borde (sin recorte por color que las borra)
      perforated - como complete pero conserva huecos internos (white_bg con agujeros blancos)
    """
    if mode == "perforated":
        fill_internal_holes = False
        if smooth_contour is True:
            smooth_contour = False
    elif fill_internal_holes is None:
        fill_internal_holes = True

    base = _maybe_fill_holes(largest_component(pred > 0), fill_internal_holes)
    radius = close_radius_for_area(int(base.sum()), close_divisor)
    closed = morph_close(base, radius) if radius > 0 else base

    if mode == "raw":
        out = base
    elif mode == "mask":
        out = base
    elif mode == "closed":
        out = _maybe_fill_holes(closed, fill_internal_holes)
    elif mode == "hull":
        hull_m = convex_hull_mask(base)
        out = _maybe_fill_holes(hull_m | closed | base, fill_internal_holes)
        if open_px > 0:
            out = morph_open(out, open_px)
        if erode_px > 0:
            out = morph_erode(out, erode_px)
    elif mode in ("complete", "perforated"):
        out = _maybe_fill_holes(closed, fill_internal_holes)
        if open_px > 0:
            out = morph_open(out, open_px)
        if erode_px > 0:
            out = morph_erode(out, erode_px)
        out = _maybe_fill_holes(out, fill_internal_holes)
    elif mode == "bridged":
        hull_m = convex_hull_mask(base)
        expanded = _maybe_fill_holes(hull_m | closed | base, fill_internal_holes)
        out, _bridge = bridge_exterior_gaps(expanded, max_area_growth=bridge_max_growth)
        out = _maybe_fill_holes(out, fill_internal_holes)
        if open_px > 0:
            out = morph_open(out, open_px)
        clip_color = False
    else:
        raise ValueError(f"mode refine invalido: {mode!r}")

    out = remove_small_components(out, min_component_ratio)
    out = largest_component(out)

    if bgr is not None:
        if clip_color:
            out = color_guided_clip(
                bgr,
                out,
                dilate_color_px=dilate_color_px,
                white_thresh=white_thresh,
                fill_holes=fill_internal_holes,
            )
            out = _maybe_fill_holes(largest_component(out), fill_internal_holes)
        if clip_paper_halo:
            paper = paper_like_pixels(bgr, white_thresh=white_thresh - 5)
            out = clip_exterior_paper_halo(out, paper, max_depth_px=paper_halo_depth)
            out = _maybe_fill_holes(out, fill_internal_holes)

    if smooth_contour:
        out = smooth_mask_contour(out, epsilon_factor=smooth_epsilon)
        out = _maybe_fill_holes(largest_component(out), fill_internal_holes)

    return out.astype(bool)


from contour_bridge import bridge_exterior_gaps  # noqa: E402
from mask_smooth import smooth_mask_contour  # noqa: E402


def tighten_mask_u8(mask_u8: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Etiquetas de entrenamiento mas ajustadas (menos sobre-segmentacion en bordes)."""
    if erode_px < 1:
        return mask_u8
    m = refine_leaf_mask(mask_u8 > 0, mode="mask", erode_px=0)
    m = morph_erode(m, erode_px)
    return (m.astype(np.uint8) * 255)
