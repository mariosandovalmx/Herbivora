"""Close open notches along the outer leaf margin.

Only bridges perimeter indentations (not square internal holes).
Applies petiole/apex exclusion before building bridges to avoid false positives
in those anatomical regions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

# Importar utilidades compartidas de petiolo/apice
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from petiole_utils import combined_exclusion_mask as _combined_excl
    _PETIOLE_UTILS_AVAILABLE = True
except ImportError:
    _PETIOLE_UTILS_AVAILABLE = False

    def _combined_excl(mask, **kw):  # type: ignore
        return np.zeros(mask.shape[:2], dtype=np.uint8)


def mask_solidity(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8) * 255
    cnt = _largest_contour(m)
    if cnt is None or len(cnt) < 3:
        return 1.0
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    return area / max(1.0, float(cv2.contourArea(hull)))


def _largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bbox_diag(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 1.0
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def _exterior_defect_triangles(cnt: np.ndarray, mask: np.ndarray) -> List[Tuple]:
    if len(cnt) < 4:
        return []
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []
    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return []

    area = max(1, int(mask.sum()))
    min_depth = max(2.5, 0.008 * (area**0.5))
    out: List[Tuple] = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = float(d) / 256.0
        if depth < min_depth:
            continue
        start = (int(cnt[s][0][0]), int(cnt[s][0][1]))
        end = (int(cnt[e][0][0]), int(cnt[e][0][1]))
        far = (int(cnt[f][0][0]), int(cnt[f][0][1]))
        h, w = mask.shape[:2]
        if not (0 <= far[0] < w and 0 <= far[1] < h):
            continue
        if cv2.pointPolygonTest(cnt, far, False) >= 0:
            continue
        chord = float(np.hypot(start[0] - end[0], start[1] - end[1]))
        if chord < 1.0 or depth / chord < 0.05:
            continue
        out.append((start, end, far, depth))
    out.sort(key=lambda x: x[3], reverse=True)
    return out


def _bridge_defects(
    mask: np.ndarray,
    max_area_growth: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rellena muescas exteriores via triangulos de defectos de convexidad."""
    base = mask.astype(bool)
    if not base.any():
        return base, np.zeros_like(base, dtype=bool)

    m_u8 = base.astype(np.uint8) * 255
    cnt = _largest_contour(m_u8)
    if cnt is None:
        return base, np.zeros_like(base, dtype=bool)

    base_area = int(base.sum())
    completed = base.copy()
    bridge = np.zeros_like(base, dtype=bool)

    for start, end, far, _ in _exterior_defect_triangles(cnt, base):
        tri = np.array([start, end, far], dtype=np.int32)
        layer = np.zeros_like(base, dtype=np.uint8)
        cv2.fillConvexPoly(layer, tri, 255)
        add = (layer > 0) & ~completed
        if not add.any():
            continue
        trial = completed | add
        growth = (int(trial.sum()) - base_area) / float(base_area)
        if growth > max_area_growth:
            continue
        completed = trial
        bridge |= add

    return completed, bridge


def _contour_tips(cnt: np.ndarray, step: int = 5, angle_thresh: float = 48.0) -> List[Tuple[int, int]]:
    if len(cnt) < step * 2 + 1:
        return [(int(cnt[i][0][0]), int(cnt[i][0][1])) for i in range(len(cnt))]
    tips: List[Tuple[int, int]] = []
    n = len(cnt)
    for i in range(n):
        p_prev = cnt[(i - step) % n][0].astype(np.float64)
        p = cnt[i][0].astype(np.float64)
        p_next = cnt[(i + step) % n][0].astype(np.float64)
        v1, v2 = p_prev - p, p_next - p
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
        if ang < angle_thresh:
            tips.append((int(p[0]), int(p[1])))
    return tips


def _bridge_tip_pairs(
    mask: np.ndarray,
    max_area_growth: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    base = mask.astype(bool)
    if not base.any():
        return base, np.zeros_like(base, dtype=bool)

    cnt = _largest_contour(base.astype(np.uint8) * 255)
    if cnt is None:
        return base, np.zeros_like(base, dtype=bool)

    tips = _contour_tips(cnt)
    if len(tips) < 2:
        return base, np.zeros_like(base, dtype=bool)

    base_area = int(base.sum())
    completed = base.copy()
    bridge = np.zeros_like(base, dtype=bool)
    diag = _bbox_diag(base)

    candidates: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            p0, p1 = tips[i], tips[j]
            dist = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
            if dist < 6 or dist > 0.9 * diag:
                continue
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            if 0 <= mid[1] < base.shape[0] and 0 <= mid[0] < base.shape[1]:
                if completed[mid[1], mid[0]]:
                    continue
            candidates.append((dist, p0, p1))
    candidates.sort(key=lambda x: x[0])

    for dist, p0, p1 in candidates:
        thick = max(2, int(round(dist * 0.08)) + 2)
        layer = np.zeros(base.shape, dtype=np.uint8)
        cv2.line(layer, p0, p1, 255, thickness=thick, lineType=cv2.LINE_AA)
        k = max(3, thick)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        layer = cv2.dilate(layer, kernel, iterations=1)
        add = (layer > 0) & ~completed
        if not add.any():
            continue
        trial = completed | add
        growth = (int(trial.sum()) - base_area) / float(base_area)
        if growth > max_area_growth:
            continue
        completed = trial
        bridge |= add
        if mask_solidity(completed) >= 0.92:
            break

    return completed, bridge


def bridge_exterior_gaps(
    mask: np.ndarray,
    max_area_growth: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cierra muescas abiertas al borde exterior. No rellena huecos internos cerrados por tejido.

    Aplica exclusion de petiolo y apice: los puentes que caen en esas zonas
    se anulan para evitar cerrar la union del petiolo o la punta del apice.

    Returns:
        completed_mask, bridge_pixels (solo lo anadido)
    """
    base = mask.astype(bool)
    if not base.any():
        return base, np.zeros_like(base, dtype=bool)

    filled_orig = ndimage.binary_fill_holes(base)
    internal = filled_orig & ~base

    m1, b1 = _bridge_defects(base, max_area_growth=max_area_growth)
    m2, b2 = _bridge_tip_pairs(m1, max_area_growth=max_area_growth)
    bridge = (b1 | b2) & ~internal

    # Eliminar puentes en zonas de petiolo/apice
    mask_u8 = base.astype(np.uint8) * 255
    excl = _combined_excl(mask_u8)
    if excl.any():
        bridge[excl > 0] = False

    completed = (m2 | base)
    completed[excl > 0] = base[excl > 0]  # revert exclusion zones to original
    bridge = bridge & ~base
    return completed.astype(bool), bridge.astype(bool)
