"""Morphology-aware post-refinement for UNET shape completion masks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contour.inference.gap_detector import classify_morphology
from leaf_contour.contour_bridge import bridge_exterior_gaps

try:
    from leaf_contour.petiole_utils import combined_exclusion_mask as _combined_excl
except ImportError:  # pragma: no cover
    def _combined_excl(mask, **kw):  # type: ignore
        return np.zeros(mask.shape[:2], dtype=np.uint8)

# Morphologies that historically used bridging (kept for callers/tests).
REFINE_MORPHOLOGIES = frozenset({"serrated", "lobed", "smooth"})

# Per-type reconstruction profiles. Tune one row at a time.
# Entire (smooth): U-Net often hallucinates far into white paper; constrain
# additions to the convex hull of the *damaged* silhouette (shape prior) and
# do not expand further with hull_margin_fill of the refined mask.
MORPHOLOGY_PROFILES: dict[str, dict] = {
    "auto": {
        "bridge": False,
        "bridge_max_growth": 0.08,
        "bridge_from_partial": False,
        "hull_margin_fill": False,
        "clip_to_partial_hull": False,
        "partial_hull_dilate_px": 0,
        "gentle_partial": False,
        "clip_color": True,
    },
    "smooth": {
        "bridge": True,
        "bridge_max_growth": 0.12,
        "bridge_from_partial": True,
        "hull_margin_fill": False,
        "clip_to_partial_hull": True,
        "partial_hull_dilate_px": 4,
        "drop_untouched_islands": True,
        "fill_internal_holes": True,
        "gentle_partial": False,
        # Clip exterior paper FPs but keep herbivory hole fills (same as serrated).
        "clip_color": True,
        "clip_preserve_holes": True,
    },
    # Serrated: keep teeth (no hull / no paper-clip). U-Net already fills
    # most internal herbivory holes; clip_color was wiping those fills because
    # hole interiors look like white paper. Fill remaining enclosed holes,
    # drop floating islands, and bridge only deep margin bites (tooth-aware).
    "serrated": {
        "bridge": False,
        "bridge_max_growth": 0.08,
        "bridge_from_partial": False,
        "hull_margin_fill": False,
        "clip_to_partial_hull": False,
        "partial_hull_dilate_px": 0,
        "drop_untouched_islands": True,
        "fill_internal_holes": True,
        "serrated_deep_bite_fill": True,
        "tooth_close_px": 15,
        "bite_min_depth_px": 28.0,
        # Skip chord-fill on complex serrated margins (protect natural sinuses).
        "bite_min_solidity": 0.90,
        "gentle_partial": True,
        # Clip exterior paper FPs, but keep fills inside herbivory holes.
        "clip_color": True,
        "clip_preserve_holes": True,
    },
    # Lobed: NEVER bridge deep sinuses. Recover enclosed herbivory holes, then
    # fill only *shallow* open margin bites on lobe edges (depth-capped so
    # natural lobe sinuses stay open).
    "lobed": {
        "bridge": False,
        "bridge_max_growth": 0.06,
        "bridge_from_partial": False,
        "hull_margin_fill": False,
        "clip_to_partial_hull": False,
        "partial_hull_dilate_px": 0,
        "drop_untouched_islands": True,
        "fill_internal_holes": True,
        "lobed_margin_bite_fill": True,
        # Two-tier shallow bites; dc floor rejects natural lobe valleys.
        "lobed_bite_min_depth_px": 6.0,
        "lobed_bite_max_depth_px": 22.0,
        "lobed_bite_min_area_px": 60,
        "lobed_bite_max_area_frac": 0.020,
        "lobed_bite_strong_min_depth": 15.0,
        "lobed_bite_strong_min_dc": 0.20,
        "lobed_bite_strong_max_chord_frac": 0.22,
        "lobed_bite_strong_max_arc": 1.48,
        "lobed_bite_weak_min_dc": 0.14,
        "lobed_bite_weak_max_area": 1500,
        "lobed_bite_weak_min_r": 0.65,
        "lobed_bite_weak_max_chord_frac": 0.16,
        "gentle_partial": True,
        "clip_color": True,
        "clip_preserve_holes": True,
    },
    "compound": {
        "bridge": False,
        "bridge_max_growth": 0.08,
        "bridge_from_partial": False,
        "hull_margin_fill": False,
        "clip_to_partial_hull": False,
        "partial_hull_dilate_px": 0,
        "gentle_partial": False,
        "clip_color": True,
    },
}


def get_morphology_profile(morphology: str) -> dict:
    """Return the reconstruction profile for a morphology key."""
    return dict(MORPHOLOGY_PROFILES.get(morphology, MORPHOLOGY_PROFILES["smooth"]))


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
    *,
    preserve_internal_holes: bool = False,
) -> np.ndarray:
    """Remove newly added pixels that look like white background.

    When ``preserve_internal_holes`` is True (serrated), paper pixels that
    fall inside enclosed holes of the damaged leaf are kept — those are the
    herbivory fills we want — while exterior paper hallucinations are dropped.
    """
    out = mask.astype(bool)
    partial_bool = partial > 0
    added = out & ~partial_bool
    if not added.any():
        return out

    paper = _paper_like_pixels(bgr, white_thresh=white_thresh)
    remove = added & paper
    if preserve_internal_holes:
        holes = ndimage.binary_fill_holes(partial_bool) & ~partial_bool
        remove = remove & ~holes
    out[remove] = False
    return out


def _clip_to_segmentation_roi(
    mask_bool: np.ndarray,
    seg_mask: np.ndarray,
    *,
    dilate_px: int = 7,
) -> np.ndarray:
    """Restrict contour to the Step-2 leaf segmentation mask (+ small margin)."""
    seg = seg_mask > 0
    if not seg.any():
        return mask_bool
    k = max(3, int(dilate_px))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    roi = cv2.dilate(seg.astype(np.uint8), ker, iterations=1) > 0
    return mask_bool & roi


def _convex_hull_u8(mask_bool: np.ndarray) -> np.ndarray:
    m = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if not cnts:
        return out
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    cv2.fillPoly(out, [hull], 255)
    return out


def _ellipse_envelope(partial_bool: np.ndarray, *, scale: float = 1.05) -> np.ndarray:
    """Fallback ovate envelope from cv2.fitEllipse."""
    m = (partial_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(partial_bool, dtype=bool)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return _convex_hull_u8(partial_bool) > 0
    try:
        (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
    except cv2.error:
        return _convex_hull_u8(partial_bool) > 0
    out = np.zeros_like(m)
    cv2.ellipse(
        out,
        ((float(cx), float(cy)), (max(float(ma) * scale, 1.0), max(float(MA) * scale, 1.0)), float(angle)),
        255,
        thickness=-1,
    )
    return out > 0


def _symmetric_entire_envelope(
    partial_bool: np.ndarray,
    *,
    dilate_px: int = 3,
    width_expand: float = 1.04,
    n_bins: int | None = None,
    clip_to_hull: bool = True,
) -> np.ndarray:
    """PCA midrib + mirrored half-width prior for Entire leaves.

    Per station, half-width = max(left margin, right margin) of remaining
    tissue so a unilateral bite is filled from the intact opposite side.

    When ``clip_to_hull`` is False, the prior may extend beyond the convex
    hull chord of a deep bite (needed to recover a natural curved margin).
    """
    ys, xs = np.where(partial_bool)
    if len(xs) < 80:
        return _ellipse_envelope(partial_bool, scale=1.05) | partial_bool

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    tang = vt[0]
    tang = tang / (np.linalg.norm(tang) + 1e-12)
    normal = np.array([-tang[1], tang[0]], dtype=np.float64)

    t = centered @ tang
    n = centered @ normal
    t_min = float(t.min())
    t_max = float(t.max())
    span = max(t_max - t_min, 1.0)
    nb = int(n_bins) if n_bins is not None else max(40, int(span / 2.0))
    edges = np.linspace(t_min, t_max, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_w = np.full(nb, np.nan, dtype=np.float64)
    for i in range(nb):
        sel = (t >= edges[i]) & (t < edges[i + 1] if i < nb - 1 else t <= edges[i + 1])
        if not np.any(sel):
            continue
        ni = n[sel]
        left = ni[ni <= 0]
        right = ni[ni >= 0]
        left_w = float(np.percentile(-left, 97)) if left.size else 0.0
        right_w = float(np.percentile(right, 97)) if right.size else 0.0
        half_w[i] = max(left_w, right_w) * width_expand

    valid = np.isfinite(half_w) & (half_w > 1.0)
    if not np.any(valid):
        return _ellipse_envelope(partial_bool, scale=1.05) | partial_bool
    filled_hw = half_w.copy()
    filled_hw[~np.isfinite(filled_hw)] = np.nan
    known = np.isfinite(filled_hw)
    filled_hw[~known] = np.interp(centers[~known], centers[known], filled_hw[known])
    pad = np.pad(filled_hw, (2, 2), mode="edge")
    roll_max = np.max(
        np.stack([pad[i : i + nb] for i in range(5)], axis=0),
        axis=0,
    )
    half_w = np.maximum(filled_hw, 0.90 * roll_max)
    med = float(np.median(half_w[np.isfinite(half_w)]))
    half_w = np.maximum(half_w, 0.75 * med)

    if nb >= 5:
        kernel = np.array([0.15, 0.7, 0.15], dtype=np.float64)
        padded = np.pad(half_w, (1, 1), mode="edge")
        half_w = np.convolve(padded, kernel, mode="valid")

    h, w = partial_bool.shape
    yy, xx = np.indices((h, w))
    rel_x = xx.astype(np.float64) - mean[0]
    rel_y = yy.astype(np.float64) - mean[1]
    tt = rel_x * tang[0] + rel_y * tang[1]
    nn = rel_x * normal[0] + rel_y * normal[1]
    hw = np.interp(tt, centers, half_w, left=0.0, right=0.0)
    envelope = (tt >= t_min) & (tt <= t_max) & (np.abs(nn) <= hw)

    if dilate_px > 0:
        kd = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        envelope = cv2.dilate(envelope.astype(np.uint8), ker, iterations=1) > 0

    if not clip_to_hull:
        return envelope | partial_bool

    hull = _convex_hull_u8(partial_bool) > 0
    if dilate_px > 0:
        kd = 2 * int(dilate_px) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        hull = cv2.dilate(hull.astype(np.uint8), ker, iterations=1) > 0
    return (envelope & hull) | partial_bool


def _clip_additions_to_partial_hull(
    mask_bool: np.ndarray,
    partial_bool: np.ndarray,
    *,
    dilate_px: int = 4,
) -> np.ndarray:
    """Clip U-Net FPs; force-fill Entire hull gaps except tip/petiole crescents.

    Lateral margin bites are filled via hull(partial) \\ tissue. Tip and
    petiole hull crescents are dropped (triangular hallucinations). A gap
    component is kept only if it is mostly outside tip/petiole excl *and*
    its centroid is not in the top/bottom ends of the leaf.
    """
    if not partial_bool.any():
        return mask_bool

    hull = _convex_hull_u8(partial_bool) > 0
    filled = ndimage.binary_fill_holes(partial_bool)
    hull_gap = hull & ~filled

    excl = (
        _combined_excl(
            (partial_bool.astype(np.uint8) * 255),
            neck_fraction=0.22,
            apex_arc_fraction=0.045,
        )
        > 0
    )

    ys_p, _ = np.where(partial_bool)
    y0 = float(ys_p.min())
    y1 = float(ys_p.max())
    span = max(y1 - y0, 1.0)
    tip_cut = y0 + 0.12 * span
    base_cut = y1 - 0.18 * span

    hull_gap_keep = np.zeros_like(hull_gap)
    if hull_gap.any():
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            hull_gap.astype(np.uint8), connectivity=8
        )
        for i in range(1, n):
            comp = labels == i
            area_c = int(stats[i, cv2.CC_STAT_AREA])
            if area_c < 1:
                continue
            frac_excl = float((comp & excl).sum()) / float(area_c)
            cy = float(cents[i, 1])
            if frac_excl >= 0.45:
                continue
            if cy <= tip_cut or cy >= base_cut:
                continue
            hull_gap_keep |= comp

    prior = _symmetric_entire_envelope(
        partial_bool,
        dilate_px=max(1, dilate_px),
        width_expand=1.08,
        clip_to_hull=True,
    )
    allowed = (prior | hull) & ~excl
    clipped = partial_bool | (mask_bool & allowed)
    out = clipped | hull_gap_keep

    area = float(partial_bool.sum())
    k = max(5, int(0.035 * float(np.sqrt(area))))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, ker) > 0
    near = cv2.dilate(hull.astype(np.uint8), ker, iterations=1) > 0
    close_add = closed & near & ~excl & ~partial_bool
    out = out | close_add

    # Final strip of tip/petiole excl additions only (end bands already
    # handled by component centroid filter so mid-bite chords are intact).
    added = out & ~partial_bool
    out = partial_bool | (added & ~excl)
    return out


def _drop_untouched_added_islands(
    mask_bool: np.ndarray,
    partial_bool: np.ndarray,
) -> np.ndarray:
    """Remove added connected components that do not touch the partial leaf."""
    added = mask_bool & ~partial_bool
    if not added.any():
        return mask_bool
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        added.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask_bool
    # Dilate partial by 1 px so touching-at-edge counts as contact.
    touch = cv2.dilate(partial_bool.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
    keep_added = np.zeros_like(added)
    for i in range(1, n):
        comp = labels == i
        if np.any(comp & touch):
            keep_added |= comp
    return partial_bool | keep_added

def _hull_exterior_margin_fill(
    mask: np.ndarray,
    *,
    max_area_growth: float = 0.15,
    min_depth_px: float = 8.0,
    min_area_px: int = 80,
) -> np.ndarray:
    """
    Fill open margin bites for entire leaves via convex hull.

    Only keeps exterior gap components deep/large enough to look like real
    herbivory (not thin crescents along a slightly non-convex healthy margin).
    Caps total growth and respects petiole/apex exclusion.
    """
    base = mask.astype(bool)
    if not base.any():
        return base

    hull = _convex_hull_u8(base) > 0
    filled = ndimage.binary_fill_holes(base)
    # Outside filled tissue but inside hull = open margin gaps (not enclosed holes).
    exterior_gap = hull & ~filled
    if not exterior_gap.any():
        return base

    excl = _combined_excl((base.astype(np.uint8) * 255))
    if excl.any():
        exterior_gap = exterior_gap & (excl == 0)

    # Depth from the current outer edge into the hull gap.
    dist = cv2.distanceTransform(exterior_gap.astype(np.uint8), cv2.DIST_L2, 5)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        exterior_gap.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(exterior_gap)
    base_area = int(base.sum())
    budget = int(max_area_growth * base_area)
    used = 0

    # Prefer deep, large bites first.
    scored: list[tuple[float, int, int]] = []
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        depth = float(dist[comp].max()) if comp.any() else 0.0
        if area < min_area_px or depth < min_depth_px:
            continue
        scored.append((depth, area, i))
    scored.sort(reverse=True)

    for depth, area, i in scored:
        if used + area > budget:
            continue
        kept |= labels == i
        used += area

    return base | kept


def _fill_lobed_margin_bites(
    mask: np.ndarray,
    *,
    min_depth_px: float = 6.0,
    max_depth_px: float = 22.0,
    min_area_px: int = 60,
    max_area_frac: float = 0.020,
    max_area_growth: float = 0.06,
    strong_min_depth: float = 15.0,
    strong_min_dc: float = 0.20,
    strong_max_chord_frac: float = 0.22,
    strong_max_arc: float = 1.48,
    weak_min_dc: float = 0.14,
    weak_max_area: int = 1500,
    weak_min_r: float = 0.65,
    weak_max_chord_frac: float = 0.16,
) -> np.ndarray:
    """Fill shallow open margin bites on lobed leaves.

    Hull-gap depth alone confuses herbivory notches with natural shallow
    lobe valleys. Keep components only if they pass a two-tier shape gate:

    - *strong*: deeper notches with high depth/chord (semicircular bites)
    - *weak*: shallower outer-margin notches with compact chord/area

    Deep lobe sinuses stay open (low depth/chord or too inward).
    """
    base = mask.astype(bool)
    if not base.any():
        return base

    hull = _convex_hull_u8(base) > 0
    filled = ndimage.binary_fill_holes(base)
    exterior_gap = hull & ~filled
    if not exterior_gap.any():
        return base

    excl = _combined_excl(
        (base.astype(np.uint8) * 255),
        neck_fraction=0.28,
        apex_arc_fraction=0.05,
    )
    if excl.any():
        exterior_gap = exterior_gap & (excl == 0)

    ys, xs = np.where(base)
    cy, cx = float(ys.mean()), float(xs.mean())
    rmax = float(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max())
    rmax = max(rmax, 1.0)

    # Exterior contour once (arc-length gate for strong bites).
    cnts_ext, _ = cv2.findContours(
        base.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cnt_xy = (
        max(cnts_ext, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        if cnts_ext
        else np.zeros((0, 2), dtype=np.float64)
    )

    dist = cv2.distanceTransform(exterior_gap.astype(np.uint8), cv2.DIST_L2, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        exterior_gap.astype(np.uint8), connectivity=8
    )
    base_area = int(base.sum())
    max_one = max(int(min_area_px), int(max_area_frac * base_area))
    budget = int(max_area_growth * base_area)
    used = 0
    kept = np.zeros_like(exterior_gap)

    def _chord_of(comp: np.ndarray) -> float:
        dil = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        border = dil & base
        by, bx = np.where(border)
        if by.size < 2:
            return 0.0
        pts = np.stack([bx.astype(np.float64), by.astype(np.float64)], axis=1)
        dmat = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        i1, i2 = np.unravel_index(int(dmat.argmax()), dmat.shape)
        return float(np.linalg.norm(pts[i1] - pts[i2]))

    scored: list[tuple[float, int, int]] = []
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        depth = float(dist[comp].max()) if comp.any() else 0.0
        if area < int(min_area_px) or area > max_one:
            continue
        if depth < float(min_depth_px) or depth > float(max_depth_px):
            continue
        chord = _chord_of(comp)
        dc = depth / max(chord, 1.0)
        gy, gx = np.where(comp)
        r_norm = float(np.sqrt((gy.mean() - cy) ** 2 + (gx.mean() - cx) ** 2) / rmax)
        # Contour arc / chord (true bites ~1.1–1.45; shallow lobe valleys higher).
        arc_ratio = 0.0
        if cnt_xy.size and chord > 0:
            dil = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            border = dil & base
            by, bx = np.where(border)
            if by.size >= 2:
                pts = np.stack([bx.astype(np.float64), by.astype(np.float64)], axis=1)
                dmat = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
                i1, i2 = np.unravel_index(int(dmat.argmax()), dmat.shape)
                p1, p2 = pts[i1], pts[i2]

                def _ni(p: np.ndarray) -> int:
                    return int(
                        np.argmin((cnt_xy[:, 0] - p[0]) ** 2 + (cnt_xy[:, 1] - p[1]) ** 2)
                    )

                s_i, e_i = _ni(p1), _ni(p2)
                nct = len(cnt_xy)
                short = min((e_i - s_i) % nct, (s_i - e_i) % nct)
                arc_ratio = float(short) / max(chord, 1.0)

        tier = ""
        if (
            depth > float(strong_min_depth)
            and dc >= float(strong_min_dc)
            and chord <= float(strong_max_chord_frac) * rmax
            and arc_ratio <= float(strong_max_arc)
        ):
            tier = "strong"
        elif (
            depth <= float(strong_min_depth)
            and dc >= float(weak_min_dc)
            and area <= int(weak_max_area)
            and r_norm >= float(weak_min_r)
            and chord <= float(weak_max_chord_frac) * rmax
        ):
            tier = "weak"

        if not tier:
            continue
        scored.append((depth, area, i))
    scored.sort(reverse=True)

    for depth, area, i in scored:
        if used + area > budget:
            continue
        kept |= labels == i
        used += area

    return base | kept



def _contour_defect_depths(mask_bool: np.ndarray) -> np.ndarray:
    """Convexity-defect depths (px) of the largest exterior contour."""
    m_u8 = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros(0, dtype=np.float64)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 4:
        return np.zeros(0, dtype=np.float64)
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return np.zeros(0, dtype=np.float64)
    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return np.zeros(0, dtype=np.float64)
    defects = np.asarray(defects, dtype=np.int32).reshape(-1, 4)
    return np.asarray([float(d) / 256.0 for *_, d in defects], dtype=np.float64)


def _mask_solidity(mask_bool: np.ndarray) -> float:
    m_u8 = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull_a = float(cv2.contourArea(cv2.convexHull(cnt)))
    return area / max(hull_a, 1.0)


def _bridge_serrated_margin_bites(
    mask_bool: np.ndarray,
    *,
    tooth_close_px: int = 15,
    min_depth_px: float = 18.0,
    max_area_growth: float = 0.12,
    min_solidity: float = 0.90,
) -> np.ndarray:
    """Fill deep margin herbivory bites without erasing serration valleys.

    On complex (low-solidity) serrated leaves, deep natural sinuses look like
    bites after a small morph-close — those leaves skip this step. On more
    ovate leaves, close kernel and depth threshold scale with typical tooth
    depth so only outliers deeper than teeth are filled.
    """
    base = mask_bool.astype(bool)
    if not base.any():
        return base

    sol = _mask_solidity(base)
    tooth_depths = _contour_defect_depths(base)
    tooth_p75 = float(np.percentile(tooth_depths, 75)) if tooth_depths.size >= 3 else float(min_depth_px)

    # Complex serrated / lobed-like margins: do not chord-fill natural sinuses.
    if sol < float(min_solidity):
        return base

    # Close at least ~tooth depth so valleys between peaks disappear before
    # defect detection; cap so true large bites are not closed away.
    area = float(base.sum())
    close_px = max(int(tooth_close_px), int(np.ceil(tooth_p75 * 1.25)))
    close_px = min(close_px, max(int(tooth_close_px), int(0.055 * np.sqrt(area))))
    adaptive_min = max(float(min_depth_px), 1.75 * tooth_p75)

    k = 2 * max(1, int(close_px)) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(base.astype(np.uint8) * 255, cv2.MORPH_CLOSE, ker) > 0
    m_u8 = closed.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return base
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 4:
        return base

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return base
    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return base
    defects = np.asarray(defects, dtype=np.int32).reshape(-1, 4)

    excl = _combined_excl(
        (base.astype(np.uint8) * 255),
        neck_fraction=0.28,
        apex_arc_fraction=0.05,
    ) > 0
    ys = np.where(base)[0]
    y0, y1 = float(ys.min()), float(ys.max())
    span = max(y1 - y0, 1.0)
    base_cut = y1 - 0.18 * span
    tip_cut = y0 + 0.10 * span

    base_area = int(base.sum())
    budget = int(max_area_growth * base_area)
    used = 0
    out = base.copy()

    scored: list[tuple[float, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for s, e, f, d in defects:
        depth = float(d) / 256.0
        if depth < adaptive_min:
            continue
        start = (int(cnt[s][0][0]), int(cnt[s][0][1]))
        end = (int(cnt[e][0][0]), int(cnt[e][0][1]))
        far = (int(cnt[f][0][0]), int(cnt[f][0][1]))
        h, w = base.shape[:2]
        if not (0 <= far[0] < w and 0 <= far[1] < h):
            continue
        if far[1] >= base_cut or far[1] <= tip_cut:
            continue
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        if 0 <= mid[1] < h and 0 <= mid[0] < w and base[mid[1], mid[0]]:
            continue
        chord = float(np.hypot(start[0] - end[0], start[1] - end[1]))
        # Wide herbivory notches can have depth/chord slightly under 0.25;
        # keep a floor so shallow natural waists stay rejected (complex leaves
        # already skip this step via min_solidity).
        if chord < 10.0 or depth / chord < 0.20:
            continue
        scored.append((depth, start, end, far))
    scored.sort(key=lambda t: t[0], reverse=True)

    hull_orig = _convex_hull_u8(base) > 0
    max_one = max(80, int(0.030 * base_area))

    for _depth, start, end, far in scored:
        layer = np.zeros(base.shape, dtype=np.uint8)
        cv2.fillConvexPoly(layer, np.array([start, end, far], dtype=np.int32), 255)
        add = (layer > 0) & ~out & hull_orig
        if excl.any():
            add &= ~excl
        area = int(add.sum())
        if area < 50 or area > max_one or used + area > budget:
            continue
        out |= add
        used += area

    return out


def refine_unet_mask(
    mask: np.ndarray,
    partial: np.ndarray,
    bgr: np.ndarray | None = None,
    *,
    morphology: str | None = None,
    bridge_max_growth: float | None = None,
    clip_color: bool | None = None,
    white_thresh: int = 240,
    seg_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """
    Refine UNET output according to the morphology profile.

    Entire (smooth): bridge U-Net + damaged partial, then exterior hull fill
    for remaining open margin bites (no paper-clip). Serrated/lobed: bridging
    with paper-clip. Compound: U-Net only. Never applies approxPolyDP smoothing.

    Returns:
        (refined_mask_u8, morphology_used)
    """
    partial_u8 = (partial > 0).astype(np.uint8) * 255
    morph = morphology or classify_morphology(partial_u8)
    profile = get_morphology_profile(morph)

    do_bridge = bool(profile["bridge"])
    growth = (
        float(bridge_max_growth)
        if bridge_max_growth is not None
        else float(profile["bridge_max_growth"])
    )
    do_clip = profile["clip_color"] if clip_color is None else bool(clip_color)
    bridge_from_partial = bool(profile.get("bridge_from_partial", False))
    hull_margin_fill = bool(profile.get("hull_margin_fill", False))
    clip_to_partial_hull = bool(profile.get("clip_to_partial_hull", False))
    drop_untouched = bool(profile.get("drop_untouched_islands", False))
    hull_dilate = int(profile.get("partial_hull_dilate_px", 0))
    fill_holes = bool(profile.get("fill_internal_holes", False))
    serrated_bites = bool(profile.get("serrated_deep_bite_fill", False))
    lobed_bites = bool(profile.get("lobed_margin_bite_fill", False))
    # Morph-specific growth (do not let the global YAML override wipe the profile).
    if serrated_bites or lobed_bites:
        growth = float(profile["bridge_max_growth"])

    base = (mask > 0).astype(bool)
    if (
        not do_bridge
        and not hull_margin_fill
        and not clip_to_partial_hull
        and not drop_untouched
        and not fill_holes
        and not serrated_bites
        and not lobed_bites
    ):
        return base.astype(np.uint8) * 255, morph

    refined = base.copy()
    partial_bool = partial_u8 > 0

    if do_bridge:
        bridged, _bridge_px = bridge_exterior_gaps(refined, max_area_growth=growth)
        refined = refined | bridged
        if bridge_from_partial:
            bridged_p, _ = bridge_exterior_gaps(partial_bool, max_area_growth=growth)
            refined = refined | bridged_p

    if hull_margin_fill:
        refined = _hull_exterior_margin_fill(
            refined,
            max_area_growth=growth,
            min_depth_px=float(profile.get("hull_min_depth_px", 20.0)),
            min_area_px=int(profile.get("hull_min_area_px", 400)),
        )

    if clip_to_partial_hull:
        refined = _clip_additions_to_partial_hull(
            refined, partial_bool, dilate_px=hull_dilate
        )

    if drop_untouched:
        refined = _drop_untouched_added_islands(refined, partial_bool)

    if bgr is not None and do_clip:
        refined = _clip_added_paper(
            refined.astype(np.uint8) * 255,
            partial_u8,
            bgr,
            white_thresh=white_thresh,
            preserve_internal_holes=bool(profile.get("clip_preserve_holes", False)),
        )

    # Fill holes after paper-clip so enclosed herbivory voids stay closed even
    # when clip_color is on (serrated uses preserve_holes + this pass).
    if fill_holes:
        refined = ndimage.binary_fill_holes(refined)

    # Margin bites after clip/fill so paper-clip cannot erase them.
    if serrated_bites:
        refined = _bridge_serrated_margin_bites(
            refined,
            tooth_close_px=int(profile.get("tooth_close_px", 15)),
            min_depth_px=float(profile.get("bite_min_depth_px", 18.0)),
            max_area_growth=growth,
            min_solidity=float(profile.get("bite_min_solidity", 0.90)),
        )
        # Triangles can enclose residual pockets — close them without a second clip.
        refined = ndimage.binary_fill_holes(refined)

    if lobed_bites:
        refined = _fill_lobed_margin_bites(
            refined,
            min_depth_px=float(profile.get("lobed_bite_min_depth_px", 6.0)),
            max_depth_px=float(profile.get("lobed_bite_max_depth_px", 22.0)),
            min_area_px=int(profile.get("lobed_bite_min_area_px", 60)),
            max_area_frac=float(profile.get("lobed_bite_max_area_frac", 0.020)),
            max_area_growth=growth,
            strong_min_depth=float(profile.get("lobed_bite_strong_min_depth", 15.0)),
            strong_min_dc=float(profile.get("lobed_bite_strong_min_dc", 0.20)),
            strong_max_chord_frac=float(profile.get("lobed_bite_strong_max_chord_frac", 0.22)),
            strong_max_arc=float(profile.get("lobed_bite_strong_max_arc", 1.48)),
            weak_min_dc=float(profile.get("lobed_bite_weak_min_dc", 0.14)),
            weak_max_area=int(profile.get("lobed_bite_weak_max_area", 1500)),
            weak_min_r=float(profile.get("lobed_bite_weak_min_r", 0.65)),
            weak_max_chord_frac=float(profile.get("lobed_bite_weak_max_chord_frac", 0.16)),
        )

    refined = refined.astype(bool) | (partial_u8 > 0)

    if seg_mask is not None:
        refined = _clip_to_segmentation_roi(refined, seg_mask)

    return refined.astype(np.uint8) * 255, morph
