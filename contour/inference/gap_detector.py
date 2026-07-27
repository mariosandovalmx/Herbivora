"""
Unified leaf-edge gap detector.

Replaces the three independent gap-detection approaches previously scattered
across preprocessing.py, auto_mask.py, and analyze_leaves.py with a single
class that combines five evidence signals and adapts to leaf morphology.

Usage
-----
    from contour.inference.gap_detector import GapDetector

    detector = GapDetector(leaf_mask, rgb_image=white_bg_rgb)
    result   = detector.detect(threshold=0.55)
    # result.damage_mask  — uint8 (H,W), 255 = confirmed gap
    # result.gap_scores   — list of GapCandidate with per-signal scores
    # result.morphology   — "smooth" | "serrated" | "lobed" | "compound"
    # result.debug_overlay — BGR overlay (requires rgb_image)

Signal weights (each 0-1, aggregated as weighted average):
  1. coarse_curvature  (sigma=20) — large-amplitude concavity at macro scale
  2. defect_depth      — convexity defect depth in pixels (>=15 = real bite)
  3. aspect_ratio      — gap width/depth ratio (< 3 = narrow bite)
  4. whiteness         — mean pixel lightness inside gap region (>220 = background)
  5. aperiodicity      — FFT autocorrelation: isolated gap vs. periodic serrations
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Make sure petiole_utils is importable from this package
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from petiole_utils import combined_exclusion_mask, detect_petiole_zone


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GapCandidate:
    """A single candidate gap along the contour, with per-signal scores."""
    start_idx: int
    end_idx: int
    contour_pts: np.ndarray
    defect_depth_px: float = 0.0
    width_px: float = 0.0
    coarse_curvature_score: float = 0.0
    defect_depth_score: float = 0.0
    aspect_ratio_score: float = 0.0
    whiteness_score: float = 0.0
    aperiodicity_score: float = 0.0
    composite_score: float = 0.0
    confirmed: bool = False


@dataclass
class DetectionResult:
    """Output of GapDetector.detect()."""
    damage_mask: np.ndarray
    gap_candidates: List[GapCandidate] = field(default_factory=list)
    morphology: str = "unknown"
    severity: float = 0.0
    is_damaged: bool = False
    debug_overlay: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "coarse_curvature": 0.25,
    "defect_depth":     0.30,
    "aspect_ratio":     0.15,
    "whiteness":        0.20,
    "aperiodicity":     0.10,
}

_DEFAULT_THRESHOLD = 0.55
_N_CONTOUR_PTS = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _resample_contour(pts: np.ndarray, n: int = _N_CONTOUR_PTS) -> np.ndarray:
    """Resample to n equidistant points by arc length."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts[:, 0, :]
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = arc[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1)).astype(np.float32)
    new_arc = np.linspace(0, total, n, endpoint=False)
    x = np.interp(new_arc, arc, pts[:, 0])
    y = np.interp(new_arc, arc, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _curvature_at_sigma(pts: np.ndarray, sigma: float) -> np.ndarray:
    n = len(pts)
    idx_prev = (np.arange(n) - 1) % n
    idx_next = (np.arange(n) + 1) % n
    x = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
    y = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
    dx  = (x[idx_next] - x[idx_prev]) / 2.0
    dy  = (y[idx_next] - y[idx_prev]) / 2.0
    ddx = x[idx_next] - 2 * x + x[idx_prev]
    ddy = y[idx_next] - 2 * y + y[idx_prev]
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return (dx * ddy - dy * ddx) / denom


def _adaptive_k_threshold(k_fine: np.ndarray) -> float:
    positive = k_fine[k_fine > 0]
    if len(positive) < 10:
        return -0.04
    mu  = float(positive.mean())
    std = float(positive.std())
    return float(-(mu + 2.5 * std))


def _aperiodicity_score(damaged_flags: np.ndarray) -> float:
    if not np.any(damaged_flags):
        return 1.0
    changes = np.diff(damaged_flags.astype(int))
    n_runs = int((changes == 1).sum())
    if damaged_flags[0]:
        n_runs += 1
    if n_runs <= 2:
        return 1.0
    if n_runs >= 8:
        return 0.0
    return float(np.clip(1.0 - (n_runs - 2) / 6.0, 0.0, 1.0))


def _whiteness_in_region(
    rgb: Optional[np.ndarray],
    poly_pts: np.ndarray,
    h: int,
    w: int,
) -> float:
    if rgb is None or poly_pts is None or len(poly_pts) < 2:
        return 0.5
    region = np.zeros((h, w), dtype=np.uint8)
    pts_int = poly_pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(region, [pts_int], 255)
    if region.sum() == 0:
        return 0.5
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    sampled = gray[region > 0].astype(float)
    return float(sampled.mean()) / 255.0


# ---------------------------------------------------------------------------
# Morphology classifier
# ---------------------------------------------------------------------------

def classify_morphology(
    leaf_mask: np.ndarray,
    contour: Optional[np.ndarray] = None,
) -> str:
    """
    Rule-based leaf morphology classification using geometric features.

    Returns one of: "smooth", "serrated", "lobed", "compound"
    """
    mask_u8 = (leaf_mask > 0).astype(np.uint8) * 255

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels > 2:
        total_area = float(stats[1:, cv2.CC_STAT_AREA].sum())
        largest = float(stats[1:, cv2.CC_STAT_AREA].max())
        if largest / max(total_area, 1) < 0.80:
            return "compound"

    if contour is None:
        contour = _largest_contour(mask_u8)
        if contour is None:
            return "smooth"

    area = float(cv2.contourArea(contour))
    if area < 100:
        return "smooth"

    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    perim      = float(cv2.arcLength(contour, closed=True))
    hull_perim = float(cv2.arcLength(hull, closed=True))

    solidity = area / max(hull_area, 1.0)
    perim_ratio = perim / max(hull_perim, 1.0)

    if solidity < 0.70 or perim_ratio > 1.45:
        return "lobed"
    if perim_ratio > 1.25:
        return "serrated"
    if solidity < 0.85 and perim_ratio > 1.15:
        return "lobed"
    return "smooth"


# ---------------------------------------------------------------------------
# GapDetector (identical logic, updated import path)
# ---------------------------------------------------------------------------

class GapDetector:
    """
    Unified leaf-edge gap detector combining curvature, convexity defects,
    aspect-ratio, color whiteness, and aperiodicity signals.
    """

    def __init__(
        self,
        leaf_mask: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        morphology: Optional[str] = None,
    ):
        self._mask_u8 = (leaf_mask > 0).astype(np.uint8) * 255
        self._rgb = rgb_image
        self._h, self._w = self._mask_u8.shape

        self._excl = combined_exclusion_mask(self._mask_u8)

        cnt = _largest_contour(self._mask_u8)
        self._contour = cnt
        self.morphology = morphology or classify_morphology(self._mask_u8, cnt)

        if cnt is not None and len(cnt) >= 5:
            self._pts = _resample_contour(cnt.reshape(-1, 2))
            self._k_fine   = _curvature_at_sigma(self._pts, sigma=2.0)
            self._k_medium = _curvature_at_sigma(self._pts, sigma=8.0)
            self._k_coarse = _curvature_at_sigma(self._pts, sigma=20.0)
            self._k_thresh = _adaptive_k_threshold(self._k_fine)
        else:
            self._pts = None
            self._k_fine = self._k_medium = self._k_coarse = None
            self._k_thresh = -0.04

        self._defects, self._defect_cnt = self._compute_defects()

    def detect(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        min_depth_px: float = 10.0,
        min_segment_pts: int = 6,
        merge_gap_pts: int = 4,
        draw_debug: bool = False,
    ) -> DetectionResult:
        if self._pts is None or self._k_coarse is None:
            return DetectionResult(
                damage_mask=np.zeros((self._h, self._w), dtype=np.uint8),
                morphology=self.morphology,
            )

        if self.morphology == "serrated":
            min_depth_px = max(min_depth_px, 18.0)
            min_segment_pts = max(min_segment_pts, 7)
        elif self.morphology == "lobed":
            min_depth_px = max(min_depth_px, 22.0)
            min_segment_pts = max(min_segment_pts, 10)

        candidates = self._find_coarse_curvature_candidates(min_segment_pts, merge_gap_pts)

        aperiodicity = _aperiodicity_score(self._k_coarse < self._k_thresh)
        for cand in candidates:
            cand.coarse_curvature_score = self._score_coarse_curvature(cand)
            cand.defect_depth_score     = self._score_defect_depth(cand, min_depth_px)
            cand.aspect_ratio_score     = self._score_aspect_ratio(cand)
            cand.whiteness_score        = self._score_whiteness(cand)
            cand.aperiodicity_score     = aperiodicity
            cand.composite_score        = self._aggregate(cand)
            cand.confirmed              = (
                cand.composite_score >= threshold
                and not self._in_exclusion_zone(cand)
            )

        damage_mask = self._build_damage_mask(candidates)
        severity = self._compute_severity(damage_mask)

        overlay = None
        if draw_debug and self._rgb is not None:
            overlay = self._draw_debug(candidates)

        return DetectionResult(
            damage_mask=damage_mask,
            gap_candidates=candidates,
            morphology=self.morphology,
            severity=severity,
            is_damaged=bool(damage_mask.any()),
            debug_overlay=overlay,
        )

    def _find_coarse_curvature_candidates(
        self, min_segment_pts: int, merge_gap_pts: int = 4
    ) -> List[GapCandidate]:
        flagged = self._k_coarse < self._k_thresh
        n = len(flagged)
        raw_runs: List[tuple] = []
        in_gap = False
        seg_start = 0
        for i in range(n + 1):
            idx = i % n
            if flagged[idx] and not in_gap:
                in_gap = True
                seg_start = idx
            elif not flagged[idx] and in_gap:
                in_gap = False
                raw_runs.append((seg_start, (i - 1) % n))
        if in_gap:
            raw_runs.append((seg_start, n - 1))

        merged: List[tuple] = []
        for run in raw_runs:
            if not merged:
                merged.append(run)
                continue
            prev_end = merged[-1][1]
            gap = (run[0] - prev_end - 1) % n
            if gap <= merge_gap_pts:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)

        candidates: List[GapCandidate] = []
        for start, end in merged:
            if end >= start:
                indices = np.arange(start, end + 1)
            else:
                indices = np.concatenate([np.arange(start, n), np.arange(0, end + 1)])
            if len(indices) < min_segment_pts:
                continue
            cand = GapCandidate(
                start_idx=int(start),
                end_idx=int(end),
                contour_pts=self._pts[indices],
            )
            cand.width_px = float(np.linalg.norm(
                self._pts[int(start)] - self._pts[int(end)]
            ))
            cand.defect_depth_px = float(-self._k_coarse[indices].min() * 50.0)
            candidates.append(cand)

        return candidates

    def _score_coarse_curvature(self, cand: GapCandidate) -> float:
        n = len(self._k_coarse)
        if cand.end_idx >= cand.start_idx:
            indices = np.arange(cand.start_idx, min(cand.end_idx + 1, n))
        else:
            indices = np.concatenate([np.arange(cand.start_idx, n), np.arange(0, cand.end_idx + 1)])
        if len(indices) == 0:
            return 0.0
        k_min = float(self._k_coarse[indices].min())
        score = float(np.clip(
            (self._k_thresh - k_min) / (2.0 * abs(self._k_thresh) + 1e-9),
            0.0, 1.0,
        ))
        return score

    def _score_defect_depth(self, cand: GapCandidate, min_depth_px: float) -> float:
        best_depth = self._best_defect_for_candidate(cand)
        if best_depth < min_depth_px:
            return 0.0
        return float(np.clip((best_depth - min_depth_px) / (2.0 * min_depth_px + 1e-9), 0.0, 1.0))

    def _score_aspect_ratio(self, cand: GapCandidate) -> float:
        depth = max(cand.defect_depth_px, 1.0)
        ratio = cand.width_px / depth
        return float(np.clip(1.0 - (ratio - 1.5) / 5.0, 0.0, 1.0))

    def _score_whiteness(self, cand: GapCandidate) -> float:
        if self._rgb is None:
            return 0.5
        start_pt = self._pts[cand.start_idx]
        end_pt   = self._pts[cand.end_idx]
        poly = np.vstack([cand.contour_pts, end_pt, start_pt])
        raw = _whiteness_in_region(self._rgb, poly, self._h, self._w)
        return float(np.clip((raw - 0.80) / 0.20, 0.0, 1.0))

    def _aggregate(self, cand: GapCandidate) -> float:
        return (
            _WEIGHTS["coarse_curvature"] * cand.coarse_curvature_score
            + _WEIGHTS["defect_depth"]   * cand.defect_depth_score
            + _WEIGHTS["aspect_ratio"]   * cand.aspect_ratio_score
            + _WEIGHTS["whiteness"]      * cand.whiteness_score
            + _WEIGHTS["aperiodicity"]   * cand.aperiodicity_score
        )

    def _build_damage_mask(self, candidates: List[GapCandidate]) -> np.ndarray:
        damage = np.zeros((self._h, self._w), dtype=np.uint8)
        hull_mask = self._hull_mask()
        missing = cv2.bitwise_and(hull_mask, cv2.bitwise_not(self._mask_u8))

        for cand in candidates:
            if not cand.confirmed:
                continue
            best_depth, best_tri = self._best_defect_triangle(cand)
            if best_tri is not None:
                tri = np.array(best_tri, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(damage, [tri], 255)
            if best_tri is None:
                start_pt = self._pts[cand.start_idx].astype(np.int32)
                end_pt   = self._pts[cand.end_idx].astype(np.int32)
                arc_pts  = cand.contour_pts.astype(np.int32)
                poly = np.vstack([arc_pts, end_pt, start_pt]).reshape(-1, 1, 2)
                cv2.fillPoly(damage, [poly], 255)

        if damage.any():
            damage = cv2.bitwise_and(damage, missing)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            damage = cv2.dilate(damage, k, iterations=1)
            damage = cv2.bitwise_and(damage, missing)
            damage[self._excl > 0] = 0

        return damage

    def _best_defect_triangle(self, cand: GapCandidate) -> tuple:
        if self._defects is None or self._defect_cnt is None:
            return 0.0, None
        pts = cand.contour_pts
        if len(pts) == 0:
            return 0.0, None

        MAX_DIST = 60.0
        best_depth = 0.0
        best_tri = None

        for defect in self._defects:
            s, e, f, d = defect[0]
            depth = float(d) / 256.0
            sx = float(self._defect_cnt[s][0][0]); sy = float(self._defect_cnt[s][0][1])
            ex = float(self._defect_cnt[e][0][0]); ey = float(self._defect_cnt[e][0][1])
            fx = float(self._defect_cnt[f][0][0]); fy = float(self._defect_cnt[f][0][1])
            key_pts = np.array([[sx, sy], [ex, ey], [fx, fy]])
            matched = False
            for kp in key_pts:
                if np.linalg.norm(pts - kp, axis=1).min() <= MAX_DIST:
                    matched = True
                    break
            if matched and depth > best_depth:
                best_depth = depth
                best_tri = [(int(sx), int(sy)), (int(ex), int(ey)), (int(fx), int(fy))]

        return best_depth, best_tri

    def _compute_defects(self) -> tuple:
        if self._contour is None or len(self._contour) < 5:
            return None, None
        n_pixels = int((self._mask_u8 > 0).sum())
        k_size = max(5, min(19, int(n_pixels ** 0.5) // 20))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        smoothed = cv2.morphologyEx(self._mask_u8, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 4:
            return None, None
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return None, None
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            return None, None
        return defects, cnt

    def _best_defect_for_candidate(self, cand: GapCandidate) -> float:
        if self._defects is None or self._defect_cnt is None:
            return 0.0
        pts = cand.contour_pts
        if len(pts) == 0:
            return 0.0

        MAX_DIST = 60.0
        best = 0.0
        for defect in self._defects:
            s, e, f, d = defect[0]
            depth = float(d) / 256.0
            key_pts = np.array([
                [float(self._defect_cnt[s][0][0]), float(self._defect_cnt[s][0][1])],
                [float(self._defect_cnt[e][0][0]), float(self._defect_cnt[e][0][1])],
                [float(self._defect_cnt[f][0][0]), float(self._defect_cnt[f][0][1])],
            ])
            for kp in key_pts:
                dists = np.linalg.norm(pts - kp, axis=1)
                if dists.min() <= MAX_DIST:
                    best = max(best, depth)
                    break
        return best

    def _in_exclusion_zone(self, cand: GapCandidate) -> bool:
        if not cand.contour_pts.size:
            return False
        cx, cy = cand.contour_pts.mean(axis=0)
        cy_i, cx_i = int(np.clip(cy, 0, self._h - 1)), int(np.clip(cx, 0, self._w - 1))
        return bool(self._excl[cy_i, cx_i] > 0)

    def _hull_mask(self) -> np.ndarray:
        hull_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        if self._contour is not None:
            hull = cv2.convexHull(self._contour)
            cv2.fillPoly(hull_mask, [hull], 255)
        return hull_mask

    def _compute_severity(self, damage_mask: np.ndarray) -> float:
        if self._contour is None:
            return 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(self._contour)))
        mask_area = float((self._mask_u8 > 0).sum())
        if hull_area < 1:
            return 0.0
        return float(np.clip((hull_area - mask_area) / hull_area, 0.0, 1.0))

    def _draw_debug(self, candidates: List[GapCandidate]) -> np.ndarray:
        if self._rgb is None:
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)
        overlay = cv2.cvtColor(self._rgb, cv2.COLOR_RGB2BGR).copy()
        for cand in candidates:
            color = (0, 200, 0) if cand.confirmed else (0, 165, 255)
            pts = cand.contour_pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=2)
        excl_vis = np.zeros_like(overlay)
        excl_vis[self._excl > 0] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 1.0, excl_vis, 0.25, 0)
        return overlay


# ---------------------------------------------------------------------------
# Convenience wrapper (backwards-compatible)
# ---------------------------------------------------------------------------

def detect_gaps(
    binary_mask: np.ndarray,
    rgb_image: Optional[np.ndarray] = None,
    threshold: float = _DEFAULT_THRESHOLD,
    min_depth_px: float = 10.0,
    min_segment_pts: int = 6,
) -> dict:
    """Drop-in replacement for the old detect_damage() API."""
    det = GapDetector(binary_mask, rgb_image=rgb_image)
    result = det.detect(
        threshold=threshold,
        min_depth_px=min_depth_px,
        min_segment_pts=min_segment_pts,
    )

    gap_indices = [
        np.arange(c.start_idx, c.end_idx + 1) % _N_CONTOUR_PTS
        for c in result.gap_candidates
        if c.confirmed
    ]

    n_pts = _N_CONTOUR_PTS
    intact_indices = np.ones(n_pts, dtype=bool)
    for idx_arr in gap_indices:
        intact_indices[idx_arr] = False

    pts = det._pts if det._pts is not None else np.zeros((n_pts, 2), dtype=np.float32)
    k   = det._k_coarse if det._k_coarse is not None else np.zeros(n_pts)

    return {
        "is_damaged":           result.is_damaged,
        "damage_mask":          result.damage_mask,
        "n_damage_regions":     len(gap_indices),
        "severity":             result.severity,
        "contour_gap_indices":  gap_indices,
        "curvature_profile":    k,
        "contour_all_pts":      pts,
        "intact_indices":       intact_indices,
        "morphology":           result.morphology,
        "gap_candidates":       result.gap_candidates,
    }
