"""Scale calibration math and damage quantification."""

from __future__ import annotations


def apply_scale(area_px: float, scale_mm2_per_px2: float | None) -> float | None:
    if scale_mm2_per_px2 is None:
        return None
    return area_px * scale_mm2_per_px2


def compute_damage(silhouette_area_px: int,
                   remaining_area_px: int) -> tuple[int, float]:
    """Return (damage_px, damage_percent)."""
    damage_px = max(0, silhouette_area_px - remaining_area_px)
    pct = (damage_px / silhouette_area_px * 100.0) if silhouette_area_px > 0 else 0.0
    return damage_px, round(pct, 4)
