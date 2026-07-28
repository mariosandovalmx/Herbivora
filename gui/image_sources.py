"""Image sources for the carousel per tab."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import re

from gui.paths import (
    VALID_EXT,
    _natural_key,
    analyzed_dir,
    leaf_roi_preview_dir,
    list_images,
    masks_dir,
    segmentation_dir,
    white_bg_dir,
    work_dir,
)
from image_io import is_image_path
from gui.widgets.contour_editor import CONTOUR_EDIT_KEY


def _glob_sorted(folder: Path, pattern: str = "*") -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.glob(pattern) if p.is_file() and is_image_path(p)),
        key=_natural_key,
    )


def segment_sources(state, *, include_input: bool = False) -> dict[str, Callable[[], list[Path]]]:
    def white_bg() -> list[Path]:
        out = state.output_path()
        if out is None:
            return []
        flat = list_images(white_bg_dir(out))
        if flat:
            return flat
        seg = segmentation_dir(out)
        if not seg.is_dir():
            return []
        return sorted(
            (p for p in seg.rglob("white_bg/*") if p.is_file() and is_image_path(p)),
            key=_natural_key,
        )

    def masks() -> list[Path]:
        out = state.output_path()
        if out is None:
            return []
        md = masks_dir(out)
        flat = sorted(md.glob("*_mask.png"), key=_natural_key) if md.is_dir() else []
        if flat:
            return flat
        seg = segmentation_dir(out)
        if not seg.is_dir():
            return []
        return sorted(seg.rglob("masks/*_mask.png"), key=_natural_key)

    def input_photos() -> list[Path]:
        inp = state.input_path()
        return list_images(inp) if inp is not None else []

    sources: dict[str, Callable[[], list[Path]]] = {}
    if include_input:
        sources["Input photos"] = input_photos
    sources["Leaves (white_bg)"] = white_bg
    sources["Masks"] = masks
    return sources


def contour_sources(state) -> dict[str, Callable[[], list[Path]]]:
    def overlays() -> list[Path]:
        """Only real Contour outputs (after Run contour), never segmentation white_bg.

        Falling back to white_bg made Tab 3 draw live mask outlines from
        segmentation masks and looked like Contour had already run.
        """
        out = state.output_path()
        if out is None:
            return []
        ov = leaf_roi_preview_dir(out) / "overlays"
        return _glob_sorted(ov, "*_leaf_overlay.jpg") or _glob_sorted(ov)

    def white_bg() -> list[Path]:
        out = state.output_path()
        return list_images(white_bg_dir(out)) if out else []

    return {
        CONTOUR_EDIT_KEY: overlays,
        "Leaf (white_bg)": white_bg,
    }


def analyze_sources(state) -> dict[str, Callable[[], list[Path]]]:
    def analyzed() -> list[Path]:
        out = state.output_path()
        if out is None:
            return []
        ad = analyzed_dir(out)
        jpgs = _glob_sorted(ad, "*_analyzed.jpg")
        return jpgs if jpgs else _glob_sorted(ad)

    return {
        "Analyzed Results": analyzed,
    }
