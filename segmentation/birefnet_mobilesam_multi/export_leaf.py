"""Export one segmented leaf to white_bg / masks / metadata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def leaf_output_stem(parent_stem: str, leaf_index: int) -> str:
    """Canonical multi-leaf stem: ``{photo}_leaf_{n}`` (1-based)."""
    return f"{parent_stem}_leaf_{leaf_index}"


def export_leaf_outputs(
    *,
    image_bgr: np.ndarray,
    M_final: np.ndarray,
    M_solid: np.ndarray,
    output_dir: Path,
    stem: str,
    cfg: dict,
    meta: dict,
    circle_info: dict | None = None,
    save_debug: bool = True,
    save_debug_overlay_fn=None,
) -> tuple[Path, Path]:
    """Composite, crop, letterbox, and write white_bg + mask + metadata.

    Imports I/O helpers from the single-leaf package without modifying it.
    Returns (white_bg_path, mask_path).
    """
    # Import from sibling single-leaf package (path prepared by caller)
    from utils.io_utils import (  # type: ignore[import-not-found]
        composite_on_white,
        crop_to_bbox,
        resize_letterbox,
        save_metadata,
    )
    from utils.circle_utils import crop_circle  # type: ignore[import-not-found]

    bg = tuple(cfg["background_color"])
    composite = composite_on_white(image_bgr, M_final, bg)
    cropped, crop_bbox = crop_to_bbox(composite, M_solid, pad_frac=0.02)
    output_img, scale_factor, offset = resize_letterbox(
        cropped, cfg["output_size"], pad_fraction=0.03, bg_color=bg
    )

    mask_crop, _ = crop_to_bbox(
        (M_solid.astype(np.uint8) * 255), M_solid, pad_frac=0.02
    )
    mask_lb, _, _ = resize_letterbox(
        mask_crop, cfg["output_size"], pad_fraction=0.03, bg_color=(0, 0, 0)
    )
    _, mask_bin = cv2.threshold(mask_lb, 127, 255, cv2.THRESH_BINARY)

    wb_dir = output_dir / "white_bg"
    masks_out = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    wb_dir.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    out_img = wb_dir / f"{stem}.png"
    out_mask = masks_out / f"{stem}_mask.png"
    cv2.imwrite(str(out_img), output_img)
    cv2.imwrite(str(out_mask), mask_bin)

    if (
        circle_info
        and circle_info.get("found")
        and cfg.get("save_circle_crop")
        and stem.endswith("_leaf_1")
    ):
        # Save circle crop once per parent photo (on first leaf)
        circles_dir = output_dir / "circles"
        circles_dir.mkdir(parents=True, exist_ok=True)
        parent = stem.rsplit("_leaf_", 1)[0]
        circle_crop = crop_circle(image_bgr, circle_info["bbox"])
        cv2.imwrite(str(circles_dir / f"{parent}_circle.png"), circle_crop)

    if save_debug and cfg.get("save_debug_overlays") and save_debug_overlay_fn is not None:
        debug_dir = output_dir / "debug"
        save_debug_overlay_fn(
            image_bgr, M_solid, circle_info or {"found": False},
            debug_dir / f"{stem}_dbg.png",
        )

    meta = dict(meta)
    meta["image_id"] = stem
    meta["scale_factor"] = round(float(scale_factor), 6)
    meta["letterbox_offset"] = list(offset)
    meta["crop_bbox"] = list(crop_bbox)
    meta["output_size"] = [cfg["output_size"], cfg["output_size"]]
    meta["output_path"] = str(out_img)
    save_metadata(meta, meta_dir / f"{stem}.json")
    return out_img, out_mask
