"""
whitebg_masks.py — ROI masks for images already on white background (without FastSAM).

Creates output/<white_bg>/ + output/<masks>/ ready for analyze_leaves.py.

Usage:
    python whitebg_masks.py --input test --output output_segmentation_test --clean-output
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "leaf_contour"))
from artifact_filters import remove_blue_from_mask  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from image_io import VALID_IMAGE_EXTENSIONS as VALID_EXT, load_bgr  # noqa: E402
NESTED_DUP_RE = re.compile(r"_white_bg_leaf_\d+_white_bg$", re.IGNORECASE)
DEFAULT_MAX_LEAF_AREA_RATIO = 0.62


def canonical_leaf_id(stem: str) -> str:
    nested = re.match(r"^(.+)_white_bg_leaf_\d+_white_bg$", stem, re.IGNORECASE)
    if nested:
        return nested.group(1)
    if stem.lower().endswith("_white_bg"):
        return stem[: -len("_white_bg")]
    return stem


def foliage_mask_hsv(bgr: np.ndarray, stressed: bool = False) -> np.ndarray:
    """Pixels with leaf tissue appearance (excludes augmented background gray)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if stressed:
        green = cv2.inRange(hsv, (18, 22, 15), (100, 255, 255))
    else:
        green = cv2.inRange(hsv, (25, 28, 20), (95, 255, 255))
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    chroma = ((sat >= 28) & (val >= 25) & (val <= 250)).astype(np.uint8) * 255
    return cv2.bitwise_or(green, chroma)


def leaf_mask_from_white_bg(
    bgr: np.ndarray,
    white_thresh: int = 250,
    min_area_ratio: float = 0.0005,
    max_leaf_area_ratio: float = DEFAULT_MAX_LEAF_AREA_RATIO,
    require_foliage: bool = True,
    fill_holes: bool = False,
    remove_blue: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Leaf mask: white/gray background under low saturation excluded; only main component.

    fill_holes=False (recommended for analysis): keeps internal herbivory holes in the
    mask so internal_holes_mask can count them. fill_holes=True bridges holes for legacy
    U-Net-only workflows.
    """
    h, w = bgr.shape[:2]
    img_area = h * w
    white = (
        (bgr[:, :, 0] >= white_thresh)
        & (bgr[:, :, 1] >= white_thresh)
        & (bgr[:, :, 2] >= white_thresh)
    )
    candidate = (~white).astype(np.uint8) * 255

    if require_foliage:
        foliage = foliage_mask_hsv(bgr)
        candidate = cv2.bitwise_and(candidate, foliage)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(80, int(min_area_ratio * img_area))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if n_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8), {"reason": "no_component"}

    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros((h, w), dtype=np.uint8)
    out[labels == best_label] = 255

    holes_filled_px = 0
    if fill_holes:
        before = int(cv2.countNonZero(out))
        filled = ndimage.binary_fill_holes(out > 0).astype(np.uint8) * 255
        holes_filled_px = int(cv2.countNonZero(filled)) - before
        out = filled

    area_ratio = cv2.countNonZero(out) / float(img_area)
    meta = {
        "area_ratio": round(area_ratio, 4),
        "area_px": int(cv2.countNonZero(out)),
        "holes_filled_px": holes_filled_px,
    }

    if area_ratio > max_leaf_area_ratio:
        # Second stricter attempt (augmented with gray spots)
        strict = cv2.bitwise_and((~white).astype(np.uint8) * 255, foliage_mask_hsv(bgr, stressed=False))
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, kernel, iterations=2)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(strict, connectivity=8)
        if n2 > 1:
            bl = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            out2 = np.zeros((h, w), dtype=np.uint8)
            out2[lab2 == bl] = 255
            if fill_holes:
                out2 = ndimage.binary_fill_holes(out2 > 0).astype(np.uint8) * 255
            ar2 = cv2.countNonZero(out2) / float(img_area)
            if ar2 < area_ratio and ar2 >= min_area / img_area:
                out = out2
                area_ratio = ar2
                meta["retried_strict"] = True
        meta["area_ratio"] = round(area_ratio, 4)
        meta["area_px"] = int(cv2.countNonZero(out))
        if area_ratio > max_leaf_area_ratio:
            meta["warning"] = "mask_too_large"

    if remove_blue:
        before_px = int(cv2.countNonZero(out))
        out = remove_blue_from_mask(out, bgr)
        removed = before_px - int(cv2.countNonZero(out))
        if removed > 0:
            meta["blue_removed_px"] = removed
            meta["area_px"] = int(cv2.countNonZero(out))
            meta["area_ratio"] = round(cv2.countNonZero(out) / float(img_area), 4)

    return out, meta


def leaf_mask_from_rgb(
    orig_rgb: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    return leaf_mask_from_white_bg(bgr, **kwargs)


def whiten_enclosed_holes(
    composite_bgr: np.ndarray,
    tissue_mask: np.ndarray,
    solid_silhouette: np.ndarray | None = None,
    *,
    dark_thresh: int = 200,
    min_area: int = 6,
) -> np.ndarray:
    """
    Paint white on internal holes and dark non-foliage regions inside the leaf silhouette.

    Handles cases where the tissue mask still marks herbivory holes as leaf pixels.
    """
    out = composite_bgr.copy()
    tissue = tissue_mask.astype(bool)
    if not tissue.any():
        return out

    if solid_silhouette is None:
        solid = ndimage.binary_fill_holes(tissue)
    else:
        solid = solid_silhouette.astype(bool)

    internal = ndimage.binary_fill_holes(tissue) & ~tissue

    foliage = foliage_mask_hsv(composite_bgr) > 0
    gray = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark = (gray < dark_thresh) & ~foliage & solid
    # Dark low-chroma regions (soil/shadow visible through herbivory holes misclassified as tissue)
    dark_low_chroma = solid & (val < 95) & (sat < 55)

    candidates = (internal | dark | dark_low_chroma).astype(np.uint8)
    if min_area > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        filtered = np.zeros_like(candidates)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 1
        candidates = filtered.astype(bool)
    else:
        candidates = candidates.astype(bool)

    out[candidates] = (255, 255, 255)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate masks from white background images.")
    parser.add_argument("--input", type=Path, default=Path("test"))
    parser.add_argument("--output", type=Path, default=Path("output_segmentation_test"))
    parser.add_argument("--white-thresh", type=int, default=250)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-leaf-area-ratio", type=float, default=DEFAULT_MAX_LEAF_AREA_RATIO)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete white_bg/ and masks/ before regenerating.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill internal holes in the mask (legacy). Default: keep perforated mask for analysis.",
    )
    parser.add_argument(
        "--bg-distance", type=float, default=42.0
    )
    parser.add_argument(
        "--remove-blue",
        action="store_true",
        help="Exclude blue scale reference dots and ink from the mask (HSV).",
    )
    parser.add_argument(
        "--output-size", 
        type=int, 
        default=512, 
        help="Pad to square and resize to this size (0 to disable)",
    )
    args = parser.parse_args()

    input_dir = args.input.resolve()
    out_root = args.output.resolve()
    white_bg_dir = out_root / "white_bg"
    masks_dir = out_root / "masks"

    if args.clean_output and out_root.exists():
        shutil.rmtree(white_bg_dir, ignore_errors=True)
        shutil.rmtree(masks_dir, ignore_errors=True)

    white_bg_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    seen_ids: set[str] = set()
    print(f"Input: {input_dir} ({len(images)} files)")
    print(f"Output : {out_root}")

    for img_path in images:
        stem = img_path.stem
        if NESTED_DUP_RE.search(stem):
            print(f"  SKIP nested name: {img_path.name}")
            continue

        leaf_id = canonical_leaf_id(stem)
        if leaf_id in seen_ids:
            print(f"  SKIP duplicate: {img_path.name}")
            continue
        seen_ids.add(leaf_id)

        bgr = load_bgr(img_path)
        if bgr is None:
            print(f"  SKIP cannot read: {img_path.name}")
            continue

        mask, meta = leaf_mask_from_white_bg(
            bgr,
            white_thresh=args.white_thresh,
            min_area_ratio=args.min_area_ratio,
            max_leaf_area_ratio=args.max_leaf_area_ratio,
            fill_holes=args.fill_holes,
            remove_blue=args.remove_blue,
        )

        warn = f"  WARN {meta.get('warning')}" if meta.get("warning") else ""
        filled = meta.get("holes_filled_px", 0)
        filled_note = f"  [+{filled:,} px filled]" if filled > 0 else ""
        bgr_out = bgr.copy()
        bgr_out[mask == 0] = 255

        if args.output_size > 0:
            h, w = bgr_out.shape[:2]
            side = max(h, w)
            padded_bgr = np.ones((side, side, 3), dtype=np.uint8) * 255
            padded_mask = np.zeros((side, side), dtype=np.uint8)
            ph = (side - h) // 2
            pw = (side - w) // 2
            padded_bgr[ph: ph + h, pw: pw + w] = bgr_out
            padded_mask[ph: ph + h, pw: pw + w] = mask
            bgr_out = cv2.resize(padded_bgr, (args.output_size, args.output_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(padded_mask, (args.output_size, args.output_size), interpolation=cv2.INTER_NEAREST)

        out_img_name = f"{leaf_id}_white_bg{img_path.suffix.lower()}"
        cv2.imwrite(str(white_bg_dir / out_img_name), bgr_out)
        cv2.imwrite(str(masks_dir / f"{leaf_id}_mask.png"), mask)
        print(
            f"  OK {img_path.name} -> {leaf_id}_mask.png "
            f"({meta.get('area_px', 0):,} px, {meta.get('area_ratio', 0)*100:.1f}% img){filled_note}{warn}"
        )

    print(f"\nDone ({len(seen_ids)} unique leaves).")


if __name__ == "__main__":
    main()
