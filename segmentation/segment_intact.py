"""
Segmentation of intact leaves from white-background photographs.

For each image:
  1. Otsu threshold on LAB-L channel  → separates non-white from background
  2. Filter connected components by mean HSV saturation
       high S  → green leaf tissue  (keep)
       low  S  → cast shadow / gray artifact  (discard)
  3. Keep the largest surviving component  → main leaf
  4. Fill internal holes
  5. Morphological closing to smooth the border

Outputs
-------
<out_dir>/masks/    — binary PNG masks  (0 / 255)
<out_dir>/white_bg/ — original image with background set to pure white

Usage
-----
python segment_intact.py --input  D:/Herbivory_software/unet2/leaf_unet_dataset/images_normalized
                         --output D:/Herbivory_software/unet2/leaf_unet_dataset/segmented
                         --preview 25
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from image_io import VALID_IMAGE_EXTENSIONS, load_bgr  # noqa: E402
from segmentation.scale_metadata import write_scale_metadata  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Resize so the longest side equals max_side, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def segment_leaf_white_bg(
    image_bgr: np.ndarray,
    *,
    sat_min: int = 15,
    close_k: int = 7,
    min_area_ratio: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the main leaf from a white-background image, avoiding shadows and blue dots.

    Parameters
    ----------
    image_bgr   : BGR image as read by cv2.imread.
    sat_min     : Pixel-level saturation threshold to help disconnect gray shadows.
    close_k     : Morphological closing kernel size to smooth the contour.
    min_area_ratio : Minimum component area as fraction of image to be kept.

    Returns
    -------
    mask     : (H, W) uint8 binary mask, values {0, 255}.
    white_bg : (H, W, 3) uint8 BGR image with background set to white.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w

    # -- Step 1: non-background mask via Otsu on LAB-L ----------------------
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    _, otsu_mask = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # -- Step 2: Pixel-level shadow and color filtering ---------------------
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # 0-179
    sat = hsv[:, :, 1]  # 0-255
    
    # a channel: green (<128) to magenta (>128)
    a_chan = lab[:, :, 1].astype(np.float32)
    # Greenness score per pixel: positive means greener
    greenness = 128.0 - a_chan

    # Remove gray shadows: require a minimum saturation to be considered part of the leaf/foreground
    # This disconnects the shadow from the leaf before we find connected components.
    color_mask = sat > sat_min

    # Exclude blue pixels explicitly (Hue between 90 and 150) to remove the scale dot
    blue_mask = (hue > 90) & (hue < 150) & (sat > 40)
    
    # Final candidate mask: foreground from Otsu, AND has color, AND is not blue
    candidate_mask = (otsu_mask > 0) & color_mask & (~blue_mask)
    candidate_mask = candidate_mask.astype(np.uint8) * 255

    # If the thresholding removed too much (e.g. very pale leaf), fallback to Otsu without sat filter, but still no blue
    if cv2.countNonZero(candidate_mask) < int(min_area_ratio * img_area):
        candidate_mask = (otsu_mask > 0) & (~blue_mask)
        candidate_mask = candidate_mask.astype(np.uint8) * 255

    # -- Step 3: Score connected components to find the true leaf -----------
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    min_area = int(min_area_ratio * img_area)
    best_label = -1
    best_score = -1e9

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
            
        comp_pixels = labels == i
        
        # Calculate mean greenness
        mean_greenness = float(greenness[comp_pixels].mean())
        
        # Score = Area * Greenness
        # A green leaf will have high positive greenness (e.g. +10 to +40)
        # A yellow shadow will have near zero or negative greenness.
        score = area * mean_greenness
        
        if score > best_score:
            best_score = score
            best_label = i

    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    
    if best_label > 0:
        leaf_mask[labels == best_label] = 255
    elif n_labels > 1:
        # Fallback if everything scored poorly: just take the largest area
        largest = max(range(1, n_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        leaf_mask[labels == largest] = 255

    # -- Step 4: fill internal holes ----------------------------------------
    filled = binary_fill_holes(leaf_mask > 0).astype(np.uint8) * 255

    # -- Step 5: morphological closing to smooth jagged edges ---------------
    # Since we removed shadows, the edge might be jagged. We use closing.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k, iterations=2)

    # -- Step 6: compose white-background image -----------------------------
    white_bg = image_bgr.copy()
    white_bg[mask == 0] = 255

    return mask, white_bg


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    sat_min: int = 20,
    close_k: int = 7,
    preview: int = 0,
    output_size: int = 768,
    scale_source: str = "original_photo",
    exts: tuple[str, ...] = tuple(VALID_IMAGE_EXTENSIONS),
) -> None:
    masks_dir    = output_dir / "masks"
    whitebg_dir  = output_dir / "white_bg"
    metadata_dir = output_dir / "metadata"
    masks_dir.mkdir(parents=True, exist_ok=True)
    whitebg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in exts
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images  |  sat_min={sat_min}  close_k={close_k}")

    failed = []

    for img_path in tqdm(image_files, desc="Segmenting", unit="img"):
        mask_out    = masks_dir    / f"{img_path.stem}_mask.png"
        whitebg_out = whitebg_dir  / f"{img_path.stem}_white_bg.png"
        meta_out    = metadata_dir / f"{img_path.stem}.json"

        # Skip already processed (metadata included, so scale stays recoverable)
        if mask_out.exists() and whitebg_out.exists() and meta_out.exists():
            continue

        img = load_bgr(img_path)
        if img is None:
            failed.append(img_path.name)
            continue

        try:
            mask, white_bg = segment_leaf_white_bg(
                img, sat_min=sat_min, close_k=close_k
            )
        except Exception as exc:
            failed.append(f"{img_path.name} ({exc})")
            continue

        src_h, src_w = white_bg.shape[:2]
        if output_size > 0:
            mask     = _resize_to_max(mask,     output_size)
            white_bg = _resize_to_max(white_bg, output_size)
        cv2.imwrite(str(mask_out),    mask)
        cv2.imwrite(str(whitebg_out), white_bg)
        out_h, out_w = white_bg.shape[:2]
        write_scale_metadata(
            meta_out,
            image_id=img_path.stem,
            source_size=(src_w, src_h),
            output_size=(out_w, out_h),
            scale_factor=out_w / float(src_w),
            scale_source=scale_source,
            output_path=whitebg_out,
        )

    if failed:
        print(f"\nFailed ({len(failed)}): {failed[:10]}{'...' if len(failed)>10 else ''}")

    # -- Optional preview grid -------------------------------------------
    if preview > 0:
        _save_preview(image_files, masks_dir, output_dir, n=preview)

    n_done = len(list(masks_dir.glob("*_mask.png")))
    print(f"\nDone: {n_done}/{len(image_files)} masks saved to {output_dir}")


# ---------------------------------------------------------------------------
# Preview grid
# ---------------------------------------------------------------------------

def _save_preview(
    image_files: list[Path],
    masks_dir: Path,
    output_dir: Path,
    n: int = 25,
) -> None:
    """Save a grid of N random (image | white_bg | mask) triplets for QC."""
    sample = random.sample(image_files, min(n, len(image_files)))
    cols = 3  # original | white_bg | mask
    rows = len(sample)
    cell_h, cell_w = 200, 300

    grid = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)

    for r, img_path in enumerate(sample):
        # Original
        orig = load_bgr(img_path)
        if orig is None:
            continue
        orig_small = cv2.resize(orig, (cell_w, cell_h))
        grid[r*cell_h:(r+1)*cell_h, 0:cell_w] = orig_small

        # White-bg
        wb_path = output_dir / "white_bg" / f"{img_path.stem}_white_bg.png"
        if wb_path.exists():
            wb = cv2.imread(str(wb_path))
            if wb is not None:
                grid[r*cell_h:(r+1)*cell_h, cell_w:2*cell_w] = cv2.resize(wb, (cell_w, cell_h))

        # Mask overlaid on original
        mask_path = masks_dir / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                overlay = orig_small.copy()
                contours, _ = cv2.findContours(
                    cv2.resize(mask, (cell_w, cell_h)),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                grid[r*cell_h:(r+1)*cell_h, 2*cell_w:3*cell_w] = overlay

    # Column headers
    for col, label in enumerate(["Original", "White BG", "Mask contour"]):
        cv2.putText(grid, label, (col*cell_w + 5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)

    preview_path = output_dir / "preview.jpg"
    cv2.imwrite(str(preview_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Preview saved: {preview_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment leaves from white-background images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory with source images.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory (creates masks/ and white_bg/ subdirs).",
    )
    parser.add_argument(
        "--sat-min", type=int, default=20,
        help="Min mean HSV-S for a component to be kept as leaf (not shadow). "
             "Increase (e.g. 35) if shadows are incorrectly included.",
    )
    parser.add_argument(
        "--close-k", type=int, default=7,
        help="Morphological closing kernel size. Larger = smoother contour.",
    )
    parser.add_argument(
        "--preview", type=int, default=25,
        help="Save a grid of N random results for visual QC. 0 = skip.",
    )
    parser.add_argument(
        "--output-size", type=int, default=768,
        help="Resize output so longest side = N px (0 = native resolution).",
    )
    parser.add_argument(
        "--scale-source", choices=("original_photo", "derived_image"),
        default="original_photo",
        help="Whether --input holds the user's photos or images derived from them.",
    )
    args = parser.parse_args()

    process_batch(
        input_dir    = Path(args.input),
        output_dir   = Path(args.output),
        sat_min      = args.sat_min,
        close_k      = args.close_k,
        preview      = args.preview,
        output_size  = args.output_size,
        scale_source = args.scale_source,
    )
