"""I/O helpers: letterbox resize, composite, metadata JSON."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def resize_letterbox(
    image: np.ndarray,
    target: int = 512,
    pad_fraction: float = 0.03,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image to target×target preserving aspect ratio with white padding.

    Adds an extra `pad_fraction` margin (relative to the shorter side) before
    placing on the canvas, so the leaf never touches the output border.

    Returns
    -------
    canvas      : uint8 array of shape (target, target, C)
    scale       : uniform scale factor (same in x and y)
    offset      : (left_pad_px, top_pad_px) on the canvas
    """
    h, w = image.shape[:2]
    # Inner region leaving pad_fraction border on each side
    inner = int(target * (1.0 - 2 * pad_fraction))
    inner = max(inner, target - 4)  # safety floor
    scale = inner / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Build white canvas
    if image.ndim == 3:
        canvas = np.full((target, target, image.shape[2]),
                         fill_value=0, dtype=np.uint8)
        canvas[:] = (bg_color[2], bg_color[1], bg_color[0])  # BGR
    else:
        canvas = np.full((target, target), fill_value=bg_color[0], dtype=np.uint8)

    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas, scale, (pad_left, pad_top)


def crop_to_bbox(image: np.ndarray, mask: np.ndarray,
                 pad_frac: float = 0.05) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to the bounding box of mask with proportional padding.

    pad_frac: fraction of the bbox's larger dimension added on each side.
    Returns the cropped image and (x1, y1, x2, y2) in the original image coords.
    """
    ys, xs = np.where(mask)
    H, W = image.shape[:2]
    if len(xs) == 0:
        return image, (0, 0, W, H)

    bx1, bx2 = int(xs.min()), int(xs.max())
    by1, by2 = int(ys.min()), int(ys.max())
    bw = bx2 - bx1 + 1
    bh = by2 - by1 + 1
    pad = max(4, int(max(bw, bh) * pad_frac))

    x1 = max(0, bx1 - pad)
    y1 = max(0, by1 - pad)
    x2 = min(W, bx2 + pad + 1)
    y2 = min(H, by2 + pad + 1)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def composite_on_white(image_bgr: np.ndarray, mask: np.ndarray,
                       bg_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Place the masked region of image on a white canvas.

    Pixels outside the (perforated) tissue mask receive the background color;
    internal herbivory holes (mask=False) are therefore white.
    """
    bg = np.full_like(image_bgr, (bg_color[2], bg_color[1], bg_color[0]))  # RGB→BGR
    out = bg.copy()
    out[mask] = image_bgr[mask]
    return out


def save_metadata(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
