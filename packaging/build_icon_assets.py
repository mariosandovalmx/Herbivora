"""Build professional HerbivoR icon assets from the approved source artwork.

Removes only the light margin connected to the image corners (outer white
border). The white fill of the leaf is preserved because it is enclosed by
green / ink and is not flood-reachable from the corners.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent / "assets"
CANDIDATES = [
    ROOT / "herbivor_icon_source.png",
]


def _is_outer_light(rgb: np.ndarray) -> np.ndarray:
    """Cream / off-white outer margin (not green, not dark ink)."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    lum = (r + g + b) / 3.0
    sat = np.stack([r, g, b], axis=-1).std(axis=-1)
    return (lum > 195) & (sat < 22)


def _flood_from_corners(seed_mask: np.ndarray) -> np.ndarray:
    """4-connected flood fill from the four corners through seed_mask."""
    h, w = seed_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if seed_mask[y, x]:
            stack.append((y, x))
            reachable[y, x] = True
        else:
            # If corner itself is not light, search a small neighborhood
            for dy in range(0, min(8, h)):
                for dx in range(0, min(8, w)):
                    for yy, xx in (
                        (dy, dx),
                        (dy, w - 1 - dx),
                        (h - 1 - dy, dx),
                        (h - 1 - dy, w - 1 - dx),
                    ):
                        if seed_mask[yy, xx] and not reachable[yy, xx]:
                            stack.append((yy, xx))
                            reachable[yy, xx] = True

    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and seed_mask[ny, nx] and not reachable[ny, nx]:
                reachable[ny, nx] = True
                stack.append((ny, nx))
    return reachable


def _make_transparent_outside_green(im: Image.Image) -> Image.Image:
    arr = np.array(im.convert("RGBA"))
    rgb = arr[..., :3]
    outer = _is_outer_light(rgb)
    remove = _flood_from_corners(outer)
    arr[remove, 3] = 0

    # Crop to remaining opaque content
    ys, xs = np.where(arr[..., 3] > 0)
    if len(xs) == 0:
        return Image.fromarray(arr)
    pad = 2
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    tile = arr[y0:y1, x0:x1]

    h, w = tile.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = tile
    return Image.fromarray(canvas, "RGBA")


def main() -> None:
    src = next(p for p in CANDIDATES if p.is_file())
    print("source:", src)

    cleaned = _make_transparent_outside_green(Image.open(src))
    print("cleaned:", cleaned.size, "corner alpha=", cleaned.getpixel((0, 0))[3])

    master = cleaned.resize((1024, 1024), Image.Resampling.LANCZOS)
    r, g, b, a = master.split()
    rgb = Image.merge("RGB", (r, g, b))
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    rgb = ImageEnhance.Color(rgb).enhance(1.03)
    master = Image.merge("RGBA", (*rgb.split(), a))

    out_icon = ROOT / "herbivor_icon.png"
    out_256 = ROOT / "herbivor_256.png"
    out_ico = ROOT / "herbivor.ico"
    master.save(out_icon, optimize=True)
    master.resize((256, 256), Image.Resampling.LANCZOS).save(out_256, optimize=True)

    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [master.resize((s, s), Image.Resampling.LANCZOS) for s in sizes]
    frames[-1].save(
        out_ico,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[:-1],
    )

    sample = Image.open(out_256)
    a = np.array(sample)
    print("corners", sample.getpixel((0, 0)), sample.getpixel((255, 255)))
    print("green", sample.getpixel((40, 40)))
    print("leaf white", sample.getpixel((110, 130)))
    cream = (a[..., 3] > 200) & (a[..., :3].mean(2) > 220)
    # cream near true corners only (outer 12px)
    border = np.zeros(a.shape[:2], dtype=bool)
    border[:12, :] = True
    border[-12:, :] = True
    border[:, :12] = True
    border[:, -12:] = True
    print("cream-in-border-strip", int((cream & border).sum()))
    print("wrote", out_icon.name, out_256.name, out_ico.name)


if __name__ == "__main__":
    main()
