"""Generate synthetic herbivory on intact Entire leaves for U-Net fine-tune.

Reads RGB leaves on white background, writes:

  Same folder (visual inspection, original orientation only):
    {stem}_clean_mask.png
    {stem}_art_{kind}_v{nn}.png
  Training pairs (includes geometric copies):
    unet_clean_masks/{stem}.png, {stem}_r90.png, {stem}_r180.png, ...
    unet_damaged_masks/{stem}_v{nn}.png and matching _r90 / _fh / ...

Kinds per leaf:
  nicks       several shallow margin scallops (the small holes the model missed)
  photo       mixed: medium irregular + nicks + internal holes
  small       one medium-small scallop
  large       C-shaped open bite (~15-25%)
  combo       one large C-bite plus several shallow nicks
  holes       internal holes only

Usage:
    python contour/scripts/prepare_entire_finetune.py
    python contour/scripts/prepare_entire_finetune.py --input D:/Herbivory_software/contour_finetune2/Entire_finetune_smallcontourdamage
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contour.inference.predict import extract_partial_mask

VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SKIP_TOKENS = ("_art_", "_clean_mask", "_preview")
DEFAULT_INPUT = Path(r"D:\Herbivory_software\contour_finetune2\Entire_finetune_smallcontourdamage")

GEOM = (
    ("", lambda im: im),
    ("_r90", lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)),
    ("_r180", lambda im: cv2.rotate(im, cv2.ROTATE_180)),
    ("_r270", lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ("_fh", lambda im: cv2.flip(im, 1)),
    ("_fv", lambda im: cv2.flip(im, 0)),
)


def _is_original_rgb(path: Path) -> bool:
    if path.suffix.lower() not in VALID_EXT:
        return False
    name = path.name.lower()
    return not any(tok in name for tok in SKIP_TOKENS)


def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
    m = (mask_bool.astype(np.uint8)) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return mask_bool
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _lateral_contour_points(mask_bool: np.ndarray) -> np.ndarray:
    """Contour samples away from tip/petiole bands (mid 70% of leaf height)."""
    m_u8 = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
    ys = np.where(mask_bool)[0]
    y0, y1 = float(ys.min()), float(ys.max())
    span = max(y1 - y0, 1.0)
    lo = y0 + 0.12 * span
    hi = y1 - 0.18 * span
    xs = np.where(mask_bool)[1]
    mid_x = float(xs.mean())
    width = max(float(xs.max() - xs.min()), 1.0)
    keep = []
    for x, y in cnt:
        if y < lo or y > hi:
            continue
        if abs(float(x) - mid_x) < 0.18 * width:
            continue
        keep.append((int(x), int(y)))
    if len(keep) < 20:
        keep = [(int(x), int(y)) for x, y in cnt if lo <= y <= hi]
    return np.asarray(keep, dtype=np.int32) if keep else cnt.astype(np.int32)


def _outward_unit(px: float, py: float, cx: float, cy: float) -> tuple[float, float]:
    vx, vy = float(px) - cx, float(py) - cy
    nrm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    return vx / nrm, vy / nrm


def _inward_width(mask_bool: np.ndarray, px: int, py: int, ux: float, uy: float) -> int:
    h, w = mask_bool.shape
    for s in range(3, 160):
        x = int(round(px - ux * s))
        y = int(round(py - uy * s))
        if x < 0 or y < 0 or x >= w or y >= h or not mask_bool[y, x]:
            return max(s - 1, 4)
    return 80


def _centroid(mask_bool: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask_bool)
    return float(xs.mean()), float(ys.mean())


def _margin_scallop(
    mask_bool: np.ndarray,
    rng: random.Random,
    px: int,
    py: int,
    ux: float,
    uy: float,
    radius: float,
) -> np.ndarray:
    """Shallow circular/elliptic nick from the margin (photo-like chewing)."""
    h, w = mask_bool.shape
    r = max(3.0, float(radius))
    ox = int(round(px + ux * r * rng.uniform(0.05, 0.38)))
    oy = int(round(py + uy * r * rng.uniform(0.05, 0.38)))
    cut = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(cut, (ox, oy), int(round(r)), 255, thickness=-1)
    if rng.random() < 0.55:
        tx, ty = -uy, ux
        r2 = r * rng.uniform(0.38, 0.75)
        ox2 = int(round(ox + tx * r * rng.uniform(-0.5, 0.5)))
        oy2 = int(round(oy + ty * r * rng.uniform(-0.5, 0.5)))
        cv2.circle(cut, (ox2, oy2), max(3, int(round(r2))), 255, thickness=-1)
    return _largest_component(mask_bool & (cut == 0))


def _place_scallops(
    mask_bool: np.ndarray,
    rng: random.Random,
    n_lo: int,
    n_hi: int,
    r_lo: float,
    r_hi: float,
    *,
    min_sep: float = 14.0,
    cluster: bool = False,
) -> np.ndarray:
    pts = _lateral_contour_points(mask_bool)
    if len(pts) < 5:
        return mask_bool.copy()
    cx, cy = _centroid(mask_bool)
    n = rng.randint(n_lo, n_hi)
    damaged = mask_bool.copy()
    if cluster:
        px0, py0 = pts[rng.randrange(len(pts))]
        ux0, uy0 = _outward_unit(px0, py0, cx, cy)
        tx, ty = -uy0, ux0
        for i in range(n):
            along = rng.uniform(-1.1, 1.1) * (r_hi + 6.0) * (0.4 + 0.12 * i)
            px = int(round(px0 + tx * along))
            py = int(round(py0 + ty * along))
            r = rng.uniform(r_lo, r_hi)
            nxt = _margin_scallop(damaged, rng, px, py, ux0, uy0, r)
            if int(nxt.sum()) < int(damaged.sum()) - 3:
                damaged = nxt
        return damaged

    used: list[tuple[int, int]] = []
    for _ in range(n):
        placed = False
        for _try in range(14):
            px, py = pts[rng.randrange(len(pts))]
            if used and any((px - ax) ** 2 + (py - ay) ** 2 < min_sep ** 2 for ax, ay in used):
                continue
            ux, uy = _outward_unit(px, py, cx, cy)
            r = rng.uniform(r_lo, r_hi)
            nxt = _margin_scallop(damaged, rng, int(px), int(py), ux, uy, r)
            if int(damaged.sum()) - int(nxt.sum()) < 4:
                continue
            damaged = nxt
            used.append((int(px), int(py)))
            placed = True
            break
        if not placed:
            break
    return damaged


def _jagged_notch(
    mask_bool: np.ndarray,
    rng: random.Random,
    px: int,
    py: int,
    ux: float,
    uy: float,
    depth: float,
    width: float,
) -> np.ndarray:
    """Irregular open margin bite (photo-like chewing, not a circular C-cut)."""
    h, w = mask_bool.shape
    tx, ty = -uy, ux
    n_inner = rng.randint(7, 13)
    verts: list[list[int]] = []
    power = rng.uniform(1.05, 2.0)
    for i in range(n_inner):
        t = (i / max(n_inner - 1, 1) - 0.5) * 2.0
        envelope = max(0.08, 1.0 - abs(t) ** power)
        jagged = envelope * rng.uniform(0.55, 1.2)
        if rng.random() < 0.28:
            jagged *= rng.uniform(0.15, 0.55)
        d = depth * jagged
        x = px + tx * t * (width * 0.5) - ux * d
        y = py + ty * t * (width * 0.5) - uy * d
        verts.append([int(round(x)), int(round(y))])
    # close outside the lamina so the hole stays open to background
    for t in (1.05, 0.35, -0.35, -1.05):
        x = px + tx * t * (width * 0.55) + ux * max(4.0, depth * 0.4)
        y = py + ty * t * (width * 0.55) + uy * max(4.0, depth * 0.4)
        verts.append([int(round(x)), int(round(y))])
    poly = np.asarray(verts, dtype=np.int32)
    if len(poly) < 4:
        return mask_bool.copy()
    cut = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cut, [poly], 255)
    damaged = mask_bool & (cut == 0)
    return _largest_component(damaged)


def _place_jagged_bite(
    mask_bool: np.ndarray,
    rng: random.Random,
    frac_lo: float,
    frac_hi: float,
    pts: np.ndarray | None = None,
    avoid: list[tuple[int, int]] | None = None,
    min_sep: float = 0.0,
) -> np.ndarray:
    if pts is None:
        pts = _lateral_contour_points(mask_bool)
    if len(pts) < 5:
        return mask_bool.copy()
    cx, cy = _centroid(mask_bool)
    leaf_area = int(mask_bool.sum())
    target = rng.uniform(frac_lo, frac_hi) * leaf_area
    avoid = avoid or []
    best = mask_bool.copy()
    best_err = 1e18
    for _try in range(18):
        px, py = pts[rng.randrange(len(pts))]
        if avoid and any((px - ax) ** 2 + (py - ay) ** 2 < min_sep ** 2 for ax, ay in avoid):
            continue
        ux, uy = _outward_unit(px, py, cx, cy)
        local_w = float(_inward_width(mask_bool, int(px), int(py), ux, uy))
        depth = min(local_w * rng.uniform(0.45, 0.95), max(6.0, (target / 0.35) ** 0.5))
        width = depth * rng.uniform(1.15, 2.4)
        damaged = _jagged_notch(mask_bool, rng, int(px), int(py), ux, uy, depth, width)
        damaged = damaged & mask_bool
        lost = int(mask_bool.sum()) - int(damaged.sum())
        if lost < 6:
            continue
        err = abs(lost - target)
        frac = lost / max(leaf_area, 1)
        if frac_lo <= frac <= frac_hi and err < best_err:
            return damaged
        if err < best_err:
            best_err = err
            best = damaged
    return best


def _open_margin_bite(
    mask_bool: np.ndarray,
    rng: random.Random,
    frac_lo: float,
    frac_hi: float,
) -> np.ndarray:
    """Cut a large open (background-connected) C-shaped notch on the lateral margin."""
    pts = _lateral_contour_points(mask_bool)
    if len(pts) < 5:
        return mask_bool.copy()
    cx, cy = _centroid(mask_bool)
    leaf_area = int(mask_bool.sum())
    target = rng.uniform(frac_lo, frac_hi) * leaf_area
    h, w = mask_bool.shape

    best = mask_bool.copy()
    best_err = 1e18
    for _try in range(14):
        px, py = pts[rng.randrange(len(pts))]
        ux, uy = _outward_unit(px, py, cx, cy)
        r_guess = max(8.0, (target / 0.45) ** 0.5)
        for scale in (0.55, 0.75, 1.0, 1.25, 1.55, 1.9):
            r = max(6.0, r_guess * scale)
            ox = int(round(px + ux * r * 0.25))
            oy = int(round(py + uy * r * 0.25))
            cut = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cut, (ox, oy), int(round(r)), 255, thickness=-1)
            if rng.random() < 0.55:
                r2 = r * rng.uniform(0.35, 0.7)
                ox2 = int(round(ox + rng.uniform(-0.4, 0.4) * r))
                oy2 = int(round(oy + rng.uniform(-0.4, 0.4) * r))
                cv2.circle(cut, (ox2, oy2), int(round(r2)), 255, thickness=-1)
            damaged = mask_bool & (cut == 0)
            damaged = _largest_component(damaged)
            lost = int(mask_bool.sum()) - int(damaged.sum())
            if lost < 8:
                continue
            err = abs(lost - target)
            frac = lost / max(leaf_area, 1)
            if frac_lo <= frac <= frac_hi and err < best_err:
                return damaged
            if err < best_err:
                best_err = err
                best = damaged
    return best


def _multi_jagged_edge(
    mask_bool: np.ndarray,
    rng: random.Random,
    n_lo: int,
    n_hi: int,
    bite_fracs: list[tuple[float, float]],
) -> np.ndarray:
    pts = _lateral_contour_points(mask_bool)
    if len(pts) < 8:
        return mask_bool.copy()
    ys = np.where(mask_bool)[0]
    min_sep = max(18.0, 0.10 * float(ys.max() - ys.min()))
    n = rng.randint(n_lo, n_hi)
    damaged = mask_bool.copy()
    used: list[tuple[int, int]] = []
    for i in range(n):
        lo, hi = bite_fracs[i % len(bite_fracs)]
        nxt = _place_jagged_bite(damaged, rng, lo, hi, pts=pts, avoid=used, min_sep=min_sep)
        if int(nxt.sum()) >= int(damaged.sum()) - 4:
            continue
        lost_map = damaged & ~nxt
        if lost_map.any():
            yb, xb = np.where(lost_map)
            used.append((int(xb.mean()), int(yb.mean())))
        damaged = nxt
    return damaged


def _internal_holes(
    mask_bool: np.ndarray,
    rng: random.Random,
    n_lo: int = 2,
    n_hi: int = 6,
    r_lo: int = 4,
    r_hi: int = 22,
) -> np.ndarray:
    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.where(dist >= 8.0)
    if xs.size < 10:
        return mask_bool.copy()
    n = rng.randint(n_lo, n_hi)
    layer = np.ones(mask_bool.shape, dtype=np.uint8)
    for _ in range(n):
        i = int(rng.randrange(xs.size))
        r = min(int(dist[ys[i], xs[i]] * 0.85), rng.randint(r_lo, r_hi))
        r = max(3, r)
        cv2.circle(layer, (int(xs[i]), int(ys[i])), r, 0, thickness=-1)
    return mask_bool & (layer > 0)


def _damage_fraction(clean: np.ndarray, damaged: np.ndarray) -> float:
    a = float(clean.sum())
    if a < 1:
        return 0.0
    return max(0.0, 1.0 - float(damaged.sum()) / a)


def _paint_rgb(bgr: np.ndarray, clean: np.ndarray, damaged: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[clean & ~damaged] = (255, 255, 255)
    return out


def _wipe_generated(src: Path) -> None:
    for p in src.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if "_art_" in name or name.endswith("_clean_mask.png") or name == "manifest.csv":
            p.unlink()
    for sub in ("unet_clean_masks", "unet_damaged_masks"):
        d = src / sub
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def process_leaf(
    rgb_path: Path,
    out_dir: Path,
    clean_dir: Path,
    damaged_dir: Path,
    rng: random.Random,
    white_thresh: int,
    with_geom: bool,
) -> list[dict]:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    clean_u8 = extract_partial_mask(bgr, threshold=white_thresh)
    clean = clean_u8 > 0
    if int(clean.sum()) < 500:
        print(f"  [skip] too little tissue: {rgb_path.name}")
        return []

    stem = rgb_path.stem
    geoms = GEOM if with_geom else GEOM[:1]
    for gname, gfn in geoms:
        cv2.imwrite(str(clean_dir / f"{stem}{gname}.png"), gfn(clean_u8))
    cv2.imwrite(str(out_dir / f"{stem}_clean_mask.png"), clean_u8)

    specs: list[tuple[int, str, str]] = [
        (0, "nicks", "nicks"),
        (1, "nicks", "nicks"),
        (2, "photo", "photo"),
        (3, "photo", "photo"),
        (4, "small", "small"),
        (5, "large", "large"),
        (6, "combo", "combo"),
        (7, "holes", "holes"),
    ]
    rows: list[dict] = []
    for v, kind, mode in specs:
        if mode == "nicks":
            damaged = _place_scallops(clean, rng, 4, 9, 6.0, 16.0, min_sep=12.0)
        elif mode == "photo":
            damaged = _place_jagged_bite(clean, rng, 0.03, 0.08)
            damaged = _place_scallops(damaged, rng, 3, 6, 6.0, 15.0, min_sep=12.0)
            damaged = _place_scallops(
                damaged, rng, 3, 5, 8.0, 20.0, min_sep=8.0, cluster=True
            )
            damaged = _internal_holes(damaged, rng, n_lo=1, n_hi=3, r_lo=4, r_hi=14)
        elif mode == "small":
            damaged = _place_scallops(clean, rng, 1, 2, 16.0, 32.0, min_sep=20.0)
        elif mode == "large":
            damaged = _open_margin_bite(clean, rng, 0.15, 0.25)
        elif mode == "combo":
            damaged = _open_margin_bite(clean, rng, 0.15, 0.22)
            damaged = _place_scallops(damaged, rng, 4, 8, 6.0, 16.0, min_sep=12.0)
            if rng.random() < 0.5:
                damaged = _internal_holes(damaged, rng, n_lo=1, n_hi=2, r_lo=4, r_hi=12)
        else:
            damaged = _internal_holes(clean, rng)

        damaged = damaged & clean
        damaged = _largest_component(damaged)
        frac = _damage_fraction(clean, damaged)
        min_frac = 0.0015 if kind in ("nicks", "holes", "small") else 0.004
        if frac < min_frac:
            continue

        dmg_u8 = (damaged.astype(np.uint8)) * 255
        rgb_dmg = _paint_rgb(bgr, clean, damaged)
        viz_name = f"{stem}_art_{kind}_v{v:02d}.png"
        cv2.imwrite(str(out_dir / viz_name), rgb_dmg)
        for gname, gfn in geoms:
            mask_name = f"{stem}{gname}_v{v:02d}.png"
            cv2.imwrite(str(damaged_dir / mask_name), gfn(dmg_u8))
            rows.append(
                {
                    "stem": f"{stem}{gname}",
                    "kind": kind,
                    "variant": v,
                    "geom": gname[1:] if gname else "id",
                    "damage_frac": f"{frac:.4f}",
                    "rgb_preview": viz_name if not gname else "",
                    "damaged_mask": f"unet_damaged_masks/{mask_name}",
                    "clean_mask": f"unet_clean_masks/{stem}{gname}.png",
                }
            )
    return rows


def add_geom_copies(src: Path) -> int:
    """Write rot/flip copies of existing unet mask pairs without regenerating damage."""
    clean_dir = src / "unet_clean_masks"
    damaged_dir = src / "unet_damaged_masks"
    if not clean_dir.is_dir() or not damaged_dir.is_dir():
        raise SystemExit(f"Missing unet_* folders in {src}")
    geom_sfx = ("_r90", "_r180", "_r270", "_fh", "_fv")
    n_new = 0

    def _strip_v(stem: str) -> str:
        for i in range(40):
            suf = f"_v{i:02d}"
            if stem.endswith(suf):
                return stem[: -len(suf)]
        return stem

    for cp in list(clean_dir.glob("*.png")):
        if any(cp.stem.endswith(s) for s in geom_sfx):
            continue
        im = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        for gname, gfn in GEOM[1:]:
            out = clean_dir / f"{cp.stem}{gname}.png"
            if out.exists():
                continue
            cv2.imwrite(str(out), gfn(im))
            n_new += 1

    for dp in list(damaged_dir.glob("*.png")):
        base = _strip_v(dp.stem)
        if any(base.endswith(s) for s in geom_sfx):
            continue
        v_suf = dp.stem[len(base) :]
        im = cv2.imread(str(dp), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        for gname, gfn in GEOM[1:]:
            out = damaged_dir / f"{base}{gname}{v_suf}.png"
            if out.exists():
                continue
            cv2.imwrite(str(out), gfn(im))
            n_new += 1
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Entire margin damage for U-Net fine-tune.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--white-thresh", type=int, default=240)
    parser.add_argument("--no-geom", action="store_true", help="Skip rot/flip copies of the mask pairs.")
    parser.add_argument(
        "--add-geom",
        action="store_true",
        help="Only write rot/flip copies of existing unet masks (does not regenerate damage).",
    )
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")

    if args.add_geom:
        n_new = add_geom_copies(src)
        print(f"[prepare] added {n_new} geometric mask copies under {src}")
        return

    originals = sorted(p for p in src.iterdir() if p.is_file() and _is_original_rgb(p))
    if not originals:
        raise SystemExit(f"No original RGB leaves in {src}")

    _wipe_generated(src)
    clean_dir = src / "unet_clean_masks"
    damaged_dir = src / "unet_damaged_masks"

    rng = random.Random(args.seed)
    all_rows: list[dict] = []
    print(f"[prepare] {len(originals)} intact leaves in {src}")
    for i, path in enumerate(originals, 1):
        print(f"  ({i}/{len(originals)}) {path.name}")
        all_rows.extend(
            process_leaf(
                path,
                src,
                clean_dir,
                damaged_dir,
                rng,
                args.white_thresh,
                with_geom=not args.no_geom,
            )
        )

    man_path = src / "manifest.csv"
    with man_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "kind",
                "variant",
                "geom",
                "damage_frac",
                "rgb_preview",
                "damaged_mask",
                "clean_mask",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)

    identity = [r for r in all_rows if r["geom"] == "id"]
    by_kind: dict[str, list[float]] = {}
    for r in identity:
        by_kind.setdefault(r["kind"], []).append(float(r["damage_frac"]))
    print(f"[prepare] wrote {len(all_rows)} mask pairs ({len(identity)} identity + geom copies)")
    print(f"  previews in {src}")
    for k, fracs in sorted(by_kind.items()):
        print(
            f"  {k:8s}  n={len(fracs):3d}  "
            f"frac mean={sum(fracs)/len(fracs):.3f}  "
            f"min={min(fracs):.3f}  max={max(fracs):.3f}"
        )


if __name__ == "__main__":
    main()
