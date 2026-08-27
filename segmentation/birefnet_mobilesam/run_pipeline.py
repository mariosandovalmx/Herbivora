"""BiRefNet_lite + MobileSAM leaf segmentation pipeline.

Outputs:
  <output>/white_bg/<stem>.png    — 512×512 leaf on white background (letterbox)
  <output>/masks/<stem>_mask.png  — binary leaf mask (512×512)
  <output>/metadata/<stem>.json   — scale calibration + area measurements
  <output>/debug/<stem>_dbg.png   — QA overlay (when --save-debug)
  <output>/circles/<stem>_circle.png — cropped blue dot (when circle found)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_io import VALID_IMAGE_EXTENSIONS as VALID_EXT, load_bgr  # noqa: E402

from utils.circle_utils import detect_blue_circle, crop_circle
from utils.mask_utils import (
    box_region_mask,
    component_at_point,
    dilate_mask,
    fill_holes,
    largest_component,
    largest_component_centroid,
    mask_iou,
    remove_region,
)
from utils.scale_utils import apply_scale, compute_damage
from utils.io_utils import (
    composite_on_white,
    crop_to_bbox,
    resize_letterbox,
    save_metadata,
)

from image_io import VALID_IMAGE_EXTENSIONS as VALID_EXT, load_bgr  # noqa: E402


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _load_cfg(path: Path, overrides: dict) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "known_diameter_mm" in overrides:
        cfg["circle"]["known_diameter_mm"] = overrides["known_diameter_mm"]
    if "hybrid_mode" in overrides:
        cfg["hybrid"]["mode"] = overrides["hybrid_mode"]
    if "seg_resolution" in overrides:
        cfg["segmentation_resolution"] = overrides["seg_resolution"]
    if "output_size" in overrides:
        cfg["output_size"] = overrides["output_size"]
    if "agreement_threshold" in overrides:
        cfg["hybrid"]["agreement_threshold"] = overrides["agreement_threshold"]
    return cfg


# --------------------------------------------------------------------------- #
# Debug overlay                                                                #
# --------------------------------------------------------------------------- #

def _save_debug_overlay(image_bgr: np.ndarray, mask: np.ndarray,
                         circle_info: dict, path: Path) -> None:
    overlay = image_bgr.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 180
    overlay[mask] = cv2.addWeighted(overlay, 0.55, green, 0.45, 0)[mask]
    if circle_info.get("found"):
        cx, cy = circle_info["center_px"]
        r = circle_info["diameter_px"] / 2
        cv2.circle(overlay, (int(cx), int(cy)), int(r), (0, 0, 255), 3)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    h, w = overlay.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


# --------------------------------------------------------------------------- #
# Segmentation with multi-level fallback                                       #
# --------------------------------------------------------------------------- #

def _segment_with_fallback(
    image: np.ndarray,
    circle_info: dict,
    cfg: dict,
    birefnet,
    mobilesam,
    device,
    box_prior: tuple[int, int, int, int] | None = None,
    point_prior: tuple[int, int] | None = None,
    mask_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Run BiRefNet + MobileSAM with robust fallbacks.

    Fallback chain:
      1. birefnet_primary merge (BiRefNet & dilated-SAM)
      2. If result < min_area_px → BiRefNet alone
      3. If still < min_area_px → MobileSAM from image centre
      4. If still < min_area_px → MobileSAM with 5-point grid prompts, pick best

    box_prior: optional (x1, y1, x2, y2) detector box (e.g. from a YOLO leaf
    detector pre-step). When given, segmentation is hard-constrained to a
    margin around this box: BiRefNet is intersected with the region before
    computing the SAM prompt, MobileSAM is prompted with the box itself
    (a much stronger geometric prior than a single point in cluttered scene
    photos), and every fallback candidate — including the final mask — is
    clipped to the region. This prevents background clutter (soil, twigs)
    outside the detected leaf from ever being counted as foreground, which
    otherwise happens when BiRefNet's generic saliency picks up high-contrast
    background in natural scene photos.

    point_prior: optional (x, y) user click (interactive GUI). When given and
    box_prior is absent, MobileSAM is prompted at that click instead of the
    BiRefNet centroid. BiRefNet still runs and is merged for edge precision.

    mask_prior: optional boolean H×W mask from an interactive MobileSAM click.
    When given, this ROI is preserved as the segmentation floor; BiRefNet only
    refines edges inside a margin around it (mobilesam_primary merge).

    Returns (final_mask, seg_meta_dict).
    """
    from utils.segmentation_utils import (
        run_birefnet, run_mobilesam_point, run_mobilesam_box, merge_masks
    )

    min_px = cfg["leaf"].get("min_area_px", 5000)
    seg_res = cfg.get("segmentation_resolution", 1024)
    H, W = image.shape[:2]

    region_mask = None
    region_box = None
    if box_prior is not None:
        margin_frac = cfg["leaf"].get("box_prior_margin_frac", 0.25)
        region_mask, region_box = box_region_mask((H, W), box_prior, margin_frac)

    # ---- BiRefNet ----
    print(f"  BiRefNet_lite ({seg_res}px)...", end=" ", flush=True)
    t0 = time.time()
    M_bi = run_birefnet(image, birefnet, size=seg_res, device=device)
    print(f"{time.time()-t0:.2f}s  area={M_bi.sum()}")

    # Remove blue circle from BiRefNet mask
    if cfg["leaf"].get("exclude_blue_region") and circle_info.get("found"):
        M_bi = remove_region(M_bi, circle_info["bbox"])
        M_bi = largest_component(M_bi)

    if region_mask is not None:
        # Discard anything BiRefNet marked as salient outside the detector's
        # box neighborhood before it can influence the SAM prompt or centroid.
        M_bi = M_bi & region_mask

    # ---- Interactive ROI: user MobileSAM mask is the floor ----
    if mask_prior is not None:
        M_sam = mask_prior.astype(bool)
        if M_sam.shape[:2] != (H, W):
            raise ValueError(
                f"mask_prior shape {M_sam.shape[:2]} != image {(H, W)}"
            )
        margin_k = int(cfg["leaf"].get("mask_prior_margin_px", 20))
        roi = dilate_mask(M_sam, k=margin_k)
        M_bi = M_bi & roi
        M_refined = merge_masks(M_bi, M_sam, "mobilesam_primary")
        M_final = (M_refined | M_sam) & roi
        if point_prior is not None:
            M_final = component_at_point(M_final, point_prior[0], point_prior[1])
        else:
            M_final = largest_component(M_final)
        iou_final = mask_iou(M_bi, M_final)
        seg_meta = {
            "hybrid_mode": "mobilesam_primary",
            "iou_models": round(float(iou_final), 4),
            "low_confidence": bool(
                iou_final < cfg["hybrid"]["agreement_threshold"]
            ),
            "fallback_used": "mask_prior",
            "box_prior_used": box_prior is not None,
            "point_prior_used": point_prior is not None,
            "point_prior": list(point_prior) if point_prior is not None else None,
            "mask_prior_used": True,
        }
        return M_final, seg_meta

    # ---- MobileSAM attempt 1 ----
    if region_box is not None:
        print(f"  MobileSAM (box={region_box})...", end=" ", flush=True)
        t0 = time.time()
        M_sam = run_mobilesam_box(image, mobilesam, box=region_box)
        print(f"{time.time()-t0:.2f}s  area={M_sam.sum()}")
    elif point_prior is not None:
        px = int(max(0, min(W - 1, point_prior[0])))
        py = int(max(0, min(H - 1, point_prior[1])))
        print(f"  MobileSAM (user click={px},{py})...", end=" ", flush=True)
        t0 = time.time()
        M_sam = run_mobilesam_point(image, mobilesam, point=(px, py))
        print(f"{time.time()-t0:.2f}s  area={M_sam.sum()}")
    else:
        cx, cy = largest_component_centroid(M_bi)
        print(f"  MobileSAM (prompt={cx},{cy})...", end=" ", flush=True)
        t0 = time.time()
        M_sam = run_mobilesam_point(image, mobilesam, point=(cx, cy))
        print(f"{time.time()-t0:.2f}s  area={M_sam.sum()}")

    mode = cfg["hybrid"]["mode"]
    M_final = merge_masks(M_bi, M_sam, mode)
    if region_mask is not None:
        M_final = M_final & region_mask
    iou = mask_iou(M_bi, M_sam)
    fallback_used = "none"

    # ---- Fallback 1: merge too small → use BiRefNet alone ----
    if M_final.sum() < min_px:
        print(f"  Merge result too small ({M_final.sum()} px, IoU={iou:.4f}). "
              f"Fallback 1: BiRefNet alone.")
        M_final = M_bi.copy()
        fallback_used = "birefnet_only"

    # ---- Fallback 2: BiRefNet also too small → SAM from user click / centre ----
    if M_final.sum() < min_px:
        if point_prior is not None:
            cx_c = int(max(0, min(W - 1, point_prior[0])))
            cy_c = int(max(0, min(H - 1, point_prior[1])))
        elif region_box is not None:
            cx_c = (region_box[0] + region_box[2]) // 2
            cy_c = (region_box[1] + region_box[3]) // 2
        else:
            cx_c, cy_c = W // 2, H // 2
        print(f"  BiRefNet mask too small ({M_final.sum()} px). "
              f"Fallback 2: MobileSAM from centre ({cx_c},{cy_c}).")
        M_sam_c = run_mobilesam_point(image, mobilesam, point=(cx_c, cy_c))
        if region_mask is not None:
            M_sam_c = M_sam_c & region_mask
        if M_sam_c.sum() > M_final.sum():
            M_final = M_sam_c
            fallback_used = "sam_center"

    # ---- Fallback 3: still too small → grid of points, pick largest ----
    if M_final.sum() < min_px:
        print(f"  Still too small ({M_final.sum()} px). "
              f"Fallback 3: point grid.")
        if region_box is not None:
            rx1, ry1, rx2, ry2 = region_box
            rw, rh = rx2 - rx1, ry2 - ry1
            grid_points = [
                (rx1 + rw // 2, ry1 + rh // 2),
                (rx1 + rw // 4, ry1 + rh // 4), (rx1 + 3 * rw // 4, ry1 + rh // 4),
                (rx1 + rw // 4, ry1 + 3 * rh // 4), (rx1 + 3 * rw // 4, ry1 + 3 * rh // 4),
            ]
        else:
            grid_points = [
                (W // 2, H // 2),
                (W // 4, H // 4), (3 * W // 4, H // 4),
                (W // 4, 3 * H // 4), (3 * W // 4, 3 * H // 4),
            ]
        if point_prior is not None:
            grid_points = [
                (int(max(0, min(W - 1, point_prior[0]))),
                 int(max(0, min(H - 1, point_prior[1])))),
                *grid_points,
            ]
        candidates = []
        for px, py in grid_points:
            m = run_mobilesam_point(image, mobilesam, point=(px, py))
            if region_mask is not None:
                m = m & region_mask
            m_comp = largest_component(m)
            candidates.append(m_comp)
        best = max(candidates, key=lambda m: m.sum())
        if best.sum() > M_final.sum():
            M_final = best
            fallback_used = "sam_grid"
        print(f"  Grid best: {M_final.sum()} px")

    # Safety net: never let any fallback path leak foreground far outside
    # the detector's box neighborhood, regardless of which stage produced it.
    if region_mask is not None:
        M_final = M_final & dilate_mask(region_mask, k=5)

    # Recompute IoU for metadata (birefnet vs final)
    iou_final = mask_iou(M_bi, M_final)
    low_conf = iou_final < cfg["hybrid"]["agreement_threshold"]

    M_final = largest_component(M_final)

    seg_meta = {
        "hybrid_mode": mode,
        "iou_models": round(float(iou_final), 4),
        "low_confidence": bool(low_conf),
        "fallback_used": fallback_used,
        "box_prior_used": box_prior is not None,
        "point_prior_used": point_prior is not None,
        "point_prior": list(point_prior) if point_prior is not None else None,
    }
    return M_final, seg_meta


# --------------------------------------------------------------------------- #
# Per-image processing                                                         #
# --------------------------------------------------------------------------- #

def _load_box_prior(box_prior_dir: Path | None, stem: str) -> tuple[int, int, int, int] | None:
    """Reads an optional <stem>.json sidecar with {"x1","y1","x2","y2"} (e.g. from
    a YOLO leaf-detection pre-step). Returns None if not given/found — segmentation
    then behaves exactly as before (unconstrained, whole-image saliency)."""
    if box_prior_dir is None:
        return None
    path = box_prior_dir / f"{stem}.json"
    if not path.is_file():
        return None
    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
        return int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"])
    except (OSError, KeyError, ValueError, TypeError) as e:
        print(f"  WARNING: could not read box prior {path}: {e}")
        return None


def _load_point_prior(point_prior_dir: Path | None, stem: str) -> tuple[int, int] | None:
    """Reads optional <stem>.json sidecar with {"x","y"} user click (interactive GUI)."""
    if point_prior_dir is None:
        return None
    path = point_prior_dir / f"{stem}.json"
    if not path.is_file():
        return None
    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
        return int(data["x"]), int(data["y"])
    except (OSError, KeyError, ValueError, TypeError) as e:
        print(f"  WARNING: could not read point prior {path}: {e}")
        return None


def process_image(
    image_path: Path,
    output_dir: Path,
    cfg: dict,
    birefnet,
    mobilesam,
    remove_blue: bool = True,
    save_debug: bool = True,
    box_prior_dir: Path | None = None,
    point_prior_dir: Path | None = None,
    point_prior: tuple[int, int] | None = None,
    circle_prior: dict | None = None,
    mask_prior: np.ndarray | None = None,
    output_stem: str | None = None,
) -> dict:
    from utils.segmentation_utils import get_device

    stem = output_stem or image_path.stem
    device = get_device()

    image = load_bgr(image_path)
    if image is None:
        print(f"  [SKIP] Cannot read: {image_path}")
        return {}
    orig_h, orig_w = image.shape[:2]
    box_prior = _load_box_prior(box_prior_dir, stem)
    if point_prior is None:
        point_prior = _load_point_prior(point_prior_dir, stem)

    if cfg.get("resize_before_segmentation"):
        # Strategy B: resize first (faster but less precise)
        from utils.io_utils import resize_letterbox as _rlb
        image, pre_scale, _ = _rlb(image, cfg["output_size"])
        pre_ratio = (pre_scale, pre_scale)
    else:
        pre_ratio = (1.0, 1.0)

    # ---- Step 1: blue circle ----
    circle_info: dict = {"found": False}
    if circle_prior is not None and circle_prior.get("found"):
        # Interactive / manual prior (coords must match current image space)
        circle_info = dict(circle_prior)
        if circle_info.get("mm2_per_px2") is None:
            from utils.circle_utils import compute_circle_scale
            known = float(cfg.get("known_diameter_mm", 6.0))
            mm_per_px, mm2 = compute_circle_scale(
                float(circle_info["diameter_px"]), known
            )
            circle_info["mm_per_px"] = mm_per_px
            circle_info["mm2_per_px2"] = mm2
        print(
            f"  Circle: prior d={circle_info['diameter_px']:.1f}px  "
            f"method={circle_info.get('method', 'prior')}  "
            f"scale={circle_info.get('mm2_per_px2', 'N/A')}"
        )
    elif remove_blue:
        circle_info = detect_blue_circle(image, cfg)
        if circle_info["found"]:
            print(f"  Circle: d={circle_info['diameter_px']:.1f}px  "
                  f"circ={circle_info['circularity']:.3f}  "
                  f"scale={circle_info.get('mm2_per_px2', 'N/A')}")
        else:
            if cfg["circle"].get("required"):
                raise RuntimeError(f"Blue circle not found in {image_path.name}")
            print("  Circle: not found")
    else:
        print("  Circle detection: disabled")

    # ---- Step 2: segmentation with fallback ----
    M_final, seg_meta = _segment_with_fallback(
        image, circle_info, cfg, birefnet, mobilesam, device,
        box_prior=box_prior, point_prior=point_prior, mask_prior=mask_prior,
    )

    if M_final.sum() == 0:
        print(f"  ERROR: all fallbacks failed, skipping {image_path.name}")
        return {}

    # ---- Step 3: refine + composite ----
    M_solid = fill_holes(M_final) if cfg["leaf"].get("fill_internal_holes") else M_final

    silhouette_px = int(M_solid.sum())
    remaining_px = int(M_final.sum())
    damage_px, damage_pct = compute_damage(silhouette_px, remaining_px)

    scale = circle_info.get("mm2_per_px2")
    silhouette_mm2 = apply_scale(silhouette_px, scale)
    damage_mm2 = apply_scale(damage_px, scale)

    # Composite on white (herbivory holes preserved via M_final)
    composite = composite_on_white(image, M_final, cfg["background_color"])

    # Tight crop with small 2% buffer (avoids edge artefacts on serrated margins)
    cropped, crop_bbox = crop_to_bbox(composite, M_solid, pad_frac=0.02)

    # Letterbox resize → 512×512, aspect-ratio preserved.
    # pad_fraction=0.03 reserves 3% white border on every side of the output
    # so the leaf never touches any edge regardless of its shape.
    bg = tuple(cfg["background_color"])
    output_512, scale_factor, offset = resize_letterbox(
        cropped, cfg["output_size"], pad_fraction=0.03, bg_color=bg
    )

    # Same crop + letterbox for the binary mask
    mask_crop, _ = crop_to_bbox(
        (M_solid.astype(np.uint8) * 255), M_solid, pad_frac=0.02
    )
    mask_512, _, _ = resize_letterbox(mask_crop, cfg["output_size"],
                                      pad_fraction=0.03, bg_color=(0, 0, 0))
    _, mask_bin = cv2.threshold(mask_512, 127, 255, cv2.THRESH_BINARY)

    # ---- Step 4: save ----
    wb_dir = output_dir / "white_bg"
    masks_out = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    wb_dir.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    out_img = wb_dir / f"{stem}.png"
    out_mask = masks_out / f"{stem}_mask.png"
    cv2.imwrite(str(out_img), output_512)
    cv2.imwrite(str(out_mask), mask_bin)

    if circle_info.get("found") and cfg.get("save_circle_crop"):
        circles_dir = output_dir / "circles"
        circles_dir.mkdir(parents=True, exist_ok=True)
        circle_crop = crop_circle(image, circle_info["bbox"])
        cv2.imwrite(str(circles_dir / f"{stem}_circle.png"), circle_crop)

    if save_debug and cfg.get("save_debug_overlays"):
        debug_dir = output_dir / "debug"
        _save_debug_overlay(image, M_solid, circle_info,
                            debug_dir / f"{stem}_dbg.png")

    meta: dict = {
        "image_id": stem,
        "original_size": [orig_w, orig_h],
        "output_size": [cfg["output_size"], cfg["output_size"]],
        "scale_factor": round(scale_factor, 6),
        "scale_source": "original_photo",
        "letterbox_offset": list(offset),
        "crop_bbox": list(crop_bbox),
        "pre_ratio": list(pre_ratio),
        "circle": {
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in circle_info.items()},
            "known_diameter_mm": cfg["circle"].get("known_diameter_mm"),
        },
        "leaf": {
            "silhouette_area_px": silhouette_px,
            "remaining_leaf_area_px": remaining_px,
            "damage_area_px": damage_px,
            "silhouette_area_mm2": round(silhouette_mm2, 4) if silhouette_mm2 is not None else None,
            "damage_area_mm2": round(damage_mm2, 4) if damage_mm2 is not None else None,
            "damage_percent": damage_pct,
        },
        "segmentation": seg_meta,
        "output_path": str(out_img),
        "measurement_space": "pre_resized" if cfg.get("resize_before_segmentation")
                              else "original_resolution",
    }
    save_metadata(meta, meta_dir / f"{stem}.json")
    print(f"  Saved → {out_img.name}  "
          f"silhouette={silhouette_px}px  damage={damage_pct:.2f}%  "
          f"fallback={seg_meta['fallback_used']}")
    return meta


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BiRefNet_lite + MobileSAM leaf segmentation"
    )
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(_HERE / "config.yaml"))
    parser.add_argument("--known-diameter-mm", type=float)
    parser.add_argument("--hybrid-mode",
                        choices=["birefnet_primary", "mobilesam_primary",
                                 "intersection", "union"])
    parser.add_argument("--seg-resolution", type=int)
    parser.add_argument("--output-size", type=int)
    parser.add_argument("--agreement-threshold", type=float)
    parser.add_argument("--remove-blue", action="store_true", default=True)
    parser.add_argument("--no-remove-blue", dest="remove_blue", action="store_false")
    parser.add_argument("--save-debug", action="store_true", default=True)
    parser.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    parser.add_argument(
        "--box-prior-dir", default=None,
        help="Folder with <stem>.json {x1,y1,x2,y2} sidecars (e.g. from a YOLO leaf "
             "detector pre-step) to hard-constrain segmentation near each detected box. "
             "Omit for unconstrained whole-image saliency segmentation (default).",
    )
    parser.add_argument(
        "--point-prior-dir", default=None,
        help="Folder with <stem>.json {x,y} click sidecars (interactive GUI). "
             "When present, MobileSAM is prompted at the click instead of the "
             "BiRefNet centroid. Ignored when --box-prior-dir applies to the stem.",
    )
    parser.add_argument(
        "--mobilesam-weights",
        default=None,
        help="Path to MobileSAM .pt weights (default: models/mobile_sam.pt next to this package).",
    )
    parser.add_argument("--allow-non-venv", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _HERE / "config.yaml"

    overrides: dict = {}
    if args.known_diameter_mm is not None:
        overrides["known_diameter_mm"] = args.known_diameter_mm
    if args.hybrid_mode is not None:
        overrides["hybrid_mode"] = args.hybrid_mode
    if args.seg_resolution is not None:
        overrides["seg_resolution"] = args.seg_resolution
    if args.output_size is not None:
        overrides["output_size"] = args.output_size
    if args.agreement_threshold is not None:
        overrides["agreement_threshold"] = args.agreement_threshold

    cfg = _load_cfg(cfg_path, overrides)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"\nBiRefNet_lite + MobileSAM pipeline")
    print(f"  Input : {input_dir}  ({len(images)} images)")
    print(f"  Output: {output_dir}")
    print(f"  Mode  : {cfg['hybrid']['mode']}  |  "
          f"Seg res: {cfg['segmentation_resolution']}px  |  "
          f"Out: {cfg['output_size']}px  (letterbox, aspect-preserved)\n")

    from utils.segmentation_utils import load_birefnet, load_mobilesam, get_device
    device = get_device()
    print(f"Device: {device}")

    print("Loading BiRefNet_lite...", end=" ", flush=True)
    t0 = time.time()
    birefnet = load_birefnet(device)
    print(f"{time.time()-t0:.1f}s")

    print("Loading MobileSAM...", end=" ", flush=True)
    t0 = time.time()
    mobilesam = load_mobilesam(device, weights=args.mobilesam_weights)
    print(f"{time.time()-t0:.1f}s\n")

    box_prior_dir = Path(args.box_prior_dir) if args.box_prior_dir else None
    if box_prior_dir is not None:
        print(f"Box prior: {box_prior_dir}  (constraining segmentation near detected boxes)")
    point_prior_dir = Path(args.point_prior_dir) if args.point_prior_dir else None
    if point_prior_dir is not None:
        print(f"Point prior: {point_prior_dir}  (MobileSAM prompted at user clicks)")

    ok, failed = 0, 0
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            result = process_image(
                img_path, output_dir, cfg,
                birefnet, mobilesam,
                remove_blue=args.remove_blue,
                save_debug=args.save_debug,
                box_prior_dir=box_prior_dir,
                point_prior_dir=point_prior_dir,
            )
            if result:
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {ok} succeeded, {failed} failed.")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
