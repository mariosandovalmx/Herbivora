"""Multi-leaf BiRefNet_lite + MobileSAM pipeline (connected components).

Outputs (one set per leaf instance):
  <output>/white_bg/{stem}_leaf_{n}.png
  <output>/masks/{stem}_leaf_{n}_mask.png
  <output>/metadata/{stem}_leaf_{n}.json

Does not modify birefnet_mobilesam/run_pipeline.py. Reuses its utils via import.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_SINGLE_LEAF = _HERE.parent / "birefnet_mobilesam"
_REPO_ROOT = _HERE.parent.parent

# Prefer single-leaf package utils + repo image_io without mutating that package
for _p in (str(_SINGLE_LEAF), str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from image_io import VALID_IMAGE_EXTENSIONS as VALID_EXT, load_bgr  # noqa: E402

try:
    from .cc_instances import LeafInstance, extract_leaf_instances  # noqa: E402
    from .export_leaf import export_leaf_outputs, leaf_output_stem  # noqa: E402
except ImportError:
    from cc_instances import LeafInstance, extract_leaf_instances  # noqa: E402
    from export_leaf import export_leaf_outputs, leaf_output_stem  # noqa: E402

from utils.circle_utils import detect_blue_circle  # noqa: E402
from utils.mask_utils import (  # noqa: E402
    box_region_mask,
    dilate_mask,
    fill_holes,
    largest_component,
    mask_iou,
    remove_region,
)
from utils.scale_utils import apply_scale, compute_damage  # noqa: E402


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
    if "min_area_px" in overrides and overrides["min_area_px"] is not None:
        cfg["leaf"]["min_area_px"] = overrides["min_area_px"]
    if "max_leaves" in overrides:
        cfg["leaf"]["max_leaves"] = overrides["max_leaves"]
    return cfg


def _save_debug_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    circle_info: dict,
    path: Path,
) -> None:
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


def _refine_instance(
    image: np.ndarray,
    instance: LeafInstance,
    M_bi_full: np.ndarray,
    cfg: dict,
    mobilesam,
) -> tuple[np.ndarray, dict]:
    """MobileSAM box refine + hybrid merge for one connected component."""
    from utils.segmentation_utils import run_mobilesam_box, merge_masks

    H, W = image.shape[:2]
    min_px = cfg["leaf"].get("min_area_px", 5000)
    margin_frac = cfg["leaf"].get("box_prior_margin_frac", 0.15)
    region_mask, region_box = box_region_mask((H, W), instance.bbox, margin_frac)

    # BiRefNet evidence for this leaf = CC mask (optionally intersect full map)
    M_bi = instance.mask & M_bi_full
    if M_bi.sum() == 0:
        M_bi = instance.mask

    print(
        f"    leaf_{instance.leaf_index}: MobileSAM box={region_box} "
        f"(cc_area={instance.area_px})...",
        end=" ",
        flush=True,
    )
    t0 = time.time()
    M_sam = run_mobilesam_box(image, mobilesam, box=region_box)
    print(f"{time.time() - t0:.2f}s  sam_area={int(M_sam.sum())}")

    mode = cfg["hybrid"]["mode"]
    M_final = merge_masks(M_bi, M_sam, mode)
    M_final = M_final & region_mask
    iou = mask_iou(M_bi, M_sam)
    fallback_used = "none"

    if M_final.sum() < min_px:
        print(f"    leaf_{instance.leaf_index}: merge small → BiRefNet CC")
        M_final = M_bi.copy()
        fallback_used = "birefnet_cc"

    if M_final.sum() < min_px:
        print(f"    leaf_{instance.leaf_index}: still small → SAM ∩ region")
        M_sam_c = M_sam & region_mask
        if M_sam_c.sum() > M_final.sum():
            M_final = M_sam_c
            fallback_used = "sam_box"

    M_final = M_final & dilate_mask(region_mask, k=5)
    M_final = largest_component(M_final)

    iou_final = mask_iou(M_bi, M_final)
    low_conf = iou_final < cfg["hybrid"]["agreement_threshold"]
    seg_meta = {
        "hybrid_mode": mode,
        "iou_models": round(float(iou_final), 4),
        "iou_bi_sam": round(float(iou), 4),
        "low_confidence": bool(low_conf),
        "fallback_used": fallback_used,
        "multi_leaf": True,
        "leaf_index": instance.leaf_index,
        "cc_area_px": instance.area_px,
        "bbox": list(instance.bbox),
    }
    return M_final, seg_meta


def process_image_multi(
    image_path: Path,
    output_dir: Path,
    cfg: dict,
    birefnet,
    mobilesam,
    *,
    remove_blue: bool = True,
    save_debug: bool = True,
) -> list[dict]:
    """Segment all non-touching leaves in one photo. Returns list of per-leaf metas."""
    from utils.segmentation_utils import get_device, run_birefnet

    stem = image_path.stem
    device = get_device()

    image = load_bgr(image_path)
    if image is None:
        print(f"  [SKIP] Cannot read: {image_path}")
        return []
    orig_h, orig_w = image.shape[:2]

    # ---- Blue circle (shared scale for all leaves from this photo) ----
    circle_info: dict = {"found": False}
    if remove_blue:
        circle_info = detect_blue_circle(image, cfg)
        if circle_info["found"]:
            print(
                f"  Circle: d={circle_info['diameter_px']:.1f}px  "
                f"scale={circle_info.get('mm2_per_px2', 'N/A')}"
            )
        else:
            if cfg["circle"].get("required"):
                raise RuntimeError(f"Blue circle not found in {image_path.name}")
            print("  Circle: not found")
    else:
        print("  Circle detection: disabled")

    # ---- BiRefNet (full image; do NOT collapse to largest CC) ----
    seg_res = cfg.get("segmentation_resolution", 1024)
    print(f"  BiRefNet_lite ({seg_res}px)...", end=" ", flush=True)
    t0 = time.time()
    M_bi = run_birefnet(image, birefnet, size=seg_res, device=device)
    print(f"{time.time() - t0:.2f}s  area={int(M_bi.sum())}")

    if cfg["leaf"].get("exclude_blue_region") and circle_info.get("found"):
        M_bi = remove_region(M_bi, circle_info["bbox"])

    min_area = int(cfg["leaf"].get("min_area_px", 5000))
    max_leaves = cfg["leaf"].get("max_leaves")
    if max_leaves is not None:
        try:
            max_leaves = int(max_leaves)
        except (TypeError, ValueError):
            max_leaves = None

    instances = extract_leaf_instances(
        M_bi, min_area_px=min_area, max_leaves=max_leaves
    )
    if not instances:
        print(f"  ERROR: no leaf components ≥ {min_area} px in {image_path.name}")
        return []

    print(f"  Found {len(instances)} leaf instance(s)")

    results: list[dict] = []
    scale = circle_info.get("mm2_per_px2")

    for inst in instances:
        out_stem = leaf_output_stem(stem, inst.leaf_index)
        M_final, seg_meta = _refine_instance(image, inst, M_bi, cfg, mobilesam)
        if M_final.sum() == 0:
            print(f"    leaf_{inst.leaf_index}: empty mask, skip")
            continue

        M_solid = fill_holes(M_final) if cfg["leaf"].get("fill_internal_holes") else M_final
        silhouette_px = int(M_solid.sum())
        remaining_px = int(M_final.sum())
        damage_px, damage_pct = compute_damage(silhouette_px, remaining_px)
        silhouette_mm2 = apply_scale(silhouette_px, scale)
        damage_mm2 = apply_scale(damage_px, scale)

        meta: dict = {
            "image_id": out_stem,
            "parent_image_id": stem,
            "leaf_index": inst.leaf_index,
            "original_size": [orig_w, orig_h],
            "scale_source": "original_photo",
            "pre_ratio": [1.0, 1.0],
            "circle": {
                **{
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in circle_info.items()
                },
                "known_diameter_mm": cfg["circle"].get("known_diameter_mm"),
            },
            "leaf": {
                "silhouette_area_px": silhouette_px,
                "remaining_leaf_area_px": remaining_px,
                "damage_area_px": damage_px,
                "silhouette_area_mm2": (
                    round(silhouette_mm2, 4) if silhouette_mm2 is not None else None
                ),
                "damage_area_mm2": (
                    round(damage_mm2, 4) if damage_mm2 is not None else None
                ),
                "damage_percent": damage_pct,
            },
            "segmentation": seg_meta,
            "measurement_space": "original_resolution",
            "multi_leaf": True,
        }

        out_img, _out_mask = export_leaf_outputs(
            image_bgr=image,
            M_final=M_final,
            M_solid=M_solid,
            output_dir=output_dir,
            stem=out_stem,
            cfg=cfg,
            meta=meta,
            circle_info=circle_info,
            save_debug=save_debug,
            save_debug_overlay_fn=_save_debug_overlay,
        )
        print(
            f"    Saved → {out_img.name}  silhouette={silhouette_px}px  "
            f"damage={damage_pct:.2f}%  fallback={seg_meta['fallback_used']}"
        )
        results.append(meta)

    return results


def run_folder_batch_multi(
    input_dir: Path,
    output_dir: Path,
    *,
    birefnet=None,
    mobilesam=None,
    known_diameter_mm: float = 6.0,
    hybrid_mode: str = "birefnet_primary",
    seg_resolution: int = 1024,
    output_size: int = 1024,
    agreement_threshold: float = 0.85,
    min_area_px: int | None = None,
    max_leaves: int | None = None,
    remove_blue: bool = True,
    mobilesam_weights: str | None = None,
    config_path: Path | None = None,
    log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Batch multi-leaf segmentation. Returns (images_ok, images_failed).

    Counts success per *source image* that produced ≥1 leaf. When models are
    omitted, loads them (CLI / cold start).
    """
    from utils.segmentation_utils import get_device, load_birefnet, load_mobilesam

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"Input folder not found: {input_dir}")

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    cfg_path = Path(config_path) if config_path else (_HERE / "config.yaml")
    overrides: dict = {
        "known_diameter_mm": known_diameter_mm,
        "hybrid_mode": hybrid_mode,
        "seg_resolution": seg_resolution,
        "output_size": output_size,
        "agreement_threshold": agreement_threshold,
    }
    if min_area_px is not None:
        overrides["min_area_px"] = min_area_px
    if max_leaves is not None:
        overrides["max_leaves"] = max_leaves
    cfg = _load_cfg(cfg_path, overrides)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    if birefnet is None or mobilesam is None:
        _log(f"Device: {device}")
        if birefnet is None:
            _log("Loading BiRefNet_lite...")
            t0 = time.time()
            birefnet = load_birefnet(device)
            _log(f"  BiRefNet loaded in {time.time() - t0:.1f}s")
        if mobilesam is None:
            _log("Loading MobileSAM...")
            t0 = time.time()
            mobilesam = load_mobilesam(device, weights=mobilesam_weights)
            _log(f"  MobileSAM loaded in {time.time() - t0:.1f}s")

    _log(
        f"Multi-leaf BiRefNet + MobileSAM\n"
        f"  Input : {input_dir}  ({len(images)} images)\n"
        f"  Output: {output_dir}\n"
        f"  Mode  : {cfg['hybrid']['mode']}  |  "
        f"Seg res: {cfg['segmentation_resolution']}px  |  "
        f"Out: {cfg['output_size']}px  |  "
        f"min_area: {cfg['leaf'].get('min_area_px')}px"
    )

    ok = failed = 0
    n = len(images)
    for i, img_path in enumerate(images, 1):
        if should_cancel and should_cancel():
            _log("Multi-leaf batch cancelled.")
            break
        _log(f"[{i}/{n}] {img_path.name}")
        try:
            results = process_image_multi(
                img_path,
                output_dir,
                cfg,
                birefnet,
                mobilesam,
                remove_blue=remove_blue,
                save_debug=True,
            )
            if results:
                ok += 1
                _log(f"  → {len(results)} leaf file(s)")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            _log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    _log(f"Done: {ok} image(s) succeeded, {failed} failed.")
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-leaf BiRefNet_lite + MobileSAM (connected components)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(_HERE / "config.yaml"))
    parser.add_argument("--known-diameter-mm", type=float, default=None)
    parser.add_argument(
        "--hybrid-mode",
        choices=["birefnet_primary", "mobilesam_primary", "intersection", "union"],
        default=None,
    )
    parser.add_argument("--seg-resolution", type=int, default=None)
    parser.add_argument("--output-size", type=int, default=None)
    parser.add_argument("--agreement-threshold", type=float, default=None)
    parser.add_argument("--min-area-px", type=int, default=None)
    parser.add_argument("--max-leaves", type=int, default=None)
    parser.add_argument("--remove-blue", action="store_true", default=True)
    parser.add_argument("--no-remove-blue", dest="remove_blue", action="store_false")
    parser.add_argument("--mobilesam-weights", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _HERE / "config.yaml"
    base = _load_cfg(cfg_path, {})

    ok, failed = run_folder_batch_multi(
        Path(args.input),
        Path(args.output),
        known_diameter_mm=(
            args.known_diameter_mm
            if args.known_diameter_mm is not None
            else float(base["circle"].get("known_diameter_mm", 6.0))
        ),
        hybrid_mode=(
            args.hybrid_mode
            if args.hybrid_mode is not None
            else str(base["hybrid"].get("mode", "birefnet_primary"))
        ),
        seg_resolution=(
            args.seg_resolution
            if args.seg_resolution is not None
            else int(base.get("segmentation_resolution", 1024))
        ),
        output_size=(
            args.output_size
            if args.output_size is not None
            else int(base.get("output_size", 1024))
        ),
        agreement_threshold=(
            args.agreement_threshold
            if args.agreement_threshold is not None
            else float(base["hybrid"].get("agreement_threshold", 0.85))
        ),
        min_area_px=args.min_area_px,
        max_leaves=args.max_leaves,
        remove_blue=args.remove_blue,
        mobilesam_weights=args.mobilesam_weights,
        config_path=cfg_path,
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
