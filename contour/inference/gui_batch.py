"""UNET Mask-to-Mask batch inference for the HerbivoR GUI Contour step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contour.inference.overlay_viz import overlay_leaf
from contour.inference.predict import load_unet_shape, reconstruct_mask
from image_io import VALID_IMAGE_EXTENSIONS, load_bgr


def process_folder(
    input_dir: Path,
    output_dir: Path,
    model,
    device: torch.device,
    img_size: int,
    threshold: float,
    white_thresh: int,
    refine: bool = True,
    bridge_max_growth: float = 0.08,
    file: str | None = None,
) -> int:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTENSIONS)
    if file:
        candidate = input_dir / file
        paths = [candidate] if candidate in paths else []
    if not paths:
        print(f"[UNET-Shape] No images found in {input_dir}")
        return 0

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i, src in enumerate(paths, 1):
        stem = src.stem
        print(f"[UNET-Shape] ({i}/{len(paths)}) {src.name}")
        bgr = load_bgr(src)
        if bgr is None:
            print("  [skip] unreadable")
            continue
        try:
            res = reconstruct_mask(
                bgr,
                model,
                device,
                img_size=img_size,
                threshold=threshold,
                white_thresh=white_thresh,
                refine=refine,
                bridge_max_growth=bridge_max_growth,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {exc}")
            continue
        if res is None:
            print("  [skip] no leaf tissue detected")
            continue

        mask = res["mask"]
        cv2.imwrite(str(masks_dir / f"{stem}_leaf_mask.png"), mask)
        partial_bool = res["partial_mask"] > 0
        leaf_bool = mask > 0
        overlay = overlay_leaf(bgr, leaf_bool, raw_bool=partial_bool)
        cv2.imwrite(
            str(overlays_dir / f"{stem}_leaf_overlay.jpg"),
            overlay,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        ok += 1

    print(f"[UNET-Shape] Done: {ok}/{len(paths)} -> {output_dir}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="UNET shape completion batch for HerbivoR GUI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default="models/best_unet_shape.pth",
    )
    parser.add_argument(
        "--config",
        default="contour/configs/config_train_f2lsm.yaml",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--file", default=None)
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable morphology-aware post-refinement (serrated/lobed bridge).",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: UNET checkpoint not found: {ckpt}")
        print("Entrena primero: contour\\ENTRENAR_UNET_F2LSM.bat")
        return 1

    cfg_path = Path(args.config)
    cfg: dict = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    infer = cfg.get("inference", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_unet_shape(
        ckpt,
        encoder=model_cfg.get("encoder", "resnet34"),
        device=device,
    )

    return 0 if process_folder(
        Path(args.input),
        Path(args.output),
        model,
        device,
        img_size=int(data_cfg.get("image_size", 512)),
        threshold=float(infer.get("threshold", 0.5)),
        white_thresh=int(infer.get("white_thresh", 240)),
        refine=bool(infer.get("refine", True)) and not args.no_refine,
        bridge_max_growth=float(infer.get("bridge_max_growth", 0.08)),
        file=args.file,
    ) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
