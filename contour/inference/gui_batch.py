"""UNET Mask-to-Mask batch inference for the Herbivora GUI Contour step."""

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
from contour.inference.predict import (
    extract_partial_mask,
    load_unet_shape,
    reconstruct_mask,
)
from gui.paths import (
    DEFAULT_UNET_SHAPE_MODEL,
    resolve_unet_shape_checkpoint,
)
from image_io import VALID_IMAGE_EXTENSIONS, load_bgr

MORPHOLOGY_CHOICES = ("auto", "smooth", "serrated", "lobed", "compound")


def _load_segmentation_mask(input_dir: Path, stem: str) -> np.ndarray | None:
    """Load Step-2 segmentation mask aligned with a white_bg crop."""
    import cv2

    masks_dir = input_dir.parent / "masks"
    for name in (f"{stem}_mask.png", f"{stem}.png"):
        path = masks_dir / name
        if path.is_file():
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
    return None


class _ModelCache:
    """Lazy cache of Contour U-Net specialists keyed by resolved checkpoint path."""

    def __init__(self, encoder: str, device: torch.device) -> None:
        self.encoder = encoder
        self.device = device
        self._models: dict[str, object] = {}

    def get(self, checkpoint: Path):
        key = str(checkpoint.resolve())
        if key not in self._models:
            print(f"[UNET-Shape] loading {checkpoint.name}")
            self._models[key] = load_unet_shape(
                checkpoint, encoder=self.encoder, device=self.device
            )
        return self._models[key]


def process_folder(
    input_dir: Path,
    output_dir: Path,
    default_checkpoint: Path,
    encoder: str,
    device: torch.device,
    img_size: int,
    threshold: float,
    white_thresh: int,
    refine: bool = True,
    bridge_max_growth: float = 0.08,
    morphology: str = "auto",
    file: str | None = None,
) -> int:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTENSIONS)
    if file:
        candidate = input_dir / file
        paths = [candidate] if candidate in paths else []
        if not paths:
            print(f"[UNET-Shape] Image not found in {input_dir}: {file}")
            return 0
    if not paths:
        print(f"[UNET-Shape] No images found in {input_dir}")
        return 0

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    cache = _ModelCache(encoder=encoder, device=device)
    # Preload the checkpoint that pipeline already selected for non-auto morph.
    if morphology != "auto":
        ckpt0 = resolve_unet_shape_checkpoint(morphology, fallback=default_checkpoint)
        cache.get(ckpt0)

    ok = 0
    for i, src in enumerate(paths, 1):
        stem = src.stem
        print(f"[UNET-Shape] ({i}/{len(paths)}) {src.name}")
        bgr = load_bgr(src)
        if bgr is None:
            print("  [skip] unreadable")
            continue

        morph_for_model = morphology
        if morphology == "auto":
            from contour.inference.gap_detector import classify_morphology

            partial = extract_partial_mask(bgr, threshold=white_thresh)
            if partial.max() == 0:
                print("  [skip] no leaf tissue detected")
                continue
            morph_for_model = classify_morphology(partial)

        ckpt = resolve_unet_shape_checkpoint(morph_for_model, fallback=default_checkpoint)
        model = cache.get(ckpt)

        seg_mask = _load_segmentation_mask(input_dir, stem)

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
                morphology=morphology,
                seg_mask=seg_mask,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {exc}")
            continue
        if res is None:
            print("  [skip] no leaf tissue detected")
            continue

        morph = res.get("morphology", "unknown")
        src_tag = res.get("morphology_source", "auto")
        print(f"  morph={morph} (source={src_tag})  ckpt={ckpt.name}")

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
    parser = argparse.ArgumentParser(description="UNET shape completion batch for Herbivora GUI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_UNET_SHAPE_MODEL),
    )
    parser.add_argument(
        "--config",
        default="contour/configs/config_train_f2lsm.yaml",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--file", default=None)
    parser.add_argument(
        "--morphology",
        default="auto",
        choices=list(MORPHOLOGY_CHOICES),
        help="Leaf type for reconstruction profiles (auto = classify from mask).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable morphology-aware post-refinement (serrated/lobed bridge).",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        # Fall back to default / specialist if the explicit path is missing.
        ckpt = resolve_unet_shape_checkpoint(args.morphology, fallback=DEFAULT_UNET_SHAPE_MODEL)
    if not ckpt.is_file():
        print(f"ERROR: UNET checkpoint not found: {args.checkpoint}")
        print("Download models with: python download_models.py")
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
    encoder = model_cfg.get("encoder", "resnet34")

    print(f"[UNET-Shape] morphology={args.morphology}  default_ckpt={ckpt.name}")
    n_masks = process_folder(
        Path(args.input),
        Path(args.output),
        default_checkpoint=ckpt,
        encoder=encoder,
        device=device,
        img_size=int(data_cfg.get("image_size", 512)),
        threshold=float(infer.get("threshold", 0.5)),
        white_thresh=int(infer.get("white_thresh", 240)),
        refine=bool(infer.get("refine", True)) and not args.no_refine,
        bridge_max_growth=float(infer.get("bridge_max_growth", 0.08)),
        morphology=args.morphology,
        file=args.file,
    )
    if n_masks == 0:
        print("[UNET-Shape] ERROR: no leaf masks were produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
