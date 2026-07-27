"""
Step 5: Leaf-UNet inference — leaf mask + post-processing and background normalization.

Usage:
    python leaf_contour/predict_leaf_roi.py --input test --output leaf_roi_test --save-raw
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_LEAF_DIR = Path(__file__).resolve().parent
_ROOT = _LEAF_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LEAF_DIR))

from whitebg_masks import VALID_EXT  # noqa: E402
from bg_normalize import normalize_white_background_bgr  # noqa: E402
from leaf_mask_postprocess import refine_leaf_mask  # noqa: E402
from overlay_viz import overlay_leaf  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "resnet34"
REFINE_CHOICES = ("raw", "mask", "closed", "hull", "complete", "bridged", "perforated")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(path: Path, num_classes: int = 2):
    """Load model, auto-detect 1ch (sigmoid) vs 2ch (softmax). Applies GPU optimizations."""
    import segmentation_models_pytorch as smp

    state = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    # Detect output channels from state_dict
    out_channels = num_classes
    for k, v in state.items():
        if "segmentation_head" in k and "weight" in k and v.dim() >= 1:
            out_channels = v.shape[0]
            break

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=out_channels,
        activation=None,
    )
    model.load_state_dict(state)
    model._out_channels = out_channels
    model = model.to(DEVICE).eval()

    # GPU-specific optimizations
    if DEVICE.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model


def preprocess_rgb(rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize + ImageNet-normalize a single RGB image → (3, H, W) float32 tensor."""
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = resized.astype(np.float32) / 255.0
    t = (t - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1))  # HWC → CHW


@torch.inference_mode()
def predict_batch(
    model,
    rgb_list: list[np.ndarray],
    sizes: list[tuple[int, int]],
    img_size: int,
    amp: bool = True,
) -> list[np.ndarray]:
    """Batched U-Net inference with AMP. Returns list of uint8 predictions at original sizes."""
    tensors = [preprocess_rgb(rgb, img_size) for rgb in rgb_list]
    batch = torch.stack(tensors).to(DEVICE, non_blocking=True)
    use_cuda = DEVICE.startswith("cuda")
    if use_cuda:
        batch = batch.to(memory_format=torch.channels_last)

    with torch.autocast(device_type=DEVICE.split(":")[0], enabled=amp and use_cuda):
        out = model(batch)

    out_ch = getattr(model, "_out_channels", out.shape[1])

    # Sigmoid/argmax over the full batch on GPU → one CPU transfer instead of N
    if out_ch == 1:
        pred_batch = (out[:, 0].float().sigmoid() > 0.5).to(torch.uint8).mul(255)
    else:
        pred_batch = out.float().argmax(dim=1).to(torch.uint8).mul(255)
    pred_np = pred_batch.cpu().numpy()  # (B, H, W) uint8 0/255

    predictions = []
    for i, (h0, w0) in enumerate(sizes):
        pred = pred_np[i]
        if pred.shape != (h0, w0):
            # INTER_LINEAR on a 0/255 mask gives smooth edges; re-binarize at midpoint
            pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
            pred = (pred > 127).astype(np.uint8)
        predictions.append(pred)

    return predictions


# Legacy single-image wrapper (kept for external callers)
def predict_mask(model, rgb: np.ndarray, img_size: int) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    return predict_batch(model, [rgb], [(h0, w0)], img_size)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict leaf lamina mask (Leaf-UNet).")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--file", type=str, default=None, help="Process only this specific file if provided")
    parser.add_argument("--output", type=Path, default=_ROOT / "leaf_roi_test")
    parser.add_argument("--model", type=Path, default=_ROOT / "best_leaf_model.pth")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument(
        "--refine",
        choices=REFINE_CHOICES,
        default="complete",
        help="raw=raw prediction; complete=holes+closing+color; bridged=bridge edge damage (for highly curved/damaged leaves).",
    )
    parser.add_argument(
        "--bridge-max-growth",
        type=float,
        default=0.6,
        help="Bridged mode: maximum area expansion ratio (0.6 = 60%%).",
    )
    parser.add_argument("--erode", type=int, default=1, help="Erosion after closing (reduces white boundary halo).")
    parser.add_argument("--close-divisor", type=float, default=12.0)
    parser.add_argument(
        "--normalize-bg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize gray/creamy background to white before running the model (default ON).",
    )
    parser.add_argument(
        "--no-clip-color",
        action="store_true",
        help="Do not clip the prediction mask using the background threshold silhouette.",
    )
    parser.add_argument(
        "--no-clip-paper-halo",
        action="store_true",
        help="Do not remove paper page borders/halos on the outer edge.",
    )
    parser.add_argument("--white-thresh", type=int, default=248)
    parser.add_argument("--bg-distance", type=float, default=42.0)
    parser.add_argument(
        "--smooth-contour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth final contour coordinates (default ON).",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw mask and orange overlay representing unrefined predictions.",
    )
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Save a copy of the normalized background BGR image under output/normalized/.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for GPU inference (default: 16). Higher = faster but uses more VRAM.",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    masks_dir = out / "masks"
    overlays_dir = out / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_raw_dir = out / "masks_raw" if args.save_raw else None
    norm_dir = out / "normalized" if args.save_normalized else None
    if masks_raw_dir is not None:
        masks_raw_dir.mkdir(parents=True, exist_ok=True)
    if norm_dir is not None:
        norm_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}. Place best_leaf_model.pth in the project root.")

    model = load_model(args.model.resolve())
    if args.file:
        images = [inp / args.file] if (inp / args.file).is_file() else []
    else:
        images = sorted(p for p in inp.iterdir() if p.suffix.lower() in VALID_EXT)
    clip_color = not args.no_clip_color
    clip_paper = not args.no_clip_paper_halo
    use_cuda = DEVICE.startswith("cuda")
    amp_tag = "AMP fp16" if use_cuda else "fp32"
    print(
        f"Model: {args.model.name}  |  refine={args.refine}  "
        f"|  norm_bg={args.normalize_bg}  |  clip_color={clip_color}  "
        f"|  batch={args.batch_size}  |  {amp_tag}  |  n={len(images)}"
    )

    # ── Process images in batches ────────────────────────────────────────────
    batch_items: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []

    pbar = tqdm(total=len(images), desc="Leaf-UNet", unit="img")

    def _flush_batch() -> None:
        if not batch_items:
            return
        rgb_list = [item[3] for item in batch_items]
        sizes = [(rgb.shape[0], rgb.shape[1]) for rgb in rgb_list]
        preds = predict_batch(model, rgb_list, sizes, args.img_size, amp=use_cuda)

        for (img_path, bgr_orig, bgr, _rgb), pred_raw in zip(batch_items, preds):
            raw_bool = pred_raw > 0
            leaf_bool = refine_leaf_mask(
                pred_raw,
                mode=args.refine,
                close_divisor=args.close_divisor,
                erode_px=args.erode,
                bgr=bgr if (clip_color or clip_paper) else None,
                clip_color=clip_color,
                clip_paper_halo=clip_paper,
                white_thresh=args.white_thresh,
                smooth_contour=args.smooth_contour,
                bridge_max_growth=args.bridge_max_growth,
            )
            stem = img_path.stem
            cv2.imwrite(str(masks_dir / f"{stem}_leaf_mask.png"), leaf_bool.astype(np.uint8) * 255)
            if masks_raw_dir is not None:
                cv2.imwrite(str(masks_raw_dir / f"{stem}_leaf_mask_raw.png"), raw_bool.astype(np.uint8) * 255)

            vis = overlay_leaf(bgr_orig, leaf_bool, raw_bool if args.save_raw else None)
            cv2.imwrite(str(overlays_dir / f"{stem}_leaf_overlay.jpg"), vis)

            area_raw = 100.0 * raw_bool.sum() / raw_bool.size
            area_ref = 100.0 * leaf_bool.sum() / leaf_bool.size
            pbar.set_postfix_str(f"raw={area_raw:.1f}% final={area_ref:.1f}%")

        batch_items.clear()

    def _load_one(img_path: Path):
        """Load + preprocess one image (runs in thread pool, parallel to GPU inference)."""
        bgr_orig = cv2.imread(str(img_path))
        if bgr_orig is None:
            return None
        bgr = bgr_orig
        if args.normalize_bg:
            bgr, _ = normalize_white_background_bgr(bgr, distance_thresh=args.bg_distance)
        if norm_dir is not None:
            cv2.imwrite(str(norm_dir / img_path.name), bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return img_path, bgr_orig, bgr, rgb

    # Prefetch images in background threads while GPU processes current batch.
    # max_workers=4 covers typical disk + CPU preprocessing overlap without excess memory.
    n_workers = min(4, max(1, len(images)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_load_one, images):
            pbar.update(1)
            if result is None:
                continue
            batch_items.append(result)
            if len(batch_items) >= args.batch_size:
                _flush_batch()

    _flush_batch()  # remaining partial batch
    pbar.close()

    print(f"\nOutput: {out}/")
    print("  overlays/  -> green contour overlay on ORIGINAL image")
    print("  masks/     -> final refined mask")
    if args.normalize_bg:
        print("  (inference is run on normalized background; overlay shows original for comparison)")


if __name__ == "__main__":
    main()
