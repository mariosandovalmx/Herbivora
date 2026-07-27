"""Model loading and inference for BiRefNet_lite + MobileSAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import torch

# --------------------------------------------------------------------------- #
# Local model storage                                                         #
# BiRefNet HF cache lives under segmentation/birefnet_mobilesam/models/.      #
# MobileSAM weights live in the unified repo-root models/ folder.             #
# --------------------------------------------------------------------------- #

_PKG_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BIREFNET_CACHE_DIR = _PKG_MODELS_DIR / "hf_cache"
_BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet_lite"
MOBILESAM_WEIGHTS = _REPO_ROOT / "models" / "mobile_sam.pt"


def _birefnet_is_cached() -> bool:
    repo_dir = _BIREFNET_CACHE_DIR / f"models--{_BIREFNET_REPO_ID.replace('/', '--')}"
    return any(repo_dir.glob("snapshots/*/model.safetensors"))


# --------------------------------------------------------------------------- #
# Device selection                                                             #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# BiRefNet_lite                                                                #
# --------------------------------------------------------------------------- #

def load_birefnet(device: torch.device):
    """Load BiRefNet_lite, cached under segmentation/birefnet_mobilesam/models/hf_cache.

    Downloads from HuggingFace only the first time. Once cached, loads fully
    offline (local_files_only) with no network round-trip.
    """
    try:
        from transformers import AutoModelForImageSegmentation
    except ImportError as e:
        raise RuntimeError(
            "transformers package not found. "
            "Install with: pip install transformers huggingface_hub"
        ) from e

    _PKG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _BIREFNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _birefnet_is_cached()

    model = AutoModelForImageSegmentation.from_pretrained(
        _BIREFNET_REPO_ID,
        trust_remote_code=True,
        cache_dir=str(_BIREFNET_CACHE_DIR),
        local_files_only=cached,
    )
    model.to(device).eval()
    return model


def run_birefnet(image_bgr: np.ndarray, model,
                 size: int = 1024,
                 device: torch.device | None = None) -> np.ndarray:
    """Run BiRefNet_lite and return a boolean mask (H, W) at original resolution."""
    from torchvision import transforms
    from PIL import Image

    if device is None:
        device = get_device()

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)
        # BiRefNet returns a list of predictions; last one is finest
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        else:
            pred = preds
        pred = pred.sigmoid().cpu().squeeze()  # (size, size)

    mask_small = pred.numpy()
    mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return mask_full > 0.5


# --------------------------------------------------------------------------- #
# MobileSAM                                                                   #
# --------------------------------------------------------------------------- #

def load_mobilesam(device: torch.device, weights: str | Path | None = None):
    """Load MobileSAM via ultralytics. Returns the SAM model.

    Default weights: ``models/mobile_sam.pt`` at the repo root. Pass ``weights``
    to use a custom checkpoint.
    """
    try:
        from ultralytics import SAM
    except ImportError as e:
        raise RuntimeError(
            "ultralytics package not found. "
            "Install with: pip install ultralytics"
        ) from e

    if weights is not None and str(weights).strip():
        path = Path(str(weights).strip().strip("\"'"))
        if not path.is_file():
            raise FileNotFoundError(f"MobileSAM weights not found: {path}")
    else:
        path = MOBILESAM_WEIGHTS
        if not path.is_file():
            raise FileNotFoundError(
                f"MobileSAM weights not found: {path}\n"
                "Run: python download_models.py"
            )

    model = SAM(str(path))
    return model


def run_mobilesam_point(image_bgr: np.ndarray, model,
                        point: tuple[int, int]) -> np.ndarray:
    """Run MobileSAM with a single foreground point prompt.

    Returns a boolean mask (H, W) at original image resolution.
    """
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = model(
            image_rgb,
            points=[[list(point)]],
            labels=[[1]],
            verbose=False,
        )
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()  # (N, h, w)
            # Pick the largest mask
            areas = [m.sum() for m in masks_data]
            best_mask = masks_data[int(np.argmax(areas))]
            mask_resized = cv2.resize(
                best_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            return mask_resized.astype(bool)
    except Exception as e:
        print(f"[MobileSAM] inference error: {e}")

    # Fallback: entire image as positive
    return np.ones((H, W), dtype=bool)


def run_mobilesam_box(image_bgr: np.ndarray, model,
                      box: tuple[int, int, int, int]) -> np.ndarray:
    """Run MobileSAM with a bounding-box prompt.

    Box prompts are a much stronger geometric prior than a single point —
    a point can land on background clutter (soil, twigs) in messy scene
    photos, silently segmenting the wrong object. A box constrains SAM to
    "the object roughly inside this box", which is far more reliable when
    the crop isn't a clean, isolated leaf on a uniform background.

    Returns a boolean mask (H, W) at original image resolution. On error,
    falls back to the box region itself (never the whole image) so a SAM
    failure can't silently mark unrelated background as foreground.
    """
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = box

    try:
        results = model(
            image_rgb,
            bboxes=[[x1, y1, x2, y2]],
            verbose=False,
        )
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()  # (N, h, w)
            areas = [m.sum() for m in masks_data]
            best_mask = masks_data[int(np.argmax(areas))]
            mask_resized = cv2.resize(
                best_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            return mask_resized.astype(bool)
    except Exception as e:
        print(f"[MobileSAM] box inference error: {e}")

    # Fallback: the box region itself, never the whole frame.
    m = np.zeros((H, W), dtype=bool)
    m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m


# --------------------------------------------------------------------------- #
# Mask merging                                                                 #
# --------------------------------------------------------------------------- #

def merge_masks(M_bi: np.ndarray, M_sam: np.ndarray,
                mode: str, dilate_k: int = 15) -> np.ndarray:
    from .mask_utils import dilate_mask, refine_boundary

    if mode == "birefnet_primary":
        # BiRefNet edges restricted to the SAM-selected object
        return M_bi & dilate_mask(M_sam, k=dilate_k)
    elif mode == "mobilesam_primary":
        return refine_boundary(M_sam, M_bi)
    elif mode == "intersection":
        return M_bi & M_sam
    elif mode == "union":
        return M_bi | M_sam
    else:
        return M_bi & dilate_mask(M_sam, k=dilate_k)
