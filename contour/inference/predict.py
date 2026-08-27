"""Mask-to-Mask UNET shape completion inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "segmentation-models-pytorch is required: pip install segmentation-models-pytorch"
    ) from exc


def extract_partial_mask(
    bgr: np.ndarray,
    threshold: int = 240,
    *,
    gentle: bool = False,
) -> np.ndarray:
    """Binary leaf mask (255=visible tissue) from a white-background crop."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if gentle:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.zeros_like(mask)
    clean[labels == largest] = 255
    return clean


def load_unet_shape(
    checkpoint: Path,
    encoder: str = "resnet34",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a 1-channel Mask-to-Mask U-Net trained with train_unet.py."""
    dev = torch.device(device)
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    model.load_state_dict(state)
    model.to(dev).eval()

    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    return model


@torch.inference_mode()
def reconstruct_mask(
    bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
    *,
    img_size: int = 512,
    threshold: float = 0.5,
    white_thresh: int = 240,
    amp: bool = True,
    refine: bool = True,
    bridge_max_growth: float | None = None,
    morphology: str | None = None,
    seg_mask: np.ndarray | None = None,
) -> dict | None:
    """Complete a damaged leaf silhouette from a white-background RGB crop.

    Parameters
    ----------
    morphology
        User-selected leaf type (``smooth`` / ``serrated`` / ``lobed`` /
        ``compound``), or ``None`` / ``\"auto\"`` to classify from the mask.
    """
    from contour.inference.gap_detector import classify_morphology
    from contour.inference.mask_refine import (
        get_morphology_profile,
        refine_unet_mask,
    )

    partial = extract_partial_mask(bgr, threshold=white_thresh)
    if partial.max() == 0:
        return None

    if seg_mask is not None and seg_mask.shape[:2] == partial.shape[:2]:
        from contour.inference.mask_refine import _clip_to_segmentation_roi

        partial_bool = partial > 0
        clipped = _clip_to_segmentation_roi(partial_bool, seg_mask, dilate_px=3)
        partial = (clipped.astype(np.uint8)) * 255
        if partial.max() == 0:
            return None

    user_morph = (morphology or "auto").strip().lower()
    if user_morph in ("", "auto", "none"):
        resolved = classify_morphology(partial)
        morph_source = "auto"
    else:
        resolved = user_morph
        morph_source = "user"

    profile = get_morphology_profile(resolved)
    if profile.get("gentle_partial"):
        partial = extract_partial_mask(bgr, threshold=white_thresh, gentle=True)

    h0, w0 = partial.shape[:2]
    resized = cv2.resize(partial, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None].to(device)

    use_cuda = device.type == "cuda"
    if use_cuda:
        x = x.to(memory_format=torch.channels_last)

    with torch.autocast(device_type=device.type, enabled=(amp and use_cuda)):
        logits = model(x)

    prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8) * 255

    if pred.shape != (h0, w0):
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
        pred = (pred > 127).astype(np.uint8) * 255

    # Preserve all tissue that was already visible in the damaged leaf.
    full = cv2.bitwise_or(pred, partial)

    if refine:
        growth = (
            float(bridge_max_growth)
            if bridge_max_growth is not None
            else float(profile["bridge_max_growth"])
        )
        full, resolved = refine_unet_mask(
            full,
            partial,
            bgr,
            morphology=resolved,
            bridge_max_growth=growth,
            white_thresh=white_thresh,
            seg_mask=seg_mask,
        )

    new_pixels = (full > 0) & (partial == 0)
    if new_pixels.sum() > 0:
        confidence = float(prob.max() if prob.size else 0.5)
    else:
        confidence = 0.99

    return {
        "mask": full,
        "partial_mask": partial,
        "prob_map": prob,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "morphology": resolved,
        "morphology_source": morph_source,
    }
