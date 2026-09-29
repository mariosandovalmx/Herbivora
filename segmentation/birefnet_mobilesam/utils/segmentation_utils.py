"""Model loading and inference for BiRefNet_lite + MobileSAM."""

from __future__ import annotations

import json
import os
import shutil
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

# Hub config for BiRefNet_lite must keep auto_map so transformers can load
# remote code. Older/partial caches sometimes only have weights + a bare
# config.json → "Should have a model_type key in its config.json".
_BIREFNET_AUTO_MAP = {
    "AutoConfig": "BiRefNet_config.BiRefNetConfig",
    "AutoModelForImageSegmentation": "birefnet.BiRefNet",
}
_BIREFNET_REMOTE_FILES = ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors")


def _birefnet_snapshot_dirs() -> list[Path]:
    repo_dir = _BIREFNET_CACHE_DIR / f"models--{_BIREFNET_REPO_ID.replace('/', '--')}"
    if not repo_dir.is_dir():
        return []
    return sorted(p for p in repo_dir.glob("snapshots/*") if p.is_dir())


def _config_supports_remote_birefnet(config_path: Path) -> bool:
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict):
        return False
    return all(auto_map.get(k) == v for k, v in _BIREFNET_AUTO_MAP.items())


def _birefnet_snapshot_is_complete(snapshot: Path) -> bool:
    try:
        if not all((snapshot / name).is_file() for name in _BIREFNET_REMOTE_FILES):
            return False
    except OSError:
        return False
    return _config_supports_remote_birefnet(snapshot / "config.json")


def _birefnet_is_cached() -> bool:
    return any(_birefnet_snapshot_is_complete(s) for s in _birefnet_snapshot_dirs())


def _repair_birefnet_configs() -> None:
    """Ensure local BiRefNet config.json can be loaded by modern transformers."""
    for snapshot in _birefnet_snapshot_dirs():
        config_path = snapshot / "config.json"
        try:
            if not config_path.is_file():
                continue
        except OSError:
            continue
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        auto_map = cfg.get("auto_map")
        if not isinstance(auto_map, dict):
            cfg["auto_map"] = dict(_BIREFNET_AUTO_MAP)
            changed = True
        else:
            for key, value in _BIREFNET_AUTO_MAP.items():
                if auto_map.get(key) != value:
                    auto_map[key] = value
                    changed = True
            cfg["auto_map"] = auto_map
        # Class attribute in BiRefNet_config.py; also write it into JSON so
        # transformers does not fall through to the "missing model_type" error
        # when remote-code resolution fails for any reason.
        if cfg.get("model_type") != "SegformerForSemanticSegmentation":
            cfg["model_type"] = "SegformerForSemanticSegmentation"
            changed = True
        if cfg.get("architectures") != ["BiRefNet"]:
            cfg["architectures"] = ["BiRefNet"]
            changed = True
        if changed:
            try:
                config_path.write_text(
                    json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                continue


def _invalidate_incomplete_birefnet_cache() -> None:
    """Drop incomplete snapshots so the next load re-downloads from the Hub."""
    for snapshot in _birefnet_snapshot_dirs():
        if not _birefnet_snapshot_is_complete(snapshot):
            shutil.rmtree(snapshot, ignore_errors=True)


_WINERROR_UNTRUSTED_MOUNT = 448


def _is_untrusted_mount_error(exc: BaseException) -> bool:
    if getattr(exc, "winerror", None) == _WINERROR_UNTRUSTED_MOUNT:
        return True
    text = str(exc).lower()
    return "untrusted mount point" in text or "punto de montaje no confiable" in text


def _disable_hf_cache_symlinks() -> None:
    """Ask huggingface_hub to copy files instead of creating Windows reparse points.

    HF_HUB_DISABLE_SYMLINKS is read from huggingface_hub.constants at import; we
    also set the module flag so a late call still wins after transformers loaded.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import constants as hf_c

        hf_c.HF_HUB_DISABLE_SYMLINKS = True
        hf_c.HF_HUB_DISABLE_SYMLINKS_WARNING = True
    except Exception:
        pass


def _materialize_snapshot_symlinks() -> None:
    """Replace Hugging Face snapshot symlinks with real files.

    Windows 11 (WinError 448) can refuse to traverse newly created reparse
    points under AppData. Reading the blob and rewriting a regular file avoids
    that for an already-downloaded cache.
    """
    for snapshot in _birefnet_snapshot_dirs():
        try:
            entries = list(snapshot.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                is_link = path.is_symlink()
            except OSError:
                is_link = True
            if not is_link:
                continue
            data: bytes | None = None
            try:
                data = path.read_bytes()
            except OSError:
                try:
                    target = os.readlink(path)
                    blob = Path(target) if os.path.isabs(target) else (path.parent / target)
                    data = blob.read_bytes()
                except OSError:
                    data = None
            if data is None:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            path.write_bytes(data)


# --------------------------------------------------------------------------- #
# Device selection                                                             #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    if os.name == "nt":
        _disable_hf_cache_symlinks()
        _materialize_snapshot_symlinks()

    _repair_birefnet_configs()
    cached = _birefnet_is_cached()

    def _from_pretrained(*, local_only: bool):
        return AutoModelForImageSegmentation.from_pretrained(
            _BIREFNET_REPO_ID,
            trust_remote_code=True,
            cache_dir=str(_BIREFNET_CACHE_DIR),
            local_files_only=local_only,
        )

    try:
        model = _from_pretrained(local_only=cached)
    except Exception as first_err:
        # Common failure: partial cache (weights present, config/auto_map missing)
        # with local_files_only=True. Repair config, invalidate junk, retry online.
        _repair_birefnet_configs()
        err_text = str(first_err)
        if "model_type" in err_text or "Unrecognized model" in err_text:
            _invalidate_incomplete_birefnet_cache()
        if os.name == "nt" and _is_untrusted_mount_error(first_err):
            _disable_hf_cache_symlinks()
            _materialize_snapshot_symlinks()
        try:
            model = _from_pretrained(local_only=False)
        except Exception as second_err:
            raise RuntimeError(
                "Failed to load BiRefNet_lite from Hugging Face "
                f"({_BIREFNET_REPO_ID}).\n"
                f"First error: {first_err}\n"
                f"Retry error: {second_err}\n"
                "Try deleting the folder "
                f"{_BIREFNET_CACHE_DIR} and run again with internet access."
            ) from second_err

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


def _resize_sam_mask(mask_hw: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize of a SAM mask to original image size."""
    return cv2.resize(
        mask_hw.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _pick_sam_mask(
    masks_resized: list[np.ndarray],
    point: tuple[int, int],
    *,
    single_object: bool,
) -> np.ndarray | None:
    """Choose one SAM candidate. Interactive clicks prefer the smallest object under the point."""
    if not masks_resized:
        return None
    px, py = point
    H, W = masks_resized[0].shape[:2]
    px = int(max(0, min(W - 1, px)))
    py = int(max(0, min(H - 1, py)))
    nonempty = [m for m in masks_resized if m.any()]
    if not nonempty:
        return None
    containing = [m for m in nonempty if bool(m[py, px])]
    pool = containing or nonempty
    if single_object:
        return min(pool, key=lambda m: int(m.sum()))
    return max(pool, key=lambda m: int(m.sum()))


def _flood_similar_in_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    *,
    tol: int = 22,
) -> np.ndarray:
    """Lab flood-fill from ``seed``, constrained to ``mask`` (OpenCV floodFill)."""
    H, W = mask.shape[:2]
    px = int(max(0, min(W - 1, seed[0])))
    py = int(max(0, min(H - 1, seed[1])))
    if not mask[py, px]:
        return np.zeros((H, W), dtype=bool)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    flood_mask[1 : H + 1, 1 : W + 1] = np.where(mask, 0, 1).astype(np.uint8)
    lo = (int(tol), int(tol), int(tol))
    hi = lo
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(lab.copy(), flood_mask, (px, py), 0, lo, hi, flags)
    return flood_mask[1 : H + 1, 1 : W + 1] == 255


def _bbox_xyxy(mask: np.ndarray, *, margin_frac: float = 0.12) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    H, W = mask.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(max(x2 - x1, y2 - y1, 1) * margin_frac)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(W, x2 + pad),
        min(H, y2 + pad),
    )


def run_mobilesam_point(
    image_bgr: np.ndarray,
    model,
    point: tuple[int, int],
    *,
    single_object: bool = False,
    negative_points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Run MobileSAM with a foreground point prompt.

    Returns a boolean mask (H, W) at original image resolution.

    ``single_object=True`` restricts the result to the object under the click
    (used by interactive segmentation when several leaves share a photo).
    ``negative_points`` are SAM background prompts (label=0), typically the
    click coordinates of other leaves already marked on the same photo.
    """
    from .mask_utils import isolate_clicked_object

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    px = int(max(0, min(W - 1, round(point[0]))))
    py = int(max(0, min(H - 1, round(point[1]))))

    pts = [[px, py]]
    lbls = [1]
    if negative_points:
        for nx, ny in negative_points:
            nxi = int(max(0, min(W - 1, round(nx))))
            nyi = int(max(0, min(H - 1, round(ny))))
            if nxi == px and nyi == py:
                continue
            pts.append([nxi, nyi])
            lbls.append(0)

    try:
        predict_kwargs: dict = {
            "points": [pts],
            "labels": [lbls],
            "verbose": False,
        }
        if single_object:
            # Keep lower-score (often smaller) masks; default conf=0.25 drops them.
            predict_kwargs["conf"] = 0.05
        else:
            # Reset predictor conf if a previous interactive call lowered it.
            predict_kwargs["conf"] = 0.25
        results = model(image_rgb, **predict_kwargs)
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()  # (N, h, w)
            resized = [_resize_sam_mask(m, W, H) for m in masks_data]
            best = _pick_sam_mask(resized, (px, py), single_object=single_object)
            if best is None:
                raise RuntimeError("MobileSAM returned empty masks")
            if not single_object:
                return best

            isolated = isolate_clicked_object(best, px, py)
            sam_area = int(isolated.sum())
            if sam_area <= 0:
                return isolated

            flood = _flood_similar_in_mask(image_bgr, isolated, (px, py))
            flood_area = int(flood.sum())
            # Flood much smaller than SAM → SAM likely swallowed a neighbour
            # (e.g. two green leaves on a petri dish). Re-run with a box.
            if 0.12 * sam_area < flood_area < 0.80 * sam_area:
                box = _bbox_xyxy(flood, margin_frac=0.18)
                boxed = run_mobilesam_box(image_bgr, model, box)
                boxed = isolate_clicked_object(boxed, px, py)
                boxed_area = int(boxed.sum())
                if boxed_area > 0 and boxed_area <= sam_area:
                    return boxed
                if flood_area > 0:
                    return flood
            return isolated
    except Exception as e:
        print(f"[MobileSAM] inference error: {e}")

    if single_object:
        return np.zeros((H, W), dtype=bool)
    # Automatic pipeline fallback: entire image as positive
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
