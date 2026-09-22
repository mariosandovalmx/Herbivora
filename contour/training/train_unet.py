"""Fine-tune the mask-to-mask contour U-Net (Entire / smooth specialist).

Pairs damaged binary masks with intact ground-truth masks at 512 px.
Warm-starts from the path in the YAML config (do not overwrite production
until the mixed large+small test is accepted).

Usage (from repo root):
    python contour/scripts/prepare_entire_finetune.py
    python contour/training/train_unet.py --config contour/configs/config_finetune_entire.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise ImportError("pip install albumentations") from exc

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError("pip install segmentation-models-pytorch") from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_kw):
        return x

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "contour" / "configs" / "config_finetune_entire.yaml"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_augmentations(img_size: int, training: bool) -> A.Compose:
    if training:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomScale(scale_limit=0.15, p=0.4),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                A.CenterCrop(img_size, img_size),
                A.ElasticTransform(alpha=30, sigma=5, p=0.2),
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(img_size, img_size), ToTensorV2()])


def _clean_damaged_stem(stem: str) -> str:
    for i in range(40):
        suffix = f"_v{i:02d}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _leaf_id_from_damaged_stem(stem: str) -> str:
    """Strip variant, geometric and folder prefixes so copies stay with the same leaf."""
    s = _clean_damaged_stem(stem)
    for suf in ("_r90", "_r180", "_r270", "_fh", "_fv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    for pre in ("big_", "small_"):
        if s.startswith(pre):
            return s[len(pre) :]
    return s


def _split_pairs_by_leaf(
    pairs: list[tuple[Path, Path]],
    val_split: float,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pair in pairs:
        groups[_leaf_id_from_damaged_stem(pair[0].stem)].append(pair)
    leaf_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(leaf_ids)
    n_val = max(1, int(round(len(leaf_ids) * val_split)))
    n_val = min(n_val, len(leaf_ids) - 1) if len(leaf_ids) > 1 else 1
    val_ids = set(leaf_ids[:n_val])
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    for lid, items in groups.items():
        (val_pairs if lid in val_ids else train_pairs).extend(items)
    return train_pairs, val_pairs


def build_mask_pairs(
    dataset_dir: Path,
    damaged_subdir: str,
    clean_subdir: str,
) -> list[tuple[Path, Path]]:
    damaged_dir = dataset_dir / damaged_subdir
    clean_dir = dataset_dir / clean_subdir
    if not damaged_dir.is_dir() or not clean_dir.is_dir():
        raise RuntimeError(
            f"Missing mask folders.\n  damaged: {damaged_dir}\n  clean: {clean_dir}\n"
            "Run: python contour/scripts/prepare_entire_finetune.py"
        )
    sample_files = sorted(
        p for p in damaged_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT
    )
    clean_map = {
        cp.stem: cp
        for cp in clean_dir.iterdir()
        if cp.is_file() and cp.suffix.lower() in VALID_EXT
    }
    pairs: list[tuple[Path, Path]] = []
    for dp in sample_files:
        clean_path = clean_map.get(_clean_damaged_stem(dp.stem))
        if clean_path is not None:
            pairs.append((dp, clean_path))
    if not pairs:
        raise RuntimeError(f"No (damaged, clean) pairs under {dataset_dir}")
    return pairs


def _dataset_dirs(data_cfg: dict) -> list[Path]:
    raw = data_cfg.get("dataset_dirs") or data_cfg.get("dataset_dir")
    if raw is None:
        raise RuntimeError("Set data.dataset_dir or data.dataset_dirs in the YAML config.")
    if isinstance(raw, (str, Path)):
        raw = [raw]
    dirs = [Path(p).resolve() for p in raw]
    missing = [p for p in dirs if not p.is_dir()]
    if missing:
        raise RuntimeError("Dataset folder(s) not found:\n  " + "\n  ".join(str(p) for p in missing))
    return dirs


def build_mask_pairs_many(
    dataset_dirs: list[Path],
    damaged_subdir: str,
    clean_subdir: str,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for dataset_dir in dataset_dirs:
        folder_pairs = build_mask_pairs(dataset_dir, damaged_subdir, clean_subdir)
        print(f"[train] {dataset_dir.name}: {len(folder_pairs)} pairs")
        pairs.extend(folder_pairs)
    if not pairs:
        raise RuntimeError("No (damaged, clean) pairs in the listed dataset folders.")
    return pairs


class LeafShapeDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], img_size: int, training: bool) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.aug = _build_augmentations(img_size, training)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        damaged_path, clean_path = self.pairs[idx]
        damaged = cv2.imread(str(damaged_path), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        if damaged is None or clean is None:
            z = torch.zeros(1, self.img_size, self.img_size)
            return z, z
        _, damaged = cv2.threshold(damaged, 127, 255, cv2.THRESH_BINARY)
        _, clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
        result = self.aug(image=damaged, mask=clean)
        x = result["image"].float() / 255.0
        y = result["mask"].float() / 255.0
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        elif y.ndim == 3 and y.shape[0] != 1:
            y = y[:1]
        return x, y


class BoundaryWeightedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.4, boundary_weight: float = 4.0, boundary_kernel: int = 5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.boundary_kernel = boundary_kernel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _boundary_map(self, targets: torch.Tensor) -> torch.Tensor:
        k = self.boundary_kernel
        kernel = torch.ones(1, 1, k, k, device=targets.device)
        dilated = torch.clamp(torch.nn.functional.conv2d(targets, kernel, padding=k // 2), 0, 1)
        eroded = 1 - torch.clamp(torch.nn.functional.conv2d(1 - targets, kernel, padding=k // 2), 0, 1)
        boundary = (dilated - eroded).clamp(0, 1)
        return 1.0 + (self.boundary_weight - 1.0) * boundary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = (self.bce(logits, targets) * self._boundary_map(targets)).mean()
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


@torch.inference_mode()
def compute_metrics(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str, amp: bool) -> dict[str, float]:
    model.eval()
    use_cuda = device.startswith("cuda")
    loss_accum = torch.tensor(0.0, device=device)
    iou_accum = torch.tensor(0.0, device=device)
    n_batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_cuda:
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp and use_cuda):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss_accum += loss.detach()
        preds = (logits.float().sigmoid() > 0.5).float()
        inter = (preds * y).sum(dim=(2, 3))
        union = (preds + y).clamp(0, 1).sum(dim=(2, 3))
        iou_accum += (inter / (union + 1e-6)).mean()
        n_batches += 1
    n = max(n_batches, 1)
    return {"loss": (loss_accum / n).item(), "iou": (iou_accum / n).item()}


def build_model(encoder: str, in_channels: int, out_channels: int) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )


def _unwrap_state_dict(model: nn.Module) -> dict:
    sd = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def _load_weights(model: nn.Module, path: Path, device: str) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if isinstance(raw, dict) and any(k.startswith("_orig_mod.") for k in raw):
        raw = {k.replace("_orig_mod.", "", 1): v for k, v in raw.items()}
    model.load_state_dict(raw, strict=False)


def train(cfg: dict, resume: bool, device_override: str | None) -> None:
    d, m, t, p = cfg["data"], cfg["model"], cfg["training"], cfg["paths"]
    dataset_dirs = _dataset_dirs(d)
    img_size = int(d["image_size"])
    val_split = float(d.get("val_split", 0.15))
    epochs = int(t.get("epochs", 40))
    batch_size = int(t.get("batch_size", 8))
    lr = float(t.get("lr", 1e-4))
    weight_decay = float(t.get("weight_decay", 1e-4))
    patience_max = int(t.get("early_stop_patience", 12))
    use_amp = bool(t.get("amp", True))
    num_workers = int(t.get("num_workers", 0))
    boundary_kernel = int(t.get("boundary_kernel", 5))

    save_best = REPO_ROOT / p["save_best"]
    save_last = REPO_ROOT / p["save_last"]
    save_best.parent.mkdir(parents=True, exist_ok=True)

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.startswith("cuda")

    all_pairs = build_mask_pairs_many(dataset_dirs, d["damaged_masks_subdir"], d["clean_masks_subdir"])
    n_total = len(all_pairs)
    train_pairs, val_pairs = _split_pairs_by_leaf(all_pairs, val_split, seed=42)

    train_ds = LeafShapeDataset(train_pairs, img_size, training=True)
    val_ds = LeafShapeDataset(val_pairs, img_size, training=False)
    lkw = dict(num_workers=num_workers, pin_memory=use_cuda, persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **lkw)
    val_loader = DataLoader(val_ds, batch_size=max(2, batch_size * 2), shuffle=False, **lkw)

    model = build_model(m.get("encoder", "resnet34"), int(m.get("in_channels", 1)), int(m.get("out_channels", 1)))
    model = model.to(device)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = BoundaryWeightedBCEDiceLoss(boundary_kernel=boundary_kernel)
    scaler = torch.amp.GradScaler("cuda" if use_cuda else "cpu", enabled=(use_amp and use_cuda))

    start_epoch = 0
    best_iou = 0.0
    patience = 0

    warm = t.get("warm_start")
    warm_path = (REPO_ROOT / warm) if warm and not Path(warm).is_absolute() else (Path(warm) if warm else None)
    if warm_path and warm_path.is_file() and not resume:
        _load_weights(model, warm_path, device)
        print(f"[train] warm-start {warm_path}")
    elif warm_path and not warm_path.is_file():
        print(f"[train] WARNING: warm-start not found: {warm_path}  (training from scratch)")

    if resume and save_last.is_file():
        ckpt = torch.load(save_last, map_location=device, weights_only=False)
        _load_weights(model, save_last, device)
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_iou = float(ckpt.get("best_iou", 0.0))
        patience = int(ckpt.get("patience", 0))
        print(f"[train] resumed epoch {start_epoch}  best_iou={best_iou:.4f}")

    print(
        f"[train] device={device}  pairs={n_total} train={len(train_pairs)} val={len(val_pairs)}  "
        f"batch={batch_size}  epochs={epochs}  lr={lr}"
    )
    print(f"[train] save_best={save_best}")

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss = torch.tensor(0.0, device=device)
        t0 = time.time()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_cuda:
                x = x.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], enabled=(use_amp and use_cuda)):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.detach()
        scheduler.step()
        avg_loss = (ep_loss / max(len(train_loader), 1)).item()
        val_m = compute_metrics(model, val_loader, loss_fn, device, use_amp)
        print(
            f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
            f"val_loss={val_m['loss']:.4f}  val_iou={val_m['iou']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.0f}s"
        )
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            patience = 0
            torch.save(_unwrap_state_dict(model), save_best)
            print(f"  [best] val_iou={best_iou:.4f} -> {save_best}")
        else:
            patience += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state": _unwrap_state_dict(model),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_iou": best_iou,
                "patience": patience,
                "cfg": cfg,
            },
            save_last,
        )
        if patience >= patience_max:
            print(f"\nEarly stopping after {patience_max} epochs without IoU improvement.")
            break
    print(f"\nDone. best val_iou={best_iou:.4f}\nSaved: {save_best}")
    print(
        "Do not copy over models/best_unet_shape_smooth.pth until the mixed "
        "large+small hole test is accepted."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Entire contour U-Net.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_file() else REPO_ROOT / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume, device_override=args.device)


if __name__ == "__main__":
    main()
