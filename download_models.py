#!/usr/bin/env python3
"""Download HerbivoR model weights from Hugging Face Hub into ./models/.

Usage:
    python download_models.py
    python download_models.py --repo mariosandovalmx/HerbivoR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPO = "mariosandovalmx/HerbivoR"
FILES = (
    "mobile_sam.pt",
    "best_unet_shape.pth",
    "best_model.pth",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HerbivoR model weights")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hugging Face model repo id (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required.")
        print("  pip install huggingface_hub")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models from {args.repo} → {MODELS_DIR}")
    ok = 0
    for name in FILES:
        dest = MODELS_DIR / name
        if dest.is_file() and not args.force:
            print(f"  [skip] {name} (already present)")
            ok += 1
            continue
        try:
            path = hf_hub_download(
                repo_id=args.repo,
                filename=name,
                local_dir=str(MODELS_DIR),
            )
            print(f"  [ok]   {Path(path).name}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    print(f"Done: {ok}/{len(FILES)} models ready in {MODELS_DIR}")
    return 0 if ok == len(FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
