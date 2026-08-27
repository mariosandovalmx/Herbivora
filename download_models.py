#!/usr/bin/env python3
"""Download Herbivora model weights into ./models/.

Herbivora-trained U-Nets come from Hugging Face Hub.
MobileSAM is fetched from the official Ultralytics assets release (Apache-2.0),
not re-hosted in the Herbivora Hub repo.

Usage:
    python download_models.py
    python download_models.py --repo mariosandovalmx/Herbivora
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _default_models_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models"
    return Path(__file__).resolve().parent / "models"


REPO_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
MODELS_DIR = _default_models_dir()
DEFAULT_REPO = "mariosandovalmx/Herbivora"

# Weights trained / packaged for Herbivora (hosted on the Hub).
HERBIVOR_FILES = (
    "best_unet_shape.pth",
    "best_model.pth",
    "best_unet_shape_smooth.pth",
    "best_unet_shape_serrated.pth",
    "best_unet_shape_lobed.pth",
    "best_unet_shape_compound.pth",
)

# Third-party MobileSAM weights (do not re-upload to Herbivora Hub).
MOBILESAM_NAME = "mobile_sam.pt"
MOBILESAM_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt"
)

LogFn = Callable[[str], None]


def ssl_context() -> ssl.SSLContext:
    """Verified TLS context using certifi when Python has no system CA file.

    The python.org macOS builds can report ``cafile=None`` until their separate
    certificate installer has been run. Herbivora already depends on certifi via
    its HTTP packages, so use that maintained CA bundle without weakening TLS.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError):
        return ssl.create_default_context()


@dataclass
class EnsureModelsResult:
    """Outcome of :func:`ensure_models`."""

    ok: int = 0
    total: int = 0
    lines: list[str] = field(default_factory=list)
    unverified: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.ok == self.total and not self.errors


def remote_size(repo_id: str, filename: str) -> int | None:
    """Expected byte size on the Hub, or None when it cannot be queried."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        meta = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=filename))
    except Exception:
        return None
    return meta.size


def url_size(url: str) -> int | None:
    """Content-Length from a HEAD request, or None if unavailable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None


def check_file(dest: Path, expected: int | None) -> str | None:
    """Reason the local file is unusable, or None when it looks complete."""
    try:
        actual = dest.stat().st_size
    except OSError as e:
        return str(e)
    if actual == 0:
        return "file is empty"
    if expected is not None and actual != expected:
        return f"{actual:,} bytes on disk, expected {expected:,}"
    return None


def download_url(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (atomic replace via temp file)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Herbivora model downloader"})
        with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as response:
            with tmp.open("wb") as output:
                shutil.copyfileobj(response, output)
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def _emit(log: LogFn | None, result: EnsureModelsResult, msg: str) -> None:
    result.lines.append(msg)
    if log is not None:
        log(msg)
    else:
        print(msg)


def _ensure_one_url(
    *,
    name: str,
    url: str,
    dest: Path,
    force: bool,
    result: EnsureModelsResult,
    log: LogFn | None,
) -> None:
    expected = url_size(url)
    if expected is None:
        result.unverified += 1
    if dest.is_file() and not force:
        problem = check_file(dest, expected)
        if problem is None:
            _emit(log, result, f"  [skip] {name} (already present)")
            result.ok += 1
            return
        _emit(log, result, f"  [redo] {name}: {problem} - re-downloading")
    try:
        download_url(url, dest)
    except Exception as e:
        msg = f"  [FAIL] {name}: {e}"
        _emit(log, result, msg)
        result.errors.append(msg)
        return
    problem = check_file(dest, expected)
    if problem is not None:
        msg = f"  [FAIL] {name}: incomplete download ({problem})"
        _emit(log, result, msg)
        result.errors.append(msg)
        return
    _emit(log, result, f"  [ok]   {name}")
    result.ok += 1


def _ensure_one_hub(
    *,
    name: str,
    repo_id: str,
    dest: Path,
    force: bool,
    result: EnsureModelsResult,
    log: LogFn | None,
) -> None:
    from huggingface_hub import hf_hub_download

    expected = remote_size(repo_id, name)
    if expected is None:
        result.unverified += 1
    if dest.is_file() and not force:
        problem = check_file(dest, expected)
        if problem is None:
            _emit(log, result, f"  [skip] {name} (already present)")
            result.ok += 1
            return
        _emit(log, result, f"  [redo] {name}: {problem} - re-downloading")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=name,
            local_dir=str(dest.parent),
            force_download=force,
        )
    except Exception as e:
        msg = f"  [FAIL] {name}: {e}"
        _emit(log, result, msg)
        result.errors.append(msg)
        return
    problem = check_file(Path(path), expected)
    if problem is not None:
        msg = f"  [FAIL] {name}: incomplete download ({problem})"
        _emit(log, result, msg)
        result.errors.append(msg)
        return
    _emit(log, result, f"  [ok]   {Path(path).name}")
    result.ok += 1


def ensure_models(
    *,
    repo: str = DEFAULT_REPO,
    models_dir: Path | None = None,
    force: bool = False,
    log: LogFn | None = None,
) -> EnsureModelsResult:
    """Ensure MobileSAM + Herbivora U-Nets exist under ``models_dir``.

    Skips files that are already present and match the remote size (unless
    ``force``). Safe to call from the GUI or CLI.
    """
    out = models_dir if models_dir is not None else MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    result = EnsureModelsResult(total=len(HERBIVOR_FILES) + 1)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        msg = "ERROR: huggingface_hub is required.  pip install huggingface_hub"
        _emit(log, result, msg)
        result.errors.append(msg)
        return result

    _emit(log, result, f"Downloading MobileSAM from Ultralytics assets -> {out}")
    _ensure_one_url(
        name=MOBILESAM_NAME,
        url=MOBILESAM_URL,
        dest=out / MOBILESAM_NAME,
        force=force,
        result=result,
        log=log,
    )

    _emit(log, result, f"Downloading Herbivora models from {repo} -> {out}")
    for name in HERBIVOR_FILES:
        _ensure_one_hub(
            name=name,
            repo_id=repo,
            dest=out / name,
            force=force,
            result=result,
            log=log,
        )

    _emit(log, result, f"Done: {result.ok}/{result.total} models ready in {out}")
    if result.unverified:
        _emit(
            log,
            result,
            f"WARNING: could not verify remote size for {result.unverified} file(s); "
            "they were only checked for being non-empty.",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Herbivora model weights")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hugging Face model repo id for Herbivora U-Nets (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()
    result = ensure_models(repo=args.repo, force=args.force)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
