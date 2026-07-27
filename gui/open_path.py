"""Cross-platform helpers for opening folders/files in the system file manager."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_path(path: str | Path) -> None:
    """Open a file or folder with the OS default application / file manager."""
    target = str(path)
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", target], check=False)
    else:
        subprocess.run(["xdg-open", target], check=False)
