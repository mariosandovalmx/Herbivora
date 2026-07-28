"""PyInstaller entry: launch HerbivoR GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# App root (writable) + MEIPASS (bundled code)
if getattr(sys, "frozen", False):
    app_dir = Path(sys.executable).resolve().parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    if hasattr(sys, "_MEIPASS"):
        meipass = str(sys._MEIPASS)
        if meipass not in sys.path:
            sys.path.insert(0, meipass)
else:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gui.main import main

if __name__ == "__main__":
    main()
