"""Entry point: python -m gui.main"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


# Ensure the repo / app root is in sys.path
_REPO = _repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# Frozen: also expose bundled package root
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    meipass = str(sys._MEIPASS)
    if meipass not in sys.path:
        sys.path.insert(0, meipass)


def main() -> None:
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print(
            "Missing customtkinter. Run install.bat / install.sh or:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    from gui.app import HerbivoRApp

    app = HerbivoRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
