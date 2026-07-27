"""Entry point: python -m gui.main"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is in sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


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
