"""Entry point: python -m gui.main"""

from __future__ import annotations

import os
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


def _ensure_tcl_tk_env() -> None:
    """Point TCL_LIBRARY/TK_LIBRARY at a usable init.tcl when discovery fails on Windows."""
    base = Path(getattr(sys, "base_prefix", sys.prefix))
    tcl_env = os.environ.get("TCL_LIBRARY")
    tk_env = os.environ.get("TK_LIBRARY")
    if (
        tcl_env
        and tk_env
        and (Path(tcl_env) / "init.tcl").is_file()
        and (Path(tk_env) / "tk.tcl").is_file()
    ):
        return

    pairs = [
        (base / "tcl" / "tcl8.6", base / "tcl" / "tk8.6"),
        (base / "lib" / "tcl8.6", base / "lib" / "tk8.6"),
        (base / "Library" / "lib" / "tcl8.6", base / "Library" / "lib" / "tk8.6"),
    ]
    for tcl_dir, tk_dir in pairs:
        if (tcl_dir / "init.tcl").is_file() and (tk_dir / "tk.tcl").is_file():
            os.environ["TCL_LIBRARY"] = str(tcl_dir)
            os.environ["TK_LIBRARY"] = str(tk_dir)
            return


def _install_crash_log() -> None:
    """Write uncaught errors to gui_error.log (pythonw has no console)."""
    log_path = _REPO / "gui_error.log"

    def _hook(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        import traceback

        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write("\n--- Herbivora crash ---\n")
                traceback.print_exception(exc_type, exc, tb, file=fh)
        except OSError:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


def main() -> None:
    _ensure_tcl_tk_env()
    _install_crash_log()

    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print(
            "Missing customtkinter. Run install.bat / install.sh or:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    # Do not create a separate tk.Tk() splash root: destroying it on Windows can
    # post WM_QUIT and make mainloop() exit right after the real window opens.
    # Splash is shown as a Toplevel inside HerbivoraApp construction instead.
    from gui.app import HerbivoraApp

    app = HerbivoraApp()
    try:
        app.mainloop()
    finally:
        # Force process exit so OpenMP/torch worker threads cannot freeze the
        # Windows desktop after the window is gone (pythonw has no console).
        try:
            app.destroy()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
