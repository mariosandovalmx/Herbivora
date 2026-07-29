#!/usr/bin/env python3
"""HerbivoR one-click bootstrap installer.

Installs a private Python (Windows), creates .venv, installs PyTorch
(auto CPU/CUDA/MPS), app dependencies, model weights, and shortcuts.

Usage (from repository root)::

    python packaging/bootstrap_install.py --gui
    python packaging/bootstrap_install.py --flavor auto --yes

Do not ask end users to install Python themselves on Windows — this script
downloads a per-user CPython build when needed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERSION = "3.12.10"
WIN_PYTHON_INSTALLER = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-amd64.exe"
)

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def detect_torch_flavor(forced: str = "auto") -> tuple[str, str]:
    """Return (flavor, human_note). flavor is cpu|cuda|macos."""
    if forced in ("cpu", "cuda"):
        note = f"Forced PyTorch flavor: {forced}"
        return forced, note

    system = platform.system()
    if system == "Darwin":
        return "macos", "macOS: PyTorch with CPU + Metal (MPS) when available"

    if system == "Windows" or system == "Linux":
        try:
            r = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if r.returncode == 0:
                return "cuda", "NVIDIA GPU detected — installing CUDA 12.4 wheels"
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return "cpu", "No NVIDIA GPU detected — installing CPU wheels"

    return "cpu", f"Unknown OS {system!r} — installing CPU wheels"


def _run(
    cmd: list[str],
    log: LogFn,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log("$ " + " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def _download(url: str, dest: Path, log: LogFn) -> None:
    log(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block: int, block_size: int, total: int) -> None:
        if total <= 0 or block % 64 != 0:
            return
        done = min(total, block * block_size)
        pct = 100.0 * done / total
        log(f"  … {pct:.0f}% ({done // (1024 * 1024)} / {total // (1024 * 1024)} MB)")

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)  # noqa: S310
    log(f"Saved {dest}")


def _python_version_ok(exe: Path | str) -> bool:
    try:
        r = subprocess.run(
            [str(exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            return False
        major, minor = (int(x) for x in r.stdout.strip().split(".")[:2])
        return (major, minor) >= (3, 10)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def windows_private_python_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "HerbivoR" / "Python"


def _configure_tcl_tk_env_for_prefix(base: Path) -> bool:
    """Set TCL_LIBRARY/TK_LIBRARY from known layouts. Return True if init.tcl was found."""
    pairs = [
        (base / "tcl" / "tcl8.6", base / "tcl" / "tk8.6"),
        (base / "lib" / "tcl8.6", base / "lib" / "tk8.6"),
        (base / "Library" / "lib" / "tcl8.6", base / "Library" / "lib" / "tk8.6"),
    ]
    for tcl_dir, tk_dir in pairs:
        if (tcl_dir / "init.tcl").is_file() and (tk_dir / "tk.tcl").is_file():
            os.environ["TCL_LIBRARY"] = str(tcl_dir)
            os.environ["TK_LIBRARY"] = str(tk_dir)
            return True
    return False


def _tkinter_usable(exe: Path | str) -> bool:
    """True if this interpreter can create a Tk root (needed for HerbivoR GUI)."""
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "base = Path(sys.base_prefix)\n"
        "pairs = [\n"
        " (base/'tcl'/'tcl8.6', base/'tcl'/'tk8.6'),\n"
        " (base/'lib'/'tcl8.6', base/'lib'/'tk8.6'),\n"
        " (base/'Library'/'lib'/'tcl8.6', base/'Library'/'lib'/'tk8.6'),\n"
        "]\n"
        "for tcl_dir, tk_dir in pairs:\n"
        "  if (tcl_dir/'init.tcl').is_file() and (tk_dir/'tk.tcl').is_file():\n"
        "    os.environ['TCL_LIBRARY']=str(tcl_dir)\n"
        "    os.environ['TK_LIBRARY']=str(tk_dir)\n"
        "    break\n"
        "import tkinter as t\n"
        "r = t.Tk(); r.withdraw(); r.destroy()\n"
    )
    try:
        r = subprocess.run(
            [str(exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_WIN_PYTHON_INSTALL_ARGS = (
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_doc=0",
    "Include_launcher=0",
    "Include_test=0",
    "Include_tools=0",
    "Include_tcltk=1",
    "Shortcuts=0",
    "AssociateFiles=0",
)


def _install_windows_private_python(log: LogFn) -> Path:
    target = windows_private_python_dir()
    log(f"Installing private Python {PYTHON_VERSION} to {target} …")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log("Removing previous private Python (incomplete or missing Tcl/Tk) …")
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="herbivor_py_") as tmp:
        installer = Path(tmp) / f"python-{PYTHON_VERSION}-amd64.exe"
        _download(WIN_PYTHON_INSTALLER, installer, log)
        _run(
            [str(installer), *_WIN_PYTHON_INSTALL_ARGS, f"TargetDir={target}"],
            log,
        )
    private = target / "python.exe"
    if not private.is_file() or not _python_version_ok(private):
        raise RuntimeError(f"Private Python install failed (missing {private})")
    if not _tkinter_usable(private):
        raise RuntimeError(
            f"Private Python at {private} was installed but tkinter/Tcl-Tk is not usable. "
            "Re-run the installer, or install Python from https://www.python.org/ "
            "with the Tcl/Tk component enabled."
        )
    log(f"Private Python ready: {private}")
    return private


def ensure_base_python(log: LogFn) -> Path:
    """Return a usable Python 3.10+ with working tkinter (private copy on Windows if needed)."""
    # Prefer an already-installed private HerbivoR Python on Windows.
    if platform.system() == "Windows":
        private = windows_private_python_dir() / "python.exe"
        if private.is_file() and _python_version_ok(private) and _tkinter_usable(private):
            log(f"Using private Python: {private}")
            return private
        if private.is_file():
            log("Private Python exists but tkinter/Tcl-Tk is broken — reinstalling …")

    # Prefer current interpreter if it is new enough and GUI-capable.
    if (
        _python_version_ok(sys.executable)
        and not getattr(sys, "frozen", False)
        and _tkinter_usable(sys.executable)
    ):
        log(f"Using current Python: {sys.executable}")
        return Path(sys.executable)

    # Search PATH for a GUI-capable interpreter.
    for name in ("py", "python3", "python"):
        exe = shutil.which(name)
        if not exe:
            continue
        if name == "py":
            # Windows launcher: prefer 3.12/3.11/3.10
            for args in (["-3.12"], ["-3.11"], ["-3.10"], ["-3"]):
                try:
                    r = subprocess.run(
                        [exe, *args, "-c", "import sys; print(sys.executable)"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if r.returncode == 0:
                        candidate = Path(r.stdout.strip())
                        if (
                            candidate.is_file()
                            and _python_version_ok(candidate)
                            and _tkinter_usable(candidate)
                        ):
                            log(f"Using Python launcher: {candidate}")
                            return candidate
                except (OSError, subprocess.TimeoutExpired):
                    continue
        elif _python_version_ok(exe) and _tkinter_usable(exe):
            log(f"Using PATH Python: {exe}")
            return Path(exe)

    if platform.system() != "Windows":
        raise RuntimeError(
            "Python 3.10+ with working tkinter was not found. On macOS install Python from "
            "https://www.python.org/downloads/ or Xcode Command Line Tools, "
            "then re-run this installer. On Linux install python3-tk via your package manager."
        )

    return _install_windows_private_python(log)

def venv_python(root: Path) -> Path:
    if platform.system() == "Windows":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def ensure_venv(base_python: Path, root: Path, log: LogFn) -> Path:
    py = venv_python(root)
    if py.is_file() and _python_version_ok(py) and _tkinter_usable(py):
        log(f"Using existing venv: {py}")
        return py
    venv_dir = root / ".venv"
    if venv_dir.exists():
        if py.is_file() and _python_version_ok(py) and not _tkinter_usable(py):
            log("Existing .venv cannot open the GUI (broken Tcl/Tk) — recreating …")
        else:
            log("Removing broken .venv …")
        shutil.rmtree(venv_dir, ignore_errors=True)
    log("Creating virtual environment .venv …")
    _run([str(base_python), "-m", "venv", str(venv_dir)], log, cwd=root)
    if not py.is_file():
        raise RuntimeError("venv created but python executable is missing")
    if not _tkinter_usable(py):
        raise RuntimeError(
            "Virtual environment was created but tkinter/Tcl-Tk still fails. "
            "On Windows, delete .venv and re-run Install_HerbivoR.bat so a private "
            "Python with Tcl/Tk can be installed."
        )
    return py

def install_torch(py: Path, flavor: str, log: LogFn) -> None:
    log("Upgrading pip …")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], log)

    if flavor == "cuda":
        log("Installing PyTorch CUDA 12.4 wheels …")
        _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ],
            log,
        )
    elif flavor == "macos":
        log("Installing PyTorch (macOS default wheels, MPS-capable) …")
        _run([str(py), "-m", "pip", "install", "torch", "torchvision"], log)
    else:
        log("Installing PyTorch CPU wheels …")
        _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            log,
        )


def install_requirements(py: Path, root: Path, log: LogFn) -> None:
    req = root / "requirements.txt"
    if not req.is_file():
        raise RuntimeError(f"Missing {req}")
    log("Installing HerbivoR packages from requirements.txt …")
    _run([str(py), "-m", "pip", "install", "-r", str(req)], log, cwd=root)


def download_models(py: Path, root: Path, log: LogFn) -> None:
    script = root / "download_models.py"
    if not script.is_file():
        raise RuntimeError(f"Missing {script}")
    log("Downloading model weights (~226 MB, first time only) …")
    try:
        _run([str(py), str(script)], log, cwd=root)
    except RuntimeError as exc:
        log(f"WARNING: model download failed: {exc}")
        log("You can retry later with Project → Check installation, or:")
        log(f"  {py} download_models.py")


def create_shortcuts(root: Path, log: LogFn) -> None:
    system = platform.system()
    if system == "Windows":
        ps1 = root / "packaging" / "create_windows_shortcut.ps1"
        if not ps1.is_file():
            log("WARNING: shortcut script missing; skip .lnk creation")
            return
        try:
            _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ps1),
                ],
                log,
                cwd=root,
            )
        except RuntimeError as exc:
            log(f"WARNING: could not create shortcuts: {exc}")
        return

    if system == "Darwin":
        sh = root / "packaging" / "create_macos_app.sh"
        if sh.is_file():
            try:
                _run(["bash", str(sh)], log, cwd=root)
            except RuntimeError as exc:
                log(f"WARNING: could not create HerbivoR.app: {exc}")
                log("You can still run: ./herbivor.sh")
        return


def verify_torch(py: Path, log: LogFn) -> None:
    code = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available()); "
        "mps=getattr(torch.backends,'mps',None); "
        "print('mps', bool(mps and mps.is_available()))"
    )
    try:
        _run([str(py), "-c", code], log)
    except RuntimeError as exc:
        log(f"WARNING: torch verification failed: {exc}")


def run_install(
    root: Path,
    flavor_arg: str,
    log: LogFn,
    *,
    skip_models: bool = False,
    skip_shortcuts: bool = False,
) -> str:
    root = root.resolve()
    if not (root / "gui" / "main.py").is_file():
        raise RuntimeError(
            f"Does not look like a HerbivoR source tree: {root}\n"
            "Extract the Release ZIP or clone the repository, then run the installer from that folder."
        )

    flavor, note = detect_torch_flavor(flavor_arg)
    log(note)
    log(f"Install root: {root}")

    base = ensure_base_python(log)
    py = ensure_venv(base, root, log)
    install_torch(py, flavor, log)
    install_requirements(py, root, log)
    if not skip_models:
        download_models(py, root, log)
    verify_torch(py, log)
    if not skip_shortcuts:
        create_shortcuts(root, log)

    log("")
    log("=" * 44)
    log(f"  Installation completed  [{flavor}]")
    log("=" * 44)
    if platform.system() == "Windows":
        log("Open HerbivoR with: HerbivoR.lnk  (or HerbivoR.bat)")
    elif platform.system() == "Darwin":
        log("Open HerbivoR with: HerbivoR.app  (or ./herbivor.sh)")
    else:
        log("Open HerbivoR with: ./herbivor.sh")
    return flavor


def _run_gui(root: Path, flavor_arg: str) -> int:
    _configure_tcl_tk_env_for_prefix(Path(sys.base_prefix))
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    win = tk.Tk()
    win.title("HerbivoR Installer")
    win.geometry("720x480")
    win.minsize(560, 360)

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="HerbivoR setup", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(
        frm,
        text="This installer sets up a private environment, downloads PyTorch and models,\n"
        "and creates a desktop shortcut. Internet required (about 5–20 minutes).",
    ).pack(anchor="w", pady=(4, 8))

    flavor_var = tk.StringVar(value=flavor_arg)
    opts = ttk.Frame(frm)
    opts.pack(anchor="w", fill="x")
    ttk.Label(opts, text="Compute:").pack(side="left")
    for label, val in (
        ("Auto-detect GPU", "auto"),
        ("CPU only", "cpu"),
        ("NVIDIA CUDA", "cuda"),
    ):
        ttk.Radiobutton(opts, text=label, value=val, variable=flavor_var).pack(side="left", padx=6)

    status = ttk.Label(frm, text="Ready.")
    status.pack(anchor="w", pady=(8, 4))
    bar = ttk.Progressbar(frm, mode="indeterminate")
    bar.pack(fill="x", pady=4)

    log_box = scrolledtext.ScrolledText(frm, height=18, wrap="word", font=("Consolas", 9))
    log_box.pack(fill="both", expand=True, pady=8)
    log_box.configure(state="disabled")

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill="x")
    start_btn = ttk.Button(btn_row, text="Install")
    start_btn.pack(side="left")
    close_btn = ttk.Button(btn_row, text="Close", command=win.destroy)
    close_btn.pack(side="right")

    lines: list[str] = []

    def append_log(msg: str) -> None:
        lines.append(msg)

        def _ui() -> None:
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")

        win.after(0, _ui)

    def worker() -> None:
        try:
            run_install(root, flavor_var.get(), append_log)

            def _ok() -> None:
                bar.stop()
                status.configure(text="Installation completed.")
                start_btn.configure(state="normal")
                messagebox.showinfo(
                    "HerbivoR",
                    "Installation completed.\n\n"
                    "Use the HerbivoR shortcut (leaf icon) to open the app.",
                )

            win.after(0, _ok)
        except Exception as exc:  # noqa: BLE001 — show any failure in the GUI

            def _err() -> None:
                bar.stop()
                status.configure(text="Installation failed.")
                start_btn.configure(state="normal")
                messagebox.showerror("HerbivoR installer", str(exc))

            append_log(f"ERROR: {exc}")
            win.after(0, _err)

    def on_start() -> None:
        start_btn.configure(state="disabled")
        status.configure(text="Installing… please wait.")
        bar.start(12)
        threading.Thread(target=worker, daemon=True).start()

    start_btn.configure(command=on_start)
    win.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HerbivoR bootstrap installer")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="HerbivoR source tree (default: repository root)",
    )
    parser.add_argument(
        "--flavor",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch build (default: auto-detect NVIDIA on Windows/Linux)",
    )
    parser.add_argument("--gui", action="store_true", help="Show a simple progress window")
    parser.add_argument("--yes", action="store_true", help="Non-interactive console install")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-shortcuts", action="store_true")
    args = parser.parse_args(argv)

    if args.gui:
        return _run_gui(args.root, args.flavor)

    if not args.yes and sys.stdin.isatty():
        print("HerbivoR bootstrap installer")
        print(f"Root: {args.root.resolve()}")
        flavor, note = detect_torch_flavor(args.flavor)
        print(note)
        ans = input("Continue? [Y/n] ").strip().lower()
        if ans in ("n", "no"):
            return 1

    try:
        run_install(
            args.root,
            args.flavor,
            _default_log,
            skip_models=args.skip_models,
            skip_shortcuts=args.skip_shortcuts,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
