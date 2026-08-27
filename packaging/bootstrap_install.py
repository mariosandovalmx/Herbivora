#!/usr/bin/env python3
"""Herbivora one-click bootstrap installer.

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
PYTHON_VERSION = "3.12.13"
PYTHON_STANDALONE_TAG = "20260303"
WIN_PYTHON_STANDALONE_URLS = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PYTHON_STANDALONE_TAG}/cpython-{PYTHON_VERSION}+{PYTHON_STANDALONE_TAG}"
    "-x86_64-pc-windows-msvc-install_only.tar.gz",
    "https://github.com/indygreg/python-build-standalone/releases/download/"
    f"{PYTHON_STANDALONE_TAG}/cpython-{PYTHON_VERSION}+{PYTHON_STANDALONE_TAG}"
    "-x86_64-pc-windows-msvc-install_only.tar.gz",
)
WIN_PYTHON_INSTALLER = (
    "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
)

LogFn = Callable[[str], None]

LICENSE_ACCEPT_PROMPT = (
    "I have read and agree to the Herbivora License "
    "(PolyForm Noncommercial 1.0.0), Required Notices, "
    "and Third-Party Notices. Commercial use requires prior written permission."
)


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def read_version(root: Path) -> str:
    """Version being installed, taken from the VERSION file next to the sources."""
    version_file = root / "VERSION"
    if version_file.is_file():
        text = version_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "unknown"


def load_installer_license_text(root: Path) -> str:
    """Full agreement text for installers (LICENSE + citation + third-party)."""
    packaged = root / "packaging" / "installer_license.txt"
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")

    builder = root / "packaging" / "build_installer_license.py"
    if builder.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location("_herbivor_build_license", builder)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.build_text()  # type: ignore[no-any-return]

    parts: list[str] = []
    lic = root / "LICENSE"
    if lic.is_file():
        parts.append(lic.read_text(encoding="utf-8"))
    third = root / "THIRD_PARTY_NOTICES.md"
    if third.is_file():
        parts.append("\n\n=== THIRD-PARTY NOTICES ===\n\n")
        parts.append(third.read_text(encoding="utf-8"))
    if not parts:
        return (
            "Herbivora — PolyForm Noncommercial License 1.0.0\n"
            "Noncommercial research/education only. Commercial use requires "
            "prior written permission. See LICENSE in the project repository."
        )
    return "".join(parts)


def prompt_console_license_acceptance(root: Path) -> bool:
    """Show license summary and require explicit agreement on a TTY."""
    text = load_installer_license_text(root)
    print("=" * 60)
    print("Herbivora — License agreement (required)")
    print("=" * 60)
    # Show enough of the full text without flooding the entire terminal.
    preview = text if len(text) <= 12000 else text[:12000] + "\n\n[... truncated; full text in LICENSE / THIRD_PARTY_NOTICES.md ...]\n"
    print(preview)
    print("=" * 60)
    print(LICENSE_ACCEPT_PROMPT)
    print("Full files: LICENSE, THIRD_PARTY_NOTICES.md, CITATION.cff")
    ans = input('Type "I AGREE" to accept and continue (or N to cancel): ').strip()
    return ans.upper() == "I AGREE"


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
    return Path(local) / "Herbivora" / "Python"


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
    """True if this interpreter can create a Tk root (needed for Herbivora GUI)."""
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
    "InstallLauncherAllUsers=0",
    "Include_test=0",
    "Include_tools=0",
    "Include_pip=1",
    "Include_tcltk=1",
    "Include_lib=1",
    "Include_exe=1",
    "Shortcuts=0",
    "AssociateFiles=0",
    "SimpleInstall=1",
)


def _install_windows_private_python_standalone(log: LogFn, target: Path) -> Path:
    """Extract Astral python-build-standalone (includes Tcl/Tk)."""
    import tarfile

    with tempfile.TemporaryDirectory(prefix="herbivor_pbs_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "python.tar.gz"
        last_err: Exception | None = None
        for url in WIN_PYTHON_STANDALONE_URLS:
            try:
                _download(url, archive, log)
                if archive.is_file() and archive.stat().st_size > 10_000_000:
                    last_err = None
                    break
            except Exception as exc:  # noqa: BLE001 — try next mirror
                last_err = exc
                log(f"Standalone download failed ({url}): {exc}")
        if last_err is not None and (
            not archive.is_file() or archive.stat().st_size <= 10_000_000
        ):
            raise RuntimeError(
                f"python-build-standalone download failed: {last_err}"
            ) from last_err

        log("Extracting portable Python …")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path)

        found = next(tmp_path.rglob("python.exe"), None)
        if found is None:
            raise RuntimeError("python.exe missing inside standalone archive")
        src = found.parent

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    private = target / "python.exe"
    if not private.is_file():
        raise RuntimeError(f"Standalone extract did not produce {private}")
    return private


def _install_windows_private_python(log: LogFn) -> Path:
    target = windows_private_python_dir()
    log(f"Installing private Python {PYTHON_VERSION} to {target} …")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        private = _install_windows_private_python_standalone(log, target)
    except Exception as standalone_exc:  # noqa: BLE001
        log(f"WARNING: portable Python install failed: {standalone_exc}")
        log("Falling back to official python.org installer …")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="herbivor_py_") as tmp:
            installer = Path(tmp) / "python-3.12.10-amd64.exe"
            _download(WIN_PYTHON_INSTALLER, installer, log)
            _run(
                [str(installer), *_WIN_PYTHON_INSTALL_ARGS, f"TargetDir={target}"],
                log,
            )
        private = target / "python.exe"
        if not private.is_file():
            default = (
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Python"
                / "Python312"
                / "python.exe"
            )
            if default.is_file():
                log(f"Copying from default install path {default.parent} …")
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(default.parent, target, dirs_exist_ok=True)
                private = target / "python.exe"

    if not private.is_file() or not _python_version_ok(private):
        raise RuntimeError(f"Private Python install failed (missing {private})")
    if not _tkinter_usable(private):
        raise RuntimeError(
            f"Private Python at {private} was installed but tkinter/Tcl-Tk is not usable. "
            "Re-run Install_Herbivora.bat with an internet connection."
        )
    log(f"Private Python ready: {private}")
    return private


def ensure_base_python(log: LogFn) -> Path:
    """Return a usable Python 3.10+ with working tkinter (private copy on Windows if needed)."""
    # Prefer an already-installed private Herbivora Python on Windows.
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
            "On Windows, delete .venv and re-run Install_Herbivora.bat so a private "
            "Python with Tcl/Tk can be installed."
        )
    return py

def install_torch(py: Path, flavor: str, log: LogFn) -> None:
    log("Upgrading pip …")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], log)

    pip_install = [
        str(py),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--timeout",
        "30",
    ]

    try:
        if flavor == "cuda":
            log("Installing PyTorch CUDA 12.4 wheels …")
            _run(
                [
                    *pip_install,
                    "torch",
                    "torchvision",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ],
                log,
            )
        elif flavor == "macos":
            log("Installing native PyTorch for macOS (Metal/MPS included) …")
            log("There is no separate Metal package; the standard macOS wheel provides MPS.")
            _run([*pip_install, "torch", "torchvision"], log)
        else:
            log("Installing PyTorch CPU wheels …")
            _run(
                [
                    *pip_install,
                    "torch",
                    "torchvision",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                log,
            )
    except RuntimeError:
        if flavor == "macos":
            log("")
            log("PyTorch Metal/MPS installation failed.")
            log("If pip reports 'from versions: none', it usually could not reach PyPI")
            log("or no wheel matched the selected Python and native architecture.")
            log("Check the internet connection and reopen Herbivora to retry.")
        raise


def install_requirements(py: Path, root: Path, log: LogFn) -> None:
    req = root / "requirements.txt"
    if not req.is_file():
        raise RuntimeError(f"Missing {req}")
    log("Installing Herbivora packages from requirements.txt …")
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
                log(f"WARNING: could not create Herbivora.app: {exc}")
                log("You can still run: ./herbivora.sh")
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
            f"Does not look like a Herbivora source tree: {root}\n"
            "Extract the Release ZIP or clone the repository, then run the installer from that folder."
        )

    log(f"Herbivora version: {read_version(root)}")
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
        log("Open Herbivora with: Herbivora.lnk  (or Herbivora.bat)")
    elif platform.system() == "Darwin":
        log("Open Herbivora with: Herbivora.app  (or ./herbivora.sh)")
    else:
        log("Open Herbivora with: ./herbivora.sh")
    return flavor


def _run_gui(root: Path, flavor_arg: str, *, app_mode: bool = False) -> int:
    _configure_tcl_tk_env_for_prefix(Path(sys.base_prefix))
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    version = read_version(root)

    win = tk.Tk()
    win.title(f"Herbivora Installer {version}")
    win.geometry("780x620")
    win.minsize(640, 480)

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    heading = "Herbivora first-time setup" if app_mode else "Herbivora setup"
    ttk.Label(
        frm, text=f"{heading} (version {version})", font=("Segoe UI", 14, "bold")
    ).pack(anchor="w")
    setup_description = (
        "This one-time setup creates a private environment and downloads PyTorch and models.\n"
        if app_mode
        else "This installer sets up a private environment, downloads PyTorch and models,\n"
    )
    ttk.Label(
        frm,
        text=setup_description
        + ("" if app_mode else "It also creates a desktop shortcut. ")
        + "Internet required (about 5–20 minutes).\n"
        "You must accept the license terms below before installing.",
    ).pack(anchor="w", pady=(4, 8))

    lic_frame = ttk.LabelFrame(frm, text="License agreement (scroll and read)")
    lic_frame.pack(fill="both", expand=True, pady=(0, 6))
    lic_box = scrolledtext.ScrolledText(lic_frame, height=12, wrap="word", font=("Consolas", 9))
    lic_box.pack(fill="both", expand=True, padx=4, pady=4)
    lic_box.insert("1.0", load_installer_license_text(root))
    lic_box.configure(state="disabled")

    agree_var = tk.BooleanVar(value=False)
    agree_chk = ttk.Checkbutton(frm, text=LICENSE_ACCEPT_PROMPT, variable=agree_var)
    agree_chk.pack(anchor="w", pady=(0, 6))

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

    status = ttk.Label(frm, text="Accept the license to enable Install.")
    status.pack(anchor="w", pady=(8, 4))
    bar = ttk.Progressbar(frm, mode="indeterminate")
    bar.pack(fill="x", pady=4)

    log_box = scrolledtext.ScrolledText(frm, height=10, wrap="word", font=("Consolas", 9))
    log_box.pack(fill="both", expand=True, pady=8)
    log_box.configure(state="disabled")

    btn_row = ttk.Frame(frm)
    btn_row.pack(fill="x")
    start_btn = ttk.Button(btn_row, text="Install", state="disabled")
    start_btn.pack(side="left")
    close_btn = ttk.Button(btn_row, text="Close", command=win.destroy)
    close_btn.pack(side="right")

    lines: list[str] = []
    install_succeeded = False

    def _sync_install_enabled(*_args: object) -> None:
        start_btn.configure(state=("normal" if agree_var.get() else "disabled"))

    agree_var.trace_add("write", _sync_install_enabled)

    def append_log(msg: str) -> None:
        lines.append(msg)

        def _ui() -> None:
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")

        win.after(0, _ui)

    def worker() -> None:
        nonlocal install_succeeded
        try:
            run_install(
                root,
                flavor_var.get(),
                append_log,
                skip_shortcuts=app_mode,
            )
            install_succeeded = True

            def _ok() -> None:
                bar.stop()
                status.configure(text="Installation completed.")
                start_btn.configure(state="normal")
                next_step = (
                    "Close this setup window; Herbivora will open automatically."
                    if app_mode
                    else "Use the Herbivora shortcut (leaf icon) to open the app."
                )
                messagebox.showinfo(
                    "Herbivora",
                    "Installation completed.\n\n"
                    + next_step
                    + "\n\nLicense files remain in the install folder:\n"
                    "LICENSE, THIRD_PARTY_NOTICES.md, CITATION.cff",
                )

            win.after(0, _ok)
        except Exception as exc:  # noqa: BLE001 — show any failure in the GUI
            error_message = str(exc)

            def _err() -> None:
                bar.stop()
                status.configure(text="Installation failed.")
                start_btn.configure(state="normal")
                messagebox.showerror("Herbivora installer", error_message)

            append_log(f"ERROR: {error_message}")
            win.after(0, _err)

    def on_start() -> None:
        if not agree_var.get():
            messagebox.showwarning(
                "License required",
                "You must accept the Herbivora license and third-party notices "
                "before installing.",
            )
            return
        start_btn.configure(state="disabled")
        status.configure(text="Installing… please wait.")
        bar.start(12)
        append_log("License accepted by user.")
        threading.Thread(target=worker, daemon=True).start()

    start_btn.configure(command=on_start)
    win.mainloop()
    return 0 if install_succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Herbivora bootstrap installer")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Herbivora source tree (default: repository root)",
    )
    parser.add_argument(
        "--flavor",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch build (default: auto-detect NVIDIA on Windows/Linux)",
    )
    parser.add_argument("--gui", action="store_true", help="Show a simple progress window")
    parser.add_argument(
        "--app-mode",
        action="store_true",
        help="Run first-launch setup from Herbivora.app without creating another app shortcut",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive console install (assumes license already accepted, e.g. Setup.exe)",
    )
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-shortcuts", action="store_true")
    args = parser.parse_args(argv)

    if args.gui:
        return _run_gui(args.root, args.flavor, app_mode=args.app_mode)

    if args.yes:
        _default_log(
            "Proceeding under Herbivora LICENSE + THIRD_PARTY_NOTICES.md "
            "(non-interactive / already accepted via Setup)."
        )
    elif sys.stdin.isatty():
        print("Herbivora bootstrap installer")
        print(f"Root: {args.root.resolve()}")
        if not prompt_console_license_acceptance(args.root):
            print("License not accepted — install cancelled.")
            return 1
        flavor, note = detect_torch_flavor(args.flavor)
        print(note)
        ans = input("Continue with dependency install? [Y/n] ").strip().lower()
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
