# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for HerbivoR (onedir). Run via packaging/build_windows.bat."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPEC).resolve().parent.parent

datas = [
    (str(ROOT / "contour" / "configs"), "contour/configs"),
    (str(ROOT / "segmentation" / "birefnet_mobilesam" / "config.yaml"), "segmentation/birefnet_mobilesam"),
    (str(ROOT / "VERSION"), "."),
    (str(ROOT / "models" / "README.md"), "models"),
]

binaries = []
hiddenimports = [
    "gui",
    "gui.main",
    "gui.app",
    "gui.state",
    "gui.paths",
    "gui.pipeline",
    "gui.runner",
    "image_io",
    "download_models",
    "analyze_leaves",
    "customtkinter",
    "PIL",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "segmentation_models_pytorch",
    "transformers",
    "huggingface_hub",
    "einops",
    "kornia",
    "timm",
    "scipy",
    "sklearn",
    "matplotlib",
    "yaml",
    "tqdm",
]

for pkg in (
    "customtkinter",
    "ultralytics",
    "segmentation_models_pytorch",
    "timm",
    "einops",
    "kornia",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

try:
    datas += collect_data_files("transformers")
except Exception:
    pass

hiddenimports += collect_submodules("gui")
hiddenimports += collect_submodules("contour")
hiddenimports += collect_submodules("leaf_contour")
hiddenimports += collect_submodules("segmentation")

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "herbivor_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "unittest", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HerbivoR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HerbivoR",
)
