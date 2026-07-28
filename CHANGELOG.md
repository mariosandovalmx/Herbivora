# Changelog

All notable changes to HerbivoR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] — 2026-07-28

### Added
- App branding: leaf icon for the GUI window, header, Windows shortcut (`.lnk`), and optional macOS `HerbivoR.app`.
- `assets/` icon set (`herbivor.ico`, `herbivor_256.png`, master PNG) plus rebuild/shortcut helpers under `packaging/`.

### Changed
- Installers create / document the leaf-icon launcher (`HerbivoR.lnk` on Windows; optional `create_macos_app.sh` on macOS).

## [1.1.0] — 2026-07-28

### Added
- `Install_CPU.bat` and `Install_CUDA.bat` (CUDA 12.4 wheels) for Windows.
- `Install.bat` menu to choose CPU vs CUDA.
- `install.sh` installs Torch for macOS (CPU + Metal/MPS) or Linux (CUDA if NVIDIA detected, else CPU).

### Changed
- `requirements.txt` no longer pins `torch`/`torchvision` (installers choose the wheel).
- GitHub Releases ship **source only** (no multi-GB PyInstaller assets).
- **Check installation** only probes packages actually used by the GUI; directs users to installers if Torch is missing.

### Deprecated
- Packaged `.exe` / split ZIP assets from v1.0.1 — use source + installers instead. `packaging/` remains for optional maintainer builds only.

## [1.0.1] — 2026-07-28

### Added
- Download HerbivoR U-Nets from Hugging Face (`mariosandovalmx/HerbivoR`); MobileSAM from Ultralytics assets.
- **Check installation** downloads missing weights into `models/` automatically.
- Packaging scripts for Windows (PyInstaller onedir) and macOS (build on a Mac).
- `VERSION`, `CHANGELOG.md`, and release / “test on another PC” docs.

### Changed
- MobileSAM is no longer hosted in the HerbivoR Hub repo (third-party weights).

## [1.0.0] — 2026-07-01

### Added
- Initial public-style release: GUI pipeline (Segmentation → Contour → Analysis), installers, and docs.
