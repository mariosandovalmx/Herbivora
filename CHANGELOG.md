# Changelog

All notable changes to HerbivoR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.3] — 2026-07-28

### Fixed
- Windows Setup: Start Menu shortcut no longer uses `/` in the name (caused
  `IPersistFile::Save failed; 0x80070003` — Windows treated `/` as a path separator).

## [1.3.2] — 2026-07-28

### Changed
- HerbivoR-trained Hub weights (`best_unet_shape.pth`, `best_model.pth`) relicensed
  to **PolyForm Noncommercial 1.0.0** (aligned with the application). Model card and
  docs updated; MobileSAM / BiRefNet remain under their own licenses.

## [1.3.1] — 2026-07-28

### Changed
- Relicensed from MIT to **PolyForm Noncommercial License 1.0.0**: free for
  noncommercial research and education; commercial use requires prior written
  permission; attribution required (see `LICENSE`, `CITATION.cff`).
- Installer welcome text and README updated to reflect the new terms.

## [1.3.0] — 2026-07-28

### Added
- One-click bootstrap installer: `Install_HerbivoR.bat` (Windows) and `Install_HerbivoR.command` (macOS/Linux).
- `packaging/bootstrap_install.py` — auto-detects NVIDIA GPU, installs private Python on Windows when needed, creates `.venv`, installs PyTorch + deps, downloads models, creates shortcuts.
- Maintainer builders: `packaging/build_windows_setup.bat` (Inno Setup → `HerbivoR-Setup-*.exe`) and `packaging/build_macos_dmg.sh` (→ `HerbivoR-*.dmg`).
- End-user documentation: `USER_GUIDE.md` (step-by-step for Windows, macOS, Linux).

### Changed
- README / INSTALL emphasize one-click install; legacy `Install_CPU.bat` / `Install_CUDA.bat` / `install.sh` remain as advanced options.

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
