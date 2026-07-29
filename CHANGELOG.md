# Changelog

All notable changes to HerbivoR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.11] — 2026-07-29

### Fixed
- BiRefNet_lite load no longer fails with `Unrecognized model … Should have a
  model_type key in its config.json` when a partial Hugging Face cache is present.
  The loader now validates/repairs `auto_map` + remote-code files and retries from
  the Hub when needed.

## [1.3.10] — 2026-07-29

### Added
- Installers now present the **full license agreement**: PolyForm Noncommercial
  terms (`LICENSE`), required attribution (`CITATION.cff` notices), and
  `THIRD_PARTY_NOTICES.md` (MobileSAM, BiRefNet, other dependencies).
- Windows Setup: copyright metadata, expanded Info Before/After pages, and a
  combined `installer_license.txt` acceptance page.
- Bootstrap GUI / console: explicit license acceptance before install (macOS DMG
  and repair installs). DMG also ships `LICENSE AGREEMENT.txt` in Finder.

## [1.3.9] — 2026-07-28

### Fixed
- Windows launchers no longer leave a black console (`cmd`) window open: shortcuts and
  Setup start `pythonw.exe -m gui.main`, and `HerbivoR.bat` detaches the GUI then exits.
  Crashes are still written to `gui_error.log`.

## [1.3.8] — 2026-07-28

### Changed
- Windows Setup always runs dependency install automatically (no “Run Install_HerbivoR.bat”
  checkbox). The Finished page shows a single optional **Launch HerbivoR** checkbox.

## [1.3.7] — 2026-07-28

### Fixed
- Windows Setup now **waits** for dependency install to finish before you can launch
  the app (avoids “`.venv` does not exist” right after Setup).
- Private Python install uses a **portable CPython** (Astral python-build-standalone
  with Tcl/Tk) instead of the silent python.org EXE, which often reported success
  without creating `python.exe` on some PCs.
- Double-clicking `HerbivoR.bat` / the shortcut auto-runs setup when `.venv` is missing
  (`Install_HerbivoR.bat /auto`) so users do not need manual repair steps.

## [1.3.6] — 2026-07-28

### Fixed
- Windows GUI no longer fails with `_tkinter.TclError: Can't find a usable init.tcl`
  when the system Python install is missing or misconfigured Tcl/Tk. The installer
  now prefers a private per-user CPython with `Include_tcltk=1`, verifies tkinter
  before creating `.venv`, and the app sets `TCL_LIBRARY`/`TK_LIBRARY` when needed.
- CustomTkinter theme is applied before the main window is created so the green
  System theme and leaf icon are stable at startup (no mid-launch recolor).

## [1.3.5] — 2026-07-28

### Fixed
- Contour (UNET Shape) no longer crashes on serrated/lobed leaves when OpenCV
  returns `convexityDefects` as shape `(N, 4)` instead of `(N, 1, 4)`
  (`cannot unpack non-iterable numpy.int32 object`). All images in a batch
  complete instead of silently dropping failures.

## [1.3.4] — 2026-07-28

### Fixed
- Contour tab no longer shows live segmentation mask outlines after Run
  segmentation. **Overlay contour** only lists real `leaf_roi_preview/overlays`
  from **Run contour**, so Segmentation and Contour stay independent.

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
