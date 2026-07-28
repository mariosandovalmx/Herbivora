# Changelog

All notable changes to HerbivoR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
