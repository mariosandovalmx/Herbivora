# HerbivoR

**HerbivoR** is a desktop GUI for quantifying leaf herbivory damage from photographs.

**Version:** see [`VERSION`](VERSION) · **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) · **Repo:** [github.com/mariosandovalmx/HerbivoR](https://github.com/mariosandovalmx/HerbivoR)

Pipeline:

1. **Segmentation** — isolate leaves (BiRefNet + MobileSAM, Intact Leaves, or Interactive)
2. **Contour / ROI** — reconstruct the leaf silhouette (UNET Shape) with interactive editing
3. **Analysis** — measure herbivory damage (Damage U-Net) with interactive Add/Remove/Line/Polygon tools

---

## Quick start

1. Download **Source code** from [Releases](https://github.com/mariosandovalmx/HerbivoR/releases) (or clone).
2. Install (creates `.venv`, installs PyTorch + deps, downloads models):

| Platform | Install |
|----------|---------|
| Windows CPU | Double-click **`Install_CPU.bat`** |
| Windows NVIDIA GPU | Double-click **`Install_CUDA.bat`** |
| Windows (menu) | Double-click **`Install.bat`** |
| macOS / Linux | `chmod +x install.sh herbivor.sh && ./install.sh` |

3. Run: **`HerbivoR.bat`** (Windows) or **`./herbivor.sh`** (macOS / Linux).

Full guide: **[INSTALL.md](INSTALL.md)**.

---

## Requirements

- Python **3.10+** (3.11–3.13 recommended)
- ~3 GB disk for the virtualenv + model weights
- Optional: NVIDIA GPU (use `Install_CUDA.bat` / CUDA on Linux). On Mac, Metal (MPS) is used automatically when available.

Model weights (~226 MB) are **not** in this repository. Installers (or **Check installation** in the GUI) download them from:

- HerbivoR U-Nets: [`mariosandovalmx/HerbivoR`](https://huggingface.co/mariosandovalmx/HerbivoR)
- MobileSAM: [Ultralytics assets](https://github.com/ultralytics/assets/releases) (third-party, Apache-2.0)

---

## Project layout

```
HerbivoR/
├── gui/                 # CustomTkinter desktop application
├── segmentation/        # BiRefNet + MobileSAM, Intact Leaves, whitebg helpers
├── contour/             # UNET Shape contour inference
├── leaf_contour/        # Shared mask post-processing
├── packaging/           # Optional PyInstaller (maintainers only; not in Releases)
├── models/              # Weights (downloaded; git-ignored)
├── Install_CPU.bat / Install_CUDA.bat / Install.bat
├── install.sh
└── HerbivoR.bat / herbivor.sh
```

---

## Typical workflow

1. **Project** — choose input and output folders; **Check installation**
2. **Segmentation** — BiRefNet + MobileSAM (recommended)
3. **Contour / ROI** — UNET Shape; optionally **Edit Contour**
4. **Analysis** — damage U-Net; optionally **Edit Damage**

Results appear under `{output}/analyzed/` (`results.csv` + overlay images).

---

## License

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
