# HerbivoR

**HerbivoR** is a desktop GUI for quantifying leaf herbivory damage from photographs.

**Version:** see [`VERSION`](VERSION) · **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) · **Repo:** [github.com/mariosandovalmx/HerbivoR](https://github.com/mariosandovalmx/HerbivoR)

Pipeline:

1. **Segmentation** — isolate leaves (BiRefNet + MobileSAM, Intact Leaves, or Interactive)
2. **Contour / ROI** — reconstruct the leaf silhouette (UNET Shape) with interactive editing
3. **Analysis** — measure herbivory damage (Damage U-Net) with interactive Add/Remove/Line/Polygon tools

---

## Quick start

| Platform | Install | Run |
|----------|---------|-----|
| Windows | Double-click `Install.bat` | Double-click `HerbivoR.bat` |
| macOS / Linux | `chmod +x install.sh herbivor.sh && ./install.sh` | `./herbivor.sh` |

Full step-by-step instructions: **[INSTALL.md](INSTALL.md)** (includes **testing on another PC** with a packaged `.exe` and **how to cut a Release**).

---

## Requirements

- Python **3.10+** (3.11–3.13 recommended)
- ~3 GB disk for the virtualenv + model weights
- Optional: NVIDIA GPU with CUDA for faster inference

Model weights (~226 MB) are **not** in this repository. They download automatically during install (or via **Check installation** in the GUI):

- HerbivoR U-Nets from [`mariosandovalmx/HerbivoR`](https://huggingface.co/mariosandovalmx/HerbivoR)
- MobileSAM from [Ultralytics assets](https://github.com/ultralytics/assets/releases) (third-party, Apache-2.0)

---

## Project layout

```
HerbivoR/
├── gui/                 # CustomTkinter desktop application
├── segmentation/        # BiRefNet + MobileSAM, Intact Leaves, whitebg helpers
├── contour/             # UNET Shape contour inference
├── leaf_contour/        # Shared mask post-processing
├── packaging/           # PyInstaller scripts (Windows / macOS test builds)
├── models/              # Weights (downloaded; git-ignored)
├── analyze_leaves.py    # Damage analysis CLI used by the GUI
├── download_models.py   # Fetch weights from Hugging Face
├── Install.bat / install.sh
└── HerbivoR.bat / herbivor.sh
```

---

## Typical workflow

1. **Project** — choose input and output folders; confirm the three models are detected
2. **Segmentation** — run BiRefNet + MobileSAM (recommended)
3. **Contour / ROI** — run UNET Shape; optionally **Edit Contour**
4. **Analysis** — run U-Net damage analysis; optionally **Edit Damage**

Results appear under `{output}/analyzed/` (`results.csv` + overlay images).

---

## License

See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
