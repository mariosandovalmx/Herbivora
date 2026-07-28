# HerbivoR — Installation Guide

Step-by-step setup for **Windows**, **macOS**, and **Linux**.

Releases contain **source code only** (small ZIP). Python packages and model weights are installed on your machine by the scripts below — there is no giant bundled `.exe` in the Release.

---

## 1. Prerequisites

1. Install **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)).
   - On Windows, check **“Add python.exe to PATH”** during setup.
2. (Optional, Windows/Linux GPU) Recent **NVIDIA drivers** if you will use `Install_CUDA.bat` / CUDA on Linux.
3. A free [Hugging Face](https://huggingface.co/) account is **not** required to download public models.

---

## 2. Get the source code

Download **Source code (zip)** from [Releases](https://github.com/mariosandovalmx/HerbivoR/releases), or clone:

```bash
git clone https://github.com/mariosandovalmx/HerbivoR.git
cd HerbivoR
```

---

## 3. Install dependencies and models

### Windows (pick one)

| Installer | When to use |
|-----------|-------------|
| **`Install_CPU.bat`** | No NVIDIA GPU, or you are unsure (recommended default) |
| **`Install_CUDA.bat`** | NVIDIA GPU + recent drivers (PyTorch **CUDA 12.4** wheels) |
| **`Install.bat`** | Menu to choose CPU or CUDA |

Double-click the installer and wait for `Installation completed`. It will:

1. Create `.venv`
2. Install the matching PyTorch wheel
3. Install packages from `requirements.txt`
4. Download models into `models/` (~226 MB)

### macOS / Linux

```bash
chmod +x install.sh herbivor.sh
./install.sh
```

- **macOS:** one installer — PyTorch includes **Metal (MPS)**; the app uses it automatically when available.
- **Linux:** CUDA 12.4 wheels if `nvidia-smi` works; otherwise CPU.

### Manual install (any OS)

```bash
python -m venv .venv

# 1) PyTorch first — pick ONE (see https://pytorch.org)
# CPU:
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.4:
# .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# macOS (default wheels, MPS-capable):
# .venv/bin/pip install torch torchvision

# 2) App packages (does not include torch)
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
```

On Windows use `.venv\Scripts\` instead of `.venv/bin/`.

---

## 4. Run HerbivoR

| Platform | Command |
|----------|---------|
| Windows | Double-click `HerbivoR.lnk` (leaf icon) or `HerbivoR.bat` |
| macOS / Linux | `./herbivor.sh` (macOS: optional `./packaging/create_macos_app.sh` for `HerbivoR.app`) |
| Any OS | `.venv/bin/python -m gui.main` (or `.venv\Scripts\python.exe -m gui.main`) |

The installer creates **`HerbivoR.lnk`** (and a Desktop shortcut on Windows) so Explorer shows the bitten-leaf icon. Plain `.bat` files cannot carry a custom icon.

If the window does not open on Windows, check `gui_error.log` in the project folder.

---

## 5. First analysis (recommended path)

1. **Project tab**
   - Set **Input folder** and **Output folder**
   - Click **Check installation** — verifies packages and downloads any missing models. All three models should show **OK**.
2. **Segmentation** → **Contour / ROI** → **Analysis** as usual.
3. Open `{output}/analyzed/results.csv` and the `*_analyzed.jpg` overlays.

---

## 6. Publishing model weights on Hugging Face (maintainers)

See previous docs / `download_models.py`. Hub repo: https://huggingface.co/mariosandovalmx/HerbivoR  
(`best_unet_shape.pth`, `best_model.pth`; MobileSAM from Ultralytics, not re-hosted).

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Python 3 not found` | Reinstall Python with PATH enabled |
| CUDA install but inference on CPU | Update NVIDIA drivers; run `check_gpu.py` |
| `pip install` fails on torch | Use the CPU installer, or pick another CUDA version on [pytorch.org](https://pytorch.org) |
| Models missing | `python download_models.py` or Project → Check installation |
| GUI blank / crash | Run `.venv\Scripts\python.exe -m gui.main` and read `gui_error.log` |

---

## 8. GPU check

```bash
.venv\Scripts\python.exe check_gpu.py   # Windows
.venv/bin/python check_gpu.py           # macOS / Linux
```

---

## 9. Testing on another PC

1. Copy the **source** ZIP from the Release (or clone) to the other PC.
2. Install **Python 3.10+**.
3. Run **`Install_CPU.bat`** (or CUDA / `./install.sh`).
4. Run **`HerbivoR.bat`** / `./herbivor.sh`.

Packaged PyInstaller `.exe` builds are **not** part of current Releases (too large; Torch belongs in the venv). Optional maintainer tooling remains under `packaging/` — see `packaging/README.md`.

---

## 10. Releasing a new version (maintainers)

1. Bump `VERSION` and `CHANGELOG.md`.
2. Commit and push `main`.
3. Tag and create a GitHub Release (**source only** — do not attach multi-GB exe ZIPs):

```bash
git tag v1.2.0
git push origin main --tags
gh release create v1.2.0 --title "HerbivoR v1.2.0" --notes-file CHANGELOG.md
```

Semver: **patch** = bugfixes; **minor** = features; **major** = breaking changes.
