# HerbivoR — Installation Guide

Step-by-step setup for **Windows**, **macOS**, and **Linux**.

---

## 1. Prerequisites

1. Install **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)).
   - On Windows, check **“Add python.exe to PATH”** during setup.
2. (Optional) NVIDIA drivers + CUDA if you want GPU acceleration.
3. A free [Hugging Face](https://huggingface.co/) account is **not** required to download public models, but you will need one if you publish private weights.

---

## 2. Get the source code

Clone or download this repository and open a terminal in the project folder:

```bash
cd HerbivoR
```

---

## 3. Install dependencies and models

### Windows

1. Double-click **`Install.bat`**
2. Wait until you see `Installation completed`
3. Confirm these files exist under `models/`:
   - `mobile_sam.pt`
   - `best_unet_shape.pth`
   - `best_model.pth`

### macOS / Linux

```bash
chmod +x install.sh herbivor.sh
./install.sh
```

### Manual install (any OS)

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python download_models.py

# macOS / Linux
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
```

---

## 4. Run HerbivoR

| Platform | Command |
|----------|---------|
| Windows | Double-click `HerbivoR.bat` |
| macOS / Linux | `./herbivor.sh` |
| Any OS | `.venv/bin/python -m gui.main` (or `.venv\Scripts\python.exe -m gui.main`) |

If the window does not open on Windows, check `gui_error.log` in the project folder.

---

## 5. First analysis (recommended path)

1. **Project tab**
   - Set **Input folder** (photos of leaves / scenes)
   - Set **Output folder**
   - Click **Check installation** — verifies dependencies and downloads any missing models into `models/` (MobileSAM + HerbivoR U-Nets). All three should show **OK**.
2. **Segmentation tab**
   - Method: **A. BiRefNet + MobileSAM [RECOMMENDED]**
   - Click **Run segmentation**
3. **Contour / ROI tab**
   - Click **Run contour**
   - Optionally **Edit Contour** (Add / Remove / Line / Polygon)
4. **Analysis tab**
   - Click **Run U-Net analysis**
   - Optionally **Edit Damage**
5. Open `{output}/analyzed/results.csv` and the `*_analyzed.jpg` overlays

---

## 6. Publishing model weights on Hugging Face (maintainers)

The installers call `download_models.py`, which:

- downloads **your** U-Nets from Hugging Face:

```
https://huggingface.co/mariosandovalmx/HerbivoR
```

| File | Role |
|------|------|
| `best_unet_shape.pth` | Contour U-Net |
| `best_model.pth` | Damage U-Net |

- downloads **MobileSAM** separately from Ultralytics assets (do **not** re-upload it to HerbivoR).

### Upload steps (HerbivoR U-Nets only)

```bash
pip install -U huggingface_hub
hf auth login
hf repo create HerbivoR --repo-type model --public   # once; license is set in the model card
hf upload mariosandovalmx/HerbivoR ./models/best_unet_shape.pth best_unet_shape.pth
hf upload mariosandovalmx/HerbivoR ./models/best_model.pth best_model.pth
hf upload mariosandovalmx/HerbivoR ./hf_model_card/README.md README.md
```

Replace `mariosandovalmx` with your Hugging Face username if different, and update `DEFAULT_REPO` in `download_models.py` accordingly.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Python 3 not found` | Reinstall Python with PATH enabled, or run from Anaconda Prompt |
| `pip install` fails on torch | Install a CUDA/CPU wheel from [pytorch.org](https://pytorch.org) first, then re-run `pip install -r requirements.txt` |
| Models missing | `python download_models.py` or copy the three files into `models/` |
| GUI blank / crash | Run `.venv\Scripts\python.exe -m gui.main` in a console and read the traceback / `gui_error.log` |
| BiRefNet download slow | First Segmentation run downloads BiRefNet_lite (~170 MB) into `segmentation/birefnet_mobilesam/models/hf_cache/` |

---

## 8. GPU check

```bash
.venv\Scripts\python.exe check_gpu.py   # Windows
.venv/bin/python check_gpu.py           # macOS / Linux
```

---

## 9. Testing on another PC (packaged build)

For a quick try **without installing Python** on a second Windows machine:

1. Download **`HerbivoR-windows-vX.Y.Z.zip`** from the [GitHub Releases](https://github.com/mariosandovalmx/HerbivoR/releases) page (private repo: sign in with an account that has access), **or** copy the ZIP via USB.
2. Extract the ZIP to a folder with write permission (e.g. `Documents\HerbivoR`).
3. Run **`HerbivoR.exe`**.
4. In **Project → Check installation**, let it download the three weights into `models/` next to the exe (needs internet the first time).
5. Run a short Segmentation → Contour → Analysis smoke test.

Notes:

- The packaged build is large (PyTorch and deps are bundled). Model weights are **not** inside the ZIP.
- Windows Defender may scan the new folder the first time; wait if the first launch is slow.
- **macOS `.app`:** build on a Mac with `packaging/build_macos.sh`, then copy that ZIP the same way. A Windows PC cannot produce a Mac app.

### Building the Windows ZIP (maintainers)

```bat
pip install -r requirements-dev.txt
packaging\build_windows.bat
```

Output: `dist\HerbivoR-windows-vX.Y.Z.zip` (attach to the GitHub Release; do not commit `dist/`).

---

## 10. Releasing a new version (maintainers)

1. Bump [`VERSION`](VERSION) and add notes to [`CHANGELOG.md`](CHANGELOG.md).
2. Commit and push to `main`.
3. Tag and push:

```bash
git tag v1.0.1
git push origin main --tags
```

4. Create a GitHub Release for that tag (UI or `gh release create`).
5. Attach `HerbivoR-windows-vX.Y.Z.zip` (and macOS ZIP if built). Source code ZIP is generated automatically.

Semver: **patch** = bugfixes; **minor** = features; **major** = breaking changes.
