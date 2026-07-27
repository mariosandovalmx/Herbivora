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
   - Click **Check installation** — all three models should show **OK**
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

The installers call `download_models.py`, which expects:

```
https://huggingface.co/mariosandovalmx/HerbivoR
```

with these files at the repo root:

| File | Role |
|------|------|
| `mobile_sam.pt` | MobileSAM (segmentation) |
| `best_unet_shape.pth` | Contour U-Net |
| `best_model.pth` | Damage U-Net |

### Upload steps

```bash
pip install -U huggingface_hub
hf auth login
hf repo create HerbivoR --repo-type model   # once
hf upload mariosandovalmx/HerbivoR ./models .
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
