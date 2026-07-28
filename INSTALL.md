# HerbivoR — Installation Reference

End users: follow the step-by-step **[USER_GUIDE.md](USER_GUIDE.md)**.

This page is a short reference plus **advanced** and **maintainer** notes.

Releases publish a **small** installer (Setup.exe / DMG when available) or **source code**. Large PyTorch wheels and model weights are **downloaded during install**, not bundled in the GitHub asset.

---

## Recommended (non-experts)

| Platform | What to run |
|----------|-------------|
| Windows | **`HerbivoR-Setup-vX.Y.Z.exe`** from [Releases](https://github.com/mariosandovalmx/HerbivoR/releases), **or** extract the source ZIP and double-click **`Install_HerbivoR.bat`** |
| macOS | **`HerbivoR-vX.Y.Z.dmg`** (if attached), **or** extract source and double-click **`Install_HerbivoR.command`** |
| Linux | `./Install_HerbivoR.command` or `./install.sh` |

The bootstrap installer:

1. Ensures Python (Windows: private per-user install if needed)
2. Creates `.venv`
3. Auto-detects NVIDIA GPU → CUDA 12.4, else CPU (macOS: default wheels + MPS)
4. Installs `requirements.txt`
5. Downloads models (~226 MB)
6. Creates shortcuts / `HerbivoR.app`

Full walkthrough: **[USER_GUIDE.md](USER_GUIDE.md)**.

---

## Advanced install (you already have Python 3.10+)

### Windows

| Script | When |
|--------|------|
| `Install_CPU.bat` | Force CPU |
| `Install_CUDA.bat` | Force CUDA 12.4 |
| `Install.bat` | Menu |
| `python packaging/bootstrap_install.py --yes --flavor auto` | Console bootstrap |

### macOS / Linux

```bash
chmod +x install.sh herbivor.sh Install_HerbivoR.command
./install.sh
```

### Manual

```bash
python -m venv .venv
# Pick ONE Torch install — see https://pytorch.org
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# macOS: .venv/bin/pip install torch torchvision
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
```

On Windows use `.venv\Scripts\` instead of `.venv/bin/`.

---

## Run

| Platform | Command |
|----------|---------|
| Windows | `HerbivoR.lnk` or `HerbivoR.bat` |
| macOS | `HerbivoR.app` or `./herbivor.sh` |
| Linux | `./herbivor.sh` |

If the GUI fails on Windows, check `gui_error.log`.

---

## First analysis checklist

1. **Project** → set Input / Output → **Check installation** (all models OK)
2. **Segmentation** → **Contour / ROI** → **Analysis**
3. Open `{output}/analyzed/results.csv`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No Python (Mac/Linux) | Install Python 3.10+ from python.org or your package manager |
| CUDA but runs on CPU | Update NVIDIA drivers; `check_gpu.py`; or reinstall CPU |
| Models missing | Project → Check installation / `download_models.py` |
| GUI crash | `.venv\Scripts\python.exe -m gui.main` + `gui_error.log` |

See also [USER_GUIDE.md](USER_GUIDE.md#troubleshooting).

---

## GPU check

```bash
.venv\Scripts\python.exe check_gpu.py   # Windows
.venv/bin/python check_gpu.py           # macOS / Linux
```

---

## Maintainers

### Build Windows Setup.exe

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php):

```bat
packaging\build_windows_setup.bat
```

Output: `dist\HerbivoR-Setup-vVERSION.exe` — attach to the GitHub Release.

### Build macOS DMG

On a Mac:

```bash
chmod +x packaging/build_macos_dmg.sh
./packaging/build_macos_dmg.sh
```

Output: `dist/HerbivoR-vVERSION.dmg`.

### Optional PyInstaller

Large onedir builds are **not** the supported user channel. See [packaging/README.md](packaging/README.md).

### Release checklist

1. Bump `VERSION` and `CHANGELOG.md`.
2. Commit and push `main`.
3. Build Setup.exe / DMG when possible; attach **only** those small bootstraps (+ source ZIP is automatic).
4. Tag and publish:

```bash
git tag v1.3.0
git push origin main --tags
gh release create v1.3.0 --title "HerbivoR v1.3.0" --notes-file CHANGELOG.md
```

Do **not** attach multi-GB PyInstaller / CUDA ZIPs to GitHub Releases (2 GB asset limit).
