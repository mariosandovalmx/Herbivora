# HerbivoR User Guide

Step-by-step instructions for installing and running **HerbivoR** on Windows, macOS, and Linux.

**HerbivoR** measures leaf herbivory damage from photographs using a desktop GUI.

| Document | Purpose |
|----------|---------|
| **This guide** | End-user install and first analysis |
| [INSTALL.md](INSTALL.md) | Short install reference + advanced / maintainer notes |
| [README.md](README.md) | Project overview |

---

## What you need

| Item | Notes |
|------|--------|
| Computer | Windows 10/11 (64-bit), macOS 11+, or Linux |
| Disk space | About **3–6 GB** free (environment + models; CUDA needs more) |
| Internet | Required for the **first** install (PyTorch + models ~ hundreds of MB to ~2 GB) |
| NVIDIA GPU (optional) | Windows/Linux only. Recent drivers. If absent, HerbivoR uses CPU automatically. |
| Apple GPU | On Mac, Metal (**MPS**) is used automatically when available — no extra step |

**You do not need to install Python yourself on Windows** when using the recommended installer. On macOS/Linux, Python 3.10+ must be available (see below).

A Hugging Face account is **not** required to download the public model weights.

---

## Recommended install (non-experts)

### Windows

#### Option A — Setup.exe (when provided on the Release page)

1. Open [HerbivoR Releases](https://github.com/mariosandovalmx/HerbivoR/releases).
2. Download **`HerbivoR-Setup-vX.Y.Z.exe`** (not the multi-GB developer bundles).
3. Double-click the file. If Windows SmartScreen appears, choose **More info** → **Run anyway** (unsigned builds may show this warning).
4. Follow the wizard. Accept the default folder (`%LOCALAPPDATA%\HerbivoR`) unless you need another location.
5. When Setup finishes copying files, the **HerbivoR Installer** window opens:
   - Leave **Auto-detect GPU** selected (recommended).
   - Click **Install** and wait (often **5–20 minutes**).
6. When you see **Installation completed**, open **HerbivoR** from the Desktop shortcut (leaf icon) or Start Menu.

#### Option B — Source ZIP + one-click installer

1. Download **Source code (zip)** from the same Releases page.
2. Right-click the ZIP → **Extract All…** to a folder you can write to (for example `Documents\HerbivoR`).
3. Open the extracted folder and double-click **`Install_HerbivoR.bat`**.
4. If Python is missing, Windows will download a **private** Python under `%LOCALAPPDATA%\HerbivoR\Python` (no PATH changes).
5. In the installer window, keep **Auto-detect GPU**, click **Install**, and wait until it finishes.
6. Start the app with **`HerbivoR.lnk`** (leaf icon) or **`HerbivoR.bat`**.

**GPU behavior (Windows):**

| Detection | Result |
|-----------|--------|
| `nvidia-smi` works | Installs PyTorch **CUDA 12.4** |
| No NVIDIA GPU / detection fails | Installs **CPU** PyTorch |
| You choose **CPU only** in the GUI | Forces CPU |
| You choose **NVIDIA CUDA** | Forces CUDA (needs a working NVIDIA driver) |

---

### macOS

#### Option A — DMG (when provided on the Release page)

1. Download **`HerbivoR-vX.Y.Z.dmg`** from Releases.
2. Open the DMG and drag the **HerbivoR leaf icon** onto the **Applications** folder shown beside it.
3. Eject the DMG, then open **HerbivoR** from Applications.
4. Approve HerbivoR once (see below).
5. On first launch, accept the license and click **Install**. HerbivoR downloads its private environment, PyTorch, and model weights; this can take 5–20 minutes.
6. When setup finishes, close the setup window. HerbivoR opens automatically and future launches start the app directly.

> **Do not double-click HerbivoR while it is still inside the DMG window.** macOS
> refuses to launch a downloaded app from a read-only disk image and reports only
> *"The application "HerbivoR.app" can't be opened."*, with no way to continue.
> Copy it to **Applications** first. The same instructions ship inside the DMG as
> `READ ME FIRST.txt`.

#### Approving HerbivoR on first launch

HerbivoR is distributed without an Apple Developer ID certificate, so macOS shows
a security prompt the first time you open it. Every recipient sees this once:

1. Double-click **HerbivoR** in **Applications**. macOS shows *"HerbivoR" Not
   Opened / Apple could not verify "HerbivoR" is free of malware*. Click **Done**
   (not *Move to Trash*).
2. Open **System Settings → Privacy & Security**, scroll to **Security**, and
   click **Open Anyway** next to the HerbivoR entry. Confirm with Touch ID or
   your password.
3. Double-click **HerbivoR** again and click **Open Anyway**.

On macOS 13 and 14 you can instead right-click **HerbivoR** in Applications,
choose **Open**, and click **Open** in the dialog.

Terminal equivalent, if you prefer one command:

```bash
xattr -dr com.apple.quarantine /Applications/HerbivoR.app
```

On Apple silicon, the standard macOS PyTorch wheel already includes the
**Metal Performance Shaders (MPS)** backend; there is no separate “Metal” wheel.
HerbivoR prefers native `arm64` execution and uses MPS automatically when it is
available. If pip ends with `from versions: none`, check the preceding messages:
this commonly means PyPI was temporarily unreachable rather than that Metal is
unsupported.

#### Option B — Source ZIP / Git clone

```bash
cd /path/to/HerbivoR
chmod +x Install_HerbivoR.command install.sh herbivor.sh packaging/create_macos_app.sh
./Install_HerbivoR.command
```

PyTorch on macOS includes **Metal (MPS)**. HerbivoR uses the GPU automatically when `torch.backends.mps.is_available()` is true.

---

### Linux

1. Install Python 3.10+ (and `venv` / `pip`) with your distribution packages.
2. Extract or clone HerbivoR.
3. Run:

```bash
chmod +x Install_HerbivoR.command install.sh herbivor.sh
./Install_HerbivoR.command
# or: ./install.sh
```

CUDA is selected automatically if `nvidia-smi` works; otherwise CPU wheels are installed.

Launch with:

```bash
./herbivor.sh
```

---

## After installation — first analysis

1. Open HerbivoR.
2. Go to the **Project** tab.
3. Set an **Input folder** (photos) and an **Output folder**.
4. Click **Check installation**.
   - This verifies packages and downloads any missing models into `models/`.
   - All required models should show **OK**.
5. Run the pipeline in order:
   1. **Segmentation** (BiRefNet + MobileSAM recommended)
   2. **Contour / ROI** (UNET Shape; optional Edit Contour)
   3. **Analysis** (damage U-Net; optional Edit Damage)
6. Results appear under `{output}/analyzed/`:
   - `results.csv`
   - `*_analyzed.jpg` overlays

### Multiple leaves in one photo

On the **Project** tab, enable **Multiple leaves per photo** when each input image
contains several leaves that do **not** touch or overlap.

- Uses BiRefNet + MobileSAM only (method A on the Segmentation tab).
- Writes one file per leaf: `{photo}_leaf_1.png`, `{photo}_leaf_2.png`, …
- Contour / Analysis treat each file as a normal leaf; `results.csv` gets one row per leaf.
- Contour and Damage models do **not** need re-training.
- Mutually exclusive with **Skip segmentation**.
- Limitation: touching or overlapping leaves may be detected as a single component.

---

## How to open HerbivoR later

| Platform | How to open |
|----------|-------------|
| Windows | Desktop / folder shortcut **`HerbivoR.lnk`** (leaf icon), or **`HerbivoR.bat`** |
| macOS | **`HerbivoR.app`**, or `./herbivor.sh` |
| Linux | `./herbivor.sh` |
| Any OS | `.venv/bin/python -m gui.main` (Windows: `.venv\Scripts\python.exe -m gui.main`) |

If the window does not appear on Windows, open `gui_error.log` in the HerbivoR folder.

---

## Repair / reinstall

| Goal | Action |
|------|--------|
| Missing models only | Project → **Check installation**, or run `download_models.py` inside `.venv` |
| Broken packages / GPU change | Run **`Install_HerbivoR.bat`** / **`Install_HerbivoR.command`** again |
| Confirm GPU | `.venv\Scripts\python.exe check_gpu.py` (Windows) or `.venv/bin/python check_gpu.py` |

---

## Advanced install (developers)

Use these only if you already manage Python yourself:

| Platform | Script |
|----------|--------|
| Windows CPU | `Install_CPU.bat` |
| Windows CUDA | `Install_CUDA.bat` |
| Windows menu | `Install.bat` |
| macOS / Linux | `./install.sh` |
| Manual | Create `.venv`, install Torch from [pytorch.org](https://pytorch.org), then `pip install -r requirements.txt` and `python download_models.py` |

Details: [INSTALL.md](INSTALL.md).

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| SmartScreen / “unknown publisher” | More info → Run anyway; prefer the official GitHub Release asset |
| Installer stuck on download | Check firewall/VPN; retry; ensure enough disk space |
| CUDA selected but inference uses CPU | Update NVIDIA drivers; run `check_gpu.py`; or reinstall with **CPU only** |
| `Python 3.10+ was not found` (Mac/Linux) | Install Python from python.org or your package manager, then re-run the installer |
| Models missing / failed download | Project → Check installation; or `.venv\Scripts\python.exe download_models.py` |
| GUI crash / blank window | Run with a visible console: `.venv\Scripts\python.exe -m gui.main` and read `gui_error.log` |
| Antivirus quarantines Setup.exe | Allow/whitelist the file from the official Release; code signing may come in a later version |
| macOS: “The application "HerbivoR.app" can't be opened.” | You launched it from inside the DMG. Drag it to **Applications** first, then approve it under **System Settings → Privacy & Security** |
| macOS: “Apple could not verify HerbivoR is free of malware” | Expected for this unsigned build. Click **Done**, then **Open Anyway** in **System Settings → Privacy & Security** |

---

## Model downloads (automatic)

Installers and **Check installation** download:

| File | Source |
|------|--------|
| `best_unet_shape.pth`, `best_model.pth` | [Hugging Face: mariosandovalmx/HerbivoR](https://huggingface.co/mariosandovalmx/HerbivoR) |
| `mobile_sam.pt` | [Ultralytics assets](https://github.com/ultralytics/assets/releases) (third-party, Apache-2.0) |

Weights are stored in the local `models/` folder (not shipped inside the small Setup/DMG).

---

## Uninstall (Windows Setup.exe)

Use **Settings → Apps → HerbivoR → Uninstall**, or the Start Menu uninstall entry. This removes the app folder under `%LOCALAPPDATA%\HerbivoR`. The optional private Python under `%LOCALAPPDATA%\HerbivoR\Python` may remain; you can delete that folder manually if you no longer need it.

For a source-folder install, delete the HerbivoR directory and (optional) `%LOCALAPPDATA%\HerbivoR\Python`.

---

## License and citation

HerbivoR **software** and **HerbivoR-trained model weights** (`best_unet_shape.pth`,
`best_model.pth`) are free for **noncommercial research and education**
([PolyForm Noncommercial License 1.0.0](LICENSE)).

- **Commercial use** (selling the software, paid services, commercial products/workflows) requires **prior written permission** from the copyright holder.
- If you use HerbivoR (or its trained weights) in a **publication, thesis, or presentation**, you **must cite / credit** it — see [CITATION.cff](CITATION.cff).
- Third-party weights (MobileSAM, BiRefNet) keep their own licenses (e.g. Apache-2.0) — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Installers display the full agreement and require acceptance. After install, keep
`LICENSE`, `THIRD_PARTY_NOTICES.md`, and `CITATION.cff` in the application folder.

Model card on Hugging Face: https://huggingface.co/mariosandovalmx/HerbivoR

---

## Getting help

- Repository: https://github.com/mariosandovalmx/HerbivoR  
- Version file in your install: `VERSION`  
- Changelog: [CHANGELOG.md](CHANGELOG.md)
