# HerbivoR

**HerbivoR** is a desktop GUI for quantifying leaf herbivory damage from photographs.

**Version:** see [`VERSION`](VERSION) · **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) · **User guide:** [`USER_GUIDE.md`](USER_GUIDE.md) · **Repo:** [github.com/mariosandovalmx/HerbivoR](https://github.com/mariosandovalmx/HerbivoR)

Pipeline:

1. **Segmentation** — isolate leaves (BiRefNet + MobileSAM, Intact Leaves, or Interactive)
2. **Contour / ROI** — reconstruct the leaf silhouette (UNET Shape) with interactive editing
3. **Analysis** — measure herbivory damage (Damage U-Net) with interactive Add/Remove/Line/Polygon tools

---

## Quick start (recommended)

1. Download from [Releases](https://github.com/mariosandovalmx/HerbivoR/releases):
   - **Windows:** `HerbivoR-Setup-vX.Y.Z.exe` (or Source ZIP + `Install_HerbivoR.bat`)
   - **macOS:** `HerbivoR-vX.Y.Z.dmg` (or Source ZIP + `Install_HerbivoR.command`)
2. On macOS, open the DMG and drag the **HerbivoR leaf icon** onto **Applications**, then open the app *from Applications*. Do not launch it from inside the DMG window. On Windows, run Setup.
   - macOS shows **"Apple could not verify HerbivoR is free of malware"** on first launch, because these builds are not signed with a paid Apple Developer ID. Click **Done**, then approve HerbivoR once under **System Settings → Privacy & Security → Open Anyway**. Full steps are in ` READ ME FIRST.txt` inside the DMG and in [USER_GUIDE.md](USER_GUIDE.md#macos).
3. Complete the first-time setup (downloads PyTorch + models automatically; GPU is auto-detected on Windows/Linux).
4. Open **HerbivoR** from Applications / the Windows shortcut / `./herbivor.sh`.

**Step-by-step for every OS:** **[USER_GUIDE.md](USER_GUIDE.md)** · short reference: **[INSTALL.md](INSTALL.md)**.

You do **not** need to install Python manually on Windows when using the recommended installer.

---

## Requirements

- ~3–6 GB free disk for the environment + weights
- Internet for the first install
- Optional: NVIDIA GPU (Windows/Linux). On Mac, Metal (MPS) is used automatically when available.

Model weights (~226 MB) are downloaded by the installer or **Check installation**:

- HerbivoR U-Nets: [`mariosandovalmx/HerbivoR`](https://huggingface.co/mariosandovalmx/HerbivoR)
- MobileSAM: [Ultralytics assets](https://github.com/ultralytics/assets/releases) (third-party, Apache-2.0)

---

## Project layout

```
HerbivoR/
├── gui/                      # CustomTkinter desktop application
├── packaging/                # Bootstrap installer, Setup/DMG builders, optional PyInstaller
├── models/                   # Weights (downloaded; git-ignored)
├── Install_HerbivoR.bat      # Windows one-click installer (recommended)
├── Install_HerbivoR.command  # Source-install fallback for macOS/Linux
├── Install_CPU.bat / Install_CUDA.bat / Install.bat   # advanced Windows
├── install.sh / herbivor.sh
└── USER_GUIDE.md
```

---

## Typical workflow

1. **Project** — choose input and output folders; **Check installation**. Optional:
   **Multiple leaves per photo** (BiRefNet + MobileSAM splits separated leaves into
   `{photo}_leaf_1`, `{photo}_leaf_2`, … — one CSV row each after Analysis).
2. **Segmentation** — BiRefNet + MobileSAM (recommended)
3. **Contour / ROI** — UNET Shape; optionally **Edit Contour**
4. **Analysis** — damage U-Net; optionally **Edit Damage**

Results appear under `{output}/analyzed/` (`results.csv` + overlay images).

**Multi-leaf note:** leaves in the same photo should not touch or overlap; otherwise
they may merge into one component. Contour and Damage models are unchanged.

---

## License

HerbivoR **software** and **HerbivoR-trained U-Net weights** are licensed under the
**PolyForm Noncommercial License 1.0.0** for **noncommercial research and education**
only. Commercial use requires prior written permission from the copyright holder.
If you use HerbivoR (or its trained weights) in a publication or presentation, you
must cite / credit it (see [`CITATION.cff`](CITATION.cff)).

Third-party components (e.g. MobileSAM, BiRefNet) remain under their own licenses —
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Windows/macOS installers show the **full license agreement** and require acceptance
before install.

See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[CITATION.cff](CITATION.cff). Model card:
https://huggingface.co/mariosandovalmx/HerbivoR

## Citation

See [CITATION.cff](CITATION.cff).
