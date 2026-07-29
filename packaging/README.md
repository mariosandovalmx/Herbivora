# Packaging notes for HerbivoR (maintainers)

## End-user installers (preferred)

| Artifact | Builder | Notes |
|----------|---------|-------|
| `HerbivoR-Setup-vVERSION.exe` | [`build_windows_setup.bat`](build_windows_setup.bat) + Inno Setup 6 | Unpacks source and runs `Install_HerbivoR.bat`. License page uses [`installer_license.txt`](installer_license.txt) (built from `LICENSE` + `THIRD_PARTY_NOTICES.md`). |
| `HerbivoR-vVERSION.dmg` | [`build_macos_dmg.sh`](build_macos_dmg.sh) via **GitHub Actions** (or any Mac) | Source tree + `Install_HerbivoR.command` |

Core logic: [`bootstrap_install.py`](bootstrap_install.py) (GPU detect, private Python on Windows, venv, Torch, deps, models, shortcuts).

Windows private Python helper: [`ensure_windows_python.ps1`](ensure_windows_python.ps1).

Attach **only** these small bootstraps to GitHub Releases. Torch and models download at install time.

### macOS DMG without a local Mac

You **cannot** build a native `.dmg` on Windows (`hdiutil` is macOS-only). Use CI:

1. Publish a GitHub Release for the version tag (with `HerbivoR-Setup-v*.exe` if you have it).
2. Workflow [`.github/workflows/macos-dmg.yml`](../.github/workflows/macos-dmg.yml) runs on `macos-latest`, builds `HerbivoR-vVERSION.dmg`, and uploads it to that Release.
3. The same workflow uploads **`SHA256SUMS`** (hashes of the DMG and any Setup.exe already on the Release).

Manual re-run: **Actions → macOS DMG → Run workflow** and enter the tag (e.g. `v1.3.5`).

Optional local build (only if you have a Mac):

```bash
chmod +x packaging/build_macos_dmg.sh
./packaging/build_macos_dmg.sh
```

### Verify integrity

```bash
# macOS / Linux
shasum -a 256 -c SHA256SUMS
```

```powershell
# Windows (PowerShell) — compare to the line in SHA256SUMS
Get-FileHash .\HerbivoR-Setup-v1.3.5.exe -Algorithm SHA256
```

## Optional PyInstaller (not for normal Releases)

Large onedir bundles; easy to exceed GitHub’s 2 GB asset limit.

```bat
pip install -r requirements-dev.txt
REM Ensure the active venv already has the desired torch (CPU or CUDA), then:
packaging\build_windows.bat
```

```bash
chmod +x packaging/build_macos.sh
./packaging/build_macos.sh
```

## Models

Weights are never bundled. Use the bootstrap installer, **Check installation**, or `download_models.py`.
