# Packaging notes for HerbivoR (maintainers)

## End-user installers (preferred)

| Artifact | Builder | Notes |
|----------|---------|-------|
| `HerbivoR-Setup-vVERSION.exe` | [`build_windows_setup.bat`](build_windows_setup.bat) + Inno Setup 6 | Unpacks source and runs `Install_HerbivoR.bat` |
| `HerbivoR-vVERSION.dmg` | [`build_macos_dmg.sh`](build_macos_dmg.sh) on macOS | Contains source + `Install_HerbivoR.command` |

Core logic: [`bootstrap_install.py`](bootstrap_install.py) (GPU detect, private Python on Windows, venv, Torch, deps, models, shortcuts).

Windows private Python helper: [`ensure_windows_python.ps1`](ensure_windows_python.ps1).

Attach **only** these small bootstraps to GitHub Releases. Torch and models download at install time.

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
