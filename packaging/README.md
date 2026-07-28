# Packaging notes for HerbivoR (maintainers only)

**Normal users should not use this.** Releases ship source code + `Install_*.bat` / `install.sh`.

PyInstaller bundles are optional, large, and not attached to GitHub Releases.

## Windows

```bat
pip install -r requirements-dev.txt
REM Ensure the active venv already has the desired torch (CPU or CUDA), then:
packaging\build_windows.bat
```

## macOS

```bash
chmod +x packaging/build_macos.sh
./packaging/build_macos.sh
```

## Models

Weights are never bundled. Use **Check installation** or `download_models.py`.
