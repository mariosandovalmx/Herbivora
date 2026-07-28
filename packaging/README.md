# Packaging notes for HerbivoR

## Windows

```bat
packaging\build_windows.bat
```

Produces:

- `dist/HerbivoR/` — onedir folder with `HerbivoR.exe`
- `dist/HerbivoR-windows-vX.Y.Z.zip` — copy this to another PC

Do **not** commit `dist/` or `build/`.

## macOS

Run on a Mac:

```bash
chmod +x packaging/build_macos.sh
./packaging/build_macos.sh
```

## Models

Weights are never bundled. On first run use **Check installation** or place files in `models/` next to the executable.
