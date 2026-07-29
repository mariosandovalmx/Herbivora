# HerbivoR app icons

Source artwork: sketch leaf + larva on green tile (`herbivor_icon_source.png`).

Rebuild derived files after editing the source:

```bat
.venv\Scripts\python packaging\build_icon_assets.py
Create_HerbivoR_Shortcut.bat
```

| File | Use |
|------|-----|
| `herbivor_icon_source.png` | Approved original (edit this, then rebuild) |
| `herbivor_icon.png` | Master 1024×1024 (cropped / polished) |
| `herbivor_256.png` | GUI header + window `iconphoto` |
| `herbivor.ico` | Windows title bar / taskbar / `.lnk` / PyInstaller |

Windows `.bat` files cannot show a custom Explorer icon and briefly show a console if
double-clicked. After install (or run `Create_HerbivoR_Shortcut.bat`), use **`HerbivoR.lnk`**,
which targets `pythonw.exe` so only the GUI window appears.

On macOS run `./packaging/create_macos_app.sh` to build **`HerbivoR.app`** with this icon.
