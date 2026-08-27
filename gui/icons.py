"""Resolve and apply the Herbivora app icon (window + optional CTkImage)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import customtkinter as ctk
    import tkinter as tk


def icon_dir() -> Path:
    """Directory containing herbivor.ico / herbivor_256.png."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "assets"
        if bundled.is_dir():
            return bundled
        return Path(sys.executable).resolve().parent / "assets"
    return Path(__file__).resolve().parent.parent / "assets"


def ico_path() -> Path:
    return icon_dir() / "herbivor.ico"


def png_path() -> Path:
    p256 = icon_dir() / "herbivor_256.png"
    if p256.is_file():
        return p256
    return icon_dir() / "herbivor_icon.png"


def apply_window_icon(root: tk.Misc) -> None:
    """Set the OS window / taskbar icon (replaces the default blue/Tk square)."""
    ico = ico_path()
    png = png_path()
    try:
        if sys.platform == "win32" and ico.is_file():
            root.iconbitmap(default=str(ico))
            root.iconbitmap(str(ico))
    except Exception:
        pass
    try:
        if png.is_file():
            from PIL import Image, ImageTk

            img = Image.open(png).convert("RGBA")
            photo = ImageTk.PhotoImage(img.resize((64, 64), Image.Resampling.LANCZOS))
            root.iconphoto(True, photo)
            # Keep a reference so Tk does not garbage-collect the image
            root._herbivor_icon_photo = photo  # type: ignore[attr-defined]
    except Exception:
        pass


def load_header_image(size: int = 28) -> ctk.CTkImage | None:
    """CTkImage for the in-app header bar, or None if assets are missing."""
    import customtkinter as ctk
    from PIL import Image

    png = png_path()
    if not png.is_file():
        return None
    try:
        im = Image.open(png).convert("RGBA")
        return ctk.CTkImage(light_image=im, dark_image=im, size=(size, size))
    except Exception:
        return None
