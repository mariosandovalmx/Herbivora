"""Lightweight startup splash (logo + Loading...) on the main CTk root."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Optional


_BG = "#1a1a1a"
_WIDTH = 320
_HEIGHT = 280


@dataclass
class SplashHandle:
    """Handle for a temporary splash Toplevel."""

    window: tk.Toplevel
    _photo: Optional[object] = None  # keep ImageTk ref alive

    def update(self) -> None:
        """Force a paint so the splash appears before heavy work."""
        try:
            self.window.update_idletasks()
            self.window.update()
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass


def show_splash_toplevel(master: tk.Misc) -> SplashHandle:
    """Splash as Toplevel on the existing CTk root (covers UI construction).

    Must not create a separate ``tk.Tk()`` root: destroying that root on Windows
    can post WM_QUIT and make ``mainloop()`` exit immediately after startup.
    """
    top = tk.Toplevel(master)
    top.overrideredirect(True)
    try:
        top.attributes("-topmost", True)
    except tk.TclError:
        pass

    top.update_idletasks()
    sw = top.winfo_screenwidth()
    sh = top.winfo_screenheight()
    x = max(0, (sw - _WIDTH) // 2)
    y = max(0, (sh - _HEIGHT) // 2)
    top.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")
    top.configure(bg=_BG)

    frame = tk.Frame(top, bg=_BG)
    frame.pack(expand=True, fill="both", padx=24, pady=24)

    photo = None
    try:
        from PIL import Image, ImageTk

        from gui.icons import png_path

        png = png_path()
        if png.is_file():
            img = Image.open(png).convert("RGBA")
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img, master=top)
            tk.Label(frame, image=photo, bg=_BG).pack(pady=(16, 12))
    except Exception:
        pass

    tk.Label(
        frame,
        text="HerbivoR",
        font=("Segoe UI", 16, "bold"),
        fg="#ecf0f1",
        bg=_BG,
    ).pack()
    tk.Label(
        frame,
        text="Loading...",
        font=("Segoe UI", 11),
        fg="#95a5a6",
        bg=_BG,
    ).pack(pady=(8, 0))

    handle = SplashHandle(window=top, _photo=photo)
    handle.update()
    return handle
