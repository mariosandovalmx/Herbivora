"""Leaf shape (morphology) selector for the Contour tab.

Shows a row of icon buttons (smooth, elliptic, serrated, lobed) plus an
"Auto" text button. The selected shape is highlighted. When the shape changes,
the provided callback is invoked with the morphology key.
"""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk
from PIL import Image

from gui.paths import (
    MORPHOLOGY_CHOICES,
    MORPHOLOGY_DESCRIPTIONS,
    MORPHOLOGY_LABELS,
    SHAPE_ICON_FILES,
)

_ICON_SIZE = (40, 58)
_SELECTED_BORDER = ("#1f6aa5", "#3b8ed0")


class ShapeSelector(ctk.CTkFrame):
    """Row of leaf-shape buttons. Tracks the selected morphology key."""

    def __init__(
        self,
        master,
        on_shape_change: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self._on_shape_change = on_shape_change
        self._selected: str = "auto"
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._images: dict[str, ctk.CTkImage] = {}

        for shape in MORPHOLOGY_CHOICES:
            btn = self._build_button(shape)
            btn.pack(side="left", padx=3)
            self._buttons[shape] = btn

        self._refresh_highlight()

    def _build_button(self, shape: str) -> ctk.CTkButton:
        label = MORPHOLOGY_LABELS.get(shape, shape)
        tooltip = MORPHOLOGY_DESCRIPTIONS.get(shape, "")
        icon = self._load_icon(shape)

        if icon is not None:
            btn = ctk.CTkButton(
                self,
                text=label,
                image=icon,
                compound="top",
                width=64,
                height=82,
                fg_color=("gray90", "gray25"),
                text_color=("gray10", "gray90"),
                hover_color=("gray80", "gray35"),
                border_width=2,
                border_color=("gray90", "gray25"),
                font=ctk.CTkFont(size=11),
                command=lambda s=shape: self._select(s),
            )
        else:
            # "auto" has no icon: a taller text button to match the icon row.
            btn = ctk.CTkButton(
                self,
                text="Auto\ndetect",
                width=64,
                height=82,
                fg_color=("gray90", "gray25"),
                text_color=("gray10", "gray90"),
                hover_color=("gray80", "gray35"),
                border_width=2,
                border_color=("gray90", "gray25"),
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda s=shape: self._select(s),
            )
        if tooltip:
            _attach_tooltip(btn, tooltip)
        return btn

    def _load_icon(self, shape: str) -> ctk.CTkImage | None:
        path = SHAPE_ICON_FILES.get(shape)
        if path is None or not path.is_file():
            return None
        try:
            pil = Image.open(path).convert("RGBA")
        except Exception:
            return None
        img = ctk.CTkImage(light_image=pil, dark_image=pil, size=_ICON_SIZE)
        self._images[shape] = img  # keep a reference
        return img

    def _select(self, shape: str) -> None:
        if shape == self._selected:
            return
        self._selected = shape
        self._refresh_highlight()
        if self._on_shape_change is not None:
            self._on_shape_change(shape)

    def _refresh_highlight(self) -> None:
        for shape, btn in self._buttons.items():
            if shape == self._selected:
                btn.configure(border_color=_SELECTED_BORDER)
            else:
                btn.configure(border_color=("gray90", "gray25"))

    # ── Public API ──────────────────────────────────────────────────────────

    def get_shape(self) -> str:
        return self._selected

    def set_shape(self, shape: str, notify: bool = False) -> None:
        if shape not in self._buttons:
            shape = "auto"
        self._selected = shape
        self._refresh_highlight()
        if notify and self._on_shape_change is not None:
            self._on_shape_change(shape)


def _attach_tooltip(widget, text: str) -> None:
    """Lightweight hover tooltip (customtkinter has no built-in one)."""
    tip: dict[str, ctk.CTkToplevel | None] = {"win": None}

    def show(_event=None) -> None:
        if tip["win"] is not None:
            return
        try:
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
        except Exception:
            return
        win = ctk.CTkToplevel(widget)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.geometry(f"+{x}+{y}")
        ctk.CTkLabel(
            win,
            text=text,
            fg_color=("gray95", "gray20"),
            corner_radius=6,
            wraplength=220,
            justify="left",
            padx=8,
            pady=4,
        ).pack()
        tip["win"] = win

    def hide(_event=None) -> None:
        if tip["win"] is not None:
            tip["win"].destroy()
            tip["win"] = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)
