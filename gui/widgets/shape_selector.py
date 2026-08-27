"""Leaf shape (morphology) selector for the Contour tab.

Shows a row of icon buttons (smooth/entire, serrated, lobed, compound) plus an
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


class _ShapeTooltip:
    """Single shared tooltip for all shape buttons (avoids orphan CTkToplevel windows)."""

    _SHOW_DELAY_MS = 350

    def __init__(self, host: ctk.CTkFrame) -> None:
        self._host = host
        self._win: ctk.CTkToplevel | None = None
        self._after_id: str | None = None
        self._pending_text = ""
        self._pending_widget: ctk.CTkButton | None = None

    def attach(self, widget: ctk.CTkButton, text: str) -> None:
        widget.bind("<Enter>", lambda _e, t=text, w=widget: self._schedule_show(w, t), add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Button-1>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def hide_all(self) -> None:
        self._hide()

    def _schedule_show(self, widget: ctk.CTkButton, text: str) -> None:
        self._hide()
        self._pending_widget = widget
        self._pending_text = text
        self._after_id = widget.after(self._SHOW_DELAY_MS, self._show)

    def _show(self) -> None:
        self._after_id = None
        widget = self._pending_widget
        text = self._pending_text
        if widget is None or not text:
            return
        try:
            if not widget.winfo_exists():
                return
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
        except Exception:
            return
        self._hide()
        win = ctk.CTkToplevel(self._host)
        win.overrideredirect(True)
        win.transient(self._host.winfo_toplevel())
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
        self._win = win

    def _on_leave(self, _event=None) -> None:
        self._hide()

    def _hide(self, _event=None) -> None:
        if self._after_id is not None:
            try:
                self._host.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        win = self._win
        if win is not None:
            try:
                win.withdraw()
                win.destroy()
            except Exception:
                pass
            self._win = None
        self._pending_widget = None
        self._pending_text = ""


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
        self._tooltip = _ShapeTooltip(self)

        for shape in MORPHOLOGY_CHOICES:
            btn = self._build_button(shape)
            btn.pack(side="left", padx=3)
            self._buttons[shape] = btn

        self._refresh_highlight()
        self.bind("<Destroy>", lambda _e: self._tooltip.hide_all())

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
            self._tooltip.attach(btn, tooltip)
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
        self._tooltip.hide_all()
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

    def hide_tooltips(self) -> None:
        """Call when the Contour tab is hidden to avoid ghost tooltips."""
        self._tooltip.hide_all()
