"""Resizable split pane (controls | preview)."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

# Default ratio of left panel (options) and log panel (right).
OPTIONS_PANEL_RATIO = 1 / 3
LOG_PANEL_RATIO = 1 / 3


def _root_width(widget: tk.Misc, fallback: int = 1320) -> int:
    root = widget.winfo_toplevel()
    try:
        root.update_idletasks()
        w = root.winfo_width()
        if w > 100:
            return w
    except tk.TclError:
        pass
    return fallback


def panel_width_for_ratio(
    widget: tk.Misc, ratio: float, *, minimum: int = 200, fallback_total: int = 1320
) -> int:
    return max(minimum, int(_root_width(widget, fallback_total) * ratio))


class SplitPane(ctk.CTkFrame):
    """Two columns with a draggable sash (controls | preview)."""

    def __init__(
        self,
        master,
        *,
        left_ratio: float = OPTIONS_PANEL_RATIO,
        left_min: int = 220,
        right_min: int = 280,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._left_ratio = left_ratio
        self._ratio_set = False

        sash = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
        if isinstance(sash, (list, tuple)):
            sash = sash[0] if ctk.get_appearance_mode() == "Light" else sash[1]

        self._paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashwidth=7,
            sashrelief=tk.RAISED,
            opaqueresize=True,
            bg=sash,
            bd=0,
        )
        self._paned.grid(row=0, column=0, sticky="nsew")

        self._left_host = tk.Frame(self._paned, bg=sash)
        self._right_host = tk.Frame(self._paned, bg=sash)
        self._paned.add(self._left_host, minsize=left_min, width=left_min)
        self._paned.add(self._right_host, minsize=right_min)
        self.after_idle(self._apply_left_ratio)

    def _apply_left_ratio(self) -> None:
        if self._ratio_set:
            return
        try:
            total = self._paned.winfo_width()
            if total < 200:
                self.after(40, self._apply_left_ratio)
                return
            target = panel_width_for_ratio(self, self._left_ratio, minimum=220)
            # 1/3 of the window width, without exceeding the preview's minimum width
            pos = min(target, max(220, total - 280))
            self._paned.sash_place(0, pos, 0)
            self._ratio_set = True
        except tk.TclError:
            self.after(40, self._apply_left_ratio)

    def set_left(self, widget: ctk.CTkBaseClass) -> None:
        widget.pack(in_=self._left_host, fill="both", expand=True, padx=(8, 4), pady=8)

    def set_right(self, widget: ctk.CTkBaseClass) -> None:
        widget.pack(in_=self._right_host, fill="both", expand=True, padx=(4, 8), pady=8)

    def bind_sash_moved(self, callback: Callable[[], None]) -> None:
        self._paned.bind("<ButtonRelease-1>", lambda _e: callback())


class MainSplitPane(ctk.CTkFrame):
    """Main area (tabs) | optional log (hidden by default)."""

    def __init__(
        self,
        master,
        *,
        log_min: int = 200,
        log_visible: bool = False,
        log_ratio: float = LOG_PANEL_RATIO,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._log_min = log_min
        self._log_ratio = log_ratio
        self._saved_log_width: int | None = None

        sash = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
        if isinstance(sash, (list, tuple)):
            sash = sash[0] if ctk.get_appearance_mode() == "Light" else sash[1]

        self._paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashwidth=7,
            sashrelief=tk.RAISED,
            opaqueresize=True,
            bg=sash,
            bd=0,
        )
        self._paned.grid(row=0, column=0, sticky="nsew")

        self._left_host = tk.Frame(self._paned, bg=sash)
        self._right_host = tk.Frame(self._paned, bg=sash)
        self._paned.add(self._left_host, minsize=400)

        self._log_visible = log_visible
        if log_visible:
            self._add_log_pane()

    def set_tabs(self, widget: ctk.CTkBaseClass) -> None:
        widget.pack(in_=self._left_host, fill="both", expand=True, padx=(8, 4), pady=8)

    def set_log(self, widget: ctk.CTkBaseClass) -> None:
        widget.pack(in_=self._right_host, fill="both", expand=True, padx=(4, 8), pady=8)

    @property
    def log_visible(self) -> bool:
        return self._log_visible

    def _default_log_width(self) -> int:
        return panel_width_for_ratio(self, self._log_ratio, minimum=self._log_min)

    def _add_log_pane(self) -> None:
        if self._saved_log_width is not None:
            width = self._saved_log_width
            self._paned.add(self._right_host, minsize=self._log_min, width=width)
            self._log_visible = True
            self.after_idle(lambda w=width: self._place_log_width(w))
        else:
            width = self._default_log_width()
            self._paned.add(self._right_host, minsize=self._log_min, width=width)
            self._log_visible = True
            self.after_idle(self._apply_log_ratio)

    def _place_log_width(self, log_width: int) -> None:
        try:
            total = self._paned.winfo_width()
            if total < 200:
                self.after(40, lambda: self._place_log_width(log_width))
                return
            self._paned.sash_place(0, max(0, total - log_width), 0)
        except tk.TclError:
            self.after(40, lambda: self._place_log_width(log_width))

    def _apply_log_ratio(self) -> None:
        """Places the sash so the log (right panel) takes up ~1/3."""
        try:
            total = self._paned.winfo_width()
            if total < 200:
                self.after(40, self._apply_log_ratio)
                return
            self._paned.sash_place(0, int(total * (1.0 - self._log_ratio)), 0)
        except tk.TclError:
            self.after(40, self._apply_log_ratio)

    def toggle_log(self) -> bool:
        """Toggles log visibility. Returns the new state (True = visible)."""
        if self._log_visible:
            try:
                total = self._paned.winfo_width()
                sash_x = self._paned.sash_coord(0)[0]
                self._saved_log_width = max(self._log_min, total - sash_x)
            except (tk.TclError, IndexError):
                pass
            self._paned.forget(self._right_host)
            self._log_visible = False
        else:
            self._add_log_pane()
        return self._log_visible

