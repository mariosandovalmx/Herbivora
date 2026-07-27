"""Label + entry + Browse row for folders or files."""

from __future__ import annotations

from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk


class PathPickerRow(ctk.CTkFrame):
    def __init__(
        self,
        master,
        label: str,
        *,
        is_dir: bool = True,
        filetypes: list[tuple[str, str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)
        self._is_dir = is_dir
        self._filetypes = filetypes or [("PyTorch", "*.pth"), ("All Files", "*.*")]
        self._change_cb: Callable[[], None] | None = None

        ctk.CTkLabel(self, text=label, width=160, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self._var = ctk.StringVar()
        self._var.trace_add("write", self._on_var_change)
        self._entry = ctk.CTkEntry(self, textvariable=self._var)
        self._entry.grid(row=0, column=1, sticky="ew", padx=4)
        ctk.CTkButton(self, text="Browse...", width=90, command=self._browse).grid(
            row=0, column=2, padx=4
        )

    def set_change_callback(self, cb: Callable[[], None]) -> None:
        """Register a callback fired whenever the path value changes."""
        self._change_cb = cb

    def _on_var_change(self, *_) -> None:
        if self._change_cb:
            self._change_cb()

    def get(self) -> str:
        return self._var.get().strip()

    def set(self, value: str) -> None:
        self._var.set(value)

    def _browse(self) -> None:
        if self._is_dir:
            path = filedialog.askdirectory(title="Select Folder")
        else:
            path = filedialog.askopenfilename(title="Select File", filetypes=self._filetypes)
        if path:
            self._var.set(path)

