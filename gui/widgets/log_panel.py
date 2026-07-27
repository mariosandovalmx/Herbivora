"""Scrollable log panel."""

from __future__ import annotations

import customtkinter as ctk


class LogPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._toggle_cb = None

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text="Log Registry", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        self._status = ctk.CTkLabel(header, text="Ready", text_color="gray")
        self._status.grid(row=0, column=1, sticky="e", padx=8)
        self._hide_btn = ctk.CTkButton(
            header,
            text="Hide",
            width=72,
            height=24,
            command=self._on_toggle,
        )
        self._hide_btn.grid(row=0, column=2, sticky="e", padx=(4, 0))

        self._text = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=12))
        self._text.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._text.configure(state="disabled")

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(btn_row, text="Clear Log", width=100, command=self.clear).pack(
            side="left", padx=4
        )
        self._cancel_btn = ctk.CTkButton(
            btn_row, text="Stop", width=100, fg_color="#8B0000", command=self._on_cancel
        )
        self._cancel_btn.pack(side="right", padx=4)
        self._cancel_cb = None

    def set_toggle_callback(self, cb) -> None:
        self._toggle_cb = cb

    def _on_toggle(self) -> None:
        if self._toggle_cb:
            self._toggle_cb()

    def set_cancel_callback(self, cb) -> None:
        self._cancel_cb = cb

    def _on_cancel(self) -> None:
        if self._cancel_cb:
            self._cancel_cb()

    def set_status(self, text: str, color: str = "gray") -> None:
        self._status.configure(text=text, text_color=color)

    def append(self, line: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", line + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

