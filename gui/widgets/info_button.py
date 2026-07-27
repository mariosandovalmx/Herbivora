"""Info button widget for parameter descriptions."""

from __future__ import annotations

from tkinter import messagebox
import customtkinter as ctk

class InfoButton(ctk.CTkButton):
    def __init__(self, master, title: str, message: str, **kwargs):
        self.title_text = title
        self.message_text = message
        super().__init__(
            master,
            text="ⓘ",
            width=24,
            height=24,
            corner_radius=12,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent",
            text_color=("gray50", "gray70"),
            hover_color=("gray85", "gray30"),
            command=self._show_info,
            **kwargs
        )

    def _show_info(self) -> None:
        messagebox.showinfo(self.title_text, self.message_text)
