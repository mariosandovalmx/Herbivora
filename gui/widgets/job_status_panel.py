"""Mini status panel under carousel navigation."""

from __future__ import annotations

import customtkinter as ctk


class JobStatusPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color=("gray92", "gray22"), corner_radius=8, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self._cancel_cb = None
        self._hide_after_id: str | None = None
        self._running = False

        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        top_row.grid_columnconfigure(0, weight=1)

        self._label = ctk.CTkLabel(
            top_row,
            text="",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self._label.grid(row=0, column=0, sticky="ew")

        self._stop_btn = ctk.CTkButton(
            top_row,
            text="STOP",
            width=64,
            height=24,
            fg_color=("#c0392b", "#922b21"),
            hover_color=("#a93226", "#7b241c"),
            text_color="white",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_stop,
        )
        self._stop_btn.grid(row=0, column=1, padx=(8, 0))
        self._stop_btn.grid_remove()

        self._progress: ctk.CTkProgressBar | None = None
        self._ensure_progress(show=False)
        self.grid_remove()

    def set_cancel_callback(self, cb) -> None:
        self._cancel_cb = cb

    def _on_stop(self) -> None:
        if self._cancel_cb:
            self._cancel_cb()

    def _cancel_hide(self) -> None:
        if self._hide_after_id is not None:
            try:
                self.after_cancel(self._hide_after_id)
            except Exception:
                pass
            self._hide_after_id = None

    def _ensure_progress(self, *, show: bool) -> ctk.CTkProgressBar:
        """Create a fresh progress bar (destroys any previous one and its after-loop)."""
        old = self._progress
        self._progress = None
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
            try:
                if getattr(old, "_loop_after_id", None) is not None:
                    try:
                        old.after_cancel(old._loop_after_id)
                    except Exception:
                        pass
                old._loop_running = False  # type: ignore[attr-defined]
                old.destroy()
            except Exception:
                pass

        bar = ctk.CTkProgressBar(self, mode="determinate", height=10)
        bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        bar.set(0)
        if not show:
            bar.grid_remove()
        self._progress = bar
        return bar

    def _stop_progress(self) -> None:
        self._running = False
        self._ensure_progress(show=False)

    def start(self, message: str = "Analyzing...") -> None:
        self._cancel_hide()
        self._running = True
        self.grid()
        self._label.configure(text=message, text_color=("gray20", "gray90"))
        bar = self._ensure_progress(show=True)
        bar.configure(mode="indeterminate")
        bar.set(0)
        bar.start()
        self._stop_btn.grid()

    def complete(self, message: str = "Completed") -> None:
        """Stop animation immediately and hide the panel (no lingering bar)."""
        self._cancel_hide()
        self._stop_progress()
        self._stop_btn.grid_remove()
        self._label.configure(text=message, text_color=("#2d6a4f", "#95d5b2"))
        self.grid()
        self._hide_after_id = self.after(800, self.hide)

    def fail(self, message: str = "Error") -> None:
        self._cancel_hide()
        self._stop_progress()
        self._stop_btn.grid_remove()
        self._label.configure(text=message, text_color=("#8B0000", "#ff6b6b"))
        self.grid()
        self._hide_after_id = self.after(2000, self.hide)

    def hide(self) -> None:
        self._hide_after_id = None
        self._stop_progress()
        self._stop_btn.grid_remove()
        self.grid_remove()
