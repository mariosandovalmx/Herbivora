"""Image carousel: navigation, zoom, pan, and optional eraser."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from typing import Callable

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from gui.paths import mask_path_for_white_bg, work_white_bg_copy
from gui.widgets.job_status_panel import JobStatusPanel
from image_io import format_error_hint, open_pil

ZOOM_WHEEL_FACTOR = 1.12
ZOOM_BUTTON_FACTOR = 1.25
MIN_ZOOM_LEVEL = 0.05
MAX_ZOOM_LEVEL = 25.0
SAVE_DEBOUNCE_MS = 120


class ImageCarousel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        title: str = "Preview",
        max_size: tuple[int, int] = (420, 320),
        show_source_selector: bool = True,
        eraser_enabled: bool = False,
        eraser_source_key: str = "Leaves (white_bg)",
        output_root_provider: Callable[[], Path | None] | None = None,
        show_job_status: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._eraser_enabled = eraser_enabled
        self._max_w, self._max_h = max_size
        self._paths: list[Path] = []
        self._index = 0
        self._pil_original: Image.Image | None = None
        self._tk_photo: ImageTk.PhotoImage | None = None
        self._source_getters: dict[str, Callable[[], list[Path]]] = {}
        self._path_provider: Callable[[], list[Path]] | None = None
        self._current_source_key = ""
        self._eraser_source_key = eraser_source_key
        self._output_root_provider = output_root_provider
        self._interaction_mode = "pan"
        self._eraser_active = False
        self._brush_size = 8
        self._brush_shape = "circle"
        self._last_erase_img_xy: tuple[float, float] | None = None
        self._save_after_id: str | None = None
        self._pending_mask_strokes: list[tuple[float, float, float, float, str]] = []
        self._edit_baseline_pil: Image.Image | None = None
        self._edit_baseline_mask: np.ndarray | None = None
        self._preview_cursor: tuple[int, int] | None = None
        self._undo_btn: ctk.CTkButton | None = None
        self._pick_dot_active = False
        self._pick_dot_press_xy: tuple[float, float] | None = None
        self._pick_dot_callback: Callable[[float, float, float], None] | None = None
        self._point_click_active = False
        self._point_click_callback: Callable[[float, float], None] | None = None
        self._preview_mask: np.ndarray | None = None
        self._preview_mask_rgb: tuple[int, int, int] = (135, 206, 250)  # light sky blue
        self._preview_mask_alpha: float = 0.45
        self._scale_circle: tuple[float, float, float] | None = None  # cx, cy, diameter (image px)
        self._image_changed_cb: Callable[[Path | None], None] | None = None
        self._source_changed_cb: Callable[[str], None] | None = None
        self._suppress_source_callback = False
        self._dirty = False  # True when eraser edits need writing to disk
        self._needs_initial_fit = False

        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_last: tuple[int, int] | None = None
        self._empty_message = "No images"

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(self._viewport_row, weight=1, minsize=200)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(header, text="Refresh", width=80, command=self.refresh).grid(
            row=0, column=1, sticky="e"
        )

        self._source_menu: ctk.CTkOptionMenu | None = None
        if show_source_selector:
            self._source_menu = ctk.CTkOptionMenu(
                header, values=["—"], command=self._on_source_change, width=200
            )
            self._source_menu.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        zoom_bar = ctk.CTkFrame(self, fg_color="transparent")
        zoom_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 4))
        ctk.CTkButton(zoom_bar, text="Zoom Out", width=88, command=self.zoom_out).pack(
            side="left", padx=2
        )
        ctk.CTkButton(zoom_bar, text="Zoom In", width=88, command=self.zoom_in).pack(
            side="left", padx=2
        )
        ctk.CTkButton(
            zoom_bar, text="Reset View", width=120, command=self.restore_view
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            zoom_bar,
            text="Scroll: zoom · Esc: reset view",
            text_color="gray",
        ).pack(side="left", padx=4)

        self._eraser_bar: ctk.CTkFrame | None = None
        self._mode_btn: ctk.CTkSegmentedButton | None = None
        self._shape_menu: ctk.CTkOptionMenu | None = None
        self._size_slider: ctk.CTkSlider | None = None
        self._size_label: ctk.CTkLabel | None = None
        if eraser_enabled:
            self._eraser_bar = ctk.CTkFrame(self, fg_color="transparent")
            self._eraser_bar.grid(row=self._tools_row, column=0, sticky="ew", padx=8, pady=(0, 4))
            ctk.CTkLabel(self._eraser_bar, text="Tool:").pack(side="left", padx=(0, 4))
            self._mode_btn = ctk.CTkSegmentedButton(
                self._eraser_bar,
                values=["Pan/Zoom", "Eraser"],
                command=self._on_mode_change,
                width=140,
            )
            self._mode_btn.set("Pan/Zoom")
            self._mode_btn.pack(side="left", padx=4)
            ctk.CTkLabel(self._eraser_bar, text="Shape:").pack(side="left", padx=(12, 4))
            self._shape_menu = ctk.CTkOptionMenu(
                self._eraser_bar,
                values=["Circle", "Square"],
                command=self._on_shape_change,
                width=100,
            )
            self._shape_menu.set("Circle")
            self._shape_menu.pack(side="left", padx=4)
            ctk.CTkLabel(self._eraser_bar, text="Size:").pack(side="left", padx=(12, 4))
            self._size_slider = ctk.CTkSlider(
                self._eraser_bar,
                from_=8,
                to=120,
                number_of_steps=28,
                command=self._on_brush_size_change,
                width=120,
                progress_color=("#2ECC71", "#1E8449"),
                button_color=("#2ECC71", "#1E8449"),
                button_hover_color=("#27AE60", "#196F3D"),
            )
            self._size_slider.set(self._brush_size)
            self._size_slider.pack(side="left", padx=4)
            self._size_label = ctk.CTkLabel(self._eraser_bar, text=f"{int(self._brush_size)} px")
            self._size_label.pack(side="left", padx=4)
            self._undo_btn = ctk.CTkButton(
                self._eraser_bar,
                text="Undo",
                width=88,
                command=self.undo_edits,
            )
            self._undo_btn.pack(side="left", padx=(12, 4))
            self._set_eraser_bar_enabled(False)

        # Previous | image | Next — side buttons stay visible in every tab.
        self._viewer_row = ctk.CTkFrame(self, fg_color="transparent")
        self._viewer_row.grid(row=self._viewport_row, column=0, sticky="nsew", padx=4, pady=4)
        self._viewer_row.grid_columnconfigure(1, weight=1)
        self._viewer_row.grid_rowconfigure(0, weight=1, minsize=200)

        self._prev_btn = ctk.CTkButton(
            self._viewer_row,
            text="◀ Prev",
            width=70,
            height=32,
            command=self.prev_image,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._prev_btn.grid(row=0, column=0, padx=(4, 2))

        self._viewport = ctk.CTkFrame(
            self._viewer_row, fg_color=("gray90", "gray20")
        )
        self._viewport.grid(row=0, column=1, sticky="nsew", padx=2)
        self._viewport.grid_columnconfigure(0, weight=1)
        self._viewport.grid_rowconfigure(0, weight=1)

        canvas_bg = "#ebebeb" if ctk.get_appearance_mode() == "Light" else "#2b2b2b"
        self._canvas = tk.Canvas(
            self._viewport,
            bg=canvas_bg,
            highlightthickness=0,
            cursor="hand2",
            width=self._max_w,
            height=self._max_h,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._v_scroll = ctk.CTkScrollbar(
            self._viewport,
            orientation="vertical",
            command=self._on_vscroll_command,
        )
        self._v_scroll.grid(row=0, column=1, sticky="ns")
        self._h_scroll = ctk.CTkScrollbar(
            self._viewport,
            orientation="horizontal",
            command=self._on_hscroll_command,
        )
        self._h_scroll.grid(row=1, column=0, sticky="ew")
        self._scroll_syncing = False

        self._next_btn = ctk.CTkButton(
            self._viewer_row,
            text="Next ▶",
            width=70,
            height=32,
            command=self.next_image,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._next_btn.grid(row=0, column=2, padx=(2, 4))

        caption_row = ctk.CTkFrame(self, fg_color="transparent")
        caption_row.grid(row=self._caption_row, column=0, sticky="ew", padx=8, pady=(0, 2))
        caption_row.grid_columnconfigure(0, weight=1)
        self._caption = ctk.CTkLabel(
            caption_row,
            text="",
            text_color="gray",
            wraplength=self._max_w,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        )
        self._caption.grid(row=0, column=0, sticky="ew")
        self._counter = ctk.CTkLabel(
            caption_row, text="0 / 0", font=ctk.CTkFont(size=16, weight="bold")
        )
        self._counter.grid(row=0, column=1, sticky="e", padx=(8, 0))

        self._job_status: JobStatusPanel | None = None
        if show_job_status:
            self._job_status = JobStatusPanel(self)
            self._job_status.grid(
                row=self._job_status_row,
                column=0,
                sticky="ew",
                padx=8,
                pady=(0, 8),
            )

        self.bind("<Left>", lambda _e: self.prev_image())
        self.bind("<Right>", lambda _e: self.next_image())
        self.bind("<Escape>", self._on_escape_reset_view)
        self._canvas.bind("<ButtonPress-1>", self._on_button1_press)
        self._canvas.bind("<B1-Motion>", self._on_button1_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_button1_release)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", self._on_wheel_linux)
        self._canvas.bind("<Button-5>", self._on_wheel_linux)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<Motion>", self._on_canvas_motion)
        self._canvas.bind("<Leave>", self._on_canvas_leave)
        self._canvas.bind("<Left>", lambda _e: self.prev_image())
        self._canvas.bind("<Right>", lambda _e: self.next_image())
        self._canvas.bind("<Escape>", self._on_escape_reset_view)

    @property
    def _tools_row(self) -> int:
        """Optional eraser / contour / damage edit toolbar (above the image)."""
        return 2

    @property
    def _viewport_row(self) -> int:
        """Image row with Previous (left) and Next (right)."""
        return 3

    @property
    def _caption_row(self) -> int:
        return 4

    @property
    def _job_status_row(self) -> int:
        return 5

    def job_status_start(self, message: str = "Analyzing...") -> None:
        if self._job_status is not None:
            self._job_status.start(message)

    def job_status_complete(self, message: str = "Completed") -> None:
        if self._job_status is not None:
            self._job_status.complete(message)

    def job_status_fail(self, message: str = "Error") -> None:
        if self._job_status is not None:
            self._job_status.fail(message)

    def job_status_hide(self) -> None:
        if self._job_status is not None:
            self._job_status.hide()

    def set_stop_callback(self, cb) -> None:
        if self._job_status is not None:
            self._job_status.set_cancel_callback(cb)

    def set_pick_dot_callback(self, cb: Callable[[float, float, float], None] | None) -> None:
        """Register a callback invoked as cb(center_x, center_y, diameter_px) in image space."""
        self._pick_dot_callback = cb

    def enable_pick_dot_mode(self, enabled: bool) -> None:
        """Toggle a one-shot mode where press-drag-release on the canvas marks a circle.

        Takes priority over eraser/pan while active. Does not modify the displayed
        image — only reports the picked center/diameter via set_pick_dot_callback.
        """
        self._pick_dot_active = enabled
        self._pick_dot_press_xy = None
        if enabled:
            self._point_click_active = False
        self._hide_pick_dot_preview()
        self._update_canvas_cursor()

    def set_point_click_callback(self, cb: Callable[[float, float], None] | None) -> None:
        """Register a callback invoked as cb(x, y) in image space (single click)."""
        self._point_click_callback = cb

    def enable_point_click_mode(self, enabled: bool) -> None:
        """Toggle single-click point mode for interactive MobileSAM prompts.

        Takes priority over eraser/pan while active, but yields to pick-dot mode.
        Fires set_point_click_callback with image-space coordinates on button release.
        """
        self._point_click_active = enabled
        if enabled:
            self._pick_dot_active = False
            self._pick_dot_press_xy = None
            self._hide_pick_dot_preview()
        self._update_canvas_cursor()

    def set_image_changed_callback(self, cb: Callable[[Path | None], None] | None) -> None:
        """Invoked after each image is shown (navigation / refresh)."""
        self._image_changed_cb = cb

    def set_source_changed_callback(self, cb: Callable[[str], None] | None) -> None:
        """Invoked after the source dropdown changes (or set_sources loads a key)."""
        self._source_changed_cb = cb

    @property
    def current_source_key(self) -> str:
        return self._current_source_key

    def set_preview_mask(
        self,
        mask: np.ndarray | None,
        *,
        rgb: tuple[int, int, int] = (135, 206, 250),
        alpha: float = 0.45,
    ) -> None:
        """Show a translucent color overlay for the given boolean/uint8 mask."""
        self._preview_mask = None if mask is None else np.asarray(mask)
        self._preview_mask_rgb = rgb
        self._preview_mask_alpha = float(max(0.0, min(1.0, alpha)))
        self._render_image()

    def clear_preview_mask(self) -> None:
        if self._preview_mask is None:
            return
        self._preview_mask = None
        self._render_image()

    def set_scale_circle(
        self,
        cx: float | None,
        cy: float | None = None,
        diameter: float | None = None,
    ) -> None:
        """Draw a persistent ring for the blue reference scale circle (image coords)."""
        if cx is None or cy is None or diameter is None or diameter <= 0:
            self._scale_circle = None
        else:
            self._scale_circle = (float(cx), float(cy), float(diameter))
        self._redraw_scale_circle()

    def clear_scale_circle(self) -> None:
        if self._scale_circle is None:
            try:
                self._canvas.delete("scale_circle")
            except Exception:
                pass
            return
        self._scale_circle = None
        try:
            self._canvas.delete("scale_circle")
        except Exception:
            pass

    def _redraw_scale_circle(self) -> None:
        try:
            self._canvas.delete("scale_circle")
        except Exception:
            pass
        if self._scale_circle is None or self._pil_original is None:
            return
        cx, cy, diameter = self._scale_circle
        scale = self._display_scale()
        if scale <= 0:
            return
        canvas_cx = self._pan_x + cx * scale
        canvas_cy = self._pan_y + cy * scale
        r = (diameter / 2.0) * scale
        color = "#00bcd4"
        self._canvas.create_oval(
            canvas_cx - r,
            canvas_cy - r,
            canvas_cx + r,
            canvas_cy + r,
            outline=color,
            width=2,
            tags="scale_circle",
        )
        self._canvas.create_line(
            canvas_cx - 5,
            canvas_cy,
            canvas_cx + 5,
            canvas_cy,
            fill=color,
            width=1,
            tags="scale_circle",
        )
        self._canvas.create_line(
            canvas_cx,
            canvas_cy - 5,
            canvas_cx,
            canvas_cy + 5,
            fill=color,
            width=1,
            tags="scale_circle",
        )
        self._canvas.tag_raise("scale_circle")

    def _composited_display_pil(self) -> Image.Image | None:
        if self._pil_original is None:
            return None
        if self._preview_mask is None:
            return self._pil_original
        base = np.asarray(self._pil_original.convert("RGB"), dtype=np.float32)
        h, w = base.shape[:2]
        mask = self._preview_mask
        if mask.shape[:2] != (h, w):
            mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_u8 > 127
        else:
            mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        if not np.any(mask_bool):
            return self._pil_original
        color = np.array(self._preview_mask_rgb, dtype=np.float32)
        a = self._preview_mask_alpha
        out = base.copy()
        out[mask_bool] = (1.0 - a) * out[mask_bool] + a * color
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def _on_mode_change(self, value: str) -> None:
        self._interaction_mode = "erase" if value == "Eraser" else "pan"
        self._update_canvas_cursor()
        if self._interaction_mode == "erase":
            self._ensure_preview_cursor()
            self._redraw_brush_preview()
        else:
            self._hide_brush_preview()

    def _on_shape_change(self, choice: str) -> None:
        self._brush_shape = "square" if choice == "Square" else "circle"
        self._redraw_brush_preview()

    def _on_brush_size_change(self, value: float) -> None:
        self._brush_size = max(8, int(value))
        if self._size_label is not None:
            self._size_label.configure(text=f"{self._brush_size} px")
        self._ensure_preview_cursor()
        self._redraw_brush_preview()

    def _ensure_preview_cursor(self) -> None:
        if self._preview_cursor is None:
            cw, ch = self._canvas_size()
            self._preview_cursor = (cw // 2, ch // 2)

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if not self._should_show_brush_preview():
            return
        self._preview_cursor = (event.x, event.y)
        self._redraw_brush_preview()

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self._hide_brush_preview()

    def _should_show_brush_preview(self) -> bool:
        return (
            self._eraser_active
            and self._interaction_mode == "erase"
            and self._pil_original is not None
            and self._preview_cursor is not None
        )

    def _brush_radius_canvas(self) -> float:
        return max(2.0, (self._brush_size / 2.0) * self._display_scale())

    def _hide_brush_preview(self) -> None:
        self._canvas.delete("brush_preview")

    def _redraw_brush_preview(self) -> None:
        self._hide_brush_preview()
        if not self._should_show_brush_preview() or self._preview_cursor is None:
            return
        cx, cy = self._preview_cursor
        r = self._brush_radius_canvas()
        color = "#e63946"
        if self._brush_shape == "circle":
            self._canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=color, width=2, tags="brush_preview",
            )
        else:
            self._canvas.create_rectangle(
                cx - r, cy - r, cx + r, cy + r,
                outline=color, width=2, tags="brush_preview",
            )
        self._canvas.tag_raise("brush_preview")  # above the image

    def undo_edits(self) -> None:
        if self._edit_baseline_pil is None or self._pil_original is None:
            return
        self._pending_mask_strokes.clear()
        self._pil_original = self._edit_baseline_pil.copy()
        self._dirty = False
        self._render_image()
        self._redraw_brush_preview()
        path = self._paths[self._index] if self._paths else None
        if path is None:
            return
        try:
            suffix = path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                self._pil_original.save(path, quality=95)
            else:
                self._pil_original.save(path)
            out_root = self._output_root_provider() if self._output_root_provider else None
            if out_root is not None:
                work_copy = work_white_bg_copy(out_root, path.name)
                if work_copy.parent.is_dir():
                    if suffix in (".jpg", ".jpeg"):
                        self._pil_original.save(work_copy, quality=95)
                    else:
                        self._pil_original.save(work_copy)
                if self._edit_baseline_mask is not None:
                    mask_path = mask_path_for_white_bg(path, out_root)
                    cv2.imwrite(str(mask_path), self._edit_baseline_mask)
        except OSError:
            pass

    def _capture_edit_baseline(self, path: Path) -> None:
        if self._pil_original is None:
            self._edit_baseline_pil = None
            self._edit_baseline_mask = None
            return
        self._edit_baseline_pil = self._pil_original.copy()
        self._edit_baseline_mask = None
        out_root = self._output_root_provider() if self._output_root_provider else None
        if out_root is not None:
            mask_path = mask_path_for_white_bg(path, out_root)
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self._edit_baseline_mask = mask.copy()

    def _set_eraser_bar_enabled(self, enabled: bool) -> None:
        self._eraser_active = enabled
        if self._eraser_bar is None:
            return
        state = "normal" if enabled else "disabled"
        for w in (self._mode_btn, self._shape_menu, self._size_slider, self._undo_btn):
            if w is not None:
                w.configure(state=state)
        if not enabled:
            self._interaction_mode = "pan"
            if self._mode_btn is not None:
                self._mode_btn.set("Pan/Zoom")
            self._hide_brush_preview()
        self._update_canvas_cursor()

    def _update_canvas_cursor(self) -> None:
        if self._pick_dot_active or self._point_click_active:
            self._canvas.configure(cursor="crosshair")
        elif not self._eraser_active or self._interaction_mode == "pan":
            self._canvas.configure(cursor="hand2")
        else:
            self._canvas.configure(cursor="dotbox")

    def _hide_pick_dot_preview(self) -> None:
        self._canvas.delete("pick_dot_preview")

    def _redraw_pick_dot_preview(self, canvas_cx: float, canvas_cy: float, canvas_r: float) -> None:
        self._hide_pick_dot_preview()
        color = "#06d6a0"
        self._canvas.create_oval(
            canvas_cx - canvas_r, canvas_cy - canvas_r,
            canvas_cx + canvas_r, canvas_cy + canvas_r,
            outline=color, width=2, tags="pick_dot_preview",
        )
        self._canvas.create_line(
            canvas_cx - 6, canvas_cy, canvas_cx + 6, canvas_cy,
            fill=color, width=1, tags="pick_dot_preview",
        )
        self._canvas.create_line(
            canvas_cx, canvas_cy - 6, canvas_cx, canvas_cy + 6,
            fill=color, width=1, tags="pick_dot_preview",
        )
        self._canvas.tag_raise("pick_dot_preview")

    def _update_eraser_for_source(self, source_key: str) -> None:
        if not self._eraser_enabled:
            return
        self._set_eraser_bar_enabled(source_key == self._eraser_source_key)

    def _canvas_size(self) -> tuple[int, int]:
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 10:
            w = self._max_w
        if h < 10:
            h = self._max_h
        return w, h

    def _fit_scale(self) -> float:
        if self._pil_original is None:
            return 1.0
        cw, ch = self._canvas_size()
        ow, oh = self._pil_original.size
        return min(cw / ow, ch / oh)

    def _display_scale(self) -> float:
        return self._fit_scale() * self._zoom_level

    def _displayed_size(self) -> tuple[int, int]:
        if self._pil_original is None:
            return 0, 0
        ow, oh = self._pil_original.size
        scale = self._display_scale()
        return max(1, int(ow * scale)), max(1, int(oh * scale))

    def _canvas_to_image_xy(self, cx: float, cy: float) -> tuple[float, float]:
        scale = self._display_scale()
        if scale <= 0:
            return 0.0, 0.0
        return (cx - self._pan_x) / scale, (cy - self._pan_y) / scale

    def _on_canvas_configure(self, _event=None) -> None:
        if self._pil_original is None:
            return
        if self._needs_initial_fit:
            self._ensure_view_fitted()
            return
        self._render_image()

    def _ensure_view_fitted(self) -> None:
        """Re-fit after layout settles (avoids blank 1×1 canvas on first paint).

        Only resets zoom/pan while ``_needs_initial_fit`` is True. Once the user
        zooms or pans, later layout callbacks must not wipe their view.
        """
        try:
            if self._pil_original is None:
                return
            if self._canvas.winfo_width() < 10 or self._canvas.winfo_height() < 10:
                if self._needs_initial_fit:
                    self.after(50, self._ensure_view_fitted)
                return
            if self._needs_initial_fit:
                self._needs_initial_fit = False
                self.restore_view()
            else:
                self._render_image()
        except Exception:
            pass

    def _on_button1_press(self, event: tk.Event) -> None:
        if self._pil_original is None:
            return
        self._canvas.focus_set()
        if self._pick_dot_active:
            self._pick_dot_press_xy = (float(event.x), float(event.y))
            self._hide_pick_dot_preview()
            return
        if self._point_click_active:
            return
        if self._eraser_active and self._interaction_mode == "erase":
            self._hide_brush_preview()
            self._last_erase_img_xy = self._canvas_to_image_xy(event.x, event.y)
            self._stamp_erase(self._last_erase_img_xy[0], self._last_erase_img_xy[1])
            return
        self._drag_last = (event.x, event.y)
        self._canvas.configure(cursor="fleur")

    def _on_button1_motion(self, event: tk.Event) -> None:
        if self._pil_original is None:
            return
        if self._pick_dot_active:
            if self._pick_dot_press_xy is not None:
                cx, cy = self._pick_dot_press_xy
                r = math.hypot(event.x - cx, event.y - cy)
                self._redraw_pick_dot_preview(cx, cy, r)
            return
        if self._point_click_active:
            return
        if self._eraser_active and self._interaction_mode == "erase":
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            if self._last_erase_img_xy is not None:
                self._stroke_erase(self._last_erase_img_xy[0], self._last_erase_img_xy[1], ix, iy)
            else:
                self._stamp_erase(ix, iy)
            self._last_erase_img_xy = (ix, iy)
            return
        if self._drag_last is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._pan_x += dx
        self._pan_y += dy
        self._needs_initial_fit = False
        self._drag_last = (event.x, event.y)
        self._draw_image()
        self._sync_scrollbars()

    def _on_button1_release(self, _event: tk.Event) -> None:
        if self._pick_dot_active:
            if self._pick_dot_press_xy is not None:
                canvas_cx, canvas_cy = self._pick_dot_press_xy
                canvas_r = math.hypot(_event.x - canvas_cx, _event.y - canvas_cy)
                if canvas_r >= 3:
                    img_cx, img_cy = self._canvas_to_image_xy(canvas_cx, canvas_cy)
                    scale = self._display_scale()
                    img_r = canvas_r / scale if scale > 0 else 0.0
                    if self._pick_dot_callback is not None and img_r > 0:
                        self._pick_dot_callback(img_cx, img_cy, img_r * 2.0)
            self._pick_dot_press_xy = None
            self._hide_pick_dot_preview()
            return
        if self._point_click_active:
            if self._pil_original is not None and self._point_click_callback is not None:
                ix, iy = self._canvas_to_image_xy(_event.x, _event.y)
                w, h = self._pil_original.size
                if 0 <= ix < w and 0 <= iy < h:
                    self._point_click_callback(ix, iy)
            return
        if self._eraser_active and self._interaction_mode == "erase":
            self._last_erase_img_xy = None
            self._flush_persist()
            self._update_canvas_cursor()
            self._preview_cursor = (_event.x, _event.y)
            self._redraw_brush_preview()
            return
        self._drag_last = None
        self._update_canvas_cursor()

    def _stamp_erase(self, ix: float, iy: float) -> None:
        if self._pil_original is None:
            return
        draw = ImageDraw.Draw(self._pil_original)
        r = self._brush_size / 2.0
        if self._brush_shape == "circle":
            draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=(255, 255, 255))
        else:
            draw.rectangle([ix - r, iy - r, ix + r, iy + r], fill=(255, 255, 255))
        self._queue_mask_stroke(ix, iy, ix, iy)
        self._render_image()
        self._schedule_persist()

    def _stroke_erase(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._pil_original is None:
            return
        draw = ImageDraw.Draw(self._pil_original)
        width = max(1, int(self._brush_size))
        draw.line([x0, y0, x1, y1], fill=(255, 255, 255), width=width)
        r = self._brush_size / 2.0
        if self._brush_shape == "circle":
            draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=(255, 255, 255))
        else:
            draw.rectangle([x1 - r, y1 - r, x1 + r, y1 + r], fill=(255, 255, 255))
        self._queue_mask_stroke(x0, y0, x1, y1)
        self._render_image()
        self._schedule_persist()

    def _queue_mask_stroke(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._pending_mask_strokes.append(
            (x0, y0, x1, y1, self._brush_shape)
        )

    def _schedule_persist(self) -> None:
        self._dirty = True
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(SAVE_DEBOUNCE_MS, self._flush_persist)

    def _flush_persist(self) -> None:
        """Write eraser edits back to disk. No-op when nothing was edited."""
        self._save_after_id = None
        if self._pil_original is None or not self._paths:
            return
        # Never rewrite images on mere navigation / refresh.
        if not self._dirty and not self._pending_mask_strokes:
            return
        path = self._paths[self._index]
        try:
            suffix = path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                self._pil_original.save(path, quality=95)
            else:
                self._pil_original.save(path)
            self._apply_mask_strokes(path)
            out_root = self._output_root_provider() if self._output_root_provider else None
            if out_root is not None:
                work_copy = work_white_bg_copy(out_root, path.name)
                if work_copy.parent.is_dir():
                    if suffix in (".jpg", ".jpeg"):
                        self._pil_original.save(work_copy, quality=95)
                    else:
                        self._pil_original.save(work_copy)
            self._dirty = False
        except (OSError, ValueError):
            pass
        except Exception:
            # Never let persistence errors break navigation / preview.
            pass

    def _apply_mask_strokes(self, white_bg_path: Path) -> None:
        out_root = self._output_root_provider() if self._output_root_provider else None
        if out_root is None or not self._pending_mask_strokes:
            self._pending_mask_strokes.clear()
            return
        mask_path = mask_path_for_white_bg(white_bg_path, out_root)
        if not mask_path.is_file():
            self._pending_mask_strokes.clear()
            return
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self._pending_mask_strokes.clear()
            return
        r = max(1, int(self._brush_size / 2))
        for x0, y0, x1, y1, shape in self._pending_mask_strokes:
            if shape == "circle":
                cv2.circle(mask, (int(x1), int(y1)), r, 0, -1)
                if (int(x0), int(y0)) != (int(x1), int(y1)):
                    cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 0, thickness=max(1, r * 2))
            else:
                cv2.rectangle(
                    mask,
                    (int(x1) - r, int(y1) - r),
                    (int(x1) + r, int(y1) + r),
                    0,
                    -1,
                )
        cv2.imwrite(str(mask_path), mask)
        self._pending_mask_strokes.clear()

    def _on_wheel(self, event: tk.Event) -> None:
        if self._pil_original is None:
            return
        factor = ZOOM_WHEEL_FACTOR if event.delta > 0 else 1.0 / ZOOM_WHEEL_FACTOR
        self._apply_zoom(factor, event.x, event.y)

    def _on_wheel_linux(self, event: tk.Event) -> None:
        if self._pil_original is None:
            return
        factor = ZOOM_WHEEL_FACTOR if event.num == 4 else 1.0 / ZOOM_WHEEL_FACTOR
        self._apply_zoom(factor, event.x, event.y)

    def _apply_zoom(self, factor: float, cx: int, cy: int) -> None:
        old_scale = self._display_scale()
        new_level = max(MIN_ZOOM_LEVEL, min(MAX_ZOOM_LEVEL, self._zoom_level * factor))
        if abs(new_level - self._zoom_level) < 1e-6:
            return
        ow, oh = self._pil_original.size
        old_dw = ow * old_scale
        old_dh = oh * old_scale
        if old_dw > 0 and old_dh > 0:
            rel_x = (cx - self._pan_x) / old_dw
            rel_y = (cy - self._pan_y) / old_dh
        else:
            rel_x = rel_y = 0.5
        self._zoom_level = new_level
        self._needs_initial_fit = False
        new_scale = self._display_scale()
        new_dw = ow * new_scale
        new_dh = oh * new_scale
        self._pan_x = cx - rel_x * new_dw
        self._pan_y = cy - rel_y * new_dh
        self._render_image()
        self._redraw_brush_preview()
        self._sync_scrollbars()

    def zoom_in(self) -> None:
        if self._pil_original is None:
            return
        cw, ch = self._canvas_size()
        self._apply_zoom(ZOOM_BUTTON_FACTOR, cw // 2, ch // 2)

    def zoom_out(self) -> None:
        if self._pil_original is None:
            return
        cw, ch = self._canvas_size()
        self._apply_zoom(1.0 / ZOOM_BUTTON_FACTOR, cw // 2, ch // 2)

    def _on_escape_reset_view(self, _event=None) -> str:
        self.restore_view()
        return "break"

    def restore_view(self) -> None:
        if self._pil_original is None:
            return
        self._zoom_level = 1.0
        self._center_image()
        self._render_image()
        self._sync_scrollbars()

    def _on_hscroll_command(self, *args) -> None:
        if self._pil_original is None or self._scroll_syncing:
            return
        cw, _ = self._canvas_size()
        dw, _ = self._displayed_size()
        if dw <= cw:
            return
        if args[0] == "moveto":
            frac = float(args[1])
            self._pan_x = -frac * dw
        elif args[0] == "scroll":
            amount = int(args[1])
            unit = args[2] if len(args) > 2 else "units"
            step = cw * 0.1 if unit == "units" else cw * 0.9
            self._pan_x -= amount * step
        self._clamp_pan()
        self._needs_initial_fit = False
        self._draw_image()
        self._sync_scrollbars()

    def _on_vscroll_command(self, *args) -> None:
        if self._pil_original is None or self._scroll_syncing:
            return
        _, ch = self._canvas_size()
        _, dh = self._displayed_size()
        if dh <= ch:
            return
        if args[0] == "moveto":
            frac = float(args[1])
            self._pan_y = -frac * dh
        elif args[0] == "scroll":
            amount = int(args[1])
            unit = args[2] if len(args) > 2 else "units"
            step = ch * 0.1 if unit == "units" else ch * 0.9
            self._pan_y -= amount * step
        self._clamp_pan()
        self._needs_initial_fit = False
        self._draw_image()
        self._sync_scrollbars()

    def _clamp_pan(self) -> None:
        cw, ch = self._canvas_size()
        dw, dh = self._displayed_size()
        if dw <= cw:
            self._pan_x = (cw - dw) / 2
        else:
            self._pan_x = min(0.0, max(cw - dw, self._pan_x))
        if dh <= ch:
            self._pan_y = (ch - dh) / 2
        else:
            self._pan_y = min(0.0, max(ch - dh, self._pan_y))

    def _sync_scrollbars(self) -> None:
        if not hasattr(self, "_h_scroll"):
            return
        self._scroll_syncing = True
        try:
            cw, ch = self._canvas_size()
            dw, dh = self._displayed_size()
            if dw <= 0 or dh <= 0 or cw <= 1 or ch <= 1:
                self._h_scroll.set(0.0, 1.0)
                self._v_scroll.set(0.0, 1.0)
                return
            if dw <= cw:
                self._h_scroll.set(0.0, 1.0)
            else:
                first = max(0.0, min(1.0, -self._pan_x / dw))
                last = max(first, min(1.0, (-self._pan_x + cw) / dw))
                self._h_scroll.set(first, last)
            if dh <= ch:
                self._v_scroll.set(0.0, 1.0)
            else:
                first = max(0.0, min(1.0, -self._pan_y / dh))
                last = max(first, min(1.0, (-self._pan_y + ch) / dh))
                self._v_scroll.set(first, last)
        finally:
            self._scroll_syncing = False

    @property
    def current_path(self) -> "Path | None":
        if self._paths and 0 <= self._index < len(self._paths):
            return self._paths[self._index]
        return None

    def _center_image(self) -> None:
        cw, ch = self._canvas_size()
        dw, dh = self._displayed_size()
        self._pan_x = (cw - dw) / 2
        self._pan_y = (ch - dh) / 2

    def set_sources(
        self, sources: dict[str, Callable[[], list[Path]]], default_key: str | None = None
    ) -> None:
        self._source_getters = sources
        keys = list(sources.keys())
        if not keys:
            if self._source_menu is not None:
                self._source_menu.configure(values=["—"], state="disabled")
            self._paths = []
            self._show_empty("No folders configured")
            return
        if self._source_menu is not None:
            self._source_menu.configure(values=keys, state="normal")
        key = default_key if default_key in sources else keys[0]
        # Load first; suppress OptionMenu command so .set() does not reload twice.
        self._suppress_source_callback = True
        try:
            self._load_source(key)
            if self._source_menu is not None:
                try:
                    self._source_menu.set(key)
                except Exception:
                    try:
                        self._source_menu.configure(values=keys)
                        self._source_menu.set(key)
                    except Exception:
                        pass
        finally:
            self._suppress_source_callback = False
        if self._source_changed_cb is not None:
            try:
                self._source_changed_cb(key)
            except Exception:
                pass
        self.after_idle(self._ensure_view_fitted)

    def select_source(self, key: str) -> bool:
        """Switch source dropdown to ``key`` if available. Returns True on success."""
        if key not in self._source_getters:
            return False
        self._suppress_source_callback = True
        try:
            if self._source_menu is not None:
                try:
                    self._source_menu.set(key)
                except Exception:
                    pass
            self._load_source(key)
        finally:
            self._suppress_source_callback = False
        if self._source_changed_cb is not None:
            try:
                self._source_changed_cb(key)
            except Exception:
                pass
        self.after_idle(self._ensure_view_fitted)
        return True

    def _on_source_change(self, choice: str) -> None:
        if self._suppress_source_callback:
            return
        self._load_source(choice)
        if self._source_changed_cb is not None:
            try:
                self._source_changed_cb(choice)
            except Exception:
                pass
        self.after_idle(self._ensure_view_fitted)

    def _load_source(self, key: str) -> None:
        self._flush_persist()
        self._dirty = False
        old_filename = ""
        if self._paths and 0 <= self._index < len(self._paths):
            old_filename = self._paths[self._index].name
        self._current_source_key = key
        getter = self._source_getters.get(key)
        self._paths = getter() if getter else []
        self._index = 0
        if old_filename:
            for idx, p in enumerate(self._paths):
                if p.name == old_filename:
                    self._index = idx
                    break
        self._update_eraser_for_source(key)
        self._show_current()
        self.after_idle(self._ensure_view_fitted)
    def refresh(self) -> None:
        if self._path_provider is not None:
            paths = self._path_provider()
            if not paths:
                self.set_paths([], empty_message=self._empty_message or "No images")
            else:
                self._paths = paths
                self._index = min(self._index, max(0, len(paths) - 1))
                self._show_current()
            return
        if self._current_source_key and self._current_source_key in self._source_getters:
            self._load_source(self._current_source_key)
        elif self._paths:
            self._index = min(self._index, max(0, len(self._paths) - 1))
            self._show_current()

    def set_path_provider(self, provider: Callable[[], list[Path]] | None) -> None:
        self._path_provider = provider

    def set_paths(
        self, paths: list[Path], *, empty_message: str | None = None, remember_empty: bool = True
    ) -> None:
        self._flush_persist()
        self._source_getters = {}
        self._current_source_key = ""
        self._dirty = False
        if remember_empty and empty_message:
            self._empty_message = empty_message
        if self._source_menu is not None:
            self._suppress_source_callback = True
            try:
                self._source_menu.configure(values=["Images"], state="disabled")
                self._source_menu.set("Images")
            except Exception:
                pass
            finally:
                self._suppress_source_callback = False
        old_name = ""
        if self._paths and 0 <= self._index < len(self._paths):
            old_name = self._paths[self._index].name
        self._paths = list(paths)
        self._index = self._index_for_name(old_name) if old_name else 0
        self._set_eraser_bar_enabled(False)
        if not self._paths:
            self._show_empty(empty_message or "No images in this folder")
        else:
            self._show_current()
        self.after_idle(self._ensure_view_fitted)

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.resolve()).lower()
        except OSError:
            return str(path).lower()

    def _index_for_path(self, paths: list[Path], old_path: Path | None) -> int:
        """Find old_path in paths by resolve()/name; keep current index if possible."""
        if not paths:
            return 0
        if old_path is None:
            return min(self._index, len(paths) - 1)
        key = self._path_key(old_path)
        for i, p in enumerate(paths):
            if self._path_key(p) == key:
                return i
        name = old_path.name
        for i, p in enumerate(paths):
            if p.name == name:
                return i
        return min(self._index, len(paths) - 1)

    def _index_for_name(self, name: str) -> int:
        if not name or not self._paths:
            return 0
        for i, p in enumerate(self._paths):
            if p.name == name:
                return i
        return 0

    def _refresh_paths_preserving_index(self) -> None:
        """Re-scan current source/provider without jumping to a wrong image."""
        old_path = (
            self._paths[self._index]
            if self._paths and 0 <= self._index < len(self._paths)
            else None
        )
        if self._current_source_key and self._current_source_key in self._source_getters:
            getter = self._source_getters[self._current_source_key]
            self._paths = list(getter() or [])
        elif self._path_provider is not None:
            self._paths = list(self._path_provider() or [])
        if not self._paths:
            return
        self._index = self._index_for_path(self._paths, old_path)

    def _update_nav_buttons(self) -> None:
        n = len(self._paths)
        if n > 1:
            self._prev_btn.configure(state="normal")
            self._next_btn.configure(state="normal")
        else:
            self._prev_btn.configure(state="disabled")
            self._next_btn.configure(state="disabled")
        if n:
            self._counter.configure(text=f"{self._index + 1} / {n}")
        else:
            self._counter.configure(text="0 / 0")

    def prev_image(self) -> None:
        if not self._paths:
            return
        self._flush_persist()
        self._refresh_paths_preserving_index()
        if not self._paths:
            self._show_empty(self._empty_message or "No images in this folder")
            return
        self._index = (self._index - 1) % len(self._paths)
        self._show_current()
        self.after_idle(self._ensure_view_fitted)

    def next_image(self) -> None:
        if not self._paths:
            return
        self._flush_persist()
        self._refresh_paths_preserving_index()
        if not self._paths:
            self._show_empty(self._empty_message or "No images in this folder")
            return
        self._index = (self._index + 1) % len(self._paths)
        self._show_current()
        self.after_idle(self._ensure_view_fitted)

    def _show_empty(self, msg: str) -> None:
        try:
            self._canvas.delete("all")
        except Exception:
            pass
        self._pil_original = None
        self._tk_photo = None
        self._empty_message = msg
        cw, ch = self._canvas_size()
        self._canvas.create_text(
            cw // 2,
            ch // 2,
            text=msg,
            fill="#666666" if ctk.get_appearance_mode() == "Light" else "#aaaaaa",
            font=("Arial", 16, "bold")
        )
        self._caption.configure(text="")
        self._update_nav_buttons()

    def _load_pil_from_path(self, path: Path) -> Image.Image:
        return open_pil(path)

    def _render_image(self) -> None:
        pil = self._composited_display_pil()
        if pil is None:
            return
        ow, oh = pil.size
        scale = self._display_scale()
        dw = max(1, int(ow * scale))
        dh = max(1, int(oh * scale))
        disp = pil.resize((dw, dh), Image.Resampling.LANCZOS)
        try:
            self._canvas.delete("all")
        except Exception:
            pass
        self._tk_photo = ImageTk.PhotoImage(disp)
        self._draw_image()

    def _draw_image(self) -> None:
        if self._tk_photo is None:
            return
        self._canvas.delete("img")
        self._canvas.create_image(
            self._pan_x,
            self._pan_y,
            image=self._tk_photo,
            anchor="nw",
            tags="img",
        )
        self._redraw_scale_circle()
        self._redraw_brush_preview()
        self._sync_scrollbars()

    def _show_current(self) -> None:
        if not self._paths:
            self._show_empty(self._empty_message or "No images in this folder")
            self._scale_circle = None
            if self._image_changed_cb is not None:
                self._image_changed_cb(None)
            return

        self._index = max(0, min(self._index, len(self._paths) - 1))
        path = self._paths[self._index]
        self._update_nav_buttons()
        self._caption.configure(text=path.name)
        self._pending_mask_strokes.clear()
        self._preview_mask = None
        self._scale_circle = None
        self._dirty = False

        try:
            self._pil_original = self._load_pil_from_path(path)
            self._capture_edit_baseline(path)
            self._preview_cursor = None
            self._needs_initial_fit = True
            self.restore_view()
            if self._eraser_active and self._interaction_mode == "erase":
                self._ensure_preview_cursor()
                self._redraw_brush_preview()
            if self._image_changed_cb is not None:
                try:
                    self._image_changed_cb(path)
                except Exception:
                    pass
            self.after_idle(self._ensure_view_fitted)
            self.after(100, self._ensure_view_fitted)
        except OSError:
            self._pil_original = None
            self._show_empty(format_error_hint(path))
            self._caption.configure(text=path.name)
            if self._image_changed_cb is not None:
                try:
                    self._image_changed_cb(None)
                except Exception:
                    pass
        except Exception:
            self._pil_original = None
            self._show_empty(format_error_hint(path))
            self._caption.configure(text=path.name)
