"""Contour viewer with interactive mask editing (Contour / ROI tab)."""

from __future__ import annotations

import shutil
import sys
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from gui.paths import (
    leaf_roi_preview_dir,
    masks_dir,
    white_bg_path_for_stem,
)
from gui.state import ProjectState
from gui.widgets.image_carousel import SAVE_DEBOUNCE_MS, ImageCarousel
from image_io import load_bgr

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "leaf_contour"))
from overlay_viz import overlay_leaf  # noqa: E402

CONTOUR_EDIT_KEY = "Overlay contour"
# Soft tint so the real leaf stays visible while editing.
_EDIT_OVERLAY_ALPHA = 0.18
_MAX_UNDO = 30

_C_GRAY = ("gray75", "gray30")
_C_BLUE = ("#3B8ED0", "#1F6AA5")
_C_GREEN = ("#2ECC71", "#1E8449")
_C_RED_DIM = ("#C0392B", "#922B21")
_C_RED_BRIGHT = ("#E74C3C", "#C0392B")
_C_ORANGE = ("#E67E22", "#D35400")
_C_PURPLE = ("#9B59B6", "#7D3C98")


class ContourEditorCarousel(ImageCarousel):
    """
    Edits the Leaf-UNet mask on the leaf (white_bg): add / remove leaf lamina region.
    Changes are saved in leaf_roi_preview/masks and segmentation/masks.
    """

    def __init__(self, master, state: ProjectState, **kwargs) -> None:
        self._state = state
        self._contour_mode = False
        self._edit_active = False
        self._mask: np.ndarray | None = None
        self._bgr_base: np.ndarray | None = None
        self._mask_path: Path | None = None
        self._overlay_path: Path | None = None
        self._white_bg_path: Path | None = None
        self._baseline_mask: np.ndarray | None = None
        self._contour_mode_var = "pan"
        self._contour_save_after_id: str | None = None
        self._contour_dirty = False
        self._undo_stack: list[np.ndarray] = []
        self._pending_pts: list[tuple[int, int]] = []
        self._geometry_cursor: tuple[float, float] | None = None
        self._stroke_started = False
        self._brush_shape = "circle"
        self._edit_active_cb = None

        super().__init__(master, title="Contour / ROI", show_source_selector=True, **kwargs)
        self._build_contour_bar()
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Escape>", self._on_contour_escape)
        self.bind("<Escape>", self._on_contour_escape)
        self._canvas.bind("<Return>", self._on_polygon_close_key)
        self.bind("<Return>", self._on_polygon_close_key)

    def _build_contour_bar(self) -> None:
        self._contour_bar = ctk.CTkFrame(self, fg_color=("gray92", "gray18"), corner_radius=8)
        self._contour_bar.grid(row=self._tools_row, column=0, sticky="ew", padx=8, pady=(0, 4))
        self._contour_bar.grid_columnconfigure(0, weight=1)

        tools = ctk.CTkFrame(self._contour_bar, fg_color="transparent")
        tools.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        self._btn_pan = ctk.CTkButton(
            tools, text="Pan/Zoom", width=88, height=30,
            fg_color=_C_BLUE,
            command=lambda: self._on_contour_mode_change("pan"),
        )
        self._btn_pan.pack(side="left", padx=(0, 2))

        self._sep1 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep1.pack(side="left", padx=4)

        self._btn_add = ctk.CTkButton(
            tools, text="Add", width=64, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_contour_mode_change("add"),
        )
        self._btn_add.pack(side="left", padx=2)
        self._btn_remove = ctk.CTkButton(
            tools, text="Remove", width=78, height=30,
            fg_color=_C_RED_DIM,
            command=lambda: self._on_contour_mode_change("remove"),
        )
        self._btn_remove.pack(side="left", padx=2)

        self._sep2 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep2.pack(side="left", padx=4)

        self._btn_line = ctk.CTkButton(
            tools, text="Line", width=64, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_contour_mode_change("line"),
        )
        self._btn_line.pack(side="left", padx=2)
        self._btn_polygon = ctk.CTkButton(
            tools, text="Polygon", width=80, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_contour_mode_change("polygon"),
        )
        self._btn_polygon.pack(side="left", padx=2)

        self._sep3 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep3.pack(side="left", padx=4)

        self._size_label = ctk.CTkLabel(tools, text="Size:")
        self._size_label.pack(side="left", padx=(0, 4))
        self._contour_size = ctk.CTkSlider(
            tools,
            from_=1,
            to=80,
            number_of_steps=79,
            command=self._on_contour_size_change,
            width=100,
            progress_color=_C_GREEN,
            button_color=_C_GREEN,
            button_hover_color=("#27AE60", "#196F3D"),
        )
        self._contour_size.set(self._brush_size)
        self._contour_size.pack(side="left", padx=2)
        self._contour_size_lbl = ctk.CTkLabel(tools, text=f"{int(self._brush_size)} px", width=48)
        self._contour_size_lbl.pack(side="left", padx=2)

        self._sep4 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep4.pack(side="left", padx=4)

        self._contour_undo = ctk.CTkButton(
            tools, text="Undo", width=70, height=30,
            command=self.undo_last_edit, state="disabled",
        )
        self._contour_undo.pack(side="left", padx=2)
        self._contour_reset_all = ctk.CTkButton(
            tools, text="Reset all", width=88, height=30,
            command=self.reset_all_edits,
        )
        self._contour_reset_all.pack(side="left", padx=2)
        self._contour_done = ctk.CTkButton(
            tools, text="Done", width=70, height=30,
            fg_color=_C_BLUE,
            command=lambda: self.set_edit_contour_active(False),
        )
        self._contour_done.pack(side="left", padx=(8, 0))

        self._hint_label = ctk.CTkLabel(
            self._contour_bar,
            text="",
            text_color=("gray40", "gray65"),
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self._hint_label.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self._set_contour_bar_visible(False)

    def set_edit_contour_active(self, active: bool) -> None:
        """Enter or leave Contour edit mode (called from Contour tab)."""
        self._edit_active = bool(active)
        if active:
            if self._current_source_key != CONTOUR_EDIT_KEY:
                try:
                    if self._source_menu is not None:
                        self._source_menu.set(CONTOUR_EDIT_KEY)
                except Exception:
                    pass
                self._load_source(CONTOUR_EDIT_KEY)
            self._set_contour_bar_visible(True)
            self._on_contour_mode_change("pan")
            self._update_hint()
        else:
            self._clear_pending_geometry()
            self._on_contour_mode_change("pan")
            self._set_contour_bar_visible(False)
            self._hide_brush_preview()
            self._flush_contour_persist()
        cb = getattr(self, "_edit_active_cb", None)
        if cb is not None:
            cb(self._edit_active)

    def set_edit_active_callback(self, cb) -> None:
        """Optional callback(active: bool) when edit mode toggles (e.g. Done)."""
        self._edit_active_cb = cb

    def is_edit_contour_active(self) -> bool:
        return self._edit_active

    def _set_contour_bar_visible(self, visible: bool) -> None:
        if visible and self._contour_mode:
            self._contour_bar.grid()
        else:
            self._contour_bar.grid_remove()
            self._hide_brush_preview()
            self._clear_geometry_preview()

    def current_white_bg_path(self) -> Path | None:
        """White-bg crop for the currently shown leaf, if any."""
        p = self._white_bg_path
        return p if p is not None and Path(p).is_file() else None

    def _tool_buttons(self) -> dict[str, ctk.CTkButton]:
        return {
            "pan": self._btn_pan,
            "add": self._btn_add,
            "remove": self._btn_remove,
            "line": self._btn_line,
            "polygon": self._btn_polygon,
        }

    def _on_contour_mode_change(self, mode: str) -> None:
        prev = self._contour_mode_var
        if mode != prev:
            self._clear_pending_geometry()
        self._contour_mode_var = mode
        colors = {
            "pan": _C_BLUE,
            "add": _C_GREEN,
            "remove": _C_RED_BRIGHT,
            "line": _C_ORANGE,
            "polygon": _C_PURPLE,
        }
        for name, btn in self._tool_buttons().items():
            btn.configure(fg_color=colors[name] if name == mode else _C_GRAY)
        if mode == "remove":
            self._btn_remove.configure(fg_color=_C_RED_BRIGHT)
        show_size = mode in ("add", "remove")
        state = "normal" if show_size else "disabled"
        self._contour_size.configure(state=state)
        self._update_canvas_cursor()
        self._update_hint()
        if mode in ("add", "remove") and self._mask is not None:
            self._ensure_preview_cursor()
            self._redraw_brush_preview()
        else:
            self._hide_brush_preview()
            self._redraw_geometry_preview()

    def _update_hint(self) -> None:
        hints = {
            "pan": "Drag to pan · Scroll or Zoom buttons to zoom · Use scrollbars to explore",
            "add": "Paint to add leaf tissue to the ROI · Release fills enclosed holes",
            "remove": "Paint to erase ROI tissue",
            "line": "Click start point, then end point — line bridges the gap and fills the enclosed area",
            "polygon": "Click vertices — starts/ends inside ROI auto-fill the area · Esc to cancel",
        }
        self._hint_label.configure(text=hints.get(self._contour_mode_var, ""))

    def _update_undo_btn(self) -> None:
        self._contour_undo.configure(
            state="normal" if self._undo_stack else "disabled"
        )

    def _leaf_stem_for_path(self, path: Path) -> str:
        """Normalize overlay / white_bg / mask filenames to the leaf stem."""
        stem = path.stem
        if stem.endswith("_leaf_overlay"):
            return stem[: -len("_leaf_overlay")]
        if stem.endswith("_leaf_mask"):
            return stem[: -len("_leaf_mask")]
        if stem.endswith("_mask"):
            return stem[: -len("_mask")]
        return stem

    def _resolve_edit_mask_path(self, stem: str, out: Path) -> Path | None:
        """Locate the leaf mask for this stem (preview masks, then segmentation)."""
        candidates = [
            leaf_roi_preview_dir(out) / "masks" / f"{stem}_leaf_mask.png",
            masks_dir(out) / f"{stem}_mask.png",
            masks_dir(out) / f"{stem}_leaf_mask.png",
        ]
        if stem.endswith("_white_bg"):
            base = stem[: -len("_white_bg")]
            candidates.extend(
                [
                    leaf_roi_preview_dir(out) / "masks" / f"{base}_white_bg_leaf_mask.png",
                    masks_dir(out) / f"{base}_white_bg_mask.png",
                    masks_dir(out) / f"{base}_mask.png",
                ]
            )
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _resolve_overlay_path(self, stem: str, out: Path) -> Path:
        return leaf_roi_preview_dir(out) / "overlays" / f"{stem}_leaf_overlay.jpg"

    def _safe_load_pil(self, path: Path | None) -> Image.Image | None:
        if path is None or not Path(path).is_file():
            return None
        try:
            return self._load_pil_from_path(Path(path))
        except Exception:
            try:
                bgr = load_bgr(path)
                if bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            except Exception:
                return None

    def _on_contour_size_change(self, value: float) -> None:
        self._brush_size = max(1, int(value))
        self._contour_size_lbl.configure(text=f"{self._brush_size} px")
        self._ensure_preview_cursor()
        self._redraw_brush_preview()

    def _brush_radius_canvas(self) -> float:
        return max(0.5, (self._brush_size / 2.0) * self._display_scale())

    def _update_canvas_cursor(self) -> None:
        if not self._contour_mode or not self._edit_active:
            super()._update_canvas_cursor()
            return
        if self._contour_mode_var == "pan":
            self._canvas.configure(cursor="hand2")
        elif self._contour_mode_var in ("line", "polygon"):
            self._canvas.configure(cursor="crosshair")
        else:
            self._canvas.configure(cursor="dotbox")

    def _should_show_brush_preview(self) -> bool:
        if (
            self._contour_mode
            and self._edit_active
            and self._contour_mode_var in ("add", "remove")
        ):
            return self._mask is not None and self._preview_cursor is not None
        return super()._should_show_brush_preview()

    def _load_source(self, key: str) -> None:
        self._flush_contour_persist()
        old_path = (
            self._paths[self._index]
            if self._paths and 0 <= self._index < len(self._paths)
            else None
        )
        self._contour_mode = key == CONTOUR_EDIT_KEY
        self._set_contour_bar_visible(self._contour_mode and self._edit_active)
        self._current_source_key = key
        getter = self._source_getters.get(key)
        self._paths = list(getter() or []) if getter else []
        self._index = self._index_for_path(self._paths, old_path)
        if self._contour_mode:
            self._show_contour_current()
        else:
            self._mask = None
            self._bgr_base = None
            self._contour_dirty = False
            self._edit_active = False
            self._clear_undo_stack()
            super()._show_current()
        self.after_idle(self._ensure_view_fitted)

    def _show_current(self) -> None:
        if self._contour_mode:
            self._show_contour_current()
        else:
            self._mask = None
            self._bgr_base = None
            self._contour_dirty = False
            super()._show_current()

    def _show_contour_current(self) -> None:
        if not self._paths:
            self._show_empty(
                "No contour overlays yet.\n"
                "Run Contour (UNET Shape) on this tab first.\n"
                "(Segmentation alone does not create these images.)"
            )
            return

        self._index = max(0, min(self._index, len(self._paths) - 1))
        current = Path(self._paths[self._index])
        self._update_nav_buttons()
        self._caption.configure(text=current.name)

        out = self._state.output_path()
        if out is None:
            self._show_empty("Define output folder")
            self._caption.configure(text=current.name)
            self._update_nav_buttons()
            return

        stem = self._leaf_stem_for_path(current)
        self._white_bg_path = white_bg_path_for_stem(stem, out)
        if self._white_bg_path is None and current.is_file():
            name = current.name.lower()
            if any(name.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                if "overlay" not in current.stem.lower() and "mask" not in current.stem.lower():
                    self._white_bg_path = current

        resolved_mask = self._resolve_edit_mask_path(stem, out)
        self._overlay_path = self._resolve_overlay_path(stem, out)
        if current.name.lower().endswith((".jpg", ".jpeg")) and "_leaf_overlay" in current.stem:
            self._overlay_path = current
        self._mask_path = resolved_mask or (
            leaf_roi_preview_dir(out) / "masks" / f"{stem}_leaf_mask.png"
        )

        self._pil_original = None
        self._bgr_base = None
        self._mask = None
        self._baseline_mask = None
        self._contour_dirty = False
        self._clear_undo_stack()
        self._clear_pending_geometry()

        if self._white_bg_path is not None and resolved_mask is not None:
            bgr = load_bgr(self._white_bg_path)
            mask = cv2.imread(str(resolved_mask), cv2.IMREAD_UNCHANGED)
            if mask is not None and mask.ndim > 2:
                mask = mask[:, :, 0]
            if mask is None:
                mask = cv2.imread(str(resolved_mask), cv2.IMREAD_GRAYSCALE)
            if bgr is not None and mask is not None:
                mask = np.ascontiguousarray(np.squeeze(mask))
                if mask.ndim > 2:
                    mask = mask[:, :, 0]
                if mask.shape[:2] != bgr.shape[:2]:
                    mask = cv2.resize(
                        mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                self._bgr_base = bgr
                self._mask = mask
                self._baseline_mask = mask.copy()
                self._mask_path = resolved_mask
                self._refresh_composite_pil()

        if self._pil_original is None:
            self._pil_original = self._safe_load_pil(self._overlay_path)

        if self._pil_original is None and self._white_bg_path is not None:
            self._pil_original = self._safe_load_pil(self._white_bg_path)
            if self._pil_original is not None and self._bgr_base is None:
                self._bgr_base = load_bgr(self._white_bg_path)

        if self._pil_original is None:
            self._pil_original = self._safe_load_pil(current)

        if self._pil_original is None:
            self._show_empty(
                "Could not load leaf image.\n"
                "Check segmentation/white_bg and\n"
                "leaf_roi_preview/overlays."
            )
            self._caption.configure(text=current.name)
            self._update_nav_buttons()
            return

        self._preview_cursor = None
        self._preview_mask = None
        self._needs_initial_fit = True
        self.restore_view()
        if (
            self._edit_active
            and self._contour_mode_var in ("add", "remove")
            and self._mask is not None
        ):
            self._ensure_preview_cursor()
            self._redraw_brush_preview()
        else:
            self._hide_brush_preview()
        self.after_idle(self._ensure_view_fitted)
        self.after(100, self._ensure_view_fitted)

    def _refresh_composite_pil(self) -> None:
        if self._bgr_base is None or self._mask is None:
            return
        leaf_bool = self._mask > 127
        vis = overlay_leaf(
            self._bgr_base, leaf_bool, raw_bool=None, green_alpha=_EDIT_OVERLAY_ALPHA
        )
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._pil_original = Image.fromarray(rgb)

    def _paint_mask_at(self, ix: float, iy: float, add: bool) -> None:
        if self._mask is None:
            return
        r = max(1, int(self._brush_size // 2))
        value = 255 if add else 0
        cv2.circle(self._mask, (int(ix), int(iy)), r, value, -1)

    def _apply_fill_holes(self) -> None:
        """Fill all interior background regions fully enclosed by the ROI mask."""
        if self._mask is None:
            return
        h, w = self._mask.shape
        binary = (self._mask > 127).astype(np.uint8) * 255
        inv = cv2.bitwise_not(binary)
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        flood = padded.copy()
        cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
        holes = flood[1:-1, 1:-1] == 255
        if holes.any():
            self._mask[holes] = 255

    def _stroke_mask(self, x0: float, y0: float, x1: float, y1: float, add: bool) -> None:
        if self._mask is None:
            return
        value = 255 if add else 0
        thickness = max(1, int(self._brush_size))
        cv2.line(
            self._mask,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            value,
            thickness=thickness,
        )
        self._paint_mask_at(x1, y1, add)

    def _push_undo(self) -> None:
        if self._mask is None:
            return
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > _MAX_UNDO:
            self._undo_stack.pop(0)
        self._update_undo_btn()

    def _clear_undo_stack(self) -> None:
        self._undo_stack.clear()
        self._update_undo_btn()

    def undo_last_edit(self) -> None:
        """Undo only the last committed edit."""
        if not self._undo_stack or self._mask is None:
            return
        prev = self._undo_stack.pop()
        self._mask[:] = prev
        self._update_undo_btn()
        self._contour_dirty = True
        self._refresh_composite_pil()
        self._render_image()
        self._redraw_brush_preview()
        self._redraw_geometry_preview()
        self._flush_contour_persist()

    def reset_all_edits(self) -> None:
        """Revert all edits on this leaf to the mask loaded at open."""
        if self._baseline_mask is None or self._mask is None:
            return
        self._mask[:] = self._baseline_mask
        self._clear_undo_stack()
        self._clear_pending_geometry()
        self._contour_dirty = True
        self._refresh_composite_pil()
        self._render_image()
        self._redraw_brush_preview()
        self._flush_contour_persist()

    # Back-compat alias
    def undo_contour_edits(self) -> None:
        self.reset_all_edits()

    def _clear_pending_geometry(self) -> None:
        self._pending_pts = []
        self._geometry_cursor = None
        self._clear_geometry_preview()

    def _clear_geometry_preview(self) -> None:
        try:
            self._canvas.delete("geom_preview")
        except Exception:
            pass

    def _img_to_canvas_xy(self, ix: float, iy: float) -> tuple[float, float]:
        scale = self._display_scale()
        return self._pan_x + ix * scale, self._pan_y + iy * scale

    def _redraw_geometry_preview(self) -> None:
        self._clear_geometry_preview()
        if not self._edit_active or self._contour_mode_var not in ("line", "polygon"):
            return
        pts = list(self._pending_pts)
        if self._geometry_cursor is not None and pts:
            pts = pts + [(int(self._geometry_cursor[0]), int(self._geometry_cursor[1]))]
        if not pts:
            return
        canvas_pts = [self._img_to_canvas_xy(float(x), float(y)) for x, y in pts]
        color = "#E67E22" if self._contour_mode_var == "line" else "#9B59B6"
        for i in range(len(canvas_pts) - 1):
            x0, y0 = canvas_pts[i]
            x1, y1 = canvas_pts[i + 1]
            self._canvas.create_line(
                x0, y0, x1, y1, fill=color, width=2, tags="geom_preview"
            )
        for x, y in canvas_pts:
            r = 4
            self._canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline=color, fill="#ffffff", width=2, tags="geom_preview",
            )
        self._canvas.tag_raise("geom_preview")

    def _commit_line_fill(self) -> None:
        if self._mask is None or len(self._pending_pts) < 2:
            return
        (x0, y0), (x1, y1) = self._pending_pts[0], self._pending_pts[1]
        self._push_undo()
        # Thin bridge only — brush thickness was adding a wide ROI strip outside
        # the intended edge. thickness=2 seals 4-connected holes reliably.
        cv2.line(self._mask, (x0, y0), (x1, y1), 255, thickness=2)
        self._apply_fill_holes()
        self._pending_pts = []
        self._geometry_cursor = None
        self._contour_dirty = True
        self._refresh_composite_pil()
        self._render_image()
        self._redraw_geometry_preview()
        self._flush_contour_persist()
        self._update_hint()

    def _point_in_roi(self, x: int, y: int) -> bool:
        """True if (x, y) lies inside the current leaf ROI mask."""
        if self._mask is None:
            return False
        h, w = self._mask.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        return bool(self._mask[y, x] > 127)

    def _commit_polygon_fill(self) -> None:
        if self._mask is None or len(self._pending_pts) < 3:
            return
        self._push_undo()
        pts = np.array(self._pending_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self._mask, [pts], 255)
        self._apply_fill_holes()
        self._pending_pts = []
        self._geometry_cursor = None
        self._contour_dirty = True
        self._refresh_composite_pil()
        self._render_image()
        self._redraw_geometry_preview()
        self._flush_contour_persist()
        self._update_hint()

    def _on_polygon_close_key(self, _event=None) -> str:
        if self._edit_active and self._contour_mode_var == "polygon":
            self._commit_polygon_fill()
            return "break"
        return ""

    def _on_contour_escape(self, _event=None) -> str:
        if self._edit_active and self._pending_pts:
            self._clear_pending_geometry()
            self._update_hint()
            return "break"
        return self._on_escape_reset_view(_event)

    def _on_double_click(self, event: tk.Event) -> None:
        if not (self._contour_mode and self._edit_active and self._mask is not None):
            return
        if self._contour_mode_var == "polygon" and len(self._pending_pts) >= 3:
            # Vertex already added by ButtonPress-1; just close the polygon.
            self._commit_polygon_fill()

    def _schedule_contour_persist(self) -> None:
        self._contour_dirty = True
        if self._contour_save_after_id is not None:
            self.after_cancel(self._contour_save_after_id)
        self._contour_save_after_id = self.after(SAVE_DEBOUNCE_MS, self._flush_contour_persist)

    def _flush_contour_persist(self) -> None:
        self._contour_save_after_id = None
        if not self._contour_dirty:
            return
        if not self._contour_mode or self._mask is None or self._mask_path is None:
            return
        if self._bgr_base is None or self._overlay_path is None:
            return
        out = self._state.output_path()
        if out is None:
            return
        try:
            self._mask_path.parent.mkdir(parents=True, exist_ok=True)
            self._overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self._mask_path), self._mask)
            leaf_bool = self._mask > 127
            vis = overlay_leaf(
                self._bgr_base, leaf_bool, raw_bool=None, green_alpha=_EDIT_OVERLAY_ALPHA
            )
            cv2.imwrite(str(self._overlay_path), vis)
            if self._white_bg_path is not None:
                from gui.paths import canonical_leaf_id

                leaf_id = canonical_leaf_id(self._white_bg_path.stem)
                seg_mask = masks_dir(out) / f"{leaf_id}_mask.png"
                seg_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._mask_path, seg_mask)
            self._contour_dirty = False
        except OSError:
            pass

    def _flush_persist(self) -> None:
        self._flush_contour_persist()
        super()._flush_persist()

    def _draw_image(self) -> None:
        super()._draw_image()
        self._redraw_geometry_preview()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if (
            self._contour_mode
            and self._edit_active
            and self._contour_mode_var in ("line", "polygon")
            and self._pending_pts
        ):
            self._geometry_cursor = self._canvas_to_image_xy(event.x, event.y)
            self._redraw_geometry_preview()
            return
        super()._on_canvas_motion(event)

    def _on_button1_press(self, event: tk.Event) -> None:
        if not (self._contour_mode and self._edit_active and self._mask is not None):
            super()._on_button1_press(event)
            return

        mode = self._contour_mode_var
        if mode == "pan":
            super()._on_button1_press(event)
            return

        ix, iy = self._canvas_to_image_xy(event.x, event.y)
        if mode in ("add", "remove"):
            self._hide_brush_preview()
            if not self._stroke_started:
                self._push_undo()
                self._stroke_started = True
            add = mode == "add"
            self._last_erase_img_xy = (ix, iy)
            self._paint_mask_at(ix, iy, add)
            self._refresh_composite_pil()
            self._render_image()
            self._schedule_contour_persist()
            return

        if mode == "line":
            self._pending_pts.append((int(ix), int(iy)))
            if len(self._pending_pts) >= 2:
                self._commit_line_fill()
            else:
                self._hint_label.configure(text="Click end point to bridge and fill")
                self._redraw_geometry_preview()
            return

        if mode == "polygon":
            self._pending_pts.append((int(ix), int(iy)))
            n = len(self._pending_pts)
            # Auto-fill when the polyline starts and ends inside the existing ROI
            # (no Close button): the polygon area is added to the mask immediately.
            if (
                n >= 3
                and self._point_in_roi(*self._pending_pts[0])
                and self._point_in_roi(*self._pending_pts[-1])
            ):
                self._commit_polygon_fill()
            else:
                self._redraw_geometry_preview()
                self._hint_label.configure(
                    text=f"{n} point(s) · End inside ROI to auto-fill · Esc to cancel"
                )
            return

        super()._on_button1_press(event)

    def _on_button1_motion(self, event: tk.Event) -> None:
        if (
            self._contour_mode
            and self._edit_active
            and self._contour_mode_var in ("add", "remove")
            and self._mask is not None
        ):
            add = self._contour_mode_var == "add"
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            if self._last_erase_img_xy is not None:
                self._stroke_mask(
                    self._last_erase_img_xy[0],
                    self._last_erase_img_xy[1],
                    ix,
                    iy,
                    add,
                )
            else:
                self._paint_mask_at(ix, iy, add)
            self._last_erase_img_xy = (ix, iy)
            self._refresh_composite_pil()
            self._render_image()
            self._schedule_contour_persist()
            return
        if (
            self._contour_mode
            and self._edit_active
            and self._contour_mode_var in ("line", "polygon")
        ):
            return
        super()._on_button1_motion(event)

    def _on_button1_release(self, event: tk.Event) -> None:
        if (
            self._contour_mode
            and self._edit_active
            and self._contour_mode_var in ("add", "remove")
            and self._mask is not None
        ):
            self._last_erase_img_xy = None
            self._stroke_started = False
            if self._contour_mode_var == "add":
                self._apply_fill_holes()
                self._refresh_composite_pil()
                self._render_image()
                self._contour_dirty = True
            self._flush_contour_persist()
            self._preview_cursor = (event.x, event.y)
            self._redraw_brush_preview()
            return
        super()._on_button1_release(event)

    def refresh(self) -> None:
        self._flush_contour_persist()
        super().refresh()
