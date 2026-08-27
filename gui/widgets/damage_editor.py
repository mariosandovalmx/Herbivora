"""Damage analysis viewer with interactive DAMAGE mask editing (Analysis tab)."""

from __future__ import annotations

import csv
import json
import sys
import threading
from pathlib import Path

import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from gui.paths import (
    analyzed_damage_mask_path,
    analyzed_leaf_roi_path,
    analyzed_meta_path,
    analyzed_stem_from_jpg,
    white_bg_dir,
    white_bg_path_for_stem,
)
from gui.state import ProjectState
from gui.widgets.image_carousel import SAVE_DEBOUNCE_MS, ImageCarousel
from image_io import load_rgb

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

_analyze_leaves_mod = None


def _analyze_leaves():
    """Lazy-import analyze_leaves (torch/SMP) so GUI startup stays light."""
    global _analyze_leaves_mod
    if _analyze_leaves_mod is None:
        import analyze_leaves as _al  # noqa: E402

        _analyze_leaves_mod = _al
    return _analyze_leaves_mod

_MAX_UNDO = 30

_C_GRAY = ("gray75", "gray30")
_C_BLUE = ("#3B8ED0", "#1F6AA5")
_C_GREEN = ("#2ECC71", "#1E8449")
_C_RED_DIM = ("#C0392B", "#922B21")
_C_RED_BRIGHT = ("#E74C3C", "#C0392B")
_C_SELECT = ("#9B59B6", "#7D3C98")
_C_ORANGE = ("#E67E22", "#D35400")
_C_PURPLE = ("#9B59B6", "#7D3C98")


class DamageEditorCarousel(ImageCarousel):
    """Edit the DAMAGE category on analyzed leaves; updates % and area live."""

    def __init__(self, master, state: ProjectState, **kwargs) -> None:
        self._state = state
        self._edit_mode = False  # True when damage sidecars are loaded
        self._edit_active = False  # True when user pressed Edit Damage
        self._damage_mask: np.ndarray | None = None
        self._leaf_roi: np.ndarray | None = None
        self._rgb_base: np.ndarray | None = None
        self._analyzed_path: Path | None = None
        self._meta: dict | None = None
        self._baseline_damage: np.ndarray | None = None
        self._edit_mode_var = "pan"
        self._brush_intent: str = "add"
        self._select_region_active = False
        self._damage_save_after_id: str | None = None
        self._select_busy = False
        self._select_gen = 0
        self._status_msg: str | None = None
        self._damage_dirty = False
        self._undo_stack: list[np.ndarray] = []
        self._pending_pts: list[tuple[int, int]] = []
        self._geometry_cursor: tuple[float, float] | None = None
        self._stroke_started = False
        self._edit_active_cb = None
        self._metrics_cb = None

        super().__init__(master, title="Damage Analysis", **kwargs)
        # Usable default brush for edge bites (carousel default can be too small)
        self._brush_size = max(12, int(getattr(self, "_brush_size", 12) or 12))
        self._build_edit_bar()
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Escape>", self._on_damage_escape)
        self.bind("<Escape>", self._on_damage_escape)
        self._canvas.bind("<Return>", self._on_polygon_close_key)
        self.bind("<Return>", self._on_polygon_close_key)

    def _build_edit_bar(self) -> None:
        self._edit_bar = ctk.CTkFrame(self, fg_color=("gray92", "gray18"), corner_radius=8)
        self._edit_bar.grid(row=self._tools_row, column=0, sticky="ew", padx=8, pady=(0, 4))
        self._edit_bar.grid_columnconfigure(0, weight=1)

        tools = ctk.CTkFrame(self._edit_bar, fg_color="transparent")
        tools.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        self._btn_pan = ctk.CTkButton(
            tools, text="Pan/Zoom", width=88, height=30,
            fg_color=_C_BLUE,
            command=lambda: self._on_edit_mode_change("Pan/Zoom"),
        )
        self._btn_pan.pack(side="left", padx=(0, 2))

        self._sep1 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep1.pack(side="left", padx=4)

        self._btn_add = ctk.CTkButton(
            tools, text="Add", width=64, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_edit_mode_change("Add"),
        )
        self._btn_add.pack(side="left", padx=2)
        self._btn_remove = ctk.CTkButton(
            tools, text="Remove", width=78, height=30,
            fg_color=_C_RED_DIM,
            command=lambda: self._on_edit_mode_change("Remove"),
        )
        self._btn_remove.pack(side="left", padx=2)
        self._btn_select = ctk.CTkButton(
            tools, text="Select Region", width=110, height=30,
            fg_color=_C_GRAY,
            command=self._on_select_region_toggle,
        )
        self._btn_select.pack(side="left", padx=2)

        self._sep2 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep2.pack(side="left", padx=4)

        self._btn_line = ctk.CTkButton(
            tools, text="Line", width=64, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_edit_mode_change("Line"),
        )
        self._btn_line.pack(side="left", padx=2)
        self._btn_polygon = ctk.CTkButton(
            tools, text="Polygon", width=80, height=30,
            fg_color=_C_GRAY,
            command=lambda: self._on_edit_mode_change("Polygon"),
        )
        self._btn_polygon.pack(side="left", padx=2)

        self._sep3 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep3.pack(side="left", padx=4)

        ctk.CTkLabel(tools, text="Shape:").pack(side="left", padx=(0, 4))
        self._edit_shape = ctk.CTkOptionMenu(
            tools,
            values=["Circle", "Square"],
            command=self._on_edit_shape_change,
            width=96,
        )
        self._edit_shape.set("Circle")
        self._edit_shape.pack(side="left", padx=4)

        self._size_label = ctk.CTkLabel(tools, text="Size:")
        self._size_label.pack(side="left", padx=(8, 4))
        self._edit_size = ctk.CTkSlider(
            tools,
            from_=4,
            to=80,
            number_of_steps=76,
            command=self._on_edit_size_change,
            width=100,
            progress_color=_C_GREEN,
            button_color=_C_GREEN,
            button_hover_color=("#27AE60", "#196F3D"),
        )
        self._edit_size.set(self._brush_size)
        self._edit_size.pack(side="left", padx=4)
        self._edit_size_lbl = ctk.CTkLabel(tools, text=f"{int(self._brush_size)} px", width=48)
        self._edit_size_lbl.pack(side="left", padx=4)

        self._sep4 = ctk.CTkLabel(tools, text="|", text_color="gray")
        self._sep4.pack(side="left", padx=4)

        self._edit_undo = ctk.CTkButton(
            tools, text="Undo", width=70, height=30,
            command=self.undo_last_edit, state="disabled",
        )
        self._edit_undo.pack(side="left", padx=2)
        self._edit_reset_all = ctk.CTkButton(
            tools, text="Reset all", width=88, height=30,
            command=self.reset_all_edits,
        )
        self._edit_reset_all.pack(side="left", padx=2)
        self._edit_done = ctk.CTkButton(
            tools, text="Done", width=70, height=30,
            fg_color=_C_BLUE,
            command=lambda: self.set_edit_damage_active(False),
        )
        self._edit_done.pack(side="left", padx=(8, 0))

        # Metrics shown on Analysis left panel (set_metrics_callback), not here '
        # the tool bar is too crowded and was hiding the % text.
        self._metrics_label = None

        self._hint_label = ctk.CTkLabel(
            self._edit_bar,
            text="",
            text_color=("gray40", "gray65"),
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self._hint_label.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self._set_edit_bar_visible(False)

    def set_edit_damage_active(self, active: bool) -> None:
        """Enter or leave Damage edit mode (called from Analysis tab)."""
        if active and not self._edit_mode:
            # No sidecars yet ' keep inactive
            active = False
        self._edit_active = bool(active)
        if self._edit_active:
            self._set_edit_bar_visible(True)
            self._on_edit_mode_change("Pan/Zoom")
            self._update_hint()
            self._set_edit_controls_enabled(True)
        else:
            self._clear_pending_geometry()
            self._select_region_active = False
            self._edit_mode_var = "pan"
            self._set_edit_bar_visible(False)
            self._hide_brush_preview()
            self._flush_damage_persist()
            self._set_edit_controls_enabled(False)
            self._update_canvas_cursor()
        cb = getattr(self, "_edit_active_cb", None)
        if cb is not None:
            cb(self._edit_active)

    def set_edit_active_callback(self, cb) -> None:
        """Optional callback(active: bool) when edit mode toggles (e.g. Done)."""
        self._edit_active_cb = cb

    def set_metrics_callback(self, cb) -> None:
        """Optional callback(text: str) whenever damage % / status text changes."""
        self._metrics_cb = cb

    def is_edit_damage_active(self) -> bool:
        return self._edit_active

    def _set_edit_bar_visible(self, visible: bool) -> None:
        if visible and self._edit_mode:
            self._edit_bar.grid()
        else:
            self._edit_bar.grid_remove()
            self._hide_brush_preview()
            self._clear_geometry_preview()

    def _can_edit(self) -> bool:
        return self._edit_mode and self._edit_active

    def _set_edit_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for w in (
            self._btn_pan, self._btn_add, self._btn_remove, self._btn_select,
            self._btn_line, self._btn_polygon,
            self._edit_shape, self._edit_size, self._edit_undo,
            self._edit_reset_all, self._edit_done,
        ):
            w.configure(state=state)
        if enabled:
            self._update_undo_btn()
            self._apply_mode_buttons()

    def _is_select_mode(self) -> bool:
        return self._edit_mode_var in ("select_add", "select_remove")

    def _is_brush_mode(self) -> bool:
        return self._edit_mode_var in ("add", "remove")

    def _is_geometry_mode(self) -> bool:
        return self._edit_mode_var in ("line", "polygon")

    def _apply_mode_buttons(self) -> None:
        mode = self._edit_mode_var
        self._btn_pan.configure(fg_color=_C_BLUE if mode == "pan" else _C_GRAY)
        self._btn_add.configure(
            fg_color=_C_GREEN if mode in ("add", "select_add") else _C_GRAY
        )
        self._btn_remove.configure(
            fg_color=_C_RED_BRIGHT if mode in ("remove", "select_remove") else _C_RED_DIM
        )
        self._btn_select.configure(
            fg_color=_C_SELECT if self._is_select_mode() else _C_GRAY
        )
        self._btn_line.configure(fg_color=_C_ORANGE if mode == "line" else _C_GRAY)
        self._btn_polygon.configure(fg_color=_C_PURPLE if mode == "polygon" else _C_GRAY)
        shape_ok = self._is_brush_mode()
        size_ok = self._is_brush_mode() or self._is_select_mode()
        self._edit_shape.configure(state="normal" if shape_ok else "disabled")
        self._edit_size.configure(state="normal" if size_ok else "disabled")
        self._update_size_label()

    def _update_size_label(self) -> None:
        unit = "tol" if self._is_select_mode() else "px"
        self._size_label.configure(text="Tolerance:" if self._is_select_mode() else "Size:")
        self._edit_size_lbl.configure(text=f"{int(self._brush_size)} {unit}")

    def _update_hint(self) -> None:
        hints = {
            "pan": "Drag or scroll to pan | Ctrl+scroll to zoom | Shift+scroll pans sideways",
            "add": "Paint damage anywhere (leaf or white bites) | Scroll pans | Ctrl+scroll zooms",
            "remove": "Paint to erase damage | Scroll pans | Ctrl+scroll zooms",
            "select_add": "Click to select a damage region (MobileSAM / color flood) | Scroll pans | Ctrl+scroll zooms",
            "select_remove": "Click a damage blob to remove it | Scroll pans | Ctrl+scroll zooms",
            "line": "Click two points across a white bite - fills that gap as damage | Scroll pans | Ctrl+scroll zooms",
            "polygon": "Trace the bite on the leaf edge | Close near start - fills the white gap | Scroll pans | Ctrl+scroll zooms",
        }
        self._hint_label.configure(text=hints.get(self._edit_mode_var, ""))


    def _update_undo_btn(self) -> None:
        self._edit_undo.configure(state="normal" if self._undo_stack else "disabled")

    def _on_select_region_toggle(self) -> None:
        if not self._can_edit():
            return
        self._clear_pending_geometry()
        if self._is_select_mode():
            self._select_region_active = False
            self._edit_mode_var = self._brush_intent
        else:
            self._select_region_active = True
            if self._brush_intent == "remove":
                self._edit_mode_var = "select_remove"
            else:
                self._edit_mode_var = "select_add"
            self._ensure_mobilesam_loaded()
        self._apply_mode_buttons()
        self._update_canvas_cursor()
        self._update_hint()
        self._hide_brush_preview()

    def _ensure_mobilesam_loaded(self) -> None:
        """Lazy-load MobileSAM once (shared session with Segmentation tab)."""
        from gui.interactive_sam_session import get_session

        session = get_session()
        if session.mobilesam_ready or session.is_loading_sam:
            return

        def worker() -> None:
            try:
                session.ensure_mobilesam(
                    log=None, weights=self._state.mobilesam_model or None
                )
                self.after(0, self._on_mobilesam_ready)
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_mobilesam_failed(err))

        self._status_msg = "Loading MobileSAM'"
        self._update_metrics_label()
        threading.Thread(target=worker, daemon=True).start()

    def _on_mobilesam_ready(self) -> None:
        self._status_msg = None
        self._update_metrics_label()

    def _on_mobilesam_failed(self, err: str) -> None:
        self._status_msg = "MobileSAM unavailable (using color fallback)"
        self._update_metrics_label()
        self.after(4000, self._clear_status_if_fallback)

    def _clear_status_if_fallback(self) -> None:
        if self._status_msg and "fallback" in self._status_msg.lower():
            self._status_msg = None
            self._update_metrics_label()

    def _on_edit_mode_change(self, value: str) -> None:
        prev = self._edit_mode_var
        if value == "Add":
            # Explicit Add = brush (exit Select Region)
            self._brush_intent = "add"
            self._select_region_active = False
            self._edit_mode_var = "add"
        elif value == "Remove":
            self._brush_intent = "remove"
            self._select_region_active = False
            self._edit_mode_var = "remove"
        elif value == "Line":
            self._select_region_active = False
            self._edit_mode_var = "line"
        elif value == "Polygon":
            self._select_region_active = False
            self._edit_mode_var = "polygon"
        else:
            self._select_region_active = False
            self._edit_mode_var = "pan"
        if self._edit_mode_var != prev:
            self._clear_pending_geometry()
        self._apply_mode_buttons()
        self._update_canvas_cursor()
        self._update_hint()
        if self._is_brush_mode():
            self._ensure_preview_cursor()
            self._redraw_brush_preview()
        else:
            self._hide_brush_preview()
            self._redraw_geometry_preview()

    def _on_edit_shape_change(self, choice: str) -> None:
        self._brush_shape = "square" if choice == "Square" else "circle"
        self._redraw_brush_preview()

    def _on_edit_size_change(self, value: float) -> None:
        self._brush_size = max(4, int(value))
        self._update_size_label()
        self._ensure_preview_cursor()
        self._redraw_brush_preview()

    def _update_canvas_cursor(self) -> None:
        if not self._can_edit():
            self._canvas.configure(cursor="hand2")
            return
        if self._is_select_mode() or self._is_geometry_mode():
            self._canvas.configure(cursor="crosshair")
        elif self._is_brush_mode():
            self._canvas.configure(cursor="dotbox")
        else:
            self._canvas.configure(cursor="hand2")

    def _should_show_brush_preview(self) -> bool:
        if self._can_edit() and self._is_brush_mode():
            return self._damage_mask is not None and self._preview_cursor is not None
        return super()._should_show_brush_preview()

    def _push_undo(self) -> None:
        if self._damage_mask is None:
            return
        self._undo_stack.append(self._damage_mask.copy())
        if len(self._undo_stack) > _MAX_UNDO:
            self._undo_stack.pop(0)
        self._update_undo_btn()

    def _clear_undo_stack(self) -> None:
        self._undo_stack.clear()
        self._update_undo_btn()

    def undo_last_edit(self) -> None:
        """Undo only the last committed edit."""
        if not self._undo_stack or self._damage_mask is None:
            return
        prev = self._undo_stack.pop()
        self._damage_mask[:] = prev
        self._update_undo_btn()
        self._damage_dirty = True
        self._refresh_damage_pil()
        self._render_image()
        self._update_metrics_label()
        self._redraw_brush_preview()
        self._redraw_geometry_preview()
        self._flush_damage_persist()

    def reset_all_edits(self) -> None:
        """Revert all edits on this leaf to the damage mask loaded at open."""
        if self._baseline_damage is None or self._damage_mask is None:
            return
        self._damage_mask[:] = self._baseline_damage
        self._clear_undo_stack()
        self._clear_pending_geometry()
        self._damage_dirty = True
        self._refresh_damage_pil()
        self._render_image()
        self._update_metrics_label()
        self._redraw_brush_preview()
        self._flush_damage_persist()

    # Back-compat alias
    def undo_damage_edits(self) -> None:
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
        if not self._can_edit() or not self._is_geometry_mode():
            return
        pts = list(self._pending_pts)
        if self._geometry_cursor is not None and pts:
            pts = pts + [(int(self._geometry_cursor[0]), int(self._geometry_cursor[1]))]
        if not pts:
            return
        canvas_pts = [self._img_to_canvas_xy(float(x), float(y)) for x, y in pts]
        color = "#E67E22" if self._edit_mode_var == "line" else "#9B59B6"
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

    def _point_in_paint_zone(self, x: int, y: int) -> bool:
        allow = self._paint_allow_mask()
        if allow is None:
            return False
        h, w = allow.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        return bool(allow[y, x])

    def _point_in_leaf_roi(self, x: int, y: int) -> bool:
        if self._leaf_roi is None:
            return False
        h, w = self._leaf_roi.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        return bool(self._leaf_roi[y, x])

    def _paint_allow_mask(self) -> np.ndarray | None:
        """Leaf ROI dilated so Add/brush can reach white edge bites just outside the silhouette."""
        if self._leaf_roi is None:
            return None
        # Large band beyond the leaf edge so deep white herbivory notches are paintable
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (201, 201))
        return cv2.dilate(self._leaf_roi.astype(np.uint8), kernel, iterations=1) > 0

    def _contour_gap_fill_from_overlay(self, overlay_u8: np.ndarray) -> np.ndarray:
        """Return white pockets enclosed by leaf_roi ? overlay (Contour fill-holes)."""
        if self._leaf_roi is None:
            return np.zeros(overlay_u8.shape, dtype=bool)
        before_leaf = self._leaf_roi.astype(bool)
        work = np.maximum(before_leaf.astype(np.uint8) * 255, overlay_u8)
        h, w = work.shape
        inv = cv2.bitwise_not(work)
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        flood = padded.copy()
        cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
        holes = flood[1:-1, 1:-1] == 255
        work = work.copy()
        work[holes] = 255
        pocket = (work > 127) & ~before_leaf
        leaf_px = int(before_leaf.sum())
        if leaf_px > 0 and int(pocket.sum()) > 0.35 * leaf_px:
            return np.zeros_like(pocket)
        return pocket

    def _commit_line_fill(self) -> None:
        """Contour-style bridge: close a white edge bite and mark that gap as damage."""
        if self._damage_mask is None or self._leaf_roi is None or len(self._pending_pts) < 2:
            return
        (x0, y0), (x1, y1) = self._pending_pts[0], self._pending_pts[1]
        self._push_undo()

        stroke = np.zeros(self._damage_mask.shape, dtype=np.uint8)
        cv2.line(stroke, (x0, y0), (x1, y1), 255, thickness=2)
        pocket = self._contour_gap_fill_from_overlay(stroke)
        self._damage_mask[pocket | (stroke > 0)] = True
        self._pending_pts = []
        self._geometry_cursor = None
        self._damage_dirty = True
        self._refresh_damage_pil()
        self._render_image()
        self._update_metrics_label()
        self._redraw_geometry_preview()
        self._flush_damage_persist()
        self._update_hint()

    def _commit_polygon_fill(self) -> None:
        """Fill polygon interior + Contour gap-fill for white bites closed by the polygon."""
        if self._damage_mask is None or self._leaf_roi is None or len(self._pending_pts) < 3:
            return
        leaf_px = int(self._leaf_roi.sum())
        pts = np.array(self._pending_pts, dtype=np.int32).reshape((-1, 1, 2))

        poly = np.zeros(self._damage_mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly, [pts], 255)
        # Closed outline on leaf ? Contour fill-holes captures white edge bites
        outline = np.zeros(self._damage_mask.shape, dtype=np.uint8)
        cv2.polylines(outline, [pts], isClosed=True, color=255, thickness=2)
        pocket = self._contour_gap_fill_from_overlay(outline)

        region = (poly > 0) | pocket | (outline > 0)
        region_px = int(region.sum())
        rejected = bool(leaf_px > 0 and region_px > 0.35 * leaf_px)
        if not rejected:
            self._push_undo()
            self._damage_mask[region] = True
        self._pending_pts = []
        self._geometry_cursor = None
        if not rejected:
            self._damage_dirty = True
            self._refresh_damage_pil()
            self._render_image()
            self._update_metrics_label()
            self._flush_damage_persist()
        self._redraw_geometry_preview()
        if rejected:
            self._hint_label.configure(
                text="Polygon too large (>35% of leaf) ' cancelled. Draw tighter around the damage."
            )
        else:
            self._update_hint()

    def _on_polygon_close_key(self, _event=None) -> str:
        if self._can_edit() and self._edit_mode_var == "polygon":
            self._commit_polygon_fill()
            return "break"
        return ""

    def _on_damage_escape(self, _event=None) -> str:
        if self._can_edit() and self._pending_pts:
            self._clear_pending_geometry()
            self._update_hint()
            return "break"
        return self._on_escape_reset_view(_event)

    def _on_double_click(self, event: tk.Event) -> None:
        if not (self._can_edit() and self._damage_mask is not None):
            return
        if self._edit_mode_var == "polygon" and len(self._pending_pts) >= 3:
            self._commit_polygon_fill()

    def _select_region_at(self, ix: float, iy: float) -> None:
        if self._damage_mask is None or self._leaf_roi is None or self._rgb_base is None:
            return
        x, y = int(ix), int(iy)
        h, w = self._damage_mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._edit_mode_var == "select_remove":
            if not self._damage_mask[y, x]:
                return
            self._push_undo()
            blob = _analyze_leaves().connected_damage_component(self._damage_mask, (x, y))
            self._damage_mask[blob] = False
            return

        if self._edit_mode_var != "select_add":
            return
        if not self._leaf_roi[y, x] or self._damage_mask[y, x]:
            return
        if self._select_busy:
            return

        self._ensure_mobilesam_loaded()
        from gui.interactive_sam_session import get_session

        session = get_session()
        model = session.mobilesam if session.mobilesam_ready else None

        rgb = self._rgb_base.copy()
        leaf_roi = self._leaf_roi.copy()
        damage = self._damage_mask.copy()
        tol = int(self._brush_size)
        analyzed = self._analyzed_path
        gen = self._select_gen + 1
        self._select_gen = gen
        self._select_busy = True
        self._status_msg = "Selecting region'"
        self._update_metrics_label()

        def worker() -> None:
            try:
                region = _analyze_leaves().hybrid_select_damage_region(
                    rgb,
                    (x, y),
                    leaf_roi,
                    tol,
                    damage_mask=damage,
                    mobilesam_model=model,
                )
            except Exception:
                region = np.zeros(leaf_roi.shape, dtype=bool)
            self.after(
                0,
                lambda r=region, g=gen, p=analyzed: self._apply_select_add_region(r, g, p),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_select_add_region(
        self,
        region: np.ndarray,
        gen: int,
        analyzed_path: Path | None,
    ) -> None:
        self._select_busy = False
        if gen != self._select_gen:
            return
        if analyzed_path is None or self._analyzed_path != analyzed_path:
            self._status_msg = None
            self._update_metrics_label()
            return
        if self._damage_mask is None or region is None or not region.any():
            self._status_msg = None
            self._update_metrics_label()
            return
        if region.shape != self._damage_mask.shape:
            self._status_msg = None
            self._update_metrics_label()
            return

        self._push_undo()
        self._damage_mask[region] = True
        self._status_msg = None
        self._damage_dirty = True
        self._refresh_damage_pil()
        self._render_image()
        self._update_metrics_label()
        self._flush_damage_persist()

    def _show_analyzed_fallback(self, message: str | None = None) -> None:
        """Show the analyzed JPG when editable sidecars cannot be composed."""
        was_active = self._edit_active
        self._edit_mode = False
        self._edit_active = False
        self._set_edit_bar_visible(False)
        self._set_edit_controls_enabled(False)
        self._clear_undo_stack()
        self._clear_pending_geometry()
        if message:
            if self._metrics_label is not None:
                self._metrics_label.configure(text=message)
            mcb = getattr(self, "_metrics_cb", None)
            if mcb is not None:
                try:
                    mcb(message)
                except Exception:
                    pass
        if was_active:
            cb = getattr(self, "_edit_active_cb", None)
            if cb is not None:
                cb(False)
        super()._show_current()

    def _resolve_white_bg(self, analyzed_path: Path, out: Path, meta: dict | None) -> Path | None:
        """Prefer the exact white_bg file recorded in meta (same as analyze_leaves)."""
        if meta:
            image_name = str(meta.get("image_name") or "").strip()
            if image_name:
                candidate = white_bg_dir(out) / image_name
                if candidate.is_file():
                    return candidate
                by_stem = white_bg_path_for_stem(Path(image_name).stem, out)
                if by_stem is not None:
                    return by_stem
        return white_bg_path_for_stem(analyzed_stem_from_jpg(analyzed_path), out)

    @staticmethod
    def _align_mask_to_hw(mask_u8: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        mask = np.ascontiguousarray(np.squeeze(mask_u8))
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        if mask.shape[:2] != hw:
            mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _show_current(self) -> None:
        self._select_gen += 1
        self._select_busy = False
        self._status_msg = None
        self._damage_dirty = False
        self._clear_undo_stack()
        self._clear_pending_geometry()
        keep_edit = self._edit_active
        if not self._paths:
            self._show_empty(self._empty_message)
            return

        self._index = max(0, min(self._index, len(self._paths) - 1))
        analyzed_path = self._paths[self._index]
        out = self._state.output_path()
        if out is None:
            self._show_empty("Define output folder")
            return

        dmg_path = analyzed_damage_mask_path(analyzed_path)
        roi_path = analyzed_leaf_roi_path(analyzed_path)
        meta_path = analyzed_meta_path(analyzed_path)

        if not (dmg_path.is_file() and roi_path.is_file() and meta_path.is_file()):
            self._show_analyzed_fallback("Re-run analysis to enable editing")
            return

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._show_analyzed_fallback("Could not read analysis metadata")
            return

        white_bg = self._resolve_white_bg(analyzed_path, out, meta)
        if white_bg is None or not white_bg.is_file():
            self._show_analyzed_fallback()
            return

        try:
            rgb = load_rgb(white_bg)
        except OSError:
            rgb = None
        damage_u8 = cv2.imread(str(dmg_path), cv2.IMREAD_GRAYSCALE)
        roi_u8 = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or damage_u8 is None or roi_u8 is None:
            self._show_analyzed_fallback("Could not read analysis data")
            return

        hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        damage_u8 = self._align_mask_to_hw(damage_u8, hw)
        roi_u8 = self._align_mask_to_hw(roi_u8, hw)
        if damage_u8.shape[:2] != hw or roi_u8.shape[:2] != hw:
            self._show_analyzed_fallback()
            return

        self._edit_mode = True
        self._analyzed_path = analyzed_path
        self._rgb_base = np.asarray(rgb, dtype=np.uint8)
        self._damage_mask = damage_u8 > 127
        self._leaf_roi = roi_u8 > 127
        self._meta = meta
        self._baseline_damage = self._damage_mask.copy()
        self._damage_dirty = False

        self._update_nav_buttons()
        self._caption.configure(text=analyzed_path.name)
        self._preview_cursor = None
        self._refresh_damage_pil()
        self._update_metrics_label()
        self.restore_view()

        # Preserve edit session across Prev/Next when user already opened Edit Damage
        if keep_edit:
            self._edit_active = True
            self._set_edit_bar_visible(True)
            self._set_edit_controls_enabled(True)
            self._on_edit_mode_change("Pan/Zoom")
        else:
            self._edit_active = False
            self._set_edit_bar_visible(False)
            self._set_edit_controls_enabled(False)

        if self._can_edit() and self._is_brush_mode():
            self._ensure_preview_cursor()
            self._redraw_brush_preview()
        self.after_idle(self._ensure_view_fitted)

    def _current_metrics(self) -> tuple[int, float]:
        if self._damage_mask is None or self._leaf_roi is None or self._meta is None:
            return 0, 0.0
        leaf_area = int(self._meta.get("leaf_area_px", self._leaf_roi.sum()))
        # Include edge bites painted outside the silhouette
        damage_px = int(self._damage_mask.sum())
        damage_pct = (damage_px / leaf_area * 100.0) if leaf_area > 0 else 0.0
        return damage_px, damage_pct

    def _update_metrics_label(self) -> None:
        if self._status_msg:
            text = self._status_msg
        elif not self._edit_mode or self._meta is None:
            text = ""
        else:
            damage_px, damage_pct = self._current_metrics()
            scale = self._meta.get("scale_cm2_per_px")
            al = _analyze_leaves()
            pct_txt = al.format_damage_pct(damage_pct)
            if scale is not None and self._state.report_area_cm2:
                damage_cm2 = damage_px * float(scale)
                text = f"Damage: {pct_txt}%  |  {damage_cm2:.2f} cm\u00b2"
            else:
                text = f"Damage: {pct_txt}%"
        if self._metrics_label is not None:
            self._metrics_label.configure(text=text)
        cb = getattr(self, "_metrics_cb", None)
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _refresh_damage_pil(self) -> None:
        if self._rgb_base is None or self._damage_mask is None or self._leaf_roi is None:
            return
        rgb = _analyze_leaves().compose_damage_rgb(
            self._rgb_base, self._damage_mask, self._leaf_roi
        )
        self._pil_original = Image.fromarray(rgb)

    def _paint_damage_at(self, ix: float, iy: float, add: bool) -> None:
        if self._damage_mask is None:
            return
        r = max(2, int(self._brush_size // 2))
        if self._brush_shape == "circle":
            tmp = np.zeros_like(self._damage_mask, dtype=np.uint8)
            cv2.circle(tmp, (int(ix), int(iy)), r, 255, -1)
            brush = tmp > 0
        else:
            y0, y1 = max(0, int(iy) - r), min(self._damage_mask.shape[0], int(iy) + r + 1)
            x0, x1 = max(0, int(ix) - r), min(self._damage_mask.shape[1], int(ix) + r + 1)
            brush = np.zeros_like(self._damage_mask, dtype=bool)
            brush[y0:y1, x0:x1] = True
        if add:
            self._damage_mask[brush] = True
        else:
            self._damage_mask[brush] = False

    def _stroke_damage(self, x0: float, y0: float, x1: float, y1: float, add: bool) -> None:
        if self._damage_mask is None:
            return
        thickness = max(4, int(self._brush_size))
        tmp = np.zeros(self._damage_mask.shape, dtype=np.uint8)
        cv2.line(tmp, (int(x0), int(y0)), (int(x1), int(y1)), 255, thickness=thickness)
        cv2.circle(tmp, (int(x1), int(y1)), max(2, thickness // 2), 255, -1)
        brush = tmp > 0
        if add:
            self._damage_mask[brush] = True
        else:
            self._damage_mask[brush] = False

    def _schedule_damage_persist(self) -> None:
        self._damage_dirty = True
        if self._damage_save_after_id is not None:
            self.after_cancel(self._damage_save_after_id)
        self._damage_save_after_id = self.after(SAVE_DEBOUNCE_MS, self._flush_damage_persist)

    def _flush_damage_persist(self) -> None:
        if self._damage_save_after_id is not None:
            try:
                self.after_cancel(self._damage_save_after_id)
            except Exception:
                pass
            self._damage_save_after_id = None
        if not self._damage_dirty:
            return
        if not self._edit_mode or self._damage_mask is None or self._analyzed_path is None:
            return
        if self._rgb_base is None or self._leaf_roi is None or self._meta is None:
            return
        out = self._state.output_path()
        if out is None:
            return

        damage_px, damage_pct = self._current_metrics()
        image_name = str(self._meta.get("image_name", analyzed_stem_from_jpg(self._analyzed_path)))
        scale = self._meta.get("scale_cm2_per_px")

        self._meta["damage_px"] = damage_px
        self._meta["damage_pct"] = _analyze_leaves().round_damage_pct(damage_pct)

        try:
            dmg_path = analyzed_damage_mask_path(self._analyzed_path)
            meta_path = analyzed_meta_path(self._analyzed_path)
            mask_u8 = (self._damage_mask.astype(np.uint8) * 255)
            if not cv2.imwrite(str(dmg_path), mask_u8):
                return
            meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
            _analyze_leaves().save_damage_preview_pil(
                self._rgb_base,
                self._damage_mask,
                self._leaf_roi,
                image_name,
                damage_pct,
                damage_px,
                str(self._analyzed_path),
                scale_cm2_per_px=scale,
            )
            self._update_results_csv(out, image_name, damage_px, damage_pct, scale)
            self._damage_dirty = False
        except OSError:
            pass

    def _flush_persist(self) -> None:
        self._flush_damage_persist()
        super()._flush_persist()

    def _update_results_csv(
        self,
        output_root: Path,
        image_name: str,
        damage_px: int,
        damage_pct: float,
        scale_cm2_per_px: float | None,
    ) -> None:
        csv_path = output_root / "analyzed" / "results.csv"
        if not csv_path.is_file():
            return
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return
                rows = list(reader)
                fieldnames = list(reader.fieldnames)
        except OSError:
            return

        leaf_area_px = int(self._meta.get("leaf_area_px", 0)) if self._meta else 0
        for row in rows:
            if row.get("image_name") != image_name:
                continue
            if "damage_pct" in fieldnames:
                row["damage_pct"] = str(_analyze_leaves().round_damage_pct(damage_pct))
            if "damage_px" in fieldnames:
                row["damage_px"] = str(int(damage_px))
            if (
                scale_cm2_per_px is not None
                and self._state.report_area_cm2
                and "damage_cm2" in fieldnames
            ):
                leaf_cm2 = leaf_area_px * float(scale_cm2_per_px)
                if "leaf_area_cm2" in fieldnames:
                    row["leaf_area_cm2"] = str(round(leaf_cm2, 4))
                row["damage_cm2"] = str(round(damage_px * float(scale_cm2_per_px), 4))
            break

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except OSError:
            pass

    def _draw_image(self) -> None:
        super()._draw_image()
        self._redraw_geometry_preview()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if (
            self._can_edit()
            and self._is_geometry_mode()
            and self._pending_pts
        ):
            self._geometry_cursor = self._canvas_to_image_xy(event.x, event.y)
            self._redraw_geometry_preview()
            return
        super()._on_canvas_motion(event)

    def _on_button1_press(self, event: tk.Event) -> None:
        if not self._can_edit():
            super()._on_button1_press(event)
            return

        if self._is_select_mode():
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            if self._edit_mode_var == "select_add":
                self._select_region_at(ix, iy)
                return
            self._select_region_at(ix, iy)
            self._damage_dirty = True
            self._refresh_damage_pil()
            self._render_image()
            self._update_metrics_label()
            self._flush_damage_persist()
            return

        if self._is_brush_mode():
            self._canvas.focus_set()
            self._hide_brush_preview()
            if not self._stroke_started:
                self._push_undo()
                self._stroke_started = True
            add = self._edit_mode_var == "add"
            self._last_erase_img_xy = self._canvas_to_image_xy(event.x, event.y)
            ix, iy = self._last_erase_img_xy
            self._paint_damage_at(ix, iy, add)
            self._refresh_damage_pil()
            self._render_image()
            self._update_metrics_label()
            self._schedule_damage_persist()
            return

        if self._edit_mode_var == "line":
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            self._pending_pts.append((int(ix), int(iy)))
            if len(self._pending_pts) >= 2:
                self._commit_line_fill()
            else:
                self._hint_label.configure(text="Click end point to bridge and fill")
                self._redraw_geometry_preview()
            return

        if self._edit_mode_var == "polygon":
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            px, py = int(ix), int(iy)
            # Close only when clicking near the first vertex (not on the 3rd point in-zone)
            if len(self._pending_pts) >= 3:
                x0, y0 = self._pending_pts[0]
                if (px - x0) * (px - x0) + (py - y0) * (py - y0) <= 18 * 18:
                    self._commit_polygon_fill()
                    return
            self._pending_pts.append((px, py))
            n = len(self._pending_pts)
            self._redraw_geometry_preview()
            self._hint_label.configure(
                text=f"{n} point(s) ' Click near start / Enter / double-click to fill ' Esc to cancel"
            )
            return

        super()._on_button1_press(event)

    def _on_button1_motion(self, event: tk.Event) -> None:
        if self._can_edit() and self._is_brush_mode():
            add = self._edit_mode_var == "add"
            ix, iy = self._canvas_to_image_xy(event.x, event.y)
            if self._last_erase_img_xy is not None:
                self._stroke_damage(
                    self._last_erase_img_xy[0],
                    self._last_erase_img_xy[1],
                    ix,
                    iy,
                    add,
                )
            else:
                self._paint_damage_at(ix, iy, add)
            self._last_erase_img_xy = (ix, iy)
            self._refresh_damage_pil()
            self._render_image()
            self._update_metrics_label()
            self._schedule_damage_persist()
            return
        if self._can_edit() and self._is_geometry_mode():
            return
        super()._on_button1_motion(event)

    def _on_button1_release(self, event: tk.Event) -> None:
        if self._can_edit() and self._is_brush_mode():
            self._last_erase_img_xy = None
            self._stroke_started = False
            self._flush_damage_persist()
            self._preview_cursor = (event.x, event.y)
            self._redraw_brush_preview()
            return
        super()._on_button1_release(event)

    def refresh(self) -> None:
        self._flush_damage_persist()
        super().refresh()
