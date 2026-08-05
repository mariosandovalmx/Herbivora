"""Project tab: paths, models, installation checks."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import customtkinter as ctk

from gui.paths import (
    MODELS_DIR,
    REPO_ROOT,
    analyzed_dir,
    auto_detect_models,
    count_masks,
    count_white_bg_leaves,
    list_images,
    segmentation_dir,
)
from gui.state import ProjectState
from gui.widgets.folder_picker import PathPickerRow
from gui.widgets.image_carousel import ImageCarousel
from gui.widgets.split_pane import SplitPane
from gui.widgets.info_button import InfoButton
from image_io import supported_formats_label


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------

class _CollapsibleSection(ctk.CTkFrame):
    """A frame with a toggle button that shows/hides its content."""

    def __init__(self, master, title: str, **kwargs) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._expanded = False

        # Header row: arrow + title button
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)

        self._arrow = ctk.CTkLabel(hdr, text="▶", width=18, font=ctk.CTkFont(size=11))
        self._arrow.grid(row=0, column=0, padx=(0, 4))

        btn = ctk.CTkButton(
            hdr,
            text=title,
            anchor="w",
            fg_color="transparent",
            text_color=("gray40", "gray70"),
            hover_color=("gray85", "gray25"),
            font=ctk.CTkFont(size=12),
            height=26,
            command=self._toggle,
        )
        btn.grid(row=0, column=1, sticky="ew")

        # Separator line under the header
        sep = ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.grid(row=1, column=0, sticky="ew", pady=(2, 6))

        # Content frame (hidden by default)
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.grid_columnconfigure(0, weight=1)
        # Not gridded yet — will be shown on toggle

    @property
    def content(self) -> ctk.CTkFrame:
        return self._content

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        if self._expanded:
            self._content.grid(row=2, column=0, sticky="ew")
            self._arrow.configure(text="▼")
        else:
            self._content.grid_remove()
            self._arrow.configure(text="▶")

    def expand(self) -> None:
        if not self._expanded:
            self._toggle()

    def collapse(self) -> None:
        if self._expanded:
            self._toggle()


# ---------------------------------------------------------------------------
# Project Tab
# ---------------------------------------------------------------------------

class ProjectTab(ctk.CTkFrame):
    def __init__(self, master, state: ProjectState, on_change, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._state = state
        self._on_change = on_change
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        split = SplitPane(self)
        split.grid(row=0, column=0, sticky="nsew")

        scroll = ctk.CTkScrollableFrame(split)
        scroll.grid_columnconfigure(0, weight=1)

        self._carousel = ImageCarousel(
            split, title="Preview (input folder)", show_source_selector=False
        )
        self._carousel.set_path_provider(self._input_image_paths)

        split.set_left(scroll)
        split.set_right(self._carousel)

        # ── Section: Project ─────────────────────────────────────────────
        row_proj = ctk.CTkFrame(scroll, fg_color="transparent")
        row_proj.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ctk.CTkLabel(
            row_proj,
            text="Project",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left")
        InfoButton(
            row_proj,
            title="Project directories",
            message=(
                "Input folder: Where your original images are located.\n"
                "Output folder: Where all generated masks and analysis results will be saved."
            ),
        ).pack(side="left", padx=8)

        self._input_picker = PathPickerRow(scroll, "Input folder")
        self._input_picker.grid(row=1, column=0, sticky="ew", pady=4)
        self._input_picker.set_change_callback(self._on_input_changed)

        ctk.CTkLabel(
            scroll,
            text=supported_formats_label(),
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 4))

        self._output_picker = PathPickerRow(scroll, "Output folder")
        self._output_picker.grid(row=3, column=0, sticky="ew", pady=4)
        self._output_picker.set_change_callback(self._on_output_changed)

        row_skip = ctk.CTkFrame(scroll, fg_color="transparent")
        row_skip.grid(row=4, column=0, sticky="ew", pady=(8, 4))
        self._skip_segmentation = ctk.CTkCheckBox(
            row_skip,
            text=(
                "Skip segmentation (pre-isolated leaves on white background; "
                "resize to 1024×1024 for Contour / ROI)"
            ),
            command=self._on_skip_segmentation_toggled,
        )
        self._skip_segmentation.pack(side="left")
        InfoButton(
            row_skip,
            title="Skip segmentation",
            message=(
                "Use when your input folder already contains individual leaf photos "
                "on (or near) white background.\n\n"
                "When enabled:\n"
                "• Tab 2 (Segmentation) is disabled.\n"
                "• Images are processed with whitebg_masks at the fixed pipeline "
                "resolution (1024×1024) into "
                "segmentation/white_bg/ and segmentation/masks/.\n"
                "• Continue with Contour / ROI and Analysis from their tabs."
            ),
        ).pack(side="left", padx=8)

        self._skip_prep_banner = ctk.CTkFrame(
            scroll, corner_radius=8, fg_color=("gray90", "gray25")
        )
        self._skip_prep_banner.grid(row=5, column=0, sticky="ew", pady=(4, 6))
        self._skip_prep_banner.grid_columnconfigure(0, weight=1)
        self._skip_prep_label = ctk.CTkLabel(
            self._skip_prep_banner,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=13),
            wraplength=380,
            justify="left",
        )
        self._skip_prep_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        self._skip_prep_progress = ctk.CTkProgressBar(
            self._skip_prep_banner, mode="determinate", height=8
        )
        self._skip_prep_progress.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
        self._skip_prep_progress.set(0)
        self._skip_prep_progress.grid_remove()
        self._skip_prep_banner.grid_remove()
        self._skip_prep_busy = False

        row_multi = ctk.CTkFrame(scroll, fg_color="transparent")
        row_multi.grid(row=6, column=0, sticky="w", pady=(4, 4))
        self._multi_leaf = ctk.CTkCheckBox(
            row_multi,
            text="Multiple leaves per photo",
            command=self._on_multi_leaf_toggled,
        )
        self._multi_leaf.pack(side="left")
        InfoButton(
            row_multi,
            title="Multiple leaves per photo",
            message=(
                "Enable when each input photo contains several separated leaves "
                "(not touching / overlapping).\n\n"
                "When enabled, Segmentation method A (BiRefNet + MobileSAM) runs a "
                "multi-leaf pipeline:\n"
                "• BiRefNet detects leaf tissue\n"
                "• Connected components separate each leaf\n"
                "• MobileSAM refines each leaf with a box prompt\n"
                "• Outputs are named {photo}_leaf_1, {photo}_leaf_2, …\n"
                "• Contour / Analysis and results.csv use one row per leaf "
                "(no Contour or Damage model re-training).\n\n"
                "Limitation: touching or overlapping leaves may be merged into one "
                "component. Mutually exclusive with Skip segmentation.\n"
                "Methods B (Otsu) and C (Interactive) stay single-leaf; enabling this "
                "option switches Segmentation to method A."
            ),
        ).pack(side="left", padx=8)

        row_scale = ctk.CTkFrame(scroll, fg_color="transparent")
        row_scale.grid(row=7, column=0, sticky="w", pady=(4, 4))
        self._remove_blue = ctk.CTkCheckBox(
            row_scale,
            text="Scale reference in photo (blue dot)",
            command=self._on_remove_blue_toggled,
        )
        self._remove_blue.pack(side="left")
        InfoButton(
            row_scale,
            title="Scale reference (blue dot)",
            message=(
                "Enable when photos contain a blue reference dot.\n"
                "The dot is excluded from masks and can be used for cm² analysis."
            ),
        ).pack(side="left", padx=8)

        # ── Models auto-status badge ──────────────────────────────────────
        self._models_status_frame = ctk.CTkFrame(scroll, corner_radius=8)
        self._models_status_frame.grid(row=8, column=0, sticky="ew", pady=(14, 4))
        self._models_status_frame.grid_columnconfigure(1, weight=1)

        self._models_icon = ctk.CTkLabel(
            self._models_status_frame, text="✅", font=ctk.CTkFont(size=16), width=30
        )
        self._models_icon.grid(row=0, column=0, padx=(10, 4), pady=8)

        self._models_status_label = ctk.CTkLabel(
            self._models_status_frame,
            text="Models detected automatically from  models/",
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self._models_status_label.grid(row=0, column=1, sticky="ew", pady=8)

        btn_rescan = ctk.CTkButton(
            self._models_status_frame,
            text="↻ Rescan",
            width=80,
            height=28,
            font=ctk.CTkFont(size=11),
            command=self._rescan_models,
        )
        btn_rescan.grid(row=0, column=2, padx=(4, 10), pady=8)

        # ── Collapsible: Advanced Options — Manual Paths ──────────────────
        self._adv_section = _CollapsibleSection(
            scroll, "⚙  Advanced Options — Manual Paths"
        )
        self._adv_section.grid(row=9, column=0, sticky="ew", pady=(12, 4))
        content = self._adv_section.content
        content.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            content,
            text=(
                "Override the auto-detected model paths below.  "
                "Leave unchanged to use the models/ folder."
            ),
            text_color=("gray45", "gray60"),
            wraplength=380,
            justify="left",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        # -- Segmentation models sub-header
        ctk.CTkLabel(
            content, text="Segmentation models", font=ctk.CTkFont(weight="bold", size=12)
        ).grid(row=1, column=0, sticky="w", pady=(4, 2))

        self._mobilesam_picker = PathPickerRow(
            content, "MobileSAM (.pt)", is_dir=False,
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
        )
        self._mobilesam_picker.grid(row=2, column=0, sticky="ew", pady=3)
        self._mobilesam_picker.set_change_callback(self._schedule_apply_manual_paths)

        # -- Contour models sub-header
        ctk.CTkLabel(
            content, text="Contour models", font=ctk.CTkFont(weight="bold", size=12)
        ).grid(row=3, column=0, sticky="w", pady=(10, 2))

        self._leaf_picker = PathPickerRow(
            content, "Contour U-Net (.pth)", is_dir=False,
            filetypes=[("PyTorch", "*.pth"), ("All files", "*.*")]
        )
        self._leaf_picker.grid(row=4, column=0, sticky="ew", pady=3)
        self._leaf_picker.set_change_callback(self._schedule_apply_manual_paths)

        # -- Analysis models sub-header
        ctk.CTkLabel(
            content, text="Analysis models", font=ctk.CTkFont(weight="bold", size=12)
        ).grid(row=5, column=0, sticky="w", pady=(10, 2))

        self._damage_picker = PathPickerRow(
            content, "Damage U-Net (.pth)", is_dir=False,
            filetypes=[("PyTorch", "*.pth"), ("All files", "*.*")]
        )
        self._damage_picker.grid(row=6, column=0, sticky="ew", pady=3)
        self._damage_picker.set_change_callback(self._schedule_apply_manual_paths)

        ctk.CTkLabel(
            content,
            text="Paths apply automatically when you Browse or edit the fields above.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
            wraplength=420,
            justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(8, 4))

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=10, column=0, sticky="ew", pady=12)
        self._check_install_btn = ctk.CTkButton(
            btn_row, text="Check installation", command=self._check_install
        )
        self._check_install_btn.pack(side="left", padx=4)
        self._install_check_busy = False

        # ── Status ────────────────────────────────────────────────────────
        self._status_label = ctk.CTkLabel(scroll, text="", justify="left", anchor="w")
        self._status_label.grid(row=11, column=0, sticky="ew", pady=8)

        self._skip_segmentation_cb = None
        self._input_refresh_job: str | None = None
        self._output_refresh_job: str | None = None
        self._model_paths_job: str | None = None
        self._suppress_model_apply = False

    # ------------------------------------------------------------------
    # Model auto-detection
    # ------------------------------------------------------------------

    def _rescan_models(self) -> None:
        """Re-scan canonical model folders and apply detected paths to state."""
        detected = auto_detect_models()
        self._apply_detected(detected)
        self._on_change()
        self._update_models_badge(detected)

    def _apply_detected(self, detected: dict) -> None:
        """Push auto-detected paths into both state and the manual pickers."""
        def _s(key: str, attr: str) -> None:
            p = detected.get(key)
            if p is not None:
                val = str(p)
                setattr(self._state, attr, val)

        _s("mobilesam", "mobilesam_model")
        _s("damage",    "damage_model")
        unet_shape = detected.get("unet_shape")
        if unet_shape is not None:
            path = str(unet_shape)
            self._state.leaf_model = path
            self._state.recon_model_unet_shape = path

        # Sync into the manual pickers (for display)
        self._suppress_model_apply = True
        try:
            self._mobilesam_picker.set(self._state.mobilesam_model)
            self._damage_picker.set(self._state.damage_model)
            self._leaf_picker.set(self._state.recon_model_unet_shape or self._state.leaf_model)
        finally:
            self._suppress_model_apply = False

    def _update_models_badge(self, detected: dict | None = None) -> None:
        """Update the status badge for the 3 models used by the GUI."""
        if detected is None:
            detected = auto_detect_models()
        found = sum(1 for p in detected.values() if p is not None and Path(str(p)).is_file())
        total = len(detected)

        if found == total:
            self._models_icon.configure(text="✅")
            self._models_status_label.configure(
                text=f"All {total} models detected automatically"
            )
        elif found > 0:
            self._models_icon.configure(text="⚠️")
            self._models_status_label.configure(
                text=f"{found}/{total} models found — check Advanced Options below"
            )
        else:
            self._models_icon.configure(text="❌")
            self._models_status_label.configure(
                text="No models found — set paths manually in Advanced Options"
            )
            self._adv_section.expand()

    def _schedule_apply_manual_paths(self) -> None:
        """Debounce path edits so typing does not thrash model reload / config save."""
        if self._suppress_model_apply:
            return
        if self._model_paths_job is not None:
            self.after_cancel(self._model_paths_job)
        self._model_paths_job = self.after(400, self._apply_manual_paths)

    def _apply_manual_paths(self) -> None:
        """Read the manual pickers and push their values into state."""
        self._model_paths_job = None
        if self._suppress_model_apply:
            return
        # Ignore empty writes during widget init / clear-before-set.
        damage = self._damage_picker.get()
        if damage:
            self._state.damage_model = damage
        mobilesam = self._mobilesam_picker.get()
        if mobilesam:
            prev = self._state.mobilesam_model
            self._state.mobilesam_model = mobilesam
            if mobilesam != prev:
                self._invalidate_mobilesam_session()
        leaf = self._leaf_picker.get()
        if leaf:
            self._set_contour_unet_path(leaf)
        self._on_change()
        self._update_models_badge()

    def _set_contour_unet_path(self, path: str) -> None:
        """Store Contour U-Net weights path (used by the Contour tab at run time)."""
        path = path.strip().strip("\"'")
        if not path:
            return
        self._state.leaf_model = path
        self._state.recon_model_unet_shape = path

    def _invalidate_mobilesam_session(self) -> None:
        """Drop cached MobileSAM so the next use loads the new weights path."""
        try:
            from gui.interactive_sam_session import get_session

            get_session().clear_mobilesam()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------

    def _sync_dirs_to_state(self) -> None:
        inp = self._input_picker.get()
        out = self._output_picker.get()
        if inp:
            self._state.input_dir = inp
        if out:
            self._state.output_dir = out
        self._on_change()

    def _on_input_changed(self) -> None:
        if self._input_refresh_job is not None:
            self.after_cancel(self._input_refresh_job)
        self._input_refresh_job = self.after(400, self._apply_input_change)

    def _apply_input_change(self) -> None:
        self._input_refresh_job = None
        inp = self._input_picker.get()
        if inp:
            self._output_picker.set(inp)
        self._sync_dirs_to_state()
        p = Path(inp) if inp else None
        if p and p.is_dir():
            self.refresh_preview()
            self.refresh_status()

    def _on_output_changed(self) -> None:
        if self._output_refresh_job is not None:
            self.after_cancel(self._output_refresh_job)
        self._output_refresh_job = self.after(400, self._apply_output_change)

    def _apply_output_change(self) -> None:
        self._output_refresh_job = None
        self._sync_dirs_to_state()
        self.refresh_status()

    # ------------------------------------------------------------------
    # State I/O
    # ------------------------------------------------------------------

    def set_skip_segmentation_callback(self, cb) -> None:
        self._skip_segmentation_cb = cb

    def revert_skip_segmentation(self) -> None:
        """Uncheck skip mode and sync state (after cancel or failed prep)."""
        self._skip_segmentation.deselect()
        self._state.skip_segmentation = False
        self.set_skip_prep_status("idle")
        self._on_change()

    def _replace_skip_prep_progress(self, *, animate: bool) -> None:
        """Recreate the skip-prep bar so CTk indeterminate loops cannot linger."""
        old = getattr(self, "_skip_prep_progress", None)
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
            try:
                if getattr(old, "_loop_after_id", None) is not None:
                    old.after_cancel(old._loop_after_id)
                old._loop_running = False
                old.destroy()
            except Exception:
                pass
        bar = ctk.CTkProgressBar(self._skip_prep_banner, mode="determinate", height=8)
        bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
        bar.set(0)
        self._skip_prep_progress = bar
        if animate:
            bar.configure(mode="indeterminate")
            bar.start()
        else:
            bar.grid_remove()

    def set_skip_prep_status(self, status: str, *, image_count: int | None = None) -> None:
        """Update skip-segmentation prep banner: idle | busy | done | failed."""
        self._skip_prep_busy = status == "busy"
        if status == "idle":
            self._replace_skip_prep_progress(animate=False)
            self._skip_prep_banner.grid_remove()
            self._skip_segmentation.configure(state="normal")
            return

        self._skip_prep_banner.grid()
        if status == "busy":
            size = self._state.segmentation_output_size() or 1024
            self._skip_prep_label.configure(
                text="Analyzing, please wait...\n"
                f"Resizing images and generating masks ({size}×{size}) for Contour / ROI.",
                text_color=("gray25", "gray90"),
            )
            self._replace_skip_prep_progress(animate=True)
            self._skip_segmentation.configure(state="disabled")
            return

        self._replace_skip_prep_progress(animate=False)
        self._skip_segmentation.configure(state="normal")
        if status == "done":
            suffix = f" ({image_count} images ready)" if image_count else ""
            self._skip_prep_label.configure(
                text=f"Done{suffix}. Open tab 3. Contour / ROI and click Run contour.",
                text_color=("#1b5e20", "#81c784"),
            )
        elif status == "failed":
            self._skip_prep_label.configure(
                text="Preparation failed. Check the log for details.",
                text_color=("#8B0000", "#ff6b6b"),
            )
        else:
            self._skip_prep_banner.grid_remove()

    def refresh_skip_prep_readiness(self) -> None:
        """Show done banner when skip mode is on and white_bg output already exists."""
        if self._skip_prep_busy or not self._state.skip_segmentation:
            if not self._state.skip_segmentation:
                self.set_skip_prep_status("idle")
            return
        out = self._state.output_path()
        if out is None:
            return
        n = count_white_bg_leaves(out)
        if n > 0:
            self.set_skip_prep_status("done", image_count=n)

    def _on_remove_blue_toggled(self) -> None:
        self._state.remove_blue = bool(self._remove_blue.get())
        if not self._state.remove_blue:
            # No blue-dot scale → Analysis can only report damage %.
            self._state.report_area_cm2 = False
        self._on_change()

    def _on_multi_leaf_toggled(self) -> None:
        """Enable multi-leaf mode; mutually exclusive with Skip segmentation."""
        from tkinter import messagebox

        checked = bool(self._multi_leaf.get())
        if checked and bool(self._skip_segmentation.get()):
            self._skip_segmentation.deselect()
            self._state.skip_segmentation = False
            self.set_skip_prep_status("idle")
            if self._skip_segmentation_cb:
                self._skip_segmentation_cb(run_prep=False)

        self._state.multi_leaf_photos = checked
        if checked:
            # Multi-leaf uses Method A only
            self._state.segmentation_method = "birefnet_mobilesam"
            messagebox.showinfo(
                "Multiple leaves per photo",
                "Multi-leaf mode is on.\n\n"
                "Segmentation will use BiRefNet + MobileSAM (method A) and export "
                "one file per leaf ({photo}_leaf_1, {photo}_leaf_2, …).\n\n"
                "Leaves should be separated (not touching). Contour / Damage models "
                "are unchanged — each leaf is analyzed like a normal single-leaf image.",
            )
        self._on_change()

    def _on_skip_segmentation_toggled(self) -> None:
        from tkinter import messagebox

        checked = bool(self._skip_segmentation.get())
        if not checked:
            self._state.skip_segmentation = False
            self.set_skip_prep_status("idle")
            self._on_change()
            if self._skip_segmentation_cb:
                self._skip_segmentation_cb(run_prep=False)
            return

        # Mutually exclusive with multi-leaf
        if bool(self._multi_leaf.get()):
            self._multi_leaf.deselect()
            self._state.multi_leaf_photos = False

        size = self._state.segmentation_output_size() or 1024
        proceed = messagebox.askyesno(
            "Skip segmentation",
            f"This will only resize your input images to {size}×{size} and generate\n"
            "the base masks needed for Contour / ROI.\n\n"
            "Full segmentation (BiRefNet / FastSAM) will not run.\n\n"
            "Do you want to continue?",
        )
        if not proceed:
            self.revert_skip_segmentation()
            return

        if not self.validate_for_pipeline():
            self.revert_skip_segmentation()
            return

        self._state.skip_segmentation = True
        self._on_change()
        if self._skip_segmentation_cb:
            self._skip_segmentation_cb(run_prep=True)

    def validate_for_pipeline(self) -> bool:
        """Input/output folders required for full pipeline (including skip mode)."""
        from tkinter import messagebox

        self.sync_to_state()
        if not self._state.input_dir or not Path(self._state.input_dir).is_dir():
            messagebox.showerror(
                "Input folder not set",
                "No input folder is configured.\n\n"
                "Select the folder containing your leaf photos.",
            )
            return False
        if not self._state.output_dir:
            messagebox.showerror(
                "Output folder not set",
                "No output folder is configured.\n\n"
                "Select the input folder first (output is set automatically).",
            )
            return False
        return True

    def load_from_state(self) -> None:
        self._input_picker.set(self._state.input_dir)
        # If output_dir is empty, mirror input_dir so the picker always shows a value
        # and state stays consistent with what output_path() returns
        if not self._state.output_dir and self._state.input_dir:
            self._state.output_dir = self._state.input_dir
        self._output_picker.set(self._state.output_dir)

        if self._state.skip_segmentation:
            self._skip_segmentation.select()
        else:
            self._skip_segmentation.deselect()
        self.refresh_skip_prep_readiness()

        if self._state.multi_leaf_photos:
            self._multi_leaf.select()
        else:
            self._multi_leaf.deselect()

        if self._state.remove_blue:
            self._remove_blue.select()
        else:
            self._remove_blue.deselect()

        # Auto-detect first; fall back to whatever is saved in state
        detected = auto_detect_models()
        self._apply_detected(detected)
        self._update_models_badge(detected)

        self.refresh_status()
        self.refresh_preview()

    def _input_image_paths(self) -> list[Path]:
        inp = self._state.input_path()
        if inp is None or not self._state.input_dir.strip():
            return []
        return list_images(inp)

    def refresh_preview(self) -> None:
        if not self._input_image_paths():
            self._carousel.set_paths([], empty_message="Please define the input folder")
            return
        self._carousel.set_paths(self._input_image_paths())

    def sync_to_state(self) -> None:
        self._state.input_dir = self._input_picker.get()
        self._state.output_dir = self._output_picker.get()
        self._state.skip_segmentation = bool(self._skip_segmentation.get())
        self._state.multi_leaf_photos = bool(self._multi_leaf.get())
        self._state.remove_blue = bool(self._remove_blue.get())
        # Always pull model paths from pickers (Browse alone used to leave state stale).
        mobilesam = self._mobilesam_picker.get()
        if mobilesam and mobilesam != self._state.mobilesam_model:
            self._state.mobilesam_model = mobilesam
            self._invalidate_mobilesam_session()
        elif mobilesam:
            self._state.mobilesam_model = mobilesam
        self._state.damage_model = self._damage_picker.get()
        self._set_contour_unet_path(self._leaf_picker.get())
        self._on_change()

    def refresh_status(self) -> None:
        out = self._state.output_path()
        if out is None or not self._state.output_dir:
            self._status_label.configure(text="Please define input and output folders.")
            return
        
        lines = [
            f"Output: {out}"
        ]
        self._status_label.configure(text="\n".join(lines))
        self.refresh_preview()

    # ------------------------------------------------------------------
    # Check installation
    # ------------------------------------------------------------------

    def _check_install(self) -> None:
        if self._install_check_busy:
            return
        self.sync_to_state()
        lines = [f"Python: {sys.version.split()[0]}", f"Project root: {REPO_ROOT}"]

        missing = []
        modules_to_check = {
            "torch": "PyTorch (torch)",
            "torchvision": "Torchvision (torchvision)",
            "cv2": "OpenCV (opencv-python)",
            "ultralytics": "Ultralytics (ultralytics)",
            "customtkinter": "CustomTkinter (customtkinter)",
            "numpy": "NumPy (numpy)",
            "PIL": "Pillow (Pillow)",
            "scipy": "SciPy (scipy)",
            "segmentation_models_pytorch": "Segmentation Models PyTorch (segmentation-models-pytorch)",
            "tqdm": "Progress Bar (tqdm)",
            "huggingface_hub": "Hugging Face Hub (huggingface_hub)",
            "yaml": "PyYAML (pyyaml)",
            "transformers": "Transformers (transformers)",
        }

        for mod, label in modules_to_check.items():
            try:
                __import__(mod)
                if mod == "torch":
                    import torch
                    lines.append(f"PyTorch: {torch.__version__}")
                    lines.append(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
                    mps = getattr(torch.backends, "mps", None)
                    if mps is not None and mps.is_available():
                        lines.append("Apple MPS (Metal): available")
                else:
                    lines.append(f"{label}: OK")
            except ImportError:
                lines.append(f"{label}: MISSING")
                missing.append(label)

        from tkinter import messagebox

        if missing:
            missing_str = "\n".join(f" - {m}" for m in missing)
            ans = messagebox.askyesno(
                "Missing Dependencies",
                f"HerbivoR detected missing dependencies:\n\n{missing_str}\n\n"
                "Would you like the app to attempt to automatically install them now?\n"
                "(This will run pip in the background. It may take a few minutes.)"
            )
            if ans:
                import subprocess
                try:
                    cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(REPO_ROOT / "requirements.txt"),
                    ]
                    # Do not install torch here — CPU/CUDA wheels come from the bootstrap installer
                    need_torch = any("PyTorch" in m or "Torchvision" in m for m in missing)
                    if need_torch:
                        messagebox.showinfo(
                            "Install PyTorch first",
                            "PyTorch is missing. Close this dialog and run:\n\n"
                            "  Windows: Install_HerbivoR.bat\n"
                            "  macOS/Linux: Install_HerbivoR.command\n\n"
                            "See USER_GUIDE.md. Those installers pick the correct Torch wheel "
                            "and then install the other packages.",
                        )
                        return
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if proc.returncode == 0:
                        messagebox.showinfo(
                            "Success",
                            "All dependencies installed successfully!\n\n"
                            "Please restart the application to apply the changes."
                        )
                    else:
                        err_log = proc.stderr or proc.stdout
                        messagebox.showerror(
                            "Installation Failed",
                            "Automatic installation failed. Error output:\n\n"
                            f"{err_log[:800]}\n\n"
                            "Please run Install_HerbivoR.bat (Windows) or "
                            "Install_HerbivoR.command (macOS/Linux). See USER_GUIDE.md."
                        )
                except Exception as e:
                    messagebox.showerror(
                        "Installation Error",
                        f"An error occurred during installation:\n{e}\n\n"
                        "Please run Install_HerbivoR.bat (Windows) or "
                        "Install_HerbivoR.command (macOS/Linux)."
                    )
                self.load_from_state()
                return
            messagebox.showinfo(
                "Manual Installation Instructions",
                "To run HerbivoR correctly, you must install the missing libraries.\n"
                "You can do this by executing one of the following options:\n\n"
                    "Option A (Recommended):\n"
                    "Run Install_HerbivoR.bat (Windows) or Install_HerbivoR.command "
                    "(macOS/Linux). See USER_GUIDE.md.\n\n"
                    "Option B (Command Line):\n"
                    "See INSTALL.md — install torch/torchvision from pytorch.org first, then:\n"
                    "  pip install -r requirements.txt\n"
                    "  python download_models.py"
            )
            return

        # Dependencies OK — ensure model weights (download missing into models/).
        self._start_model_ensure(lines)

    def _start_model_ensure(self, dep_lines: list[str]) -> None:
        """Download missing weights in a background thread, then show the report."""
        self._install_check_busy = True
        self._check_install_btn.configure(state="disabled", text="Downloading models…")
        self._status_label.configure(
            text="Checking models/ and downloading any missing weights…"
        )

        def worker() -> None:
            # Import from repo root (same layout as CLI).
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from download_models import DEFAULT_REPO, ensure_models

            def log(msg: str) -> None:
                self.after(0, lambda m=msg: self._status_label.configure(text=m[:200]))

            try:
                result = ensure_models(
                    repo=DEFAULT_REPO,
                    models_dir=MODELS_DIR,
                    force=False,
                    log=log,
                )
            except Exception as e:
                self.after(
                    0,
                    lambda err=str(e): self._finish_model_ensure(dep_lines, None, err),
                )
                return
            self.after(
                0,
                lambda: self._finish_model_ensure(dep_lines, result, None),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_model_ensure(
        self,
        dep_lines: list[str],
        result,
        error: str | None,
    ) -> None:
        self._install_check_busy = False
        self._check_install_btn.configure(state="normal", text="Check installation")

        lines = list(dep_lines)
        lines.append("")
        lines.append(f"Models folder: {MODELS_DIR}")

        if error is not None:
            lines.append(f"Model download ERROR: {error}")
            self._status_label.configure(text=f"Model download failed: {error}")
            from tkinter import messagebox
            messagebox.showerror("Check installation", "\n".join(lines))
            return

        # Refresh GUI paths after download.
        self._rescan_models()

        role_labels = {
            "mobilesam": "MobileSAM",
            "unet_shape": "Contour U-Net",
            "damage": "Damage U-Net",
        }
        detected = auto_detect_models()
        for role, path in detected.items():
            p = Path(str(path)) if path else None
            label = role_labels.get(role, role.upper())
            if p and p.is_file():
                lines.append(f"{label}: OK — {p.name}")
            else:
                lines.append(f"{label}: NOT found — {p}")

        if result is not None:
            lines.append("")
            lines.append(f"Download: {result.ok}/{result.total} ready")
            if result.errors:
                lines.append("Download issues:")
                lines.extend(result.errors)

        if result is not None and result.success:
            self._status_label.configure(text="Installation OK — all models ready.")
        else:
            self._status_label.configure(
                text="Some models are missing or failed to download. See details."
            )

        from tkinter import messagebox

        if result is not None and not result.success:
            messagebox.showwarning("Check installation", "\n".join(lines))
        else:
            messagebox.showinfo("Check installation", "\n".join(lines))
