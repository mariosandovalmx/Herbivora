"""Segmentation Tab — BiRefNet + MobileSAM, Otsu+LAB, and Leaf-UNet methods."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from gui.paths import (
    BIREFNET_DEFAULTS,
    FASTSAM_PETRI_DEFAULTS,
    PIPELINE_RESOLUTION,
    SEGMENT_METHOD_LABELS,
)
from gui.pipeline import (
    after_fastsam_merge,
    build_birefnet_args,
    build_intact_args,
    build_fastsam_args,
    build_whitebg_args,
    needs_fastsam_step,
    prepare_fastsam_dirs,
    script_path,
)
from gui.image_sources import segment_sources
from gui.state import ProjectState
from gui.widgets.image_carousel import ImageCarousel
from gui.widgets.split_pane import SplitPane
from gui.widgets.info_button import InfoButton


class SegmentTab(ctk.CTkFrame):
    def __init__(self, master, state: ProjectState, on_change, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._state = state
        self._on_change = on_change
        self._run_cb = None
        self._low_confidence_stems: list[str] = []
        self._pick_dot_stem: str | None = None
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        split = SplitPane(self)
        split.grid(row=0, column=0, sticky="nsew")

        scroll = ctk.CTkScrollableFrame(split)
        scroll.grid_columnconfigure(0, weight=1)

        right_panel = ctk.CTkFrame(split, fg_color="transparent")
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self._warn_banner = ctk.CTkFrame(right_panel, fg_color=("#fff3cd", "#4a3c1d"))
        self._warn_banner.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        self._warn_banner.grid_columnconfigure(0, weight=1)
        self._warn_label = ctk.CTkLabel(
            self._warn_banner,
            text="",
            anchor="w",
            justify="left",
            wraplength=520,
        )
        self._warn_label.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self._mark_scale_btn = ctk.CTkButton(
            self._warn_banner,
            text="Mark scale",
            width=110,
            command=self._on_mark_scale_circle,
        )
        self._mark_scale_btn.grid(row=0, column=1, padx=(4, 0), pady=6)
        self._mark_scale_btn.grid_remove()
        self._warn_btn = ctk.CTkButton(
            self._warn_banner, text="Adjust blue dot", width=140,
            command=self._on_adjust_blue_dot,
        )
        self._warn_btn.grid(row=0, column=2, padx=8, pady=6)
        self._warn_banner.grid_remove()
        self._pick_dot_interactive = False

        self._carousel = ImageCarousel(
            right_panel,
            title="Segmentation",
            eraser_enabled=False,
            eraser_source_key="Leaves (white_bg)",
            output_root_provider=lambda: self._state.output_path(),
            show_job_status=True,
        )
        self._carousel.grid(row=1, column=0, sticky="nsew")
        self._carousel.set_pick_dot_callback(self._on_pick_dot_picked)
        self._carousel.set_point_click_callback(self._on_interactive_point_click)
        self._carousel.set_image_changed_callback(self._on_carousel_image_changed)
        self._carousel.set_source_changed_callback(self._on_carousel_source_changed)

        action_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        action_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        action_bar.grid_columnconfigure(0, weight=1)
        action_bar.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            action_bar,
            text="Re-analyze this photo",
            height=36,
            fg_color=("#2d8a4e", "#1f6b3a"),
            hover_color=("#256e3e", "#18562f"),
            text_color="white",
            command=self._on_reanalyze_click,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(
            action_bar,
            text="Delete this photo",
            height=36,
            fg_color=("#c0392b", "#922b21"),
            hover_color=("#a93226", "#7b241c"),
            text_color="white",
            command=self._on_delete_photo,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        split.set_left(scroll)
        split.set_right(right_panel)

        ctk.CTkLabel(
            scroll, text="Segmentation", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        row_method = ctk.CTkFrame(scroll, fg_color="transparent")
        row_method.grid(row=1, column=0, sticky="ew", pady=4)
        ctk.CTkLabel(row_method, text="Segmentation method:").pack(side="left")
        self._segmentation_method = ctk.CTkComboBox(
            row_method,
            values=[
                "A. BiRefNet + MobileSAM [RECOMMENDED]",
                "B. Otsu + LAB [FAST]",
                "C. Interactive segmentation",
            ],
            command=self._on_method_change,
            width=270,
        )
        self._segmentation_method.set("A. BiRefNet + MobileSAM [RECOMMENDED]")
        self._segmentation_method.pack(side="left", padx=8)
        InfoButton(
            row_method,
            title="Segmentation method",
            message=(
                "A. BiRefNet + MobileSAM [Default]: Highest edge precision for serrated/lobed leaves. "
                "Detects blue reference dot automatically for mm² calibration. Requires GPU + internet (first run downloads models ~200 MB).\n\n"
                "B. Otsu + LAB [FAST]: Classical computer-vision method. "
                "No GPU or model download required. Best for clean white-background photos of intact green leaves.\n\n"
                "C. Interactive segmentation: Click each leaf to preview selection (light-blue mask).\n"
                "  • Only leaf: one click per photo (leaf only; % damage, no mm²).\n"
                "  • Leaf + scale: first click = leaf, second click = blue scale sticker "
                "(MobileSAM). Repeat for every photo, then Run segmentation."
            ),
        ).pack(side="left", padx=4)

        self._interactive_frame = ctk.CTkFrame(scroll, fg_color=("gray92", "gray18"))
        self._interactive_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._interactive_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self._interactive_frame,
            text="Interactive click mode:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        mode_row = ctk.CTkFrame(self._interactive_frame, fg_color="transparent")
        mode_row.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        self._cb_only_leaf = ctk.CTkCheckBox(
            mode_row,
            text="Only leaf",
            command=lambda: self._on_interactive_mode_change("leaf_only"),
        )
        self._cb_only_leaf.pack(side="left", padx=(0, 12))
        self._cb_leaf_scale = ctk.CTkCheckBox(
            mode_row,
            text="Leaf + scale",
            command=lambda: self._on_interactive_mode_change("leaf_scale"),
        )
        self._cb_leaf_scale.pack(side="left")
        InfoButton(
            mode_row,
            title="Interactive click mode",
            message=(
                "Only leaf: click once on the leaf. No scale reference — results in % damage only.\n\n"
                "Leaf + scale: click the leaf first, then click the blue scale sticker in the same "
                "photo (MobileSAM fits the circle). Repeat both clicks for every photo. "
                "This also enables the Project 'Scale reference' setting for mm²/cm² analysis."
            ),
        ).pack(side="left", padx=8)
        self._interactive_frame.grid_remove()

        ctk.CTkLabel(
            scroll,
            text=f"",
            text_color="gray",
            anchor="w",
        ).grid(row=3, column=0, sticky="w", pady=(10, 4))

        adv_header = ctk.CTkFrame(scroll, fg_color="transparent")
        adv_header.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self._adv_toggle = ctk.CTkButton(
            adv_header,
            text="Advanced options  ▼",
            width=160,
            height=28,
            fg_color="transparent",
            border_width=1,
            anchor="w",
            command=self._toggle_advanced,
        )
        self._adv_toggle.pack(side="left")

        _hidden = ctk.CTkFrame(self, fg_color="transparent")
        self._build_fastsam_options(_hidden)

        self._birefnet_frame = ctk.CTkFrame(scroll, fg_color=("gray92", "gray18"))
        self._birefnet_frame.grid_columnconfigure(1, weight=1)
        self._build_birefnet_options(self._birefnet_frame)

        ctk.CTkLabel(
            scroll,
            text=(
                ""
            ),
            text_color="gray",
            wraplength=360,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=8)

        ctk.CTkButton(
            scroll,
            text="Run segmentation",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            command=self._on_run,
        ).grid(row=7, column=0, sticky="ew", pady=(16, 4))

        self._reanalyze_cb = None
        self._interactive_finalize_cb = None
        self._reanalyze_temp_dirs: list[Path] = []
        self._interactive_busy = False
        self._was_interactive = False
        self._scale_click_from_button = False

        self._adv_visible = False
        self._birefnet_frame.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        if self._state.segment_advanced_expanded:
            self._show_advanced()
        else:
            self._hide_advanced()

    # ------------------------------------------------------------------
    # Re-analyze single photo
    # ------------------------------------------------------------------

    def set_reanalyze_callback(self, cb) -> None:
        self._reanalyze_cb = cb

    def set_interactive_finalize_callback(self, cb) -> None:
        """cb() starts in-process BiRefNet finalize of all click selections."""
        self._interactive_finalize_cb = cb

    def refresh_current_image(self) -> None:
        self._carousel.refresh()
        self._interactive_busy = False
        self._apply_interactive_ui()

    def cleanup_reanalyze_temp(self) -> None:
        import shutil
        for d in self._reanalyze_temp_dirs:
            if d and d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        self._reanalyze_temp_dirs = []

    def _find_source_file(self, input_dir: "Path", stem: str) -> "Path | None":
        from gui.paths import VALID_EXT
        for p in input_dir.iterdir():
            if p.stem == stem and p.suffix.lower() in VALID_EXT:
                return p
        return None

    def _on_reanalyze_click(self) -> None:
        from pathlib import Path
        from gui.paths import canonical_leaf_id
        if not self._reanalyze_cb:
            return
        current = self._carousel.current_path
        if current is None:
            messagebox.showwarning("No image", "No image is currently displayed in the viewer.")
            return
        stem = current.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        stem = canonical_leaf_id(stem)  # strip _white_bg suffix regardless of method
        input_dir = Path(self._state.input_dir) if self._state.input_dir else None
        if not input_dir or not input_dir.is_dir():
            messagebox.showerror("Input folder not set", "Configure the input folder in tab 1 first.")
            return
        source_file = self._find_source_file(input_dir, stem)
        if source_file is None:
            messagebox.showerror(
                "Source not found",
                f"Could not find original image '{stem}' in:\n{input_dir}",
            )
            return
        steps = self.build_reanalyze_steps(source_file)
        if steps:
            self._reanalyze_cb(steps)

    def _on_delete_photo(self) -> None:
        from gui.paths import mask_path_for_white_bg, work_white_bg_copy
        current = self._carousel.current_path
        if current is None:
            messagebox.showwarning("No image", "No image is currently displayed.")
            return
        if not messagebox.askyesno(
            "Delete photo",
            f"Remove '{current.name}' from the segmentation output?\n\n"
            "The leaf image and its mask will be deleted so this image is skipped "
            "in subsequent steps. The original source photo is not affected.",
            icon="warning",
        ):
            return
        out_root = self._state.output_path()
        current.unlink(missing_ok=True)
        if out_root is not None:
            mask_path_for_white_bg(current, out_root).unlink(missing_ok=True)
            work_white_bg_copy(out_root, current.name).unlink(missing_ok=True)
        self._carousel.refresh()

    def build_reanalyze_steps(self, source_file: "Path") -> list:
        import shutil, tempfile
        from pathlib import Path
        from gui.paths import segmentation_dir, BIREFNET_DEFAULTS
        from gui.pipeline import script_path
        self.cleanup_reanalyze_temp()
        temp_dir = Path(tempfile.mkdtemp(prefix="herbivory_reanalyze_"))
        shutil.copy2(source_file, temp_dir / source_file.name)
        self._reanalyze_temp_dirs = [temp_dir]
        out_root = self._state.output_path()
        if out_root is None:
            self.cleanup_reanalyze_temp()
            return []
        seg_out = segmentation_dir(out_root)

        # Remove all outputs for this stem (both naming variants) so only the new
        # method's files remain — covers both plain and _white_bg suffixed names.
        from gui.paths import white_bg_dir, masks_dir, analyzed_dir, leaf_roi_preview_dir, unlink_analyzed_artifacts
        _stem = source_file.stem
        _variants = [_stem, f"{_stem}_white_bg"]
        for _f in [
            white_bg_dir(out_root) / f"{_stem}.png",
            white_bg_dir(out_root) / f"{_stem}_white_bg.png",
            masks_dir(out_root) / f"{_stem}_mask.png",
            seg_out / "metadata" / f"{_stem}.json",
        ]:
            _f.unlink(missing_ok=True)
        # Delete prior analysis images for both stem variants so the carousel only
        # shows the result from the newly selected method.
        _analyzed = analyzed_dir(out_root)
        for _v in _variants:
            unlink_analyzed_artifacts(_analyzed, _v)
        # Delete leaf_roi_preview files so stale contour masks don't carry over.
        _preview = leaf_roi_preview_dir(out_root)
        for _sub in ("masks", "overlays"):
            _subdir = _preview / _sub
            if _subdir.is_dir():
                for _v in _variants:
                    for _pf in _subdir.glob(f"{_v}*"):
                        _pf.unlink(missing_ok=True)

        method = self._state.validate_segmentation_method()
        if method == "intact":
            return [(
                f"Re-analyze: Intact Leaves ({source_file.name})",
                script_path("segmentation/segment_intact.py"),
                ["--input", str(temp_dir), "--output", str(seg_out),
                 "--sat-min", "20", "--close-k", "7", "--preview", "0"],
            )]
        if method in ("birefnet_mobilesam", "interactive_mobilesam"):
            from gui.pipeline import build_birefnet_args
            args = build_birefnet_args(self._state, input_override=temp_dir)
            label = (
                "Interactive segmentation"
                if method == "interactive_mobilesam"
                else "BiRefNet + MobileSAM"
            )
            return [(
                f"Re-analyze: {label} ({source_file.name})",
                script_path("segmentation/birefnet_mobilesam/run_pipeline.py"),
                args,
            )]
        messagebox.showwarning(
            "Method not supported",
            "Single-photo re-analyze is only available for BiRefNet + MobileSAM, "
            "Interactive segmentation, and Intact Leaves.\n\n"
            "For FastSAM, use 'Run segmentation' on all images.",
        )
        self.cleanup_reanalyze_temp()
        return []

    def _on_carousel_image_changed(self, path: Path | None) -> None:
        """Restore MobileSAM preview + scale circle only on Input photos."""
        if not self._is_interactive_method():
            self._carousel.clear_preview_mask()
            self._carousel.clear_scale_circle()
            return
        # Results / masks view: never show the light-blue SAM overlay
        if self._carousel.current_source_key != "Input photos":
            self._carousel.clear_preview_mask()
            self._carousel.clear_scale_circle()
            return
        from gui.interactive_sam_session import get_session

        session = get_session()
        sel = session.selection_for_path(path)
        if sel is not None:
            self._carousel.set_preview_mask(sel.mask)
        else:
            self._carousel.clear_preview_mask()
        self._apply_scale_circle_overlay(path)
        self._update_interactive_banner()

    def _apply_scale_circle_overlay(self, path: Path | None) -> None:
        if not self._state.remove_blue:
            self._carousel.clear_scale_circle()
            return
        from gui.interactive_sam_session import get_session

        circle = get_session().circle_for_path(path)
        if circle and circle.get("found"):
            cx, cy = circle["center_px"]
            self._carousel.set_scale_circle(float(cx), float(cy), float(circle["diameter_px"]))
        else:
            self._carousel.clear_scale_circle()

    def _interactive_click_mode(self) -> str:
        mode = getattr(self._state, "interactive_click_mode", "leaf_scale")
        return mode if mode in ("leaf_only", "leaf_scale") else "leaf_scale"

    def _is_leaf_scale_mode(self) -> bool:
        return self._interactive_click_mode() == "leaf_scale"

    def _interactive_stage(self, path: Path | None) -> str:
        """Return 'leaf' | 'scale' | 'done' for the current interactive photo."""
        if not self._is_leaf_scale_mode():
            from gui.interactive_sam_session import get_session

            sel = get_session().selection_for_path(path)
            if sel is not None and sel.mask is not None and int(sel.mask.sum()) > 0:
                return "done"
            return "leaf"

        from gui.interactive_sam_session import get_session

        sel = get_session().selection_for_path(path)
        if sel is None or sel.mask is None or int(sel.mask.sum()) == 0:
            return "leaf"
        if not sel.has_circle:
            return "scale"
        return "done"

    def _set_interactive_checkboxes(self, mode: str) -> None:
        if mode == "leaf_only":
            self._cb_only_leaf.select()
            self._cb_leaf_scale.deselect()
        else:
            self._cb_leaf_scale.select()
            self._cb_only_leaf.deselect()

    def _on_interactive_mode_change(self, mode: str) -> None:
        if mode not in ("leaf_only", "leaf_scale"):
            mode = "leaf_scale"
        # Mutually exclusive: never leave both unchecked
        self._set_interactive_checkboxes(mode)
        self._state.interactive_click_mode = mode
        use_scale = mode == "leaf_scale"
        self._state.remove_blue = use_scale
        if not use_scale:
            self._state.report_area_cm2 = False
            self._carousel.clear_scale_circle()
        self._update_interactive_banner()
        self._on_change()

    def _update_interactive_mode_panel(self) -> None:
        if self._is_interactive_method():
            self._set_interactive_checkboxes(self._interactive_click_mode())
            self._interactive_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        else:
            self._interactive_frame.grid_remove()

    def _ensure_interactive_models_async(self) -> None:
        """Load MobileSAM once in a background thread when entering method C."""
        import threading
        from gui.interactive_sam_session import get_session

        session = get_session()
        if session.is_loading_sam:
            return

        weights = self._state.mobilesam_model or None

        def worker() -> None:
            try:
                session.ensure_mobilesam(log=None, weights=weights)
                self.after(0, self._on_mobilesam_loaded)
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_mobilesam_load_failed(err))

        # Fast path: already loaded with the requested weights
        try:
            if session.mobilesam_ready:
                session.ensure_mobilesam(log=None, weights=weights)
                self._on_mobilesam_loaded()
                return
        except Exception:
            pass

        self._warn_label.configure(text="Loading MobileSAM (once)… please wait.")
        self._mark_scale_btn.grid_remove()
        self._warn_btn.configure(text="…", state="disabled")
        self._warn_banner.grid()
        threading.Thread(target=worker, daemon=True).start()

    def _on_mobilesam_loaded(self) -> None:
        self._update_interactive_banner()
        self._carousel.enable_point_click_mode(True)

    def _on_mobilesam_load_failed(self, err: str) -> None:
        messagebox.showerror("MobileSAM load failed", err)
        self._warn_label.configure(text=f"MobileSAM failed to load: {err}")
        self._mark_scale_btn.grid_remove()
        self._warn_btn.configure(text="Retry", state="normal", command=self._ensure_interactive_models_async)
        self._warn_banner.grid()

    def _update_interactive_banner(self) -> None:
        from gui.interactive_sam_session import get_session

        session = get_session()
        n = session.n_selected
        ready = session.mobilesam_ready
        use_scale = self._is_leaf_scale_mode()
        if not ready:
            self._warn_label.configure(text="Loading MobileSAM… please wait.")
            self._mark_scale_btn.grid_remove()
            self._warn_btn.configure(text="…", state="disabled")
        else:
            path = self._carousel.current_path
            stage = self._interactive_stage(path)
            if not use_scale:
                self._warn_label.configure(
                    text=(
                        f"Only leaf — click the leaf.\n"
                        f"Selected: {n}  ·  Next → next photo  ·  then Run"
                    )
                )
                self._mark_scale_btn.grid_remove()
            elif stage == "leaf":
                self._warn_label.configure(
                    text=f"Leaf + scale — Step 1/2: click the LEAF.\nSelected: {n}"
                )
                self._mark_scale_btn.grid_remove()
            elif stage == "scale":
                self._warn_label.configure(
                    text="Leaf + scale — Step 2/2: click the SCALE sticker."
                )
                self._mark_scale_btn.grid_remove()
            else:
                circle = session.circle_for_path(path)
                if circle and circle.get("found"):
                    conf = "low-conf" if circle.get("low_confidence") else "ok"
                    scale_note = f"d={circle['diameter_px']:.0f}px ({conf})"
                else:
                    scale_note = "scale ok"
                self._warn_label.configure(
                    text=(
                        f"Done — leaf + scale ({scale_note}).\n"
                        f"Selected: {n}  ·  Next → repeat  ·  then Run"
                    )
                )
                self._mark_scale_btn.configure(
                    text="Mark scale",
                    command=self._on_mark_scale_circle,
                    state="normal",
                )
                self._mark_scale_btn.grid()
            self._warn_btn.configure(
                text="Clear selections",
                state="normal",
                command=self._clear_interactive_selections,
            )
        self._warn_banner.grid()

    def _clear_interactive_selections(self) -> None:
        from gui.interactive_sam_session import get_session

        get_session().clear_selections()
        self._carousel.clear_preview_mask()
        self._carousel.clear_scale_circle()
        self._update_interactive_banner()

    def _resolve_interactive_source_file(self) -> Path | None:
        current = self._carousel.current_path
        if current is None:
            return None
        input_dir = Path(self._state.input_dir) if self._state.input_dir else None
        if not input_dir or not input_dir.is_dir():
            return None
        if current.parent.resolve() == input_dir.resolve() and current.is_file():
            return current
        from gui.paths import canonical_leaf_id

        stem = current.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        stem = canonical_leaf_id(stem)
        return self._find_source_file(input_dir, stem)

    def _on_mark_scale_circle(self) -> None:
        """Interactive: one click on the blue sticker → MobileSAM fits the circle."""
        if not self._is_leaf_scale_mode():
            messagebox.showinfo(
                "Scale reference off",
                "Select \"Leaf + scale\" in the Interactive click mode to mark the blue sticker.",
            )
            return
        if self._interactive_busy:
            return
        source_file = self._resolve_interactive_source_file()
        if source_file is None:
            messagebox.showwarning(
                "No image",
                "Open an Input photo and configure the input folder first.",
            )
            return
        from gui.interactive_sam_session import get_session

        if not get_session().mobilesam_ready:
            self._ensure_interactive_models_async()
            messagebox.showinfo(
                "Loading model",
                "MobileSAM is still loading. Wait a moment and try Mark scale again.",
            )
            return

        self._pick_dot_stem = source_file.stem
        self._pick_dot_interactive = True  # Mark-scale mode (SAM click)
        self._scale_click_from_button = True
        self._carousel.enable_pick_dot_mode(False)
        self._carousel.enable_point_click_mode(True)
        self._mark_scale_btn.grid_remove()
        self._warn_label.configure(
            text=f"Mark scale: click the blue sticker in '{source_file.name}'."
        )
        self._warn_btn.configure(
            text="Cancel",
            state="normal",
            command=self._on_cancel_mark_scale,
        )
        self._warn_banner.grid()

    def _on_cancel_mark_scale(self) -> None:
        self._pick_dot_stem = None
        self._pick_dot_interactive = False
        self._scale_click_from_button = False
        self._mark_scale_btn.configure(text="Mark scale", command=self._on_mark_scale_circle)
        self._carousel.enable_pick_dot_mode(False)
        self._apply_interactive_ui()

    def _on_interactive_point_click(self, x: float, y: float) -> None:
        if self._interactive_busy:
            return
        if not self._is_interactive_method():
            return

        # Mark-scale mode (button): route click to MobileSAM circle fit
        if self._pick_dot_interactive and self._pick_dot_stem is not None:
            self._on_mark_scale_sam_click(x, y)
            return

        source_file = self._resolve_interactive_source_file()
        if source_file is None:
            input_dir = Path(self._state.input_dir) if self._state.input_dir else None
            if not input_dir or not input_dir.is_dir():
                messagebox.showerror(
                    "Input folder not set", "Configure the input folder in tab 1 first."
                )
            else:
                messagebox.showerror(
                    "Source not found",
                    f"Could not find original image for click in:\n{input_dir}",
                )
            return

        # Leaf + scale step 2: scale click (no auto-detect)
        if self._is_leaf_scale_mode():
            stage = self._interactive_stage(source_file)
            if stage == "scale":
                self._scale_click_from_button = False
                self._pick_dot_stem = source_file.stem
                self._on_mark_scale_sam_click(x, y)
                return

        from gui.interactive_sam_session import get_session

        session = get_session()
        if not session.mobilesam_ready:
            self._ensure_interactive_models_async()
            messagebox.showinfo(
                "Loading model",
                "MobileSAM is still loading. Wait a moment and click again.",
            )
            return

        self._interactive_busy = True
        self._carousel.enable_point_click_mode(False)
        self._warn_label.configure(text=f"Selecting leaf in {source_file.name}…")
        known_mm = float(self._state.birefnet_known_diameter_mm)

        import threading

        def worker() -> None:
            try:
                sel = session.predict_click(
                    source_file,
                    x,
                    y,
                    known_diameter_mm=known_mm,
                    detect_circle=False,
                )
                self.after(0, lambda s=sel: self._on_interactive_preview_done(s, None))
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_interactive_preview_done(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_mark_scale_sam_click(self, x: float, y: float) -> None:
        source_file = self._resolve_interactive_source_file()
        if source_file is None:
            self._on_cancel_mark_scale()
            return

        from gui.interactive_sam_session import get_session

        session = get_session()
        if self._pick_dot_stem is None:
            self._pick_dot_stem = source_file.stem
        self._interactive_busy = True
        self._carousel.enable_point_click_mode(False)
        self._warn_label.configure(text=f"MobileSAM fitting scale circle in {source_file.name}…")
        known_mm = float(self._state.birefnet_known_diameter_mm)

        import threading

        def worker() -> None:
            try:
                circle = session.predict_scale_click(
                    source_file, x, y, known_diameter_mm=known_mm
                )
                self.after(0, lambda c=circle: self._on_mark_scale_sam_done(c, None))
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_mark_scale_sam_done(None, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_mark_scale_sam_done(self, circle: dict | None, err: str | None) -> None:
        from gui.pipeline import save_circle_override
        from gui.interactive_sam_session import get_session

        self._interactive_busy = False
        stem = self._pick_dot_stem
        from_button = self._scale_click_from_button
        out_root = self._state.output_path()

        if err:
            messagebox.showerror("Scale MobileSAM error", err)
            self._carousel.enable_point_click_mode(True)
            if from_button:
                self._warn_label.configure(
                    text="Scale click failed — click the sticker again, or Cancel."
                )
                self._warn_btn.configure(text="Cancel", command=self._on_cancel_mark_scale)
            else:
                self._pick_dot_stem = None
                self._pick_dot_interactive = False
                self._scale_click_from_button = False
                self._update_interactive_banner()
            return

        if not circle or not circle.get("found"):
            reason = (circle or {}).get("reason", "unknown")
            self._carousel.enable_point_click_mode(True)
            if from_button:
                self._warn_label.configure(
                    text=(
                        f"MobileSAM did not find a small circular sticker ({reason}). "
                        f"Click again on the sticker, or Cancel."
                    )
                )
                self._warn_btn.configure(text="Cancel", command=self._on_cancel_mark_scale)
            else:
                self._pick_dot_stem = None
                self._pick_dot_interactive = False
                self._scale_click_from_button = False
                self._warn_label.configure(
                    text=(
                        f"Scale click failed ({reason}). "
                        f"Click the blue sticker again (Step 2 of 2)."
                    )
                )
                self._warn_btn.configure(
                    text="Clear selections",
                    state="normal",
                    command=self._clear_interactive_selections,
                )
                self._warn_banner.grid()
            return

        # Success
        self._pick_dot_stem = None
        self._pick_dot_interactive = False
        self._scale_click_from_button = False
        self._mark_scale_btn.configure(text="Mark scale", command=self._on_mark_scale_circle)

        cx, cy = circle["center_px"]
        d = float(circle["diameter_px"])
        if stem is not None and out_root is not None:
            try:
                save_circle_override(out_root, stem, float(cx), float(cy), d)
            except Exception:
                pass

        sel = get_session().selection_for_path(self._carousel.current_path)
        if sel is not None:
            self._carousel.set_preview_mask(sel.mask)
        self._carousel.set_scale_circle(float(cx), float(cy), d)
        self._apply_interactive_ui()
        if from_button:
            conf = " (low confidence)" if circle.get("low_confidence") else ""
            messagebox.showinfo(
                "Scale circle marked",
                f"MobileSAM scale circle for '{stem}'.\nDiameter: {d:.1f}px{conf}",
            )

    def _on_interactive_preview_done(self, sel, err: str | None) -> None:
        self._interactive_busy = False
        self._carousel.enable_point_click_mode(True)
        if err:
            messagebox.showerror("MobileSAM error", err)
            self._update_interactive_banner()
            return
        if sel is not None:
            self._carousel.set_preview_mask(sel.mask)
            if self._is_leaf_scale_mode() and sel.has_circle:
                self._carousel.set_scale_circle(
                    sel.circle_cx, sel.circle_cy, sel.circle_diameter
                )
            else:
                self._carousel.clear_scale_circle()
        self._update_interactive_banner()

    def collect_interactive_finalize_kwargs(self) -> dict:
        """State snapshot for the background BiRefNet finalize job."""
        self.sync_to_state()
        out = self._state.output_path()
        from gui.paths import segmentation_dir

        return {
            "output_dir": segmentation_dir(out) if out else None,
            "project_root": out,
            "known_diameter_mm": float(self._state.birefnet_known_diameter_mm),
            "hybrid_mode": str(self._state.birefnet_hybrid_mode),
            "seg_resolution": int(self._state.birefnet_seg_resolution),
            "output_size": int(self._state.segmentation_output_size()),
            "agreement_threshold": float(self._state.birefnet_agreement_threshold),
            "remove_blue": bool(self._state.remove_blue),
        }

    def collect_birefnet_batch_kwargs(self, input_override: Path | None = None) -> dict:
        """State snapshot for in-process BiRefNet + MobileSAM (method A)."""
        self.sync_to_state()
        out = self._state.output_path()
        from gui.paths import segmentation_dir

        inp = input_override if input_override is not None else self._state.input_path()
        return {
            "input_dir": inp,
            "output_dir": segmentation_dir(out) if out else None,
            "known_diameter_mm": float(self._state.birefnet_known_diameter_mm),
            "hybrid_mode": str(self._state.birefnet_hybrid_mode),
            "seg_resolution": int(self._state.birefnet_seg_resolution),
            "output_size": int(self._state.segmentation_output_size()),
            "agreement_threshold": float(self._state.birefnet_agreement_threshold),
            "remove_blue": bool(self._state.remove_blue),
            "mobilesam_weights": self._state.mobilesam_model or None,
        }

    def preload_birefnet_models_async(self) -> None:
        """Warm-load BiRefNet + MobileSAM in the background (method A)."""
        import threading
        from gui.interactive_sam_session import get_session

        session = get_session()
        if session.birefnet_ready and session.mobilesam_ready:
            return
        if session.is_loading_sam or session.is_loading_birefnet:
            return

        weights = self._state.mobilesam_model or None

        def worker() -> None:
            try:
                session.ensure_mobilesam(log=None, weights=weights)
                session.ensure_birefnet(log=None)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------

    def _build_birefnet_options(self, parent: ctk.CTkFrame) -> None:
        pad = {"padx": 10, "pady": 3, "sticky": "w"}
        row = 0

        ctk.CTkLabel(
            parent,
            text="BiRefNet + MobileSAM tuning",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        ctk.CTkLabel(parent, text="Merge mode:").grid(
            row=row, column=0, padx=(10, 4), pady=3, sticky="w"
        )
        self._birefnet_mode = ctk.CTkComboBox(
            parent,
            values=["birefnet_primary", "mobilesam_primary", "intersection", "union"],
            width=160,
        )
        self._birefnet_mode.grid(row=row, column=1, padx=4, pady=3, sticky="w")
        InfoButton(
            parent,
            title="Merge mode",
            message=(
                "BiRefNet provides precise leaf edges; MobileSAM selects which object is the leaf "
                "(e.g. excludes the blue reference dot).\n\n"
                "birefnet_primary (default, recommended): BiRefNet edges restricted to the SAM leaf.\n"
                "mobilesam_primary: SAM defines the leaf; BiRefNet refines the boundary band.\n"
                "intersection: keep overlap only (AND) — strictest; may shrink edges.\n"
                "union: keep either mask (OR) — most inclusive; may add background."
            ),
        ).grid(row=row, column=2, padx=4, pady=3, sticky="w")
        row += 1

        ctk.CTkLabel(
            parent,
            text=(
                "birefnet_primary (default): precise edges within SAM leaf — recommended.\n"
                "mobilesam_primary: SAM body + BiRefNet edge refine — if BiRefNet picks extras.\n"
                "intersection: overlap only (AND) — strictest; may clip borders.\n"
                "union: either mask (OR) — largest; may include artifacts."
            ),
            text_color="gray",
            wraplength=400,
            justify="left",
            anchor="w",
        ).grid(row=row, column=0, columnspan=3, padx=10, pady=(0, 6), sticky="w")
        row += 1

        self._birefnet_agreement = self._adv_entry(
            parent, row, "Agreement threshold:", "flag if IoU < threshold"
        )
        row += 1

        reset_row = ctk.CTkFrame(parent, fg_color="transparent")
        reset_row.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 10))
        ctk.CTkButton(
            reset_row,
            text="Reset BiRefNet defaults",
            width=160,
            command=self._reset_birefnet_defaults,
        ).pack(side="left")

    def _build_fastsam_options(self, parent: ctk.CTkFrame) -> None:
        pad = {"padx": 10, "pady": 3, "sticky": "w"}
        row = 0

        ctk.CTkLabel(
            parent,
            text="FastSAM tuning",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=row, column=0, columnspan=3, **pad)
        row += 1

        self._max_leaves = self._adv_entry(parent, row, "Max leaves per photo:", "(empty = no limit)")
        row += 1

        self._stressed = ctk.CTkCheckBox(parent, text="Stressed / yellow-green leaves")
        self._stressed.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        self._strict_crop = ctk.CTkCheckBox(
            parent, text="Strict crop (foliage-only bbox; excludes blue dot at edges)"
        )
        self._strict_crop.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        self._reject_dark = ctk.CTkCheckBox(parent, text="Reject dark debris / artifacts")
        self._reject_dark.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        ctk.CTkLabel(parent, text="Filter thresholds", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, columnspan=3, **pad
        )
        row += 1

        self._min_overlap = self._adv_entry(parent, row, "Min green overlap:", "0.16 recommended")
        row += 1
        self._dark_ratio = self._adv_entry(parent, row, "Dark artifact ratio:", "lower = stricter")
        row += 1
        self._dark_value = self._adv_entry(parent, row, "Dark value threshold:", "HSV V, default 62")
        row += 1
        self._crop_margin = self._adv_entry(parent, row, "Crop margin (px):", "")
        row += 1
        self._max_area = self._adv_entry(parent, row, "Max mask area ratio:", "reject Petri dish")
        row += 1

        ctk.CTkLabel(parent, text="Model inference", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, columnspan=3, **pad
        )
        row += 1

        self._conf = self._adv_entry(parent, row, "Confidence:", "")
        row += 1
        self._prior_dilate = self._adv_entry(parent, row, "Green prior dilate (px):", "damaged leaf edges")
        row += 1

        reset_row = ctk.CTkFrame(parent, fg_color="transparent")
        reset_row.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 10))
        ctk.CTkButton(
            reset_row,
            text="Reset Petri defaults",
            width=160,
            command=self._reset_petri_defaults,
        ).pack(side="left")

    def _adv_entry(
        self, parent: ctk.CTkFrame, row: int, label: str, hint: str
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, padx=(10, 4), pady=3, sticky="w")
        entry = ctk.CTkEntry(parent, width=70)
        entry.grid(row=row, column=1, padx=4, pady=3, sticky="w")
        if hint:
            ctk.CTkLabel(parent, text=hint, text_color="gray").grid(
                row=row, column=2, padx=4, pady=3, sticky="w"
            )
        return entry

    def _toggle_advanced(self) -> None:
        if self._adv_visible:
            self._hide_advanced()
        else:
            self._show_advanced()
        self._state.segment_advanced_expanded = self._adv_visible
        self._on_change()

    def _show_advanced(self) -> None:
        self._adv_visible = True
        self._adv_toggle.configure(text="Advanced options  ▲")
        self._update_advanced_panels()

    def _hide_advanced(self) -> None:
        self._adv_visible = False
        self._adv_toggle.configure(text="Advanced options  ▼")
        self._update_advanced_panels()

    def _reset_petri_defaults(self) -> None:
        for key, value in FASTSAM_PETRI_DEFAULTS.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._apply_method_label(FASTSAM_PETRI_DEFAULTS["segmentation_method"])
        self.load_from_state()
        self._on_change()

    def _apply_method_label(self, method: str) -> None:
        self._segmentation_method.set(
            SEGMENT_METHOD_LABELS.get(str(method), SEGMENT_METHOD_LABELS["birefnet_mobilesam"])
        )

    def _is_birefnet_method(self) -> bool:
        label = self._segmentation_method.get().lower()
        return "birefnet" in label or "interactive" in label

    def _is_intact_method(self) -> bool:
        return "otsu" in self._segmentation_method.get().lower()

    def _is_interactive_method(self) -> bool:
        # Combobox only — avoids keeping the Interactive panel visible after
        # switching away while state still says interactive_mobilesam.
        return "interactive" in self._segmentation_method.get().lower()

    def _reset_birefnet_defaults(self) -> None:
        for key, value in BIREFNET_DEFAULTS.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._state.segmentation_method = "birefnet_mobilesam"
        self._apply_method_label("birefnet_mobilesam")
        self.load_from_state()
        self._on_change()

    def _update_advanced_panels(self) -> None:
        self._update_interactive_mode_panel()
        if self._is_intact_method():
            self._adv_toggle.configure(state="disabled")
            self._birefnet_frame.grid_remove()
            return
        self._adv_toggle.configure(state="normal")
        if not self._adv_visible:
            self._birefnet_frame.grid_remove()
            return
        if self._is_birefnet_method():
            self._birefnet_frame.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        else:
            self._birefnet_frame.grid_remove()

    def set_run_callback(self, cb) -> None:
        self._run_cb = cb

    def _on_method_change(self, _val=None) -> None:
        # Sync method first so UI helpers see the new selection
        self.sync_to_state()
        self._update_advanced_panels()
        if self._is_interactive_method():
            mode = self._interactive_click_mode()
            self._state.interactive_click_mode = mode
            self._state.remove_blue = mode == "leaf_scale"
            if not self._state.remove_blue:
                self._state.report_area_cm2 = False
            self._set_interactive_checkboxes(mode)
            self._on_change()
        self.refresh_preview()
        method = self._state.validate_segmentation_method()
        if method == "birefnet_mobilesam":
            self.preload_birefnet_models_async()
        elif method == "intact":
            from gui.interactive_sam_session import reset_session

            reset_session(release_models=True)

    def _apply_interactive_ui(self) -> None:
        """Show input photos + click banner when Interactive method is selected."""
        if self._pick_dot_stem is not None:
            return
        self._update_interactive_mode_panel()
        interactive = self._is_interactive_method()
        if interactive:
            if not self._was_interactive:
                self._ensure_interactive_models_async()
            self._was_interactive = True
            self._update_interactive_banner()
            from gui.interactive_sam_session import get_session

            self._carousel.enable_point_click_mode(
                get_session().mobilesam_ready and not self._interactive_busy
            )
            # Restore overlay for current image if already selected
            self._on_carousel_image_changed(self._carousel.current_path)
        else:
            if self._was_interactive:
                from gui.interactive_sam_session import get_session, reset_session

                # Method A reuses the same in-GUI models — keep them warm.
                if self._state.validate_segmentation_method() == "birefnet_mobilesam":
                    get_session().clear_selections()
                else:
                    reset_session(release_models=True)
                self._was_interactive = False
            self._carousel.enable_point_click_mode(False)
            self._carousel.clear_preview_mask()
            self._carousel.clear_scale_circle()
            self._mark_scale_btn.grid_remove()
            if not self._low_confidence_stems:
                self._warn_banner.grid_remove()

    def _hide_interactive_banner(self) -> None:
        if self._is_interactive_method():
            self._warn_banner.grid_remove()
        else:
            self._warn_banner.grid_remove()

    def _set_entry(self, entry: ctk.CTkEntry, value: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, value)

    def load_from_state(self) -> None:
        self._apply_method_label(self._state.validate_segmentation_method())

        self._set_entry(
            self._max_leaves,
            "" if self._state.fastsam_max_leaves is None else str(self._state.fastsam_max_leaves),
        )
        self._set_entry(self._min_overlap, str(self._state.fastsam_min_overlap))
        self._set_entry(self._dark_ratio, str(self._state.fastsam_dark_ratio_threshold))
        self._set_entry(self._dark_value, str(self._state.fastsam_dark_value_threshold))
        self._set_entry(self._crop_margin, str(self._state.fastsam_component_margin_px))
        self._set_entry(self._max_area, str(self._state.fastsam_max_area_ratio))
        self._set_entry(self._conf, str(self._state.fastsam_conf))
        self._set_entry(self._prior_dilate, str(self._state.fastsam_prior_dilate_px))

        for cb, val in (
            (self._stressed, self._state.stressed_leaves),
            (self._strict_crop, self._state.fastsam_strict_crop),
            (self._reject_dark, self._state.reject_dark_artifacts),
        ):
            if val:
                cb.select()
            else:
                cb.deselect()

        self._birefnet_mode.set(self._state.birefnet_hybrid_mode)
        self._set_entry(self._birefnet_agreement,
                        str(self._state.birefnet_agreement_threshold))
        self._state.apply_fixed_pipeline_resolution()

        mode = getattr(self._state, "interactive_click_mode", "leaf_scale")
        if mode not in ("leaf_only", "leaf_scale"):
            mode = "leaf_scale"
        self._state.interactive_click_mode = mode
        self._set_interactive_checkboxes(mode)

        if self._state.segment_advanced_expanded:
            self._show_advanced()
        else:
            self._hide_advanced()

        self._on_method_change()
        self.refresh_preview()

    def refresh_preview(self, *, show_results: bool = False) -> None:
        interactive = self._is_interactive_method()
        if show_results:
            default_key = "Leaves (white_bg)"
        elif interactive:
            default_key = "Input photos"
        else:
            default_key = None
        self._carousel.set_sources(
            segment_sources(self._state, include_input=interactive),
            default_key=default_key,
        )
        self._update_low_confidence_banner()
        if show_results:
            self._show_post_run_results_ui()
        else:
            self._apply_interactive_ui()

    def _show_post_run_results_ui(self) -> None:
        """After Run: show Leaves (white_bg), drop SAM overlays, pause click mode."""
        self._carousel.clear_preview_mask()
        self._carousel.clear_scale_circle()
        self._carousel.enable_point_click_mode(False)
        if not self._is_interactive_method():
            return
        from gui.interactive_sam_session import get_session

        n = get_session().n_selected
        mode_note = (
            "Leaf + scale"
            if self._is_leaf_scale_mode()
            else "Only leaf"
        )
        self._warn_label.configure(
            text=(
                f"Done — viewing Leaves (white_bg).\n"
                f"{mode_note}  ·  selections kept: {n}  ·  Back to Input to add more"
            )
        )
        if self._is_leaf_scale_mode():
            self._mark_scale_btn.configure(
                text="Mark scale",
                command=self._on_mark_scale_circle,
                state="disabled",
            )
            self._mark_scale_btn.grid()
        else:
            self._mark_scale_btn.grid_remove()
        self._warn_btn.configure(
            text="Back to Input",
            state="normal",
            command=self._back_to_input_photos,
        )
        self._warn_banner.grid()

    def _back_to_input_photos(self) -> None:
        if not self._carousel.select_source("Input photos"):
            self.refresh_preview(show_results=False)

    def _on_carousel_source_changed(self, key: str) -> None:
        if not self._is_interactive_method():
            return
        if self._pick_dot_stem is not None:
            return
        if key == "Input photos":
            self._apply_interactive_ui()
        else:
            self._carousel.clear_preview_mask()
            self._carousel.clear_scale_circle()
            self._carousel.enable_point_click_mode(False)
            self._show_post_run_results_ui()

    def _update_low_confidence_banner(self) -> None:
        import json
        from gui.paths import segmentation_dir
        from gui.pipeline import has_circle_override

        self._low_confidence_stems = []
        # Scale / blue-dot warnings only apply when the Project checkbox is on.
        if not self._state.remove_blue:
            if self._pick_dot_stem is not None:
                return
            if self._is_interactive_method():
                return
            self._warn_banner.grid_remove()
            return

        out_root = self._state.output_path()
        if out_root is not None:
            meta_dir = segmentation_dir(out_root) / "metadata"
            if meta_dir.is_dir():
                for mf in sorted(meta_dir.glob("*.json")):
                    try:
                        meta = json.loads(mf.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    stem = meta.get("image_id", mf.stem)
                    if has_circle_override(out_root, stem):
                        continue
                    circle = meta.get("circle")
                    if circle is None:
                        # Backend does not look for the dot itself (whitebg/intact);
                        # scale comes from the separate scan, so nothing to flag.
                        continue
                    if not circle.get("found") or circle.get("low_confidence"):
                        self._low_confidence_stems.append(stem)

        if self._pick_dot_stem is not None:
            # Leave Mark-scale / Adjust-blue-dot messaging alone.
            return
        if self._is_interactive_method():
            # Interactive banner takes precedence.
            return

        n = len(self._low_confidence_stems)
        if n > 0:
            self._warn_label.configure(
                text=f"⚠ Blue dot detection uncertain in {n} image(s) — scale may be wrong."
            )
            self._warn_btn.configure(text="Adjust blue dot", command=self._on_adjust_blue_dot)
            self._warn_banner.grid()
        else:
            self._warn_banner.grid_remove()

    def _on_adjust_blue_dot(self) -> None:
        if not self._state.remove_blue:
            return
        if not self._low_confidence_stems:
            return
        stem = self._low_confidence_stems[0]
        out_root = self._state.output_path()
        input_dir = Path(self._state.input_dir) if self._state.input_dir else None
        if out_root is None or not input_dir or not input_dir.is_dir():
            messagebox.showerror("Input/output not set", "Configure the input and output folders first.")
            return
        source_file = self._find_source_file(input_dir, stem)
        if source_file is None:
            messagebox.showerror(
                "Source not found", f"Could not find original image '{stem}' in:\n{input_dir}"
            )
            return
        self._pick_dot_stem = stem
        self._pick_dot_interactive = False
        self._carousel.set_paths([source_file], empty_message="")
        self._carousel.enable_pick_dot_mode(True)
        self._mark_scale_btn.grid_remove()
        self._warn_label.configure(
            text=f"Click-drag on the blue/dark reference dot in '{stem}' (center to edge), then release."
        )
        self._warn_btn.configure(text="Cancel", command=self._on_cancel_pick_dot)
        self._warn_banner.grid()

    def _on_cancel_pick_dot(self) -> None:
        self._carousel.enable_pick_dot_mode(False)
        self._pick_dot_stem = None
        self._pick_dot_interactive = False
        self._scale_click_from_button = False
        self._mark_scale_btn.configure(text="Mark scale", command=self._on_mark_scale_circle)
        self.refresh_preview()
        self._apply_interactive_ui()

    def _on_pick_dot_picked(self, cx: float, cy: float, diameter_px: float) -> None:
        from gui.pipeline import save_circle_override

        stem = self._pick_dot_stem
        out_root = self._state.output_path()
        self._carousel.enable_pick_dot_mode(False)
        self._pick_dot_stem = None
        self._pick_dot_interactive = False
        self._scale_click_from_button = False
        self._mark_scale_btn.configure(text="Mark scale", command=self._on_mark_scale_circle)
        if stem is None or out_root is None:
            return

        save_circle_override(out_root, stem, cx, cy, diameter_px)
        self.refresh_preview()
        messagebox.showinfo(
            "Blue dot updated",
            f"Manual blue dot saved for '{stem}'.\nDiameter: {diameter_px:.1f}px",
        )

    def sync_to_state(self) -> None:
        sel_method = self._segmentation_method.get().lower()
        if "interactive" in sel_method:
            self._state.segmentation_method = "interactive_mobilesam"
        elif "birefnet" in sel_method:
            self._state.segmentation_method = "birefnet_mobilesam"
        elif "otsu" in sel_method:
            self._state.segmentation_method = "intact"
        else:
            self._state.segmentation_method = "birefnet_mobilesam"

        if self._state.segmentation_method == "interactive_mobilesam":
            if bool(self._cb_only_leaf.get()) and not bool(self._cb_leaf_scale.get()):
                mode = "leaf_only"
            elif bool(self._cb_leaf_scale.get()):
                mode = "leaf_scale"
            else:
                mode = self._interactive_click_mode()
            self._state.interactive_click_mode = mode
            self._state.remove_blue = mode == "leaf_scale"
            if not self._state.remove_blue:
                self._state.report_area_cm2 = False

        ml = self._max_leaves.get().strip()
        self._state.fastsam_max_leaves = int(ml) if ml else None

        defaults = FASTSAM_PETRI_DEFAULTS
        self._state.stressed_leaves = bool(self._stressed.get())
        self._state.fastsam_strict_crop = bool(self._strict_crop.get())
        self._state.reject_dark_artifacts = bool(self._reject_dark.get())
        self._state.normalize_bg = True
        self._state.fastsam_min_overlap = self._parse_float(
            self._min_overlap, float(defaults["fastsam_min_overlap"])
        )
        self._state.fastsam_dark_ratio_threshold = self._parse_float(
            self._dark_ratio, float(defaults["fastsam_dark_ratio_threshold"])
        )
        self._state.fastsam_dark_value_threshold = self._parse_int(
            self._dark_value, int(defaults["fastsam_dark_value_threshold"])
        )
        self._state.fastsam_component_margin_px = self._parse_int(
            self._crop_margin, int(defaults["fastsam_component_margin_px"])
        )
        self._state.fastsam_max_area_ratio = self._parse_float(
            self._max_area, float(defaults["fastsam_max_area_ratio"])
        )
        self._state.fastsam_conf = self._parse_float(self._conf, float(defaults["fastsam_conf"]))
        self._state.fastsam_prior_dilate_px = self._parse_int(
            self._prior_dilate, int(defaults["fastsam_prior_dilate_px"])
        )

        bdefaults = BIREFNET_DEFAULTS
        self._state.birefnet_hybrid_mode = self._birefnet_mode.get()
        self._state.birefnet_agreement_threshold = self._parse_float(
            self._birefnet_agreement,
            float(bdefaults["birefnet_agreement_threshold"]),
        )
        self._state.apply_fixed_pipeline_resolution()

        self._state.segment_advanced_expanded = self._adv_visible
        self._on_change()

    @staticmethod
    def _parse_int(entry: ctk.CTkEntry, fallback: int) -> int:
        try:
            return int(entry.get().strip())
        except ValueError:
            return fallback

    @staticmethod
    def _parse_float(entry: ctk.CTkEntry, fallback: float) -> float:
        try:
            return float(entry.get().strip())
        except ValueError:
            return fallback

    def _validate(self) -> bool:
        self.sync_to_state()
        if not self._state.input_dir or not Path(self._state.input_dir).is_dir():
            messagebox.showerror(
                "Input folder not set",
                "No input folder is configured.\n\n"
                "Go to tab  1. Project  and select the folder containing your leaf photos.\n"
                "The output folder will be set automatically.",
            )
            return False
        if not self._state.output_dir:
            messagebox.showerror(
                "Output folder not set",
                "No output folder is configured.\n\n"
                "Go to tab  1. Project  and select the input folder first.",
            )
            return False
        return True

    def _on_run(self) -> None:
        if self._state.skip_segmentation:
            messagebox.showinfo(
                "Segmentation disabled",
                "Segmentation is disabled.\n\n"
                "Use the 'Skip segmentation' option on the Project tab, "
                "or uncheck it to run segmentation.",
            )
            return
        self.sync_to_state()
        if self._state.validate_segmentation_method() == "interactive_mobilesam":
            from gui.interactive_sam_session import get_session

            session = get_session()
            if session.n_selected == 0:
                mode = self._interactive_click_mode()
                if mode == "leaf_scale":
                    tip = (
                        "Click the leaf, then the scale sticker on each photo "
                        "(light-blue preview + cyan ring).\n"
                        "When all photos are selected, press Run segmentation again."
                    )
                else:
                    tip = (
                        "Click each leaf first to mark it (light-blue preview).\n"
                        "When all photos are selected, press Run segmentation again."
                    )
                messagebox.showinfo("No selections", tip)
                self._apply_interactive_ui()
                return
            if self._state.output_path() is None:
                messagebox.showerror(
                    "Output folder not set", "Configure the output folder in tab 1 first."
                )
                return
            if not self._interactive_finalize_cb:
                return
            n = session.n_selected
            mode = self._interactive_click_mode()
            if mode == "leaf_scale":
                missing_scale = sum(
                    1
                    for sel in session.selections.values()
                    if sel.mask is not None
                    and sel.mask.size > 1
                    and int(sel.mask.sum()) > 0
                    and not sel.has_circle
                )
                if missing_scale > 0:
                    if not messagebox.askyesno(
                        "Missing scale clicks",
                        f"{missing_scale} photo(s) have a leaf click but no scale click.\n\n"
                        "Continue anyway? (those photos will have no mm² calibration)",
                    ):
                        self._apply_interactive_ui()
                        return
                confirm_msg = (
                    f"Run BiRefNet on {n} selected photo(s)?\n\n"
                    "Mode: Leaf + scale.\n"
                    "Models stay loaded in memory (loaded once)."
                )
            else:
                confirm_msg = (
                    f"Run BiRefNet on {n} selected photo(s)?\n\n"
                    "Mode: Only leaf (no scale reference).\n"
                    "Models stay loaded in memory (loaded once)."
                )
            if not messagebox.askyesno("Finalize interactive selections", confirm_msg):
                return
            self._interactive_finalize_cb()
            return
        if not self._validate() or not self._run_cb:
            return
        self._run_cb()

    def build_pipeline_steps(self, log) -> list[tuple[str, Path, list[str]]]:
        steps: list[tuple[str, Path, list[str]]] = []
        method = self._state.validate_segmentation_method()
        if method == "intact":
            steps.append(
                (
                    "Intact Leaves Segmentation (segment_intact)",
                    script_path("segmentation/segment_intact.py"),
                    build_intact_args(self._state),
                )
            )
            self._prep = None
            self._needs_merge = False
            return steps

        if method == "hybrid":
            steps.append(
                (
                    "Hybrid Single Leaf Segmentation (Otsu + Canny)",
                    script_path("segmentation/segment_intact.py"),
                    build_hybrid_args(self._state),
                )
            )
            self._prep = None
            self._needs_merge = False
            return steps

        if method in ("birefnet_mobilesam", "interactive_mobilesam"):
            # Interactive batch path should not be reached; GUI blocks Run.
            # Kept here so full-pipeline callers that somehow set the method
            # still invoke the same BiRefNet script (without point priors).
            steps.append(
                (
                    "BiRefNet + MobileSAM Segmentation (high precision)",
                    script_path("segmentation/birefnet_mobilesam/run_pipeline.py"),
                    build_birefnet_args(self._state),
                )
            )
            self._prep = None
            self._needs_merge = False
            return steps

        # Unreachable with the current method combobox (v1.0).
        from tkinter import messagebox
        messagebox.showerror(
            "Unsupported method",
            f"Segmentation method '{method}' is not available in this version.\n"
            "Choose BiRefNet + MobileSAM, Intact Leaves, or Interactive segmentation.",
        )
        self._prep = None
        self._needs_merge = False
        return []

    def after_sequence(self, log) -> None:
        if getattr(self, "_needs_merge", False):
            after_fastsam_merge(self._state, log)
        self.refresh_preview()
