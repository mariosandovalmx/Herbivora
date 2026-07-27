"""Damage Analysis Tab: analyze_leaves."""

from __future__ import annotations

import os
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from gui.paths import PIPELINE_RESOLUTION, masks_dir, segmentation_dir
from gui.pipeline import build_analyze_args, script_path
from gui.image_sources import analyze_sources
from gui.state import ProjectState
from gui.widgets.damage_editor import DamageEditorCarousel
from gui.widgets.split_pane import SplitPane
from gui.widgets.info_button import InfoButton


# Analysis always uses the Contour (tab 3) leaf mask as ROI.
# "filled" = Contour silhouette (+ hole fill inside it); never hull/closed expansion.
CONTOUR_ROI_MODE = "filled"


class AnalyzeTab(ctk.CTkFrame):
    def __init__(self, master, state: ProjectState, on_change, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._state = state
        self._on_change = on_change
        self._run_cb = None
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        split = SplitPane(self)
        split.grid(row=0, column=0, sticky="nsew")

        scroll = ctk.CTkScrollableFrame(split)
        scroll.grid_columnconfigure(0, weight=1)

        self._carousel = DamageEditorCarousel(split, self._state, show_job_status=True)
        self._carousel.set_edit_active_callback(self._on_edit_active_changed)
        self._carousel.set_metrics_callback(self._on_damage_metrics)

        split.set_left(scroll)
        split.set_right(self._carousel)

        ctk.CTkLabel(
            scroll, text="Damage Analysis", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        ctk.CTkLabel(
            scroll,
            text=f"U-Net resolution: {PIPELINE_RESOLUTION}×{PIPELINE_RESOLUTION} (fixed)",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=4)

        adv = ctk.CTkFrame(scroll, fg_color=("gray92", "gray18"))
        adv.grid(row=2, column=0, sticky="ew", pady=(8, 4))
        adv.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(adv, text="White-hole detection", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4)
        )
        self._white_hole_adaptive = ctk.CTkCheckBox(
            adv,
            text="Auto Brightness (per image, recommended)",
            command=self._on_white_hole_auto_toggle,
        )
        self._white_hole_adaptive.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 2))
        InfoButton(
            adv,
            title="Auto Brightness",
            message=(
                "When enabled, Analysis picks a Brightness threshold for each leaf "
                "from tissue brightness + a short sweep that stops before noise explodes.\n\n"
                "Recommended for mixed lighting / missing white holes.\n"
                "Uncheck to show Manual Brightness and force a fixed threshold for all images."
            ),
        ).grid(row=1, column=2, sticky="w", padx=4, pady=(4, 2))

        self._manual_brightness_label = ctk.CTkLabel(adv, text="Manual Brightness:")
        self._white_hole_brightness = ctk.CTkEntry(adv, width=70)

        self._white_hole_min_area_label = ctk.CTkLabel(adv, text="Min area (px):")
        self._white_hole_min_area = ctk.CTkEntry(adv, width=70)
        self._white_hole_min_area_info = InfoButton(
            adv,
            title="Min area (px)",
            message=(
                "Minimum size (in pixels) a bright patch must have to count as a white hole.\n\n"
                "Patches smaller than this value are ignored as noise or speckles.\n\n"
                "• Lower values (e.g. 3): detect smaller holes.\n"
                "• Higher values (e.g. 15–50): filter out tiny bright spots that are not real damage.\n\n"
                "Default: 3 (good when small herbivory holes are being missed)."
            ),
        )

        self._white_hole_edge_band_label = ctk.CTkLabel(adv, text="Edge band (px):")
        self._white_hole_edge_band = ctk.CTkEntry(adv, width=70)
        self._white_hole_edge_band_info = InfoButton(
            adv,
            title="Edge band (px)",
            message=(
                "Width (in pixels) of a margin along the leaf contour where white holes "
                "are not searched.\n\n"
                "This reduces false positives from bright edges or reflections near the "
                "leaf border.\n\n"
                "• Lower values (e.g. 1): allow holes closer to the margin.\n"
                "• Higher values (e.g. 5–10): ignore a wider border strip; may miss holes "
                "touching the contour.\n\n"
                "Default: 1 (good when holes near the edge are being missed)."
            ),
        )

        self._white_hole_hint = ctk.CTkLabel(
            adv,
            text="Defaults: Auto on · Min area 3 · Edge band 1",
            text_color="gray",
            font=ctk.CTkFont(size=11),
        )
        # Match default state: Auto on → Manual Brightness hidden.
        self._white_hole_adaptive.select()
        self._on_white_hole_auto_toggle()

        row_out = ctk.CTkFrame(scroll, fg_color="transparent")
        row_out.grid(row=3, column=0, sticky="ew", pady=(12, 4))
        ctk.CTkLabel(row_out, text="Measurement output:").pack(side="left")
        self._output_mode_var = ctk.StringVar(value="Damage % only")
        self._radio_pct = ctk.CTkRadioButton(
            row_out,
            text="Damage % only",
            variable=self._output_mode_var,
            value="Damage % only",
            command=self._on_output_mode_change,
        )
        self._radio_pct.pack(side="left", padx=(8, 4))
        self._radio_area = ctk.CTkRadioButton(
            row_out,
            text="Damage % + area (cm\u00b2)",
            variable=self._output_mode_var,
            value="Damage % + area (cm\u00b2)",
            command=self._on_output_mode_change,
        )
        self._radio_area.pack(side="left", padx=4)
        InfoButton(
            row_out,
            title="Measurement output",
            message=(
                "Damage % only: Reports herbivory as a percentage of total leaf area. No scale reference needed.\n\n"
                "Damage % + area (cm²): Also computes the absolute damage area and leaf area in cm². "
                "Requires a blue reference dot visible in the original scene photo. "
                "Enter the dot's known diameter in mm below (default: 6.0 mm = 0.6 cm reference dot)."
            )
        ).pack(side="left", padx=6)

        self._scale_area_row = ctk.CTkFrame(scroll, fg_color="transparent")
        self._scale_area_row.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(self._scale_area_row, text="Blue dot diameter (mm):", width=160, anchor="w").pack(side="left")
        self._scale_area_entry = ctk.CTkEntry(self._scale_area_row, width=70)
        self._scale_area_entry.pack(side="left", padx=8)
        ctk.CTkLabel(
            self._scale_area_row,
            text="Default: 6.0 mm = 0.6 cm reference dot",
            text_color="gray",
        ).pack(side="left", padx=4)

        ctk.CTkLabel(
            scroll,
            text=(
                "Leaf ROI always comes from the Contour tab mask "
                "(segmentation/masks/ after Contour runs). "
                "Internal holes count as damage; edge-artifact filtering is always on."
            ),
            text_color="gray",
            wraplength=360,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(8, 4))

        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=6, column=0, sticky="ew", pady=16)
        ctk.CTkButton(
            btn_row,
            text="Run U-Net analysis",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            command=self._on_run,
        ).pack(side="left", padx=4)
        self._edit_damage_btn = ctk.CTkButton(
            btn_row,
            text="Edit Damage",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            fg_color=("#2ECC71", "#1E8449"),
            hover_color=("#27AE60", "#196F3D"),
            command=self._on_edit_damage,
        )
        self._edit_damage_btn.pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Open Results", command=self._open_results).pack(
            side="left", padx=4
        )
        InfoButton(
            btn_row,
            title="Damage editing",
            message=(
                "Press Edit Damage to open the tool bar.\n\n"
                "Add / Remove: paint damage with the brush.\n\n"
                "Select Region: click to add or remove a whole damage area. "
                "Choose Add or Remove first, then enable Select Region and click on the image.\n\n"
                "Line / Polygon: bridge or fill damage regions (same workflow as Contour).\n\n"
                "Select Region + Add: MobileSAM selects the fragment under the click; "
                "if SAM is unavailable or over-selects the leaf, color flood fill is used "
                "(Tolerance controls that fallback).\n\n"
                "Select Region + Remove: removes the connected damage blob under the click."
            ),
        ).pack(side="left", padx=4)

        self._edit_hint = ctk.CTkLabel(
            scroll,
            text=(
                "Edit Damage tools: Add / Remove brush, Select Region, Line, Polygon, "
                "Undo (last step), Reset all."
            ),
            text_color=("gray45", "gray60"),
            wraplength=340,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self._edit_hint.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        self._edit_hint.grid_remove()

        self._damage_metrics = ctk.CTkLabel(
            scroll,
            text="",
            text_color=("#1b5e20", "#81c784"),
            wraplength=340,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self._damage_metrics.grid(row=8, column=0, sticky="ew", pady=(4, 12))

    def _on_damage_metrics(self, text: str) -> None:
        self._damage_metrics.configure(text=text or "")

    def _on_edit_active_changed(self, active: bool) -> None:
        if active:
            self._edit_damage_btn.configure(text="Done editing")
            self._edit_hint.grid()
        else:
            self._edit_damage_btn.configure(text="Edit Damage")
            self._edit_hint.grid_remove()

    def _on_edit_damage(self) -> None:
        active = not self._carousel.is_edit_damage_active()
        self._carousel.set_edit_damage_active(active)

    def set_run_callback(self, cb) -> None:
        self._run_cb = cb

    def _on_white_hole_auto_toggle(self) -> None:
        """Show Manual Brightness only when Auto is off; keep layout compact."""
        auto = bool(self._white_hole_adaptive.get())
        if auto:
            self._manual_brightness_label.grid_remove()
            self._white_hole_brightness.grid_remove()
            row_min, row_edge, row_hint = 2, 3, 4
        else:
            self._manual_brightness_label.grid(row=2, column=0, sticky="w", padx=10, pady=3)
            self._white_hole_brightness.grid(row=2, column=1, sticky="w", padx=4, pady=3)
            row_min, row_edge, row_hint = 3, 4, 5
        self._white_hole_min_area_label.grid(row=row_min, column=0, sticky="w", padx=10, pady=3)
        self._white_hole_min_area.grid(row=row_min, column=1, sticky="w", padx=4, pady=3)
        self._white_hole_min_area_info.grid(row=row_min, column=2, sticky="w", padx=4, pady=3)
        self._white_hole_edge_band_label.grid(row=row_edge, column=0, sticky="w", padx=10, pady=3)
        self._white_hole_edge_band.grid(row=row_edge, column=1, sticky="w", padx=4, pady=3)
        self._white_hole_edge_band_info.grid(row=row_edge, column=2, sticky="w", padx=4, pady=3)
        self._white_hole_hint.grid(
            row=row_hint, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 8)
        )

    def _on_output_mode_change(self, value: str | None = None) -> None:
        """Show/hide the blue dot area entry based on selected output mode."""
        val = value or self._output_mode_var.get()
        if val == "Damage % + area (cm\u00b2)":
            self._scale_area_row.grid()
        else:
            self._scale_area_row.grid_remove()

    def load_from_state(self) -> None:
        self._state.apply_fixed_pipeline_resolution()
        # Always on (no longer exposed in GUI).
        # Always off (option removed from GUI).
        self._state.fill_marginal = True
        self._state.edge_artifact_filter = True
        self._state.draw_hull_line = False
        self._white_hole_brightness.delete(0, "end")
        self._white_hole_brightness.insert(0, str(self._state.white_hole_brightness))
        self._white_hole_min_area.delete(0, "end")
        self._white_hole_min_area.insert(0, str(self._state.white_hole_min_area))
        self._white_hole_edge_band.delete(0, "end")
        self._white_hole_edge_band.insert(0, str(self._state.white_hole_edge_band))
        if self._state.white_hole_adaptive:
            self._white_hole_adaptive.select()
        else:
            self._white_hole_adaptive.deselect()
        self._on_white_hole_auto_toggle()
        # Measurement output
        if not self._state.remove_blue:
            self._radio_area.configure(state="disabled")
            self._output_mode_var.set("Damage % only")
            self._state.report_area_cm2 = False
        else:
            self._radio_area.configure(state="normal")
            mode = "Damage % + area (cm\u00b2)" if self._state.report_area_cm2 else "Damage % only"
            self._output_mode_var.set(mode)
        self._scale_area_entry.delete(0, "end")
        self._scale_area_entry.insert(0, str(self._state.birefnet_known_diameter_mm))
        self._on_output_mode_change()
        self._state.roi_mode = CONTOUR_ROI_MODE
        self.refresh_preview()

    def refresh_preview(self) -> None:
        self._carousel.set_sources(analyze_sources(self._state))

    def sync_to_state(self) -> None:
        self._state.apply_fixed_pipeline_resolution()
        self._state.fill_marginal = True
        self._state.edge_artifact_filter = True
        self._state.draw_hull_line = False
        try:
            self._state.white_hole_brightness = int(
                self._white_hole_brightness.get().strip() or "235"
            )
        except ValueError:
            self._state.white_hole_brightness = 235
        try:
            self._state.white_hole_min_area = int(
                self._white_hole_min_area.get().strip() or "3"
            )
        except ValueError:
            self._state.white_hole_min_area = 3
        try:
            self._state.white_hole_edge_band = int(
                self._white_hole_edge_band.get().strip() or "1"
            )
        except ValueError:
            self._state.white_hole_edge_band = 1
        self._state.white_hole_adaptive = bool(self._white_hole_adaptive.get())
        self._state.report_area_cm2 = (self._output_mode_var.get() == "Damage % + area (cm\u00b2)")
        import math as _math
        try:
            _d_mm = float(self._scale_area_entry.get().strip() or "6.0")
        except ValueError:
            _d_mm = 6.0
        self._state.birefnet_known_diameter_mm = _d_mm
        self._state.scale_area_cm2 = _math.pi * (_d_mm / 2) ** 2 / 100
        self._state.roi_mode = CONTOUR_ROI_MODE
        self._on_change()

    def _validate(self) -> bool:
        self.sync_to_state()
        out = self._state.output_path()
        if out is None:
            messagebox.showerror("Error", "Define the output folder in Project tab.")
            return False
        seg = segmentation_dir(out)
        md = masks_dir(out)
        if not seg.is_dir():
            messagebox.showerror("Error", "Run segmentation first.")
            return False
        if not md.is_dir() or not list(md.glob("*_mask.png")):
            messagebox.showerror(
                "Error",
                "No Contour masks found in segmentation/masks/.\n"
                "Please run the Contour tab first (it copies leaf masks there).",
            )
            return False
        if not Path(self._state.damage_model).is_file():
            messagebox.showerror("Error", f"Damage U-Net model not found:\n{self._state.damage_model}")
            return False
        return True

    def _on_run(self) -> None:
        if self._validate() and self._run_cb:
            self._run_cb()

    def build_steps(self) -> list[tuple[str, Path, list[str]]]:
        return [
            (
                "Damage U-Net Analysis (analyze_leaves)",
                script_path("analyze_leaves.py"),
                build_analyze_args(self._state),
            )
        ]

    def _open_results(self) -> None:
        out = self._state.output_path()
        if out is None:
            return
        an = out / "analyzed"
        if an.is_dir():
            from gui.open_path import open_path
            open_path(an)
