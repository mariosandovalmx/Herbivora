"""Contour / ROI tab — UNET Shape (Mask-to-Mask) method."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from gui.paths import DEFAULT_UNET_SHAPE_MODEL, white_bg_dir
from gui.pipeline import build_contour_step, copy_leaf_masks_to_segmentation
from gui.image_sources import contour_sources
from gui.state import ProjectState
from gui.widgets.contour_editor import ContourEditorCarousel, CONTOUR_EDIT_KEY
from gui.widgets.split_pane import SplitPane


class ContourTab(ctk.CTkFrame):
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

        self._carousel = ContourEditorCarousel(
            split, self._state, show_job_status=True
        )
        self._carousel.set_edit_active_callback(self._on_edit_active_changed)

        split.set_left(scroll)
        split.set_right(self._carousel)

        ctk.CTkLabel(
            scroll, text="Contour / ROI", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        from gui.paths import PIPELINE_RESOLUTION

        ctk.CTkLabel(
            scroll,
            text=f"Uses segmentation canvas: {PIPELINE_RESOLUTION}×{PIPELINE_RESOLUTION} (fixed)",
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(0, 4))

        row_method = ctk.CTkFrame(scroll, fg_color="transparent")
        row_method.grid(row=2, column=0, sticky="ew", pady=8)
        ctk.CTkLabel(row_method, text="Contour method:").pack(side="left")
        ctk.CTkLabel(
            row_method,
            text="A. UNET Shape [Mask-to-Mask]",
            font=ctk.CTkFont(weight="bold"),
        ).pack(side="left", padx=8)

        unet_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        unet_frame.grid(row=3, column=0, sticky="ew", pady=4)

        ctk.CTkLabel(
            unet_frame,
            text=(
                "Mask-to-Mask U-Net (512 px): extracts the partial silhouette "
                "from the segmented leaf and completes missing edges to reconstruct "
                "the intact leaf ROI. Use Edit Contour to fix gaps with Add, Remove, "
                "Line, or Polygon tools."
            ),
            text_color="gray",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        row_ckpt = ctk.CTkFrame(unet_frame, fg_color="transparent")
        row_ckpt.pack(fill="x", pady=2)
        ctk.CTkLabel(row_ckpt, text="Checkpoint:").pack(side="left")
        self._recon_unet_shape_model = ctk.CTkEntry(
            row_ckpt, width=170, placeholder_text=DEFAULT_UNET_SHAPE_MODEL.name,
        )
        self._recon_unet_shape_model.pack(side="left", padx=6)
        ctk.CTkLabel(
            unet_frame,
            text="Same path as Project → Advanced → Contour U-Net.",
            text_color="gray",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(2, 0))

        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=5, column=0, sticky="ew", pady=8)
        self._run_contour_btn = ctk.CTkButton(
            btn_row,
            text="Run contour",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            command=self._on_run,
        )
        self._run_contour_btn.pack(side="left", padx=4)
        self._edit_contour_btn = ctk.CTkButton(
            btn_row,
            text="Edit Contour",
            height=40,
            font=ctk.CTkFont(weight="bold"),
            fg_color=("#2ECC71", "#1E8449"),
            hover_color=("#27AE60", "#196F3D"),
            command=self._on_edit_contour,
        )
        self._edit_contour_btn.pack(side="left", padx=4)

        self._contour_ready_label = ctk.CTkLabel(
            scroll,
            text="",
            text_color=("gray40", "gray65"),
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self._contour_ready_label.grid(row=6, column=0, sticky="ew", pady=(0, 8))

        self._edit_hint = ctk.CTkLabel(
            scroll,
            text=(
                "Edit Contour tools: Add / Remove brush, Line (2-point bridge), "
                "Polygon fill, Undo (last step), Reset all."
            ),
            text_color=("gray45", "gray60"),
            wraplength=340,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self._edit_hint.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        self._edit_hint.grid_remove()

    def _on_edit_active_changed(self, active: bool) -> None:
        if active:
            self._edit_contour_btn.configure(text="Done editing")
            self._edit_hint.grid()
        else:
            self._edit_contour_btn.configure(text="Edit Contour")
            self._edit_hint.grid_remove()

    def _on_edit_contour(self) -> None:
        active = not self._carousel.is_edit_contour_active()
        self._carousel.set_edit_contour_active(active)

    def set_run_callback(self, cb) -> None:
        self._run_cb = cb

    def set_run_contour_enabled(self, enabled: bool) -> None:
        self._run_contour_btn.configure(state="normal" if enabled else "disabled")

    def set_contour_ready_hint(self, text: str = "", *, busy: bool = False) -> None:
        if busy:
            self._contour_ready_label.configure(
                text="Analyzing, please wait... Preparing images and masks.",
                text_color=("#1f6aa5", "#64b5f6"),
            )
            self.set_run_contour_enabled(False)
        elif text:
            self._contour_ready_label.configure(
                text=text,
                text_color=("#1b5e20", "#81c784"),
            )
            self.set_run_contour_enabled(True)
        else:
            self._contour_ready_label.configure(text="")
            self.set_run_contour_enabled(True)

    def load_from_state(self) -> None:
        ckpt = (
            getattr(self._state, "recon_model_unet_shape", "")
            or getattr(self._state, "leaf_model", "")
            or str(DEFAULT_UNET_SHAPE_MODEL)
        )
        self._recon_unet_shape_model.delete(0, "end")
        self._recon_unet_shape_model.insert(0, ckpt)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        self._carousel.set_sources(contour_sources(self._state), default_key=CONTOUR_EDIT_KEY)
        # Re-fit after the split pane finishes laying out the right panel.
        self.after(50, lambda: getattr(self._carousel, "_ensure_view_fitted", lambda: None)())
        self.after(200, lambda: getattr(self._carousel, "_ensure_view_fitted", lambda: None)())

    def sync_to_state(self) -> None:
        self._state.contour_method = "recon_unet_shape"
        self._state.leaf_normalize_bg = True
        ckpt = self._recon_unet_shape_model.get().strip().strip("\"'")
        if ckpt:
            self._state.recon_model_unet_shape = ckpt
            self._state.leaf_model = ckpt  # keep Project Contour U-Net in sync
        self._on_change()

    def _validate(self) -> bool:
        self.sync_to_state()
        out = self._state.output_path()
        if out is None:
            messagebox.showerror("Error", "Please define the output folder in Project tab.")
            return False
        wb = white_bg_dir(out)
        if not wb.is_dir() or not any(wb.iterdir()):
            if self._state.skip_segmentation:
                messagebox.showerror(
                    "Error",
                    "No prepared images found in segmentation/white_bg/.\n\n"
                    "Enable 'Skip segmentation' on the Project tab and run "
                    "the full pipeline, or run whitebg_masks prep first.\n\n"
                    f"Expected folder:\n{wb}",
                )
            else:
                messagebox.showerror(
                    "Error",
                    "Run segmentation first (Segmentation tab).\n"
                    f"No images found in:\n{wb}",
                )
            return False
        unet_path = Path(
            self._state.recon_model_unet_shape or str(DEFAULT_UNET_SHAPE_MODEL)
        )
        if not unet_path.is_file():
            messagebox.showerror(
                "Error",
                f"UNET Shape checkpoint not found:\n{unet_path}\n\n"
                "Download the models first:\n"
                "  python download_models.py",
            )
            return False
        return True

    def _on_run(self) -> None:
        if self._validate() and self._run_cb:
            self._run_cb(filename=None)

    def build_steps(self, filename: str | None = None) -> list[tuple[str, Path, list[str]]]:
        return [build_contour_step(self._state, filename=filename)]

    def post_success(self, log) -> None:
        out = self._state.output_path()
        if out:
            copy_leaf_masks_to_segmentation(out, log)
        # Force Overlay contour so the predicted U-Net result is visible
        self._carousel.set_sources(
            contour_sources(self._state),
            default_key=CONTOUR_EDIT_KEY,
        )
        n = len(getattr(self._carousel, "_paths", []) or [])
        self.after(50, lambda: getattr(self._carousel, "_ensure_view_fitted", lambda: None)())
        self.after(200, lambda: getattr(self._carousel, "_ensure_view_fitted", lambda: None)())
        if log:
            log(f"Contour viewer: showing {n} image(s) ({CONTOUR_EDIT_KEY}).")
            if n == 0:
                log(
                    "WARNING: Contour viewer has 0 images. "
                    "Check leaf_roi_preview/overlays or segmentation/white_bg."
                )
