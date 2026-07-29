"""Main HerbivoR GUI Window."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from gui.pipeline import (
    build_skip_segmentation_args,
    build_whitebg_args,
    needs_skip_segmentation_prep,
    run_scale_detection,
    script_path,
)
from gui.runner import JobResult, JobRunner, JobStatus
from gui.state import ProjectState, load_config, save_config
from gui.tabs.analyze_tab import AnalyzeTab
from gui.tabs.contour_tab import ContourTab
from gui.tabs.project_tab import ProjectTab
from gui.tabs.segment_tab import SegmentTab
from gui.widgets.log_panel import LogPanel
from gui.widgets.split_pane import MainSplitPane


class HerbivoRApp(ctk.CTk):
    def __init__(self) -> None:
        # Theme must be set before the CTk window is constructed; otherwise the
        # root keeps the default palette and then visibly recolors (icon can drop).
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("green")

        super().__init__()
        self.title("HerbivoR — Leaf Damage Analysis")
        self.geometry("1320x820")
        self.minsize(1000, 700)

        from gui.icons import apply_window_icon, load_header_image

        apply_window_icon(self)

        self._state = load_config()
        self._runner = JobRunner(self._on_log, self._on_job_done)
        self._after_fastsam = False
        self._after_contour_copy = False
        self._chain_contour = False
        self._chain_analyze = False
        self._skip_segmentation_prep = False
        self._active_job_panel: str | None = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._main_split = MainSplitPane(self)
        self._main_split.grid(row=0, column=0, sticky="nsew")

        self._left_wrap = ctk.CTkFrame(self._main_split)
        self._left_wrap.grid_rowconfigure(1, weight=1)
        self._left_wrap.grid_columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(self._left_wrap, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))

        header_img = load_header_image(32)
        if header_img is not None:
            self._header_icon_img = header_img  # keep reference
            ctk.CTkLabel(top_bar, text="", image=header_img, width=32).pack(
                side="left", padx=(0, 8)
            )
        ctk.CTkLabel(
            top_bar,
            text="HerbivoR",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")

        self._show_log_btn = ctk.CTkButton(
            top_bar,
            text="Show Log",
            width=130,
            height=26,
            command=self._toggle_log,
        )
        self._show_log_btn.pack(side="right")

        self._tabs = ctk.CTkTabview(self._left_wrap)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self._tabs.add("1. Project")
        self._tabs.add("2. Segmentation")
        self._tabs.add("3. Contour / ROI")
        self._tabs.add("4. Analysis")
        self._tabs.configure(command=self._on_tab_changed)

        self._project_tab = ProjectTab(
            self._tabs.tab("1. Project"), self._state, self._on_state_change
        )
        self._project_tab.pack(fill="both", expand=True)
        self._project_tab.set_skip_segmentation_callback(self._on_skip_segmentation_changed)

        self._segment_tab = SegmentTab(
            self._tabs.tab("2. Segmentation"), self._state, self._on_state_change
        )
        self._segment_tab.pack(fill="both", expand=True)
        self._segment_tab.set_run_callback(self._run_segmentation)
        self._segment_tab.set_reanalyze_callback(self._on_reanalyze)
        self._segment_tab.set_interactive_finalize_callback(self._run_interactive_finalize)
        self._is_reanalyze = False
        self._is_interactive_finalize = False

        self._contour_tab = ContourTab(
            self._tabs.tab("3. Contour / ROI"), self._state, self._on_state_change
        )
        self._contour_tab.pack(fill="both", expand=True)
        self._contour_tab.set_run_callback(self._run_contour)

        self._analyze_tab = AnalyzeTab(
            self._tabs.tab("4. Analysis"), self._state, self._on_state_change
        )
        self._analyze_tab.pack(fill="both", expand=True)
        self._analyze_tab.set_run_callback(self._run_analyze)

        self._segment_tab._carousel.set_stop_callback(self._runner.cancel)
        self._contour_tab._carousel.set_stop_callback(self._runner.cancel)
        self._analyze_tab._carousel.set_stop_callback(self._runner.cancel)

        self._main_split.set_tabs(self._left_wrap)

        self._log_panel = LogPanel(self._main_split)
        self._log_panel.set_cancel_callback(self._runner.cancel)
        self._log_panel.set_toggle_callback(self._toggle_log)
        self._main_split.set_log(self._log_panel)

        self._load_all_tabs()
        self._update_tab_access()
        self._refresh_carousel_for_tab(self._tabs.get())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_log(self) -> None:
        visible = self._main_split.toggle_log()
        if visible:
            self._show_log_btn.pack_forget()
        else:
            self._show_log_btn.pack(side="right")
        self.update_idletasks()

    def _carousel_for_panel(self, panel: str):
        if panel in ("contour", "skip_prep"):
            return self._contour_tab._carousel
        if panel == "segment":
            return self._segment_tab._carousel
        if panel == "analyze":
            return self._analyze_tab._carousel
        return None

    def _skip_prep_image_count(self) -> int | None:
        out = self._state.output_path()
        if out is None:
            return None
        from gui.paths import count_white_bg_leaves

        n = count_white_bg_leaves(out)
        return n if n > 0 else None

    def _skip_prep_ui_start(self) -> None:
        self._project_tab.set_skip_prep_status("busy")
        self._contour_tab.set_contour_ready_hint(busy=True)
        self._set_status("Analyzing, please wait...", "#1f6aa5")

    def _skip_prep_ui_done(self) -> None:
        n = self._skip_prep_image_count()
        self._project_tab.set_skip_prep_status("done", image_count=n)
        hint = "Done — images and masks are ready. Click Run contour."
        if n:
            hint = f"Done — {n} images prepared. Click Run contour."
        self._contour_tab.set_contour_ready_hint(hint)
        self._set_status("Ready for Contour / ROI", "#2d6a4f")

    def _skip_prep_ui_fail(self, message: str = "Preparation failed") -> None:
        self._project_tab.set_skip_prep_status("failed")
        self._contour_tab.set_contour_ready_hint(
            "Preparation failed. Re-enable Skip segmentation on the Project tab to retry."
        )
        self._set_status(message, "#8B0000")

    def _halt_all_job_panels(self) -> None:
        """Force-stop progress animation on every tab carousel."""
        for car in (
            self._segment_tab._carousel,
            self._contour_tab._carousel,
            self._analyze_tab._carousel,
        ):
            try:
                car.job_status_hide()
            except Exception:
                pass
        self._active_job_panel = None

    def _job_panel_start(self, panel: str, message: str = "Analyzing...") -> None:
        # Stop any previous panel bar so it cannot keep animating after the job moves on.
        prev = self._active_job_panel
        if prev and prev != panel:
            prev_car = self._carousel_for_panel(prev)
            if prev_car is not None:
                prev_car.job_status_hide()
        self._active_job_panel = panel
        car = self._carousel_for_panel(panel)
        if car is not None:
            car.job_status_start(message)

    def _job_panel_complete(self, panel: str, message: str = "Completed") -> None:
        car = self._carousel_for_panel(panel)
        if car is not None:
            car.job_status_complete(message)
        # Halt stray bars on other tabs.
        for other in (
            self._segment_tab._carousel,
            self._contour_tab._carousel,
            self._analyze_tab._carousel,
        ):
            if other is car:
                continue
            try:
                other.job_status_hide()
            except Exception:
                pass
        if self._active_job_panel == panel:
            self._active_job_panel = None

    def _job_panel_fail(self, panel: str, message: str = "Error") -> None:
        car = self._carousel_for_panel(panel)
        if car is not None:
            car.job_status_fail(message)
        for other in (
            self._segment_tab._carousel,
            self._contour_tab._carousel,
            self._analyze_tab._carousel,
        ):
            if other is car:
                continue
            try:
                other.job_status_hide()
            except Exception:
                pass
        if self._active_job_panel == panel:
            self._active_job_panel = None

    def _load_all_tabs(self) -> None:
        self._project_tab.load_from_state()
        self._segment_tab.load_from_state()
        self._contour_tab.load_from_state()
        self._analyze_tab.load_from_state()

    def _sync_all(self) -> None:
        self._project_tab.sync_to_state()
        self._segment_tab.sync_to_state()
        self._contour_tab.sync_to_state()
        self._analyze_tab.sync_to_state()
        save_config(self._state)

    def _on_state_change(self) -> None:
        save_config(self._state)
        self._project_tab.refresh_status()
        self._update_tab_access()

    def _on_skip_segmentation_changed(self, *, run_prep: bool = False) -> None:
        self._update_tab_access()
        if self._state.skip_segmentation:
            self._tabs.set("3. Contour / ROI")
            self._on_tab_changed()
            if run_prep:
                if not self._run_skip_segmentation_prep(
                    chain_contour=False, chain_analyze=False
                ):
                    self._project_tab.revert_skip_segmentation()
                    self._update_tab_access()
            else:
                self._project_tab.refresh_skip_prep_readiness()
                n = self._skip_prep_image_count()
                if n:
                    self._contour_tab.set_contour_ready_hint(
                        f"Done — {n} images prepared. Click Run contour."
                    )
        else:
            self._project_tab.set_skip_prep_status("idle")
            self._contour_tab.set_contour_ready_hint("")

    def _update_tab_access(self) -> None:
        seg_btn = self._tabs._segmented_button._buttons_dict.get("2. Segmentation")
        if seg_btn is None:
            return
        if self._state.skip_segmentation:
            seg_btn.configure(state="disabled")
            if self._tabs.get() == "2. Segmentation":
                self._tabs.set("3. Contour / ROI")
                self._on_tab_changed()
        else:
            seg_btn.configure(state="normal")

    def _on_tab_changed(self) -> None:
        tab_name = self._tabs.get()
        if tab_name == "2. Segmentation" and self._state.skip_segmentation:
            self._tabs.set("3. Contour / ROI")
            tab_name = "3. Contour / ROI"
        if tab_name == "1. Project":
            self._project_tab.load_from_state()
        elif tab_name == "2. Segmentation":
            self._segment_tab.load_from_state()
        elif tab_name == "3. Contour / ROI":
            self._contour_tab.load_from_state()
        elif tab_name == "4. Analysis":
            self._analyze_tab.load_from_state()
        self._refresh_carousel_for_tab(tab_name)

    def _refresh_carousel_for_tab(self, tab_name: str) -> None:
        if tab_name == "1. Project":
            self._project_tab.refresh_preview()
        elif tab_name == "2. Segmentation":
            self._segment_tab.refresh_preview()
        elif tab_name == "3. Contour / ROI":
            self._contour_tab.refresh_preview()
        elif tab_name == "4. Analysis":
            self._analyze_tab.refresh_preview()

    def _refresh_all_carousels(self, *, segment_show_results: bool = False) -> None:
        self._project_tab.refresh_preview()
        self._segment_tab.refresh_preview(show_results=segment_show_results)
        self._contour_tab.refresh_preview()
        self._analyze_tab.refresh_preview()

    def _on_log(self, line: str) -> None:
        self.after(0, lambda: self._log_panel.append(line))

    def _set_status(self, text: str, color: str = "gray") -> None:
        self.after(0, lambda: self._log_panel.set_status(text, color))

    def _on_reanalyze(self, steps: list) -> None:
        if self._runner.is_running:
            from tkinter import messagebox
            messagebox.showwarning("Busy", "Wait for the current job to finish.")
            return
        if not steps:
            return
        method = self._state.validate_segmentation_method()
        if method in ("birefnet_mobilesam", "interactive_mobilesam"):
            title, _script, args = steps[0]
            input_override = None
            for i, a in enumerate(args):
                if a == "--input" and i + 1 < len(args):
                    input_override = Path(args[i + 1])
                    break
            self._run_birefnet_inprocess(
                input_override=input_override,
                title=title,
                is_reanalyze=True,
            )
            return
        self._is_reanalyze = True
        self._is_interactive_finalize = False
        self._after_fastsam = False
        self._job_panel_start("segment")
        self._set_status("Re-analyzing...", "#1f6aa5")
        title, script, args = steps[0]
        self._runner.run_script(script, args, title=title)

    def _run_interactive_finalize(self) -> None:
        """Finalize all Interactive click selections in-process (models stay loaded)."""
        if self._runner.is_running:
            from tkinter import messagebox
            messagebox.showwarning("Busy", "Wait for the current job to finish.")
            return
        kwargs = self._segment_tab.collect_interactive_finalize_kwargs()
        if kwargs.get("output_dir") is None:
            return
        from gui.interactive_sam_session import get_session

        session = get_session()
        self._is_interactive_finalize = True
        self._is_reanalyze = False
        self._after_fastsam = False
        self._job_panel_start("segment")
        self._set_status("Interactive finalize (BiRefNet)...", "#1f6aa5")
        n = session.n_selected

        def work(log, should_cancel) -> None:
            out_dir = kwargs["output_dir"]
            log(f"Finalizing {n} interactive selection(s) with BiRefNet (models kept in RAM)...")
            ok, failed = session.finalize_batch(
                out_dir,
                known_diameter_mm=kwargs["known_diameter_mm"],
                hybrid_mode=kwargs["hybrid_mode"],
                seg_resolution=kwargs["seg_resolution"],
                output_size=kwargs["output_size"],
                agreement_threshold=kwargs["agreement_threshold"],
                remove_blue=kwargs["remove_blue"],
                project_root=kwargs.get("project_root"),
                log=log,
                should_cancel=should_cancel,
            )
            if failed and not ok:
                raise RuntimeError(f"All {failed} interactive finalizations failed.")

        self._runner.run_callable(
            work,
            title=f"Interactive segmentation finalize ({n} photo(s))",
        )

    def _run_birefnet_inprocess(
        self,
        *,
        input_override: Path | None = None,
        title: str | None = None,
        is_reanalyze: bool = False,
        chain_contour: bool = False,
        chain_analyze: bool = False,
    ) -> None:
        """Method A: BiRefNet + MobileSAM in the GUI process (models stay warm)."""
        if self._runner.is_running:
            return
        kwargs = self._segment_tab.collect_birefnet_batch_kwargs(input_override)
        if kwargs.get("input_dir") is None or kwargs.get("output_dir") is None:
            from tkinter import messagebox
            messagebox.showerror(
                "Folders required",
                "Define input and output folders in the Project tab.",
            )
            return
        from gui.interactive_sam_session import get_session

        session = get_session()
        self._is_interactive_finalize = False
        self._is_reanalyze = is_reanalyze
        self._after_fastsam = False
        self._chain_contour = chain_contour
        self._chain_analyze = chain_analyze
        self._job_panel_start("segment")
        self._set_status("Running BiRefNet + MobileSAM (in-GUI)...", "#1f6aa5")
        job_title = title or "BiRefNet + MobileSAM Segmentation (models kept warm)"

        def work(log, should_cancel) -> None:
            ok, failed = session.run_folder_batch(
                kwargs["input_dir"],
                kwargs["output_dir"],
                known_diameter_mm=kwargs["known_diameter_mm"],
                hybrid_mode=kwargs["hybrid_mode"],
                seg_resolution=kwargs["seg_resolution"],
                output_size=kwargs["output_size"],
                agreement_threshold=kwargs["agreement_threshold"],
                remove_blue=kwargs["remove_blue"],
                mobilesam_weights=kwargs.get("mobilesam_weights"),
                log=log,
                should_cancel=should_cancel,
            )
            if failed and not ok:
                raise RuntimeError(f"All {failed} BiRefNet segmentations failed.")

        self._runner.run_callable(work, title=job_title)

    def _on_job_done(self, result: JobResult) -> None:
        def finish() -> None:
            if result.status == JobStatus.SUCCESS:
                self._log_panel.set_status("Ready", "#2d6a4f")
                if getattr(self, "_is_interactive_finalize", False):
                    self._is_interactive_finalize = False
                    self._job_panel_complete("segment")
                    self._on_state_change()
                    self._segment_tab.refresh_preview(show_results=True)
                    from tkinter import messagebox
                    messagebox.showinfo(
                        "Interactive segmentation",
                        "BiRefNet finalize complete.\n"
                        "Showing Leaves (white_bg) in the Segmentation viewer.",
                    )
                    return
                if getattr(self, "_is_reanalyze", False):
                    self._is_reanalyze = False
                    self._segment_tab.cleanup_reanalyze_temp()
                    self._job_panel_complete("segment")
                    self._segment_tab.refresh_current_image()
                    return
                if self._after_fastsam:
                    self._after_fastsam = False
                    try:
                        self._segment_tab.after_sequence(self._runner.log)
                    except Exception as e:
                        messagebox.showerror("Error", f"After FastSAM: {e}")
                        self._job_panel_fail("segment", "Error")
                        self._clear_chain()
                        return
                    self._start_whitebg()
                    return
                if self._skip_segmentation_prep:
                    self._skip_segmentation_prep = False
                    self._job_panel_complete("skip_prep", message="Done")
                    self._skip_prep_ui_done()
                    self._on_state_change()
                    self._contour_tab.refresh_preview()
                    self._refresh_all_carousels()
                    self._runner.log(
                        "Skip segmentation prep complete. Ready for Contour / ROI."
                    )
                    if self._chain_contour:
                        self._chain_contour = False
                        self._start_contour()
                    elif self._chain_analyze:
                        pass
                    return
                if self._after_contour_copy:
                    self._after_contour_copy = False
                    try:
                        self._contour_tab.post_success(self._runner.log)
                    except Exception as e:
                        self._runner.log(f"WARNING: contour viewer refresh failed: {e}")
                    finally:
                        # Always clear Analyzing… even if the viewer refresh fails
                        self._job_panel_complete("contour")
                    self._on_state_change()
                    if self._chain_analyze:
                        self._chain_analyze = False
                        self._start_analyze()
                    elif self._chain_contour:
                        pass
                    return
                # Stop the progress bar FIRST: carousel refresh can throw and
                # would otherwise leave the indeterminate animation running.
                panel_done = self._active_job_panel
                do_chain_contour = bool(self._chain_contour)
                do_chain_analyze = bool(self._chain_analyze)
                if do_chain_contour:
                    self._job_panel_complete("segment")
                    self._chain_contour = False
                elif do_chain_analyze:
                    self._job_panel_complete("segment")
                    self._chain_analyze = False
                else:
                    if panel_done:
                        self._job_panel_complete(panel_done)
                    else:
                        self._halt_all_job_panels()

                try:
                    out_root = self._state.output_path()
                    if out_root:
                        from gui.paths import work_dir
                        import shutil
                        temp_d = work_dir(out_root)
                        if temp_d.is_dir():
                            self._runner.log("Cleaning up temporary extraction files...")
                            shutil.rmtree(temp_d, ignore_errors=True)

                    self._on_state_change()
                    self._refresh_all_carousels(segment_show_results=True)
                except Exception as e:
                    try:
                        self._runner.log(f"WARNING after job success UI refresh: {e}")
                    except Exception:
                        pass

                if do_chain_contour:
                    self._start_contour()
                elif do_chain_analyze:
                    self._start_analyze()
            elif result.status == JobStatus.CANCELLED:
                self._log_panel.set_status("Cancelled", "#b8860b")
                if getattr(self, "_is_interactive_finalize", False):
                    self._is_interactive_finalize = False
                    self._segment_tab.refresh_current_image()
                if getattr(self, "_is_reanalyze", False):
                    self._is_reanalyze = False
                    self._segment_tab.cleanup_reanalyze_temp()
                    self._segment_tab.refresh_current_image()
                if self._skip_segmentation_prep:
                    self._project_tab.revert_skip_segmentation()
                    self._contour_tab.set_contour_ready_hint("")
                    self._update_tab_access()
                if self._active_job_panel:
                    self._job_panel_fail(self._active_job_panel, "Cancelled")
                self._clear_chain()
            else:
                self._log_panel.set_status("Error", "#8B0000")
                if getattr(self, "_is_interactive_finalize", False):
                    self._is_interactive_finalize = False
                    self._segment_tab.refresh_current_image()
                if getattr(self, "_is_reanalyze", False):
                    self._is_reanalyze = False
                    self._segment_tab.cleanup_reanalyze_temp()
                    self._segment_tab.refresh_current_image()
                if self._skip_segmentation_prep:
                    self._project_tab.revert_skip_segmentation()
                    self._update_tab_access()
                if self._active_job_panel:
                    self._job_panel_fail(self._active_job_panel, "Error")
                self._clear_chain()
                messagebox.showerror(
                    "Process Error",
                    result.message or f"Exit code {result.returncode}",
                )

        self.after(0, finish)

    def _clear_chain(self) -> None:
        self._after_fastsam = False
        self._after_contour_copy = False
        self._chain_contour = False
        self._chain_analyze = False
        self._skip_segmentation_prep = False
        self._active_job_panel = None

    def _run_segmentation(self, *, chain_contour: bool = False, chain_analyze: bool = False) -> None:
        if self._runner.is_running:
            return
        self._sync_all()
        if self._state.skip_segmentation:
            messagebox.showinfo(
                "Segmentation disabled",
                "Segmentation is disabled.\n\n"
                "Use the 'Skip segmentation' option on the Project tab, "
                "or uncheck it to run the normal pipeline.",
            )
            return
        if not self._segment_tab._validate():
            return
        method = self._state.validate_segmentation_method()
        if method == "birefnet_mobilesam":
            self._run_birefnet_inprocess(
                chain_contour=chain_contour,
                chain_analyze=chain_analyze,
            )
            return
        self._chain_contour = chain_contour
        self._chain_analyze = chain_analyze
        self._job_panel_start("segment")
        self._set_status("Running...", "#1f6aa5")
        steps = self._segment_tab.build_pipeline_steps(self._runner.log)
        if len(steps) >= 2:
            self._after_fastsam = True
            title, script, args = steps[0]
            self._runner.run_script(script, args, title=title)
        elif len(steps) == 1:
            self._after_fastsam = False
            title, script, args = steps[0]
            self._runner.run_script(script, args, title=title)
        else:
            self._job_panel_fail("segment", "Nothing to run")
            self._runner.log("Nothing to run in segmentation.")

    def _start_whitebg(self) -> None:
        if self._runner.is_running:
            return
        self._set_status("Running whitebg...", "#1f6aa5")
        self._runner.run_script(
            script_path("segmentation/whitebg_masks.py"),
            build_whitebg_args(self._state),
            title="White background + masks (whitebg_masks)",
        )

    def _start_contour(self, filename: str | None = None) -> None:
        if self._runner.is_running:
            return
        self._contour_tab.sync_to_state()
        self._after_contour_copy = True
        self._job_panel_start("contour")
        status_msg = f"Running contour on {filename}..." if filename else "Running contour..."
        self._set_status(status_msg, "#1f6aa5")
        title, script, args = self._contour_tab.build_steps(filename=filename)[0]
        self._runner.run_script(script, args, title=title)

    def _start_analyze(self) -> None:
        if self._runner.is_running:
            return
        self._analyze_tab.sync_to_state()
        if not self._analyze_tab._validate():
            self._clear_chain()
            return
        # Run scale detection in-process just before analysis
        run_scale_detection(self._state, self._runner.log)
        self._job_panel_start("analyze")
        self._set_status("Running analysis...", "#1f6aa5")
        title, script, args = self._analyze_tab.build_steps()[0]
        self._runner.run_script(script, args, title=title)

    def _run_contour(self, filename: str | None = None) -> None:
        if self._runner.is_running:
            return
        self._sync_all()
        if not self._contour_tab._validate():
            return
        self._chain_contour = False
        self._chain_analyze = False
        # When re-analyzing a single image from Tab 3, delete its prior analysis
        # output so the carousel only shows the result from the new method.
        if filename is not None:
            out_root = self._state.output_path()
            if out_root is not None:
                from pathlib import Path
                from gui.paths import analyzed_dir, unlink_analyzed_artifacts
                stem = Path(filename).stem
                _analyzed = analyzed_dir(out_root)
                for _v in [stem, f"{stem}_white_bg",
                           stem.removesuffix("_white_bg") if stem.endswith("_white_bg") else None]:
                    if _v:
                        unlink_analyzed_artifacts(_analyzed, _v)
        self._start_contour(filename=filename)

    def _run_analyze(self) -> None:
        if self._runner.is_running:
            return
        self._sync_all()
        if not self._analyze_tab._validate():
            return
        self._chain_contour = False
        self._chain_analyze = False
        self._start_analyze()

    def _run_skip_segmentation_prep(
        self, *, chain_contour: bool = False, chain_analyze: bool = False
    ) -> bool:
        if self._runner.is_running:
            messagebox.showwarning("Busy", "Wait for the current job to finish.")
            return False
        self._sync_all()
        if not self._project_tab.validate_for_pipeline():
            return False
        args = build_skip_segmentation_args(self._state)
        if not args:
            messagebox.showerror("Error", "Could not build skip-segmentation arguments.")
            return False
        self._chain_contour = chain_contour
        self._chain_analyze = chain_analyze
        self._skip_segmentation_prep = True
        self._skip_prep_ui_start()
        self._job_panel_start("skip_prep", message="Analyzing, please wait...")
        out_size = self._state.segmentation_output_size() or 1024
        self._runner.run_script(
            script_path("segmentation/whitebg_masks.py"),
            args,
            title=f"Skip segmentation prep (whitebg_masks {out_size}×{out_size})",
        )
        return True

    def _on_close(self) -> None:
        if self._runner.is_running:
            if not messagebox.askyesno("Exit", "A process is running. Exit anyway?"):
                return
            self._runner.cancel()
        self._sync_all()
        self.destroy()

