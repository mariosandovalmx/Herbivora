"""Project state and configuration persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from gui.paths import (
    PIPELINE_RESOLUTION,
    CONTOUR_METHOD_CHOICES,
    DEFAULT_DAMAGE_MODEL,
    DEFAULT_LEAF_MODEL,
    DEFAULT_MOBILESAM,
    DEFAULT_UNET_SHAPE_CONFIG,
    DEFAULT_UNET_SHAPE_MODEL,
    LEAF_REFINE_CHOICES,
    MORPHOLOGY_MODEL_FILES,
    REPO_ROOT,
    SEGMENT_METHOD_CHOICES,
)


def _config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata) / "HerbivoraGUI"
    else:
        base = Path.home() / ".herbivora_gui"
    base.mkdir(parents=True, exist_ok=True)
    return base / "config.json"


@dataclass
class ProjectState:
    input_dir: str = ""
    output_dir: str = ""
    mobilesam_model: str = str(DEFAULT_MOBILESAM)
    damage_model: str = str(DEFAULT_DAMAGE_MODEL)
    leaf_model: str = str(DEFAULT_LEAF_MODEL)

    # Segmentation
    segmentation_method: str = "birefnet_mobilesam"
    skip_segmentation: bool = False
    multi_leaf_photos: bool = False  # Project: multiple leaves per photo → interactive multi-click
    skip_fastsam: bool = False
    fastsam_max_leaves: int | None = 2
    fastsam_output_size: int = PIPELINE_RESOLUTION
    fastsam_min_overlap: float = 0.16
    fastsam_dark_ratio_threshold: float = 0.50
    fastsam_dark_value_threshold: int = 62
    fastsam_component_margin_px: int = 4
    fastsam_strict_crop: bool = True
    fastsam_conf: float = 0.35
    fastsam_imgsz: int = PIPELINE_RESOLUTION
    fastsam_prior_dilate_px: int = 35
    fastsam_max_area_ratio: float = 0.75
    segment_advanced_expanded: bool = False
    analyze_advanced_expanded: bool = False
    stressed_leaves: bool = True
    remove_blue: bool = True
    reject_dark_artifacts: bool = True
    normalize_bg: bool = True
    # Interactive (method C): "leaf_only" | "leaf_scale"
    interactive_click_mode: str = "leaf_scale"
    # Legacy path field (kept so old config.json still loads)
    fastsam_model: str = ""

    # BiRefNet + MobileSAM segmentation
    birefnet_known_diameter_mm: float = 6.0
    birefnet_hybrid_mode: str = "birefnet_primary"
    birefnet_seg_resolution: int = PIPELINE_RESOLUTION
    birefnet_output_size: int = PIPELINE_RESOLUTION
    birefnet_agreement_threshold: float = 0.85

    def segmentation_output_size(self) -> int:
        """Fixed leaf canvas size for Segmentation → Contour → Analysis."""
        return PIPELINE_RESOLUTION

    def apply_fixed_pipeline_resolution(self) -> None:
        """Force pipeline sizes to PIPELINE_RESOLUTION (ignores legacy config values)."""
        self.birefnet_output_size = PIPELINE_RESOLUTION
        self.birefnet_seg_resolution = PIPELINE_RESOLUTION
        self.fastsam_output_size = PIPELINE_RESOLUTION
        self.fastsam_imgsz = PIPELINE_RESOLUTION
        self.unet_size = PIPELINE_RESOLUTION

    # Contour
    contour_method: str = "recon_unet_shape"
    contour_morphology: str = "auto"
    leaf_refine: str = "complete"
    leaf_img_size: int = 384
    leaf_normalize_bg: bool = True
    leaf_smooth_contour: bool = True
    roi_mode: str = "filled"
    recon_device: str = "cpu"
    recon_unet_refine: str = "complete"
    recon_model_unet_shape: str = str(DEFAULT_UNET_SHAPE_MODEL)
    recon_unet_shape_config: str = str(DEFAULT_UNET_SHAPE_CONFIG)
    # Per-morphology Contour specialists (Contour tab picks these automatically).
    recon_model_shape_smooth: str = str(MORPHOLOGY_MODEL_FILES["smooth"])
    recon_model_shape_serrated: str = str(MORPHOLOGY_MODEL_FILES["serrated"])
    recon_model_shape_lobed: str = str(MORPHOLOGY_MODEL_FILES["lobed"])
    recon_model_shape_compound: str = str(MORPHOLOGY_MODEL_FILES["compound"])

    # Analysis
    unet_size: int = PIPELINE_RESOLUTION
    fill_marginal: bool = True
    draw_hull_line: bool = False
    edge_artifact_filter: bool = True
    edge_min_inward_px: float = 3.5
    white_hole_brightness: int = 235
    white_hole_min_area: int = 3
    white_hole_edge_band: int = 2
    white_hole_adaptive: bool = True
    superficial_damage: bool = True
    report_area_cm2: bool = False
    scale_area_cm2: float = 0.2827  # area of a 6.0mm-diameter (0.6cm) blue reference dot

    def input_path(self) -> Path | None:
        return Path(self.input_dir) if self.input_dir else None

    def output_path(self) -> Path | None:
        if self.output_dir:
            return Path(self.output_dir)
        if self.input_dir:
            return Path(self.input_dir)
        return None

    def validate_roi_mode(self) -> str:
        """Analysis always uses the Contour mask (filled silhouette), never hull/closed."""
        return "filled"

    def validate_leaf_refine(self) -> str:
        if self.leaf_refine in LEAF_REFINE_CHOICES:
            return self.leaf_refine
        return "bridged"

    def validate_contour_method(self) -> str:
        if self.contour_method in CONTOUR_METHOD_CHOICES:
            return self.contour_method
        return "recon_unet_shape"

    def validate_segmentation_method(self) -> str:
        if self.segmentation_method in SEGMENT_METHOD_CHOICES:
            return self.segmentation_method
        if self.segmentation_method == "fastsam":
            return "birefnet_mobilesam"
        return "birefnet_mobilesam"


# User-facing checkboxes: never restored from disk; always unchecked on each GUI launch.
_SESSION_CHECKBOX_FIELDS: tuple[str, ...] = (
    "skip_segmentation",
    "multi_leaf_photos",
    "remove_blue",
    "stressed_leaves",
    "reject_dark_artifacts",
    "fastsam_strict_crop",
    "white_hole_adaptive",
    "segment_advanced_expanded",
    "analyze_advanced_expanded",
)


def reset_session_checkboxes(state: ProjectState) -> None:
    """Force all GUI checkboxes off at the start of each application session."""
    for key in _SESSION_CHECKBOX_FIELDS:
        setattr(state, key, False)
    state.interactive_click_mode = ""


def load_config() -> ProjectState:
    path = _config_path()
    if not path.is_file():
        state = ProjectState()
        reset_session_checkboxes(state)
        return state
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = ProjectState()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        # Always start with empty input/output paths
        state.input_dir = ""
        state.output_dir = ""
        state.apply_fixed_pipeline_resolution()
        reset_session_checkboxes(state)
        return state
    except (json.JSONDecodeError, OSError):
        state = ProjectState()
        reset_session_checkboxes(state)
        return state


def save_config(state: ProjectState) -> None:
    path = _config_path()
    state.apply_fixed_pipeline_resolution()
    data = asdict(state)
    for key in _SESSION_CHECKBOX_FIELDS:
        data.pop(key, None)
    data.pop("interactive_click_mode", None)
    data["repo_root"] = str(REPO_ROOT)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
