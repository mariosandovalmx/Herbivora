"""Shared pipeline logic for GUI subprocess steps."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Callable

from gui.paths import (
    REPO_ROOT,
    is_preextracted_leaf,
    leaf_roi_preview_dir,
    list_images,
    masks_dir,
    segmentation_dir,
    white_bg_dir,
    work_dir,
)
from dataclasses import dataclass

from gui.state import ProjectState


SCALE_JSON_NAME = "scale_reference.json"
CIRCLE_OVERRIDES_JSON_NAME = "circle_overrides.json"
SKIP_SEGMENTATION_OUTPUT_SIZE = 1024  # fixed pipeline resolution


LogFn = Callable[[str], None]


@dataclass
class FastSamPrep:
    fastsam_input: Path
    fastsam_output: Path
    white_bg_dir: Path
    n_scene_images: int


def copy_leaf_masks_to_segmentation(output_root: Path, log: LogFn | None = None) -> int:
    """Copies edited contour ROI masks (leaf_roi_preview) to segmentation/masks/.

    Returns number of copied files.
    """
    preview = leaf_roi_preview_dir(output_root)
    src_masks = preview / "masks"
    dest = masks_dir(output_root)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if not src_masks.is_dir():
        if log:
            log(f"Masks folder does not exist: {src_masks}")
        return 0
    for mf in src_masks.glob("*_leaf_mask.png"):
        stem = mf.stem
        if stem.endswith("_leaf_mask"):
            stem = stem[: -len("_leaf_mask")]
        leaf_id = stem[: -len("_white_bg")] if stem.lower().endswith("_white_bg") else stem
        shutil.copy2(mf, dest / f"{leaf_id}_mask.png")
        n += 1
    if log:
        log(f"Copied {n} masks to {dest}")
    return n


def prepare_fastsam_dirs(state: ProjectState, log: LogFn) -> FastSamPrep | None:
    """Copies images to fastsam_input. None if FastSAM is not needed."""
    inp = state.input_path()
    out_root = state.output_path()
    if inp is None or not inp.is_dir():
        raise ValueError("Select a valid input folder.")
    if out_root is None:
        raise ValueError("Select an output folder.")

    wd = work_dir(out_root)
    out_wb = wd / "white_bg"
    out_wb.mkdir(parents=True, exist_ok=True)

    images = list_images(inp)
    if not images:
        raise ValueError(f"No images found in {inp}")

    preextracted = [p for p in images if is_preextracted_leaf(p)]
    need_fastsam = [p for p in images if not is_preextracted_leaf(p)]
    log(
        f"Input: {len(images)} files | already isolated: {len(preextracted)} | "
        f"FastSAM: {len(need_fastsam)}"
    )

    for p in preextracted:
        shutil.copy2(p, out_wb / p.name)

    if not need_fastsam:
        log("All images are already individual leaves.")
        return None

    shutil.rmtree(wd / "fastsam_input", ignore_errors=True)
    shutil.rmtree(wd / "fastsam_run", ignore_errors=True)
    fastsam_in = wd / "fastsam_input"
    fastsam_tmp = wd / "fastsam_run"
    fastsam_in.mkdir(parents=True)

    for p in need_fastsam:
        shutil.copy2(p, fastsam_in / p.name)

    return FastSamPrep(
        fastsam_input=fastsam_in,
        fastsam_output=fastsam_tmp,
        white_bg_dir=out_wb,
        n_scene_images=len(need_fastsam),
    )


def build_fastsam_args(state: ProjectState, prep: FastSamPrep) -> list[str]:
    method = "fastsam"
    model_path = state.fastsam_model
    args = [
        "--input", str(prep.fastsam_input),
        "--output", str(prep.fastsam_output),
        "--model-type", method,
        "--model", str(Path(model_path).resolve()),
        "--output-size", str(state.fastsam_output_size),
        "--min-overlap", str(state.fastsam_min_overlap),
        "--dark-ratio-threshold", str(state.fastsam_dark_ratio_threshold),
        "--dark-value-threshold", str(state.fastsam_dark_value_threshold),
        "--component-margin-px", str(state.fastsam_component_margin_px),
        "--conf", str(state.fastsam_conf),
        "--imgsz", str(state.fastsam_imgsz),
        "--prior-dilate-px", str(state.fastsam_prior_dilate_px),
        "--max-area-ratio", str(state.fastsam_max_area_ratio),
    ]
    if state.remove_blue:
        args.append("--remove-blue")
    if state.fastsam_strict_crop:
        args.append("--strict-crop")
    else:
        args.append("--no-strict-crop")
    if state.reject_dark_artifacts:
        args.append("--reject-dark-artifacts")
    if state.stressed_leaves:
        args.append("--stressed-leaves")
    if state.fastsam_max_leaves is not None:
        args.extend(["--max-leaves", str(state.fastsam_max_leaves)])
    args.append("--allow-non-venv")
    return args


def needs_fastsam_step(state: ProjectState) -> bool:
    if state.skip_fastsam:
        return False
    inp = state.input_path()
    if inp is None:
        return False
    images = list_images(inp)
    return any(not is_preextracted_leaf(p) for p in images)


def after_fastsam_merge(state: ProjectState, log: LogFn) -> Path:
    """After FastSAM, copies white_bg to the work directory."""
    out_root = state.output_path()
    assert out_root is not None
    wd = work_dir(out_root)
    fastsam_tmp = wd / "fastsam_run"
    out_wb = wd / "white_bg"
    out_wb.mkdir(parents=True, exist_ok=True)

    extracted = list_images(fastsam_tmp / "white_bg")
    for p in extracted:
        shutil.copy2(p, out_wb / p.name)

    # Also copy preextracted if not already there
    inp = state.input_path()
    if inp:
        for p in list_images(inp):
            if is_preextracted_leaf(p):
                dest = out_wb / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)

    total = len(list_images(out_wb))
    if total == 0:
        raise RuntimeError("No leaves generated in white_bg/. Check FastSAM and input photos.")
    log(f"Total leaves ready: {total} in {out_wb}")
    return out_wb


def resolve_whitebg_input(state: ProjectState) -> Path:
    """Folder feeding whitebg_masks."""
    if state.skip_fastsam:
        inp = state.input_path()
        if inp is None:
            raise ValueError("Input folder not defined.")
        return inp
    out_root = state.output_path()
    if out_root is None:
        raise ValueError("Output folder not defined.")
    wb = work_dir(out_root) / "white_bg"
    if wb.is_dir() and list_images(wb):
        return wb
    inp = state.input_path()
    if inp and all(is_preextracted_leaf(p) for p in list_images(inp)):
        return inp
    return wb


def build_whitebg_args(state: ProjectState, *, output_size: int | None = None) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    seg = segmentation_dir(out_root)
    wb_input = resolve_whitebg_input(state)
    size = output_size if output_size is not None else state.segmentation_output_size()
    # Scale calibration is measured on the user's photos, so it can only be mapped
    # into white_bg pixels when whitebg_masks reads those photos rather than the
    # FastSAM crops.
    inp = state.input_path()
    scale_source = "original_photo" if inp is not None and wb_input == inp else "derived_image"
    args = [
        "--input", str(wb_input),
        "--output", str(seg),
        "--clean-output",
        "--scale-source", scale_source,
    ]
    if size is not None and size > 0:
        args.extend(["--output-size", str(size)])
    if state.remove_blue:
        args.append("--remove-blue")
    return args


def build_skip_segmentation_args(state: ProjectState) -> list[str]:
    """whitebg_masks on Project input → segmentation/ at configured output size."""
    inp = state.input_path()
    out_root = state.output_path()
    if inp is None or out_root is None:
        return []
    out_size = state.segmentation_output_size() or SKIP_SEGMENTATION_OUTPUT_SIZE
    args = [
        "--input", str(inp),
        "--output", str(segmentation_dir(out_root)),
        "--clean-output",
        "--output-size", str(out_size),
        "--scale-source", "original_photo",
    ]
    if state.remove_blue:
        args.append("--remove-blue")
    return args


def needs_skip_segmentation_prep(state: ProjectState) -> bool:
    """True when skip mode is on and segmentation/white_bg has no images."""
    if not state.skip_segmentation:
        return False
    out_root = state.output_path()
    if out_root is None:
        return True
    wb = white_bg_dir(out_root)
    return not wb.is_dir() or not list_images(wb)



def build_geometric_contour_args(state: ProjectState) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    return [
        "--input", str(white_bg_dir(out_root)),
        "--masks-dir", str(masks_dir(out_root)),
        "--output", str(leaf_roi_preview_dir(out_root)),
    ]


def build_recon_unet_shape_args(state: ProjectState) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    from gui.paths import DEFAULT_UNET_SHAPE_MODEL, DEFAULT_UNET_SHAPE_CONFIG
    args = [
        "--input", str(white_bg_dir(out_root)),
        "--output", str(leaf_roi_preview_dir(out_root)),
        "--device", getattr(state, "recon_device", "cpu"),
    ]
    ckpt = getattr(state, "recon_model_unet_shape", "").strip().strip("\"'")
    if not ckpt:
        ckpt = getattr(state, "leaf_model", "").strip().strip("\"'")
    cfg = getattr(state, "recon_unet_shape_config", "").strip().strip("\"'")
    p_ckpt = Path(ckpt) if ckpt else DEFAULT_UNET_SHAPE_MODEL
    p_cfg = Path(cfg) if cfg else DEFAULT_UNET_SHAPE_CONFIG
    if p_ckpt.is_file():
        args.extend(["--checkpoint", str(p_ckpt)])
    if p_cfg.is_file():
        args.extend(["--config", str(p_cfg)])
    return args


def build_contour_step(state: ProjectState, filename: str | None = None) -> tuple[str, Path, list[str]]:
    """Returns (title, script, args) for the Contour UNET Shape step."""
    step = (
        "UNET Shape (Mask-to-Mask 512px)",
        script_path("contour/inference/gui_batch.py"),
        build_recon_unet_shape_args(state),
    )
    if filename:
        step[2].extend(["--file", filename])
    return step


def _load_circle_overrides(out_root: Path) -> dict:
    import json as _json
    path = out_root / CIRCLE_OVERRIDES_JSON_NAME
    if not path.is_file():
        return {}
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return {}


def has_circle_override(out_root: Path, image_id: str) -> bool:
    return image_id in _load_circle_overrides(out_root)


def save_circle_override(
    out_root: Path, image_id: str, cx: float, cy: float, diameter_px: float
) -> None:
    """Persist a manual blue-dot correction, keyed by image_id.

    Manual overrides always take precedence over automatic detection in
    run_scale_detection(), so a user-corrected dot isn't silently overwritten
    by a future automatic re-run.
    """
    import json as _json
    overrides = _load_circle_overrides(out_root)
    overrides[image_id] = {
        "center_px": [cx, cy],
        "diameter_px": diameter_px,
        "method": "manual",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / CIRCLE_OVERRIDES_JSON_NAME).write_text(
        _json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _white_bg_scale(meta: dict, *, from_original: bool) -> float | None:
    """Linear factor converting source pixels into white_bg pixels.

    Segmentation writes ``scale_factor`` (measurement space → white_bg) and
    ``pre_ratio`` (source photo → measurement space). Calibrations taken on the
    user's photo (manual overrides, scale_detect scans) need both; the circle
    recorded by BiRefNet is already in measurement space and needs only the
    former. Returns None when the metadata cannot describe the conversion, so
    callers report % only instead of a calibration in the wrong pixel space.
    """
    try:
        scale_factor = float(meta.get("scale_factor"))
    except (TypeError, ValueError):
        return None
    if not scale_factor > 0:
        return None
    if not from_original:
        return scale_factor
    # Metadata predating scale_source was always produced from the user's photo.
    if meta.get("scale_source", "original_photo") != "original_photo":
        return None
    try:
        pre_scale = float((meta.get("pre_ratio") or [1.0, 1.0])[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not pre_scale > 0:
        return None
    return pre_scale * scale_factor


def run_scale_detection(state: ProjectState, log: LogFn) -> Path | None:
    """Build scale_reference.json from BiRefNet metadata or direct blue dot scan.

    Primary path: reads mm2_per_px2 from segmentation/metadata/*.json (BiRefNet output).
    Fallback: scans original input images with scale_detect.scan_folder.
    Returns path to scale_reference.json, or None when no scale found.
    Skipped entirely when Project → "Scale reference in photo" is off (% only).
    """
    out_root = state.output_path()
    if out_root is None:
        return None

    if not state.remove_blue:
        log("Scale: reference disabled in Project — reporting damage % only.")
        return None

    import json as _json
    json_path = out_root / SCALE_JSON_NAME
    meta_dir = segmentation_dir(out_root) / "metadata"

    scale_entries: dict[str, float] = {}
    overrides = _load_circle_overrides(out_root)

    # Primary: reuse circle detection already done by BiRefNet during segmentation
    if meta_dir.is_dir():
        meta_files = sorted(meta_dir.glob("*.json"))
        if meta_files:
            log(f"Scale: reading circle data from {len(meta_files)} segmentation metadata file(s)...")
            n_manual = 0
            for mf in meta_files:
                try:
                    meta = _json.loads(mf.read_text(encoding="utf-8"))
                except (OSError, _json.JSONDecodeError):
                    continue
                image_id = meta.get("image_id", mf.stem)
                # Segmentation crops to the leaf's bbox then resizes that crop into
                # the white_bg image, while analyze_leaves.py counts area in white_bg
                # pixels. Every cm²/px² therefore has to be divided by the squared
                # linear factor for its own pixel space before being written out.
                override = overrides.get(image_id)
                if override:
                    try:
                        diameter_px = float(override["diameter_px"])
                    except (KeyError, TypeError, ValueError):
                        diameter_px = 0.0
                    if not diameter_px > 0:
                        log(f"Scale: ignoring manual dot for '{image_id}' — invalid diameter.")
                        continue
                    factor = _white_bg_scale(meta, from_original=True)
                    if factor is None:
                        log(
                            f"Scale: skipping '{image_id}' — cannot map the manual dot "
                            "into white_bg pixels."
                        )
                        continue
                    known_area_mm2 = math.pi * (state.birefnet_known_diameter_mm / 2) ** 2
                    circle_area_px = math.pi * (diameter_px / 2) ** 2
                    cm2_per_px2 = (known_area_mm2 / circle_area_px) / 100.0  # mm² → cm²
                    scale_entries[image_id] = cm2_per_px2 / factor ** 2
                    n_manual += 1
                else:
                    circle = meta.get("circle", {})
                    mm2_per_px2 = circle.get("mm2_per_px2")
                    if mm2_per_px2 and circle.get("found"):
                        factor = _white_bg_scale(meta, from_original=False)
                        if factor is None:
                            log(
                                f"Scale: skipping '{image_id}' — segmentation metadata "
                                "has no usable scale_factor."
                            )
                            continue
                        cm2_per_px2 = (float(mm2_per_px2) / 100.0) / factor ** 2
                        # Stored as bare stem; lookup_scale_factor handles extension/suffix variants
                        scale_entries[image_id] = cm2_per_px2
            detected = len(scale_entries)
            manual_note = f" ({n_manual} manual override(s))" if n_manual else ""
            log(f"Scale: blue dot found in {detected}/{len(meta_files)} image(s) via segmentation metadata{manual_note}.")
            if detected == 0:
                log("Scale: no blue dots in segmentation metadata — trying direct scan...")

    # Fallback: scan original images when BiRefNet metadata has no circle data
    if not scale_entries:
        inp = state.input_path()
        if inp is None:
            log("Scale: no input folder set — skipping scale detection.")
            return None
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT))
            from scale_detect import scan_folder  # noqa: E402
        except ImportError as e:
            log(f"WARNING: Could not import scale_detect: {e}")
            return None
        known_cm2 = state.scale_area_cm2
        log(f"Scale: scanning {inp} for blue reference dots (known area={known_cm2} cm²)...")
        results = scan_folder(inp, known_cm2=known_cm2)
        detected = sum(1 for v in results.values() if v is not None)
        log(f"Scale: found blue dot in {detected}/{len(results)} image(s).")
        if detected == 0:
            log("WARNING: No blue dots detected. Results will be reported as % only.")
            return None
        # scan_folder() measures the dot on the ORIGINAL photo, so every entry must
        # be converted into white_bg-pixel space using that image's segmentation
        # metadata. Without it the conversion is unknown, and emitting the
        # original-space value would silently report wrong cm² areas.
        scale_entries = {}
        unmapped: list[str] = []
        for image_name, cm2_per_px2_orig in results.items():
            if cm2_per_px2_orig is None:
                continue
            meta = {}
            mf = meta_dir / f"{Path(image_name).stem}.json"
            if mf.is_file():
                try:
                    meta = _json.loads(mf.read_text(encoding="utf-8"))
                except (OSError, _json.JSONDecodeError):
                    meta = {}
            factor = _white_bg_scale(meta, from_original=True)
            if factor is None:
                unmapped.append(image_name)
                continue
            scale_entries[image_name] = cm2_per_px2_orig / factor ** 2
        if unmapped:
            log(
                f"WARNING: {len(unmapped)} image(s) have no usable segmentation metadata "
                "to convert the blue dot into white_bg pixels — reported as % only."
            )

    if not scale_entries:
        log("Scale: no scale data found — reporting % only.")
        return None

    out_data: dict = dict(scale_entries)

    json_path.write_text(
        _json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Scale reference saved: {json_path}")
    return json_path


def scale_json_path(output_root: Path) -> Path:
    return output_root / SCALE_JSON_NAME


def build_analyze_args(state: ProjectState) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    args = [
        "--segmentation-dir", str(segmentation_dir(out_root)),
        "--out-dir", str(out_root / "analyzed"),
        "--unet-path", str(Path(state.damage_model).resolve()),
        "--unet-size", str(state.unet_size),
        "--roi-mode", state.validate_roi_mode(),
    ]
    # fill_marginal + edge_artifact_filter are always enabled (no GUI toggles).
    # draw_hull_line is always off (option removed from GUI).
    state.fill_marginal = True
    state.edge_artifact_filter = True
    state.draw_hull_line = False
    # Clamp only extreme weak values; do not force the previous over-aggressive floors.
    state.edge_min_inward_px = float(state.edge_min_inward_px or 3.5)
    if state.edge_min_inward_px < 2.5 or state.edge_min_inward_px > 6.0:
        state.edge_min_inward_px = 3.5
    state.white_hole_edge_band = int(state.white_hole_edge_band or 2)
    if state.white_hole_edge_band < 1:
        state.white_hole_edge_band = 2
    # Cap legacy over-aggressive configs (5–8) that were eating real holes/notches.
    if state.white_hole_edge_band > 3:
        state.white_hole_edge_band = 2
    args.extend(["--edge-min-inward-px", str(state.edge_min_inward_px)])
    args.extend(["--white-hole-edge-band", str(state.white_hole_edge_band)])
    if state.white_hole_min_area != 3:
        args.extend(["--white-hole-min-area", str(state.white_hole_min_area)])
    if not state.white_hole_adaptive:
        args.append("--no-white-hole-adaptive")
        if state.white_hole_brightness != 235:
            args.extend(["--white-hole-brightness", str(state.white_hole_brightness)])
    elif state.white_hole_brightness != 235:
        # Soft hint for AUTO mode (tissue seed blend).
        args.extend(["--white-hole-brightness", str(state.white_hole_brightness)])
    # Scale file only when Project checkbox is on (else damage % only).
    if state.remove_blue:
        json_p = scale_json_path(out_root)
        if json_p.is_file():
            args.extend(["--scale-file", str(json_p)])
        if state.report_area_cm2:
            args.extend(["--scale-area-cm2", str(state.scale_area_cm2)])
    return args


def build_birefnet_args(
    state: ProjectState,
    *,
    input_override: Path | None = None,
    box_prior_dir: Path | None = None,
    point_prior_dir: Path | None = None,
) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    seg = segmentation_dir(out_root)
    inp = input_override if input_override is not None else state.input_path()
    if inp is None:
        return []
    args = [
        "--input", str(inp),
        "--output", str(seg),
        "--known-diameter-mm", str(state.birefnet_known_diameter_mm),
        "--hybrid-mode", state.birefnet_hybrid_mode,
        "--seg-resolution", str(state.birefnet_seg_resolution),
        "--output-size", str(state.segmentation_output_size()),
        "--agreement-threshold", str(state.birefnet_agreement_threshold),
    ]
    if state.remove_blue:
        args.append("--remove-blue")
    else:
        args.append("--no-remove-blue")
    if box_prior_dir is not None:
        args.extend(["--box-prior-dir", str(box_prior_dir)])
    if point_prior_dir is not None:
        args.extend(["--point-prior-dir", str(point_prior_dir)])
    ms = getattr(state, "mobilesam_model", "").strip().strip("\"'")
    if ms:
        args.extend(["--mobilesam-weights", str(Path(ms).resolve())])
    args.append("--allow-non-venv")
    return args


def build_intact_args(state: ProjectState) -> list[str]:
    out_root = state.output_path()
    if out_root is None:
        return []
    seg = segmentation_dir(out_root)
    inp = state.input_path()
    if inp is None:
        return []
    return [
        "--input", str(inp),
        "--output", str(seg),
        "--sat-min", "20",
        "--close-k", "7",
        "--preview", "0",
        "--output-size", str(state.segmentation_output_size() or 1024),
        "--scale-source", "original_photo",
    ]


def script_path(name: str) -> Path:
    return REPO_ROOT / name

