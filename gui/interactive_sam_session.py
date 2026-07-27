"""In-GUI MobileSAM session for C. Interactive segmentation.

Loads MobileSAM once (and BiRefNet once for the final batch) so models are
not reloaded for every photo. Selections are click points + preview masks
stored per image stem until "Run segmentation" finalizes with BiRefNet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_BIREFNET_DIR = _REPO / "segmentation" / "birefnet_mobilesam"


@dataclass
class InteractiveSelection:
    path: Path
    x: int
    y: int
    mask: np.ndarray  # bool HxW at original image resolution
    # Blue reference circle for scale (original-image coords)
    circle_cx: float | None = None
    circle_cy: float | None = None
    circle_diameter: float | None = None
    circle_method: str | None = None
    circle_mm2_per_px2: float | None = None
    circle_low_confidence: bool = False

    @property
    def has_circle(self) -> bool:
        return (
            self.circle_cx is not None
            and self.circle_cy is not None
            and self.circle_diameter is not None
            and self.circle_diameter > 0
        )

    def circle_prior_dict(self) -> dict | None:
        """Format accepted by process_image(..., circle_prior=...)."""
        if not self.has_circle:
            return None
        import math

        cx = float(self.circle_cx)
        cy = float(self.circle_cy)
        d = float(self.circle_diameter)
        r = d / 2.0
        return {
            "found": True,
            "center_px": (cx, cy),
            "diameter_px": d,
            "area_px": math.pi * r * r,
            "circularity": 1.0,
            "bbox": (int(cx - r), int(cy - r), max(1, int(d)), max(1, int(d))),
            "method": self.circle_method or "interactive",
            "score": 1.0 if not self.circle_low_confidence else 0.3,
            "low_confidence": bool(self.circle_low_confidence),
            "mm_per_px": None,
            "mm2_per_px2": self.circle_mm2_per_px2,
        }


def _subtract_circle_from_mask(
    mask: np.ndarray, cx: float, cy: float, diameter: float
) -> np.ndarray:
    """Remove the reference-dot disk from a boolean leaf mask."""
    import cv2

    u8 = (mask.astype(bool).astype(np.uint8)) * 255
    r = max(1, int(round(diameter / 2.0)))
    cv2.circle(u8, (int(round(cx)), int(round(cy))), r, 0, -1)
    return u8 > 0


def _apply_circle_to_selection(
    sel: InteractiveSelection,
    circle: dict,
    *,
    subtract_from_mask: bool = True,
) -> InteractiveSelection:
    if not circle.get("found"):
        return sel
    center = circle["center_px"]
    cx, cy = float(center[0]), float(center[1])
    d = float(circle["diameter_px"])
    sel.circle_cx = cx
    sel.circle_cy = cy
    sel.circle_diameter = d
    sel.circle_method = str(circle.get("method", "auto"))
    sel.circle_mm2_per_px2 = circle.get("mm2_per_px2")
    sel.circle_low_confidence = bool(circle.get("low_confidence", False))
    if subtract_from_mask and sel.mask is not None:
        sel.mask = _subtract_circle_from_mask(sel.mask, cx, cy, d)
    return sel


def build_manual_circle_info(
    cx: float,
    cy: float,
    diameter_px: float,
    known_diameter_mm: float,
    *,
    method: str = "manual",
) -> dict:
    """Build a circle_info dict compatible with process_image / detect_blue_circle."""
    import math

    mm_per_px = known_diameter_mm / diameter_px if diameter_px > 0 else None
    known_area = math.pi * (known_diameter_mm / 2.0) ** 2
    circle_area = math.pi * (diameter_px / 2.0) ** 2
    mm2_per_px2 = known_area / circle_area if circle_area > 0 else None

    r = diameter_px / 2.0
    return {
        "found": True,
        "center_px": (float(cx), float(cy)),
        "diameter_px": float(diameter_px),
        "area_px": math.pi * r * r,
        "circularity": 1.0,
        "bbox": (
            int(cx - r),
            int(cy - r),
            max(1, int(diameter_px)),
            max(1, int(diameter_px)),
        ),
        "method": method,
        "score": 1.0,
        "low_confidence": False,
        "mm_per_px": mm_per_px,
        "mm2_per_px2": mm2_per_px2,
    }


def _circle_from_mask(
    mask: np.ndarray,
    known_diameter_mm: float,
    *,
    image_hw: tuple[int, int] | None = None,
) -> dict:
    """Fit a circle to a MobileSAM mask of the reference sticker."""
    import math

    import cv2

    u8 = (mask.astype(bool).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "low_confidence": True, "reason": "empty_mask"}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 20:
        return {"found": False, "low_confidence": True, "reason": "mask_too_small"}

    if image_hw is not None:
        H, W = image_hw
        img_area = float(max(1, H * W))
        frac = area / img_area
        # Sticker is small; reject leaf-sized / image-sized masks
        if frac > 0.08:
            return {
                "found": False,
                "low_confidence": True,
                "reason": "mask_too_large",
            }

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    diameter = float(radius) * 2.0
    peri = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0

    circle = build_manual_circle_info(
        float(cx), float(cy), diameter, known_diameter_mm, method="mobilesam"
    )
    circle["area_px"] = area
    circle["circularity"] = circularity
    circle["score"] = float(circularity)
    circle["low_confidence"] = circularity < 0.65
    return circle


@dataclass
class InteractiveSamSession:
    selections: dict[str, InteractiveSelection] = field(default_factory=dict)
    # Circles marked before a leaf click (stem -> circle_info)
    pending_circles: dict[str, dict] = field(default_factory=dict)
    _device: object | None = None
    _mobilesam: object | None = None
    _mobilesam_weights: str | None = None
    _birefnet: object | None = None
    _loading_sam: bool = False
    _loading_birefnet: bool = False

    @property
    def mobilesam_ready(self) -> bool:
        return self._mobilesam is not None

    @property
    def mobilesam(self):
        """Loaded MobileSAM model, or None if not ready."""
        return self._mobilesam

    @property
    def is_loading_sam(self) -> bool:
        return self._loading_sam

    @property
    def is_loading_birefnet(self) -> bool:
        return self._loading_birefnet

    @property
    def birefnet_ready(self) -> bool:
        return self._birefnet is not None

    @property
    def n_selected(self) -> int:
        return sum(
            1
            for sel in self.selections.values()
            if sel.mask is not None and sel.mask.size > 1 and int(sel.mask.sum()) > 0
        )

    def clear_selections(self) -> None:
        self.selections.clear()
        self.pending_circles.clear()

    def release_models(self) -> None:
        """Drop model handles (e.g. when leaving Interactive method)."""
        self._mobilesam = None
        self._mobilesam_weights = None
        self._birefnet = None
        self._device = None

    def clear_mobilesam(self) -> None:
        """Drop only MobileSAM so the next ensure_mobilesam reloads weights."""
        self._mobilesam = None
        self._mobilesam_weights = None

    def _ensure_path(self) -> None:
        import sys

        for p in (str(_BIREFNET_DIR), str(_REPO)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def ensure_mobilesam(
        self,
        log: Callable[[str], None] | None = None,
        weights: str | Path | None = None,
    ) -> None:
        import time

        self._ensure_path()
        from utils.segmentation_utils import MOBILESAM_WEIGHTS, get_device, load_mobilesam

        if weights is not None and str(weights).strip():
            resolved = str(Path(str(weights).strip().strip("\"'")).resolve())
        elif self._mobilesam is not None and self._mobilesam_weights:
            if log:
                log(f"MobileSAM already in memory ({Path(self._mobilesam_weights).name}).")
            return
        else:
            resolved = str(Path(MOBILESAM_WEIGHTS).resolve())

        # Wait if another thread is loading
        while self._loading_sam:
            time.sleep(0.05)

        if self._mobilesam is not None and self._mobilesam_weights == resolved:
            if log:
                log(f"MobileSAM already in memory ({Path(resolved).name}).")
            return

        self._loading_sam = True
        try:
            if log:
                log("Loading MobileSAM (kept in memory for this GUI session)...")
            if self._device is None:
                self._device = get_device()
            self._mobilesam = load_mobilesam(self._device, weights=resolved)
            self._mobilesam_weights = resolved
            if log:
                log(f"MobileSAM ready ({Path(resolved).name}).")
        finally:
            self._loading_sam = False

    def ensure_birefnet(self, log: Callable[[str], None] | None = None) -> None:
        import time

        while self._loading_birefnet:
            time.sleep(0.05)
        if self._birefnet is not None:
            if log:
                log("BiRefNet_lite already in memory.")
            return
        self._loading_birefnet = True
        try:
            self._ensure_path()
            from utils.segmentation_utils import get_device, load_birefnet

            if self._device is None:
                self._device = get_device()
            if log:
                log("Loading BiRefNet_lite (kept in memory for this GUI session)...")
            self._birefnet = load_birefnet(self._device)
            if log:
                log("BiRefNet ready.")
        finally:
            self._loading_birefnet = False

    def detect_circle_for_image(
        self,
        image_path: Path,
        known_diameter_mm: float = 6.0,
    ) -> dict:
        """Auto-detect blue reference circle (no SAM). Returns circle_info dict."""
        self._ensure_path()
        from image_io import load_bgr
        from run_pipeline import _load_cfg
        from utils.circle_utils import detect_blue_circle

        image = load_bgr(image_path)
        if image is None:
            return {"found": False, "low_confidence": True}
        cfg = _load_cfg(
            _BIREFNET_DIR / "config.yaml",
            {"known_diameter_mm": known_diameter_mm},
        )
        return detect_blue_circle(image, cfg)

    def circle_for_path(self, path: Path | None) -> dict | None:
        """Return circle_info for a path from selection or pending mark."""
        if path is None:
            return None
        stem = path.stem
        sel = self.selections.get(stem)
        if sel is not None and sel.has_circle:
            return sel.circle_prior_dict()
        pending = self.pending_circles.get(stem)
        if pending and pending.get("found"):
            return pending
        return None

    def set_circle_on_stem(
        self,
        stem: str,
        circle: dict,
        *,
        subtract_from_mask: bool = True,
    ) -> InteractiveSelection | None:
        """Attach circle to an existing selection, or store as pending until leaf click."""
        if not circle.get("found"):
            return None
        sel = self.selections.get(stem)
        if sel is None:
            self.pending_circles[stem] = circle
            return None
        self.pending_circles.pop(stem, None)
        return _apply_circle_to_selection(sel, circle, subtract_from_mask=subtract_from_mask)

    def predict_scale_click(
        self,
        image_path: Path,
        x: float,
        y: float,
        *,
        known_diameter_mm: float = 6.0,
        log: Callable[[str], None] | None = None,
    ) -> dict:
        """Click the blue sticker → MobileSAM mask → fitted scale circle."""
        self.ensure_mobilesam(log=log)
        self._ensure_path()
        from image_io import load_bgr
        from utils.segmentation_utils import run_mobilesam_point

        image = load_bgr(image_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        H, W = image.shape[:2]
        px = int(max(0, min(W - 1, round(x))))
        py = int(max(0, min(H - 1, round(y))))
        mask = run_mobilesam_point(image, self._mobilesam, point=(px, py))
        circle = _circle_from_mask(mask, known_diameter_mm, image_hw=(H, W))
        if not circle.get("found"):
            reason = circle.get("reason", "unknown")
            if log:
                log(f"  Scale MobileSAM failed ({reason}) at ({px},{py})")
            return circle

        stem = image_path.stem
        self.set_circle_on_stem(stem, circle, subtract_from_mask=True)
        if log:
            conf = "low-conf" if circle.get("low_confidence") else "ok"
            log(
                f"  Scale MobileSAM: d={circle['diameter_px']:.1f}px "
                f"circ={circle.get('circularity', 0):.2f} ({conf})"
            )
        return circle

    def predict_click(
        self,
        image_path: Path,
        x: float,
        y: float,
        *,
        known_diameter_mm: float = 6.0,
        detect_circle: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> InteractiveSelection:
        """Run MobileSAM at (x, y), auto-detect scale circle, store selection."""
        self.ensure_mobilesam(log=log)
        self._ensure_path()
        from image_io import load_bgr
        from utils.segmentation_utils import run_mobilesam_point

        image = load_bgr(image_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        H, W = image.shape[:2]
        px = int(max(0, min(W - 1, round(x))))
        py = int(max(0, min(H - 1, round(y))))
        mask = run_mobilesam_point(image, self._mobilesam, point=(px, py))

        stem = image_path.stem
        prev = self.selections.get(stem)
        sel = InteractiveSelection(
            path=image_path.resolve(),
            x=px,
            y=py,
            mask=mask.astype(bool),
        )

        pending = self.pending_circles.pop(stem, None)
        circle: dict | None = None
        if pending and pending.get("found"):
            circle = pending
        elif (
            prev is not None
            and prev.has_circle
            and prev.circle_method in ("manual", "mobilesam")
        ):
            circle = prev.circle_prior_dict()
        elif detect_circle:
            circle = self.detect_circle_for_image(image_path, known_diameter_mm)

        if circle and circle.get("found"):
            _apply_circle_to_selection(sel, circle, subtract_from_mask=True)
            if log:
                conf = "low-conf" if circle.get("low_confidence") else "ok"
                log(
                    f"  Scale circle: d={circle['diameter_px']:.1f}px "
                    f"method={circle.get('method')} ({conf})"
                )
        elif log and detect_circle:
            log("  Scale circle: not found (use Mark scale circle)")

        self.selections[stem] = sel
        if log:
            log(f"MobileSAM preview: {image_path.name} @ ({px},{py})  area={int(sel.mask.sum())} px")
        return sel

    def selection_for_path(self, path: Path | None) -> InteractiveSelection | None:
        if path is None:
            return None
        return self.selections.get(path.stem)

    def run_folder_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        known_diameter_mm: float = 6.0,
        hybrid_mode: str = "birefnet_primary",
        seg_resolution: int = 1024,
        output_size: int = 1024,
        agreement_threshold: float = 0.85,
        remove_blue: bool = True,
        mobilesam_weights: str | None = None,
        log: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[int, int]:
        """Run BiRefNet + MobileSAM on a folder, keeping models loaded in the GUI process."""
        from image_io import VALID_IMAGE_EXTENSIONS

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        if not input_dir.is_dir():
            raise RuntimeError(f"Input folder not found: {input_dir}")

        images = sorted(
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No images found in {input_dir}")

        self.ensure_mobilesam(log=log, weights=mobilesam_weights)
        self.ensure_birefnet(log=log)
        self._ensure_path()

        from run_pipeline import _load_cfg, process_image

        cfg = _load_cfg(
            _BIREFNET_DIR / "config.yaml",
            {
                "known_diameter_mm": known_diameter_mm,
                "hybrid_mode": hybrid_mode,
                "seg_resolution": seg_resolution,
                "output_size": output_size,
                "agreement_threshold": agreement_threshold,
            },
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if log:
            log(
                f"BiRefNet + MobileSAM (in-GUI, models kept warm)\n"
                f"  Input : {input_dir}  ({len(images)} images)\n"
                f"  Output: {output_dir}\n"
                f"  Mode  : {cfg['hybrid']['mode']}  |  "
                f"Seg res: {cfg['segmentation_resolution']}px  |  "
                f"Out: {cfg['output_size']}px"
            )

        ok = failed = 0
        n = len(images)
        for i, img_path in enumerate(images, 1):
            if should_cancel and should_cancel():
                if log:
                    log("BiRefNet batch cancelled.")
                break
            if log:
                log(f"[{i}/{n}] {img_path.name}")
            try:
                result = process_image(
                    img_path,
                    output_dir,
                    cfg,
                    self._birefnet,
                    self._mobilesam,
                    remove_blue=remove_blue,
                    save_debug=True,
                )
                if result:
                    ok += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if log:
                    log(f"  ERROR: {e}")

        if log:
            log(f"Done: {ok} succeeded, {failed} failed.")
        return ok, failed

    def finalize_batch(
        self,
        output_dir: Path,
        *,
        known_diameter_mm: float = 6.0,
        hybrid_mode: str = "birefnet_primary",
        seg_resolution: int = 1024,
        output_size: int = 1024,
        agreement_threshold: float = 0.85,
        remove_blue: bool = True,
        project_root: Path | None = None,
        log: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[int, int]:
        """Refine all selections with BiRefNet + MobileSAM merge. Returns (ok, failed)."""
        if not self.selections:
            raise RuntimeError("No interactive selections to finalize.")

        self.ensure_mobilesam(log=log)
        self.ensure_birefnet(log=log)
        self._ensure_path()

        from run_pipeline import _load_cfg, process_image

        cfg_path = _BIREFNET_DIR / "config.yaml"
        cfg = _load_cfg(
            cfg_path,
            {
                "known_diameter_mm": known_diameter_mm,
                "hybrid_mode": hybrid_mode,
                "seg_resolution": seg_resolution,
                "output_size": output_size,
                "agreement_threshold": agreement_threshold,
            },
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fold pending manual circles onto selections before export (scale mode only)
        if remove_blue:
            for stem, sel in list(self.selections.items()):
                if not sel.has_circle and stem in self.pending_circles:
                    _apply_circle_to_selection(
                        sel, self.pending_circles[stem], subtract_from_mask=True
                    )

            # Persist manual/auto circles as overrides for Analysis scale detection
            if project_root is not None:
                try:
                    from gui.pipeline import save_circle_override

                    for stem, sel in self.selections.items():
                        if sel.has_circle:
                            save_circle_override(
                                project_root,
                                stem,
                                float(sel.circle_cx),
                                float(sel.circle_cy),
                                float(sel.circle_diameter),
                            )
                except Exception as e:
                    if log:
                        log(f"WARNING: could not save circle overrides: {e}")

        ok = failed = 0
        items = list(self.selections.items())
        n = len(items)
        for i, (stem, sel) in enumerate(items, 1):
            if should_cancel and should_cancel():
                if log:
                    log("Interactive batch cancelled.")
                break
            # Skip stubs that never got a leaf click (1x1 empty mask)
            if sel.mask.size <= 1 or int(sel.mask.sum()) == 0:
                if log:
                    log(f"[{i}/{n}] SKIP {sel.path.name}: no leaf click yet")
                failed += 1
                continue
            if log:
                log(f"[{i}/{n}] BiRefNet refine: {sel.path.name}  click=({sel.x},{sel.y})")
            try:
                circle_prior = (
                    sel.circle_prior_dict()
                    if remove_blue and sel.has_circle
                    else None
                )
                result = process_image(
                    sel.path,
                    output_dir,
                    cfg,
                    self._birefnet,
                    self._mobilesam,
                    remove_blue=remove_blue,
                    save_debug=True,
                    point_prior=(sel.x, sel.y),
                    circle_prior=circle_prior,
                )
                if result:
                    ok += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if log:
                    log(f"  ERROR: {e}")
        if log:
            log(f"Interactive batch done: {ok} succeeded, {failed} failed.")
        return ok, failed


# Module-level session reused while Interactive method stays selected
_SESSION: InteractiveSamSession | None = None


def get_session() -> InteractiveSamSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = InteractiveSamSession()
    return _SESSION


def reset_session(*, release_models: bool = False) -> None:
    global _SESSION
    if _SESSION is not None and release_models:
        _SESSION.release_models()
        _SESSION.clear_selections()
    _SESSION = InteractiveSamSession() if not release_models else None
