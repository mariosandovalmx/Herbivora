"""Per-image scale metadata shared by the non-BiRefNet segmentation scripts.

``analyze_leaves.py`` counts leaf and damage area in white_bg pixels, while the
blue reference dot is measured on the user's photo. Every segmentation backend
therefore has to record the linear factor between those two pixel spaces, using
the same field names as the BiRefNet pipeline (see run_pipeline.py), so
``gui.pipeline.run_scale_detection`` can convert cm²/px² regardless of which
backend produced the white_bg images.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_scale_metadata(
    path: Path,
    *,
    image_id: str,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    scale_factor: float,
    scale_source: str = "original_photo",
    output_path: Path | None = None,
) -> None:
    """Write the metadata JSON describing one source image → white_bg mapping.

    ``scale_factor`` is the linear factor from source pixels to white_bg pixels.
    ``scale_source`` is "original_photo" when the source image is the user's own
    photo, and "derived_image" when it was produced by an earlier step (in which
    case a dot measured on the photo cannot be mapped through this factor alone).
    """
    meta = {
        "image_id": image_id,
        "original_size": [int(source_size[0]), int(source_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "scale_factor": round(float(scale_factor), 6),
        "pre_ratio": [1.0, 1.0],
        "scale_source": scale_source,
        "measurement_space": "original_resolution",
    }
    if output_path is not None:
        meta["output_path"] = str(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
