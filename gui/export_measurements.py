"""Build and write Analysis measurement tables for statistical export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Keep in sync with analyze_leaves.DAMAGE_PCT_DECIMALS
_DAMAGE_PCT_DECIMALS = 3


def _round_damage_pct(pct: float) -> float:
    return round(float(pct), _DAMAGE_PCT_DECIMALS)


_PCT_FIELDS = [
    "image_name",
    "leaf_area_px",
    "damage_px",
    "undamaged_px",
    "damage_pct",
    "undamaged_pct",
]

_CM2_FIELDS = [
    "scale_cm2_per_px",
    "leaf_area_cm2",
    "damage_cm2",
    "undamaged_cm2",
]


def _stem_from_image_name(image_name: str) -> str:
    return Path(image_name).stem


def _load_meta(analyzed_dir: Path, image_name: str) -> dict | None:
    meta_path = analyzed_dir / f"{_stem_from_image_name(image_name)}_meta.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    f = _safe_float(value, None)
    if f is None:
        return default
    return int(round(f))


def build_measurement_rows(
    analyzed_dir: Path,
    *,
    include_cm2: bool,
) -> list[dict]:
    """Read results.csv (+ optional *_meta.json) into export rows."""
    analyzed_dir = Path(analyzed_dir)
    csv_path = analyzed_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No results.csv found in {analyzed_dir}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    rows: list[dict] = []
    for raw in raw_rows:
        image_name = (raw.get("image_name") or "").strip()
        if not image_name:
            continue

        meta = _load_meta(analyzed_dir, image_name)

        if meta is not None:
            leaf_area_px = _safe_int(meta.get("leaf_area_px"), 0)
            damage_px = _safe_int(meta.get("damage_px"), 0)
            damage_pct = _safe_float(meta.get("damage_pct"), 0.0) or 0.0
            scale = _safe_float(meta.get("scale_cm2_per_px"), None)
        else:
            # Fallback from summary CSV (schema varies by analysis mode)
            leaf_area_px = _safe_int(raw.get("leaf_area_px"), 0)
            damage_pct = _safe_float(raw.get("damage_pct"), None)
            scale = None
            damage_px = 0
            if damage_pct is None:
                damage_pct = 0.0
            # Reconstruct damage_px if only % is available
            if leaf_area_px > 0 and damage_pct is not None:
                damage_px = int(round(leaf_area_px * float(damage_pct) / 100.0))

        undamaged_px = max(0, leaf_area_px - damage_px)
        undamaged_pct = _round_damage_pct(max(0.0, 100.0 - float(damage_pct)))
        damage_pct = _round_damage_pct(float(damage_pct))

        row: dict = {
            "image_name": image_name,
            "leaf_area_px": leaf_area_px,
            "damage_px": damage_px,
            "undamaged_px": undamaged_px,
            "damage_pct": damage_pct,
            "undamaged_pct": undamaged_pct,
        }

        if include_cm2 and scale is not None and scale > 0:
            leaf_cm2 = leaf_area_px * scale
            damage_cm2 = damage_px * scale
            undamaged_cm2 = undamaged_px * scale
            # Prefer CSV cm² when present (already rounded by pipeline)
            csv_leaf_cm2 = _safe_float(raw.get("leaf_area_cm2"), None)
            csv_damage_cm2 = _safe_float(raw.get("damage_cm2"), None)
            row["scale_cm2_per_px"] = round(scale, 8)
            row["leaf_area_cm2"] = (
                round(csv_leaf_cm2, 4) if csv_leaf_cm2 is not None else round(leaf_cm2, 4)
            )
            row["damage_cm2"] = (
                round(csv_damage_cm2, 4) if csv_damage_cm2 is not None else round(damage_cm2, 4)
            )
            row["undamaged_cm2"] = round(undamaged_cm2, 4)
        elif include_cm2:
            # Mode asked for cm² but this leaf has no scale — leave blanks
            row["scale_cm2_per_px"] = ""
            row["leaf_area_cm2"] = ""
            row["damage_cm2"] = ""
            row["undamaged_cm2"] = ""

        rows.append(row)

    rows.sort(key=lambda r: str(r.get("image_name", "")))
    return rows


def measurement_fieldnames(*, include_cm2: bool) -> list[str]:
    fields = list(_PCT_FIELDS)
    if include_cm2:
        fields.extend(_CM2_FIELDS)
    return fields


def write_measurements(path: Path, rows: list[dict], *, include_cm2: bool) -> None:
    """Write rows to CSV (comma) or TXT (tab) based on file suffix."""
    path = Path(path)
    if not rows:
        raise ValueError("No measurement rows to export.")

    fieldnames = measurement_fieldnames(include_cm2=include_cm2)
    delimiter = "\t" if path.suffix.lower() == ".txt" else ","

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
