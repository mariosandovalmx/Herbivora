#!/usr/bin/env bash
# Build Herbivora.app on macOS (must run on a Mac; cannot cross-compile from Windows).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=python3
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
fi

VER="$(tr -d '[:space:]' < VERSION || echo 0.0.0)"

echo "[Herbivora] Installing packaging deps..."
"$PY" -m pip install -r requirements-dev.txt

echo "[Herbivora] Building PyInstaller onedir / .app..."
"$PY" -m PyInstaller --noconfirm --clean --windowed \
  --name Herbivora \
  --paths "$ROOT" \
  --add-data "contour/configs:contour/configs" \
  --add-data "segmentation/birefnet_mobilesam/config.yaml:segmentation/birefnet_mobilesam" \
  --add-data "VERSION:." \
  --add-data "models/README.md:models" \
  packaging/herbivor_entry.py

ZIP="dist/Herbivora-macos-v${VER}.zip"
rm -f "$ZIP"
(
  cd dist
  if [[ -d Herbivora.app ]]; then
    zip -r "../$ZIP" Herbivora.app
  else
    zip -r "../$ZIP" Herbivora
  fi
)

echo "[Herbivora] Done: $ROOT/$ZIP"
echo "Copy to another Mac, extract, open Herbivora, then use Check installation for models."
