#!/usr/bin/env bash
# HerbivoR installer for macOS / Linux
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================"
echo "  HerbivoR - Dependency Installation"
echo "============================================"
echo

PY=""
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: Python 3 not found."
  echo "Install Python 3.10+ and re-run this script."
  exit 1
fi

echo "Using: $PY"
"$PY" --version

if [[ ! -x ".venv/bin/python" ]]; then
  echo
  echo "Creating virtual environment .venv ..."
  "$PY" -m venv .venv
fi

echo
echo "Installing packages (may take several minutes)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Downloading model weights (~226 MB, first time only)..."
if ! .venv/bin/python download_models.py; then
  echo "WARNING: Model download failed. Retry with:"
  echo "  .venv/bin/python download_models.py"
  echo "Or place files manually in models/ — see models/README.md"
fi

echo
echo "============================================"
echo "  Installation completed"
echo "============================================"
echo
echo "Open the application with: ./herbivor.sh"
echo
