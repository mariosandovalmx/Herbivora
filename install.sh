#!/usr/bin/env bash
# HerbivoR installer for macOS / Linux
# - macOS: installs default PyTorch (CPU + Metal/MPS when available)
# - Linux: CUDA 12.4 if nvidia-smi works, otherwise CPU
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
echo "Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

OS="$(uname -s)"
TORCH_INDEX=""
TORCH_NOTE=""

if [[ "$OS" == "Darwin" ]]; then
  # Official macOS wheels include MPS (Metal) support when the hardware allows it.
  TORCH_INDEX=""
  TORCH_NOTE="macOS: PyTorch with CPU + Metal (MPS) when available"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  TORCH_NOTE="Linux: NVIDIA detected — installing CUDA 12.4 wheels"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_NOTE="Linux: no NVIDIA GPU detected — installing CPU wheels"
fi

echo
echo "$TORCH_NOTE"
if [[ -n "$TORCH_INDEX" ]]; then
  .venv/bin/python -m pip install torch torchvision --index-url "$TORCH_INDEX"
else
  .venv/bin/python -m pip install torch torchvision
fi

echo
echo "Installing HerbivoR packages from requirements.txt..."
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
if [[ "$OS" == "Darwin" ]]; then
  echo "On Apple Silicon / modern Macs, Metal (MPS) is used automatically when available."
fi
echo "Open the application with: ./herbivor.sh"
if [[ "$OS" == "Darwin" ]]; then
  echo
  echo "Optional: create a Dock/Finder app with the leaf icon:"
  echo "  chmod +x packaging/create_macos_app.sh && ./packaging/create_macos_app.sh"
fi
echo
