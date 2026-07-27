#!/usr/bin/env bash
# Launch HerbivoR GUI (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x ".venv/bin/python" ]]; then
  exec .venv/bin/python -m gui.main
fi

echo "The .venv folder does not exist or startup failed."
echo "Please run ./install.sh first."
exit 1
