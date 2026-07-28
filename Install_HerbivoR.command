#!/bin/bash
# HerbivoR one-click installer for macOS (double-click in Finder).
# Also works on Linux from a terminal: ./Install_HerbivoR.command
cd "$(dirname "$0")" || exit 1

BOOTSTRAP="packaging/bootstrap_install.py"
if [[ ! -f "$BOOTSTRAP" ]]; then
  echo "ERROR: $BOOTSTRAP not found."
  echo "Extract the full HerbivoR release and try again."
  read -r -p "Press Enter to close..."
  exit 1
fi

PY=""
if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="python3"
  fi
fi

if [[ -z "$PY" ]] && command -v python >/dev/null 2>&1; then
  if python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="python"
  fi
fi

if [[ -z "$PY" ]]; then
  echo "Python 3.10+ was not found."
  echo
  echo "On macOS:"
  echo "  1) Install from https://www.python.org/downloads/macos/"
  echo "     (or: xcode-select --install)"
  echo "  2) Double-click Install_HerbivoR.command again."
  echo
  echo "On Linux, install python3 via your package manager, then re-run."
  read -r -p "Press Enter to close..."
  exit 1
fi

echo "Starting HerbivoR installer with: $PY"
# Prefer GUI when a display is available; fall back to console.
if [[ -n "${DISPLAY:-}" || "$(uname -s)" == "Darwin" ]]; then
  if "$PY" "$BOOTSTRAP" --gui --flavor auto; then
    exit 0
  fi
  echo "GUI installer unavailable; falling back to console mode..."
fi

"$PY" "$BOOTSTRAP" --yes --flavor auto
ERR=$?
if [[ $ERR -ne 0 ]]; then
  echo "Installer exited with code $ERR"
  read -r -p "Press Enter to close..."
fi
exit $ERR
