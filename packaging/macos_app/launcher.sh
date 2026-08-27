#!/bin/bash
# Herbivora.app launcher (macOS).
#
# This script is Contents/MacOS/Herbivora inside the distributed app bundle.
# The bundle is meant to be dragged into /Applications and double-clicked.
#
#   First launch : copy the payload to ~/Library/Application Support/Herbivora,
#                  make sure a GUI-capable Python 3.10+ exists (downloading a
#                  private CPython when the Mac has none), then run the
#                  bootstrap installer (venv + PyTorch + model weights).
#   Later launches: start the GUI directly.
#
# Nothing is written inside the app bundle, so the app keeps working from a
# read-only /Applications and stays valid for code signing.

set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PAYLOAD="$BUNDLE_DIR/Contents/Resources/payload"

SUPPORT_DIR="$HOME/Library/Application Support/Herbivora"
APP_DIR="$SUPPORT_DIR/app"
PYTHON_DIR="$SUPPORT_DIR/python"
STAMP_FILE="$SUPPORT_DIR/installed_version"
LOCK_DIR="$SUPPORT_DIR/setup.lock"
LOG="$SUPPORT_DIR/launch.log"

# Private CPython used when the Mac has no usable Python (same build the
# Windows installer uses).
PBS_TAG="20260303"
PBS_PYTHON="3.12.13"

mkdir -p "$SUPPORT_DIR"
if [[ -f "$LOG" && "$(stat -f%z "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]]; then
  mv -f "$LOG" "$LOG.1"
fi
exec > >(tee -a "$LOG") 2>&1
echo "=== Herbivora launch $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "bundle: $BUNDLE_DIR"
echo "process architecture: $(uname -m)"
echo "running under Rosetta: $(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)"

# ── UI helpers ───────────────────────────────────────────────────────────────
alert() { # alert <note|stop> <message>
  /usr/bin/osascript - "$1" "$2" >/dev/null 2>&1 <<'APPLESCRIPT'
on run argv
  set theIcon to item 1 of argv
  set theText to item 2 of argv
  if theIcon is "stop" then
    display dialog theText with title "Herbivora" buttons {"OK"} default button "OK" with icon stop
  else
    display dialog theText with title "Herbivora" buttons {"OK"} default button "OK" with icon note
  end if
end run
APPLESCRIPT
}

notify() { # notify <message>
  /usr/bin/osascript - "$1" >/dev/null 2>&1 <<'APPLESCRIPT'
on run argv
  display notification (item 1 of argv) with title "Herbivora setup"
end run
APPLESCRIPT
}

fail() {
  echo "ERROR: $1"
  alert stop "$1

Details: $LOG"
  exit 1
}

# ── Python discovery ─────────────────────────────────────────────────────────
# Finder launches apps with a minimal PATH, so probe the usual install roots
# explicitly instead of relying on `command -v`.
python_candidates() {
  echo "$PYTHON_DIR/bin/python3"
  local dir name
  for dir in /opt/homebrew/bin /usr/local/bin /usr/bin; do
    # Prefer 3.12, PyTorch's broadly tested macOS version. Accept 3.13 as a
    # fallback; newer interpreters may precede wheels for app dependencies.
    for name in python3.12 python3.13 python3.11 python3.10 python3; do
      echo "$dir/$name"
    done
  done
  local fw
  for fw in $(ls -rd /Library/Frameworks/Python.framework/Versions/3.* 2>/dev/null); do
    echo "$fw/bin/python3"
  done
}

python_usable() { # python_usable <exe>
  local exe="$1"
  [[ -x "$exe" ]] || return 1
  local expected_arch="x86_64"
  if [[ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    expected_arch="arm64"
  fi
  HERBIVOR_EXPECTED_ARCH="$expected_arch" "$exe" - >/dev/null 2>&1 <<'PYEOF'
import os
import platform
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(1)
if platform.machine() != os.environ["HERBIVOR_EXPECTED_ARCH"]:
    raise SystemExit(1)
import tkinter
tkinter.Tcl()
PYEOF
}

find_python() {
  local exe
  for exe in $(python_candidates); do
    if python_usable "$exe"; then
      echo "$exe"
      return 0
    fi
  done
  return 1
}

install_private_python() {
  local arch url tmp
  # Query the hardware rather than `uname`: a script-only app can inherit a
  # Rosetta preference even on Apple silicon.
  if [[ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    arch="aarch64-apple-darwin"
  elif [[ "$(uname -m)" == "x86_64" ]]; then
    arch="x86_64-apple-darwin"
  else
    echo "Unsupported architecture: $(uname -m)"
    return 1
  fi

  tmp="$(mktemp -d "${TMPDIR:-/tmp}/herbivor_py.XXXXXX")" || return 1
  local base
  local ok=1
  for base in \
    "https://github.com/astral-sh/python-build-standalone/releases/download" \
    "https://github.com/indygreg/python-build-standalone/releases/download"
  do
    url="$base/$PBS_TAG/cpython-${PBS_PYTHON}+${PBS_TAG}-${arch}-install_only.tar.gz"
    echo "Downloading private Python: $url"
    if /usr/bin/curl -fL --retry 3 --retry-delay 2 -o "$tmp/python.tar.gz" "$url"; then
      ok=0
      break
    fi
    echo "Download failed from $base"
  done
  if [[ $ok -ne 0 ]]; then
    rm -rf "$tmp"
    return 1
  fi

  echo "Extracting private Python …"
  if ! /usr/bin/tar -xzf "$tmp/python.tar.gz" -C "$tmp"; then
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$PYTHON_DIR"
  mkdir -p "$(dirname "$PYTHON_DIR")"
  if ! mv "$tmp/python" "$PYTHON_DIR"; then
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$tmp"
  python_usable "$PYTHON_DIR/bin/python3"
}

ensure_python() {
  local exe
  if exe="$(find_python)"; then
    echo "$exe"
    return 0
  fi
  echo "No GUI-capable Python 3.10+ found — installing a private copy."
  notify "Downloading the Python runtime (one time, about 50 MB)…"
  if install_private_python >&2; then
    echo "$PYTHON_DIR/bin/python3"
    return 0
  fi
  return 1
}

# ── Payload sync ─────────────────────────────────────────────────────────────
if [[ -f "$PAYLOAD/VERSION" ]]; then
  payload_version="$(tr -d '[:space:]' < "$PAYLOAD/VERSION")"
else
  payload_version="0.0.0"
fi
if [[ -f "$STAMP_FILE" ]]; then
  installed_version="$(tr -d '[:space:]' < "$STAMP_FILE")"
else
  installed_version=""
fi

sync_payload() {
  [[ -d "$PAYLOAD" ]] || fail "This copy of Herbivora.app is incomplete (no payload). Download the DMG again."

  if [[ -d "$APP_DIR" && "$installed_version" != "$payload_version" ]]; then
    echo "Upgrading $installed_version -> $payload_version: clearing old program files."
    find "$APP_DIR" -mindepth 1 -maxdepth 1 \
      ! -name '.venv' ! -name 'models' -exec rm -rf {} + 2>/dev/null
  fi

  mkdir -p "$APP_DIR"
  echo "Copying program files to $APP_DIR"
  /usr/bin/rsync -a --exclude '.venv' --exclude 'models' "$PAYLOAD/" "$APP_DIR/" \
    || fail "Could not copy Herbivora into $APP_DIR."
  # Model weights (hundreds of MB) are downloaded once and never overwritten.
  mkdir -p "$APP_DIR/models"
  /usr/bin/rsync -a --exclude '*.pt' --exclude '*.pth' --exclude '*.safetensors' \
    "$PAYLOAD/models/" "$APP_DIR/models/" 2>/dev/null
  chmod +x "$APP_DIR"/*.sh "$APP_DIR"/*.command "$APP_DIR"/packaging/*.sh 2>/dev/null
}

# ── Run ──────────────────────────────────────────────────────────────────────
VENV_PY="$APP_DIR/.venv/bin/python"

runtime_ready() {
  [[ -x "$VENV_PY" ]] || return 1
  "$VENV_PY" - >/dev/null 2>&1 <<'PYEOF'
import customtkinter
import torch
import torchvision
PYEOF
}

needs_setup=0
runtime_ready || needs_setup=1
[[ "$installed_version" == "$payload_version" ]] || needs_setup=1

if [[ $needs_setup -eq 1 ]]; then
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another Herbivora setup is already running ($LOCK_DIR)."
    alert note "Herbivora setup is already running in another window.

If that is not true, delete this folder and try again:
$LOCK_DIR"
    exit 0
  fi
  trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

  sync_payload

  PY="$(ensure_python)" || fail "Herbivora needs Python 3.10 or newer and could not download its own copy.

Install Python from https://www.python.org/downloads/macos/ and open Herbivora again."
  echo "Using Python: $PY"

  echo "Starting the Herbivora installer …"
  if ! "$PY" "$APP_DIR/packaging/bootstrap_install.py" --root "$APP_DIR" --gui --app-mode; then
    echo "Setup was cancelled or did not finish. Open Herbivora to retry."
    exit 0
  fi
  if ! runtime_ready; then
    echo "Setup returned without installing all required runtime packages."
    alert stop "Herbivora setup is incomplete. Open Herbivora to retry."
    exit 0
  fi
  echo "$payload_version" > "$STAMP_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null
  trap - EXIT
fi

cd "$APP_DIR" || fail "Missing $APP_DIR — open Herbivora again to reinstall."
echo "Launching GUI from $APP_DIR"
exec "$VENV_PY" -m gui.main
