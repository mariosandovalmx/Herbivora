#!/usr/bin/env bash
# Build a simple HerbivoR-vVERSION.dmg on macOS (maintainers).
# Contents: source tree + Install_HerbivoR.command for double-click setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VER="$(tr -d '[:space:]' < VERSION || echo 0.0.0)"
STAGE="$ROOT/dist/dmg_stage/HerbivoR"
DMG="$ROOT/dist/HerbivoR-v${VER}.dmg"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must run on macOS."
  exit 1
fi

rm -rf "$ROOT/dist/dmg_stage"
mkdir -p "$STAGE"
rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'dist' \
  --exclude 'build' \
  --exclude '.cursor' \
  --exclude 'hf_cache' \
  --exclude 'models/*.pt' \
  --exclude 'models/*.pth' \
  --exclude '*.lnk' \
  --exclude 'gui_error.log' \
  "$ROOT/" "$STAGE/"

chmod +x "$STAGE/Install_HerbivoR.command" "$STAGE/install.sh" "$STAGE/herbivor.sh" \
  "$STAGE/packaging/create_macos_app.sh" "$STAGE/packaging/build_macos_dmg.sh" || true

# Finder-friendly alias name
ln -sf Install_HerbivoR.command "$STAGE/Install HerbivoR.command" 2>/dev/null || true

rm -f "$DMG"
hdiutil create -volname "HerbivoR $VER" -srcfolder "$ROOT/dist/dmg_stage" -ov -format UDZO "$DMG"
echo "Created $DMG"
echo "Attach this file to the GitHub Release for macOS users."
