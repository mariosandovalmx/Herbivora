#!/usr/bin/env bash
# Build the end-user Herbivora drag-to-Applications DMG on macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must run on macOS."
  exit 1
fi

VER="$(tr -d '[:space:]' < VERSION || echo 0.0.0)"
DIST="$ROOT/dist"
STAGE="$DIST/dmg_stage"
DMG="$DIST/Herbivora-v${VER}.dmg"

echo "[Herbivora] Building simple drag-to-Applications DMG v$VER"
rm -rf "$STAGE"
mkdir -p "$STAGE"

bash "$ROOT/packaging/build_macos_app_bundle.sh" "$STAGE"
ln -s /Applications "$STAGE/Applications"

# Herbivora is not notarized (that requires a paid Apple Developer ID), so every
# recipient hits Gatekeeper once. Launching the app from inside this read-only
# image is a dead end, so ship the instructions in the same window. The leading
# space sorts the file first in Finder's icon view.
sed "s/__VERSION__/$VER/g" "$ROOT/packaging/macos_app/dmg_readme.txt" \
  > "$STAGE/ READ ME FIRST.txt"

rm -f "$DMG"
hdiutil create \
  -volname "Herbivora $VER" \
  -srcfolder "$STAGE" \
  -format UDZO \
  -imagekey zlib-level=9 \
  -ov \
  "$DMG" >/dev/null

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$DMG"
fi

echo "[Herbivora] Created $DMG"
echo "[Herbivora] Contents: Herbivora.app + Applications shortcut"
