#!/usr/bin/env bash
# Build a macOS .icns file from the HerbivoR leaf artwork.
# Usage: packaging/make_icns.sh <output.icns> [source.png]
set -euo pipefail

OUT="${1:?usage: make_icns.sh <output.icns> [source.png]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${2:-}"
PREBUILT="$ROOT/assets/HerbivoR.icns"

if [[ -z "$SRC" && -f "$PREBUILT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  cp "$PREBUILT" "$OUT"
  echo "make_icns.sh: wrote $OUT (from $(basename "$PREBUILT"))"
  exit 0
fi

if [[ -z "$SRC" ]]; then
  for candidate in "$ROOT/assets/herbivor_icon.png" "$ROOT/assets/herbivor_256.png"; do
    if [[ -f "$candidate" ]]; then
      SRC="$candidate"
      break
    fi
  done
fi
[[ -f "$SRC" ]] || { echo "make_icns.sh: no source PNG found"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/herbivor_iconset.XXXXXX")"
ICONSET="$WORK/HerbivoR.iconset"
mkdir -p "$ICONSET"

# Apple's icon grid: every size the Finder, Dock and Launchpad ask for.
sips -z 16 16     "$SRC" --out "$ICONSET/icon_16x16.png"      >/dev/null
sips -z 32 32     "$SRC" --out "$ICONSET/icon_16x16@2x.png"   >/dev/null
sips -z 32 32     "$SRC" --out "$ICONSET/icon_32x32.png"      >/dev/null
sips -z 64 64     "$SRC" --out "$ICONSET/icon_32x32@2x.png"   >/dev/null
sips -z 128 128   "$SRC" --out "$ICONSET/icon_128x128.png"    >/dev/null
sips -z 256 256   "$SRC" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$SRC" --out "$ICONSET/icon_256x256.png"    >/dev/null
sips -z 512 512   "$SRC" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$SRC" --out "$ICONSET/icon_512x512.png"    >/dev/null
sips -z 1024 1024 "$SRC" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

mkdir -p "$(dirname "$OUT")"
if ! iconutil -c icns "$ICONSET" -o "$OUT"; then
  echo "make_icns.sh: iconutil rejected the iconset; trying Pillow fallback"
  python3 -c \
    'from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2], format="ICNS")' \
    "$SRC" "$OUT"
fi
rm -rf "$WORK"
echo "make_icns.sh: wrote $OUT (from $(basename "$SRC"))"
