#!/usr/bin/env bash
# Build the distributable HerbivoR.app (macOS).
#
# The bundle carries the whole HerbivoR source tree in Contents/Resources/payload
# and a launcher that installs into ~/Library/Application Support/HerbivoR on
# first run. Users drag this app into /Applications; no Terminal, no .command.
#
#   packaging/build_macos_app_bundle.sh [output_dir]      # default: dist/
#
# Optional Developer ID signing (removes the "Apple could not verify" warning
# once the DMG is also notarized):
#   CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
#     packaging/build_macos_app_bundle.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must run on macOS."
  exit 1
fi

OUT_DIR="${1:-$ROOT/dist}"
APP="$OUT_DIR/HerbivoR.app"
VER="$(tr -d '[:space:]' < VERSION || echo 0.0.0)"

echo "[HerbivoR] Building HerbivoR.app v$VER"
mkdir -p "$OUT_DIR"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/payload"

# ── Payload: the source tree the launcher installs from ──────────────────────
rsync -a \
  --exclude '.git' \
  --exclude '.github' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude 'dist' \
  --exclude 'build' \
  --exclude '.cursor' \
  --exclude 'hf_cache' \
  --exclude 'models/*.pt' \
  --exclude 'models/*.pth' \
  --exclude 'models/*.safetensors' \
  --exclude '*.lnk' \
  --exclude 'gui_error.log' \
  --exclude 'HerbivoR.app' \
  "$ROOT/" "$APP/Contents/Resources/payload/"

# Full agreement text, generated directly inside the payload so a package build
# never modifies the maintainer's source checkout.
python3 "$ROOT/packaging/build_installer_license.py" \
  "$APP/Contents/Resources/payload/packaging/installer_license.txt"

chmod +x "$APP/Contents/Resources/payload"/*.sh \
         "$APP/Contents/Resources/payload"/*.command \
         "$APP/Contents/Resources/payload"/packaging/*.sh 2>/dev/null || true

# ── Icon ─────────────────────────────────────────────────────────────────────
bash "$ROOT/packaging/make_icns.sh" "$APP/Contents/Resources/AppIcon.icns"

# ── Launcher ─────────────────────────────────────────────────────────────────
# Build a true universal executable so Finder recognizes this as a native app
# and Apple-silicon Macs do not classify the script launcher as Intel-only.
command -v clang >/dev/null 2>&1 || {
  echo "[HerbivoR] ERROR: clang/Xcode Command Line Tools are required."
  exit 1
}
clang \
  -arch arm64 \
  -arch x86_64 \
  -mmacosx-version-min=11.0 \
  -Os \
  "$ROOT/packaging/macos_app/launcher.c" \
  -o "$APP/Contents/MacOS/HerbivoR"
cp "$ROOT/packaging/macos_app/launcher.sh" "$APP/Contents/Resources/launcher.sh"
chmod +x "$APP/Contents/MacOS/HerbivoR" "$APP/Contents/Resources/launcher.sh"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>HerbivoR</string>
  <key>CFBundleDisplayName</key><string>HerbivoR</string>
  <key>CFBundleIdentifier</key><string>mx.mariosandoval.herbivor</string>
  <key>CFBundleVersion</key><string>$VER</string>
  <key>CFBundleShortVersionString</key><string>$VER</string>
  <key>CFBundleExecutable</key><string>HerbivoR</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleSignature</key><string>????</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.education</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <!-- Script-only apps may otherwise be classified as Intel apps on Apple
       silicon. Prefer the native slice while retaining Intel Mac support. -->
  <key>LSArchitecturePriority</key>
  <array>
    <string>arm64</string>
    <string>x86_64</string>
  </array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>HerbivoR — PolyForm Noncommercial License 1.0.0</string>
</dict>
</plist>
EOF

cat > "$APP/Contents/PkgInfo" <<'EOF'
APPL????
EOF

# ── Signature ────────────────────────────────────────────────────────────────
# Ad-hoc by default: gives the bundle a stable code identity, but does not skip
# Gatekeeper. Developer ID signing plus Apple notarization avoids that warning.
IDENTITY="${CODESIGN_IDENTITY:--}"
if [[ "$IDENTITY" == "-" ]]; then
  SIGN_ARGS=(--force --timestamp=none --sign -)
else
  SIGN_ARGS=(--force --options runtime --timestamp --sign "$IDENTITY")
fi
if codesign "${SIGN_ARGS[@]}" "$APP" 2>&1; then
  if [[ "$IDENTITY" == "-" ]]; then
    echo "[HerbivoR] Ad-hoc signed (unsigned distribution: users must approve once in"
    echo "           System Settings > Privacy & Security)."
  else
    echo "[HerbivoR] Signed with: $IDENTITY"
  fi
else
  echo "[HerbivoR] WARNING: codesign failed; shipping an unsigned bundle."
fi

codesign --verify --deep --strict --verbose=2 "$APP"

SIZE="$(du -sh "$APP" | cut -f1)"
echo "[HerbivoR] Created $APP ($SIZE)"
