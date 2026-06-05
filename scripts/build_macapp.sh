#!/usr/bin/env bash
# scripts/build_macapp.sh — wrap the SwiftPM-built TinyGPTApp binary into
# a proper .app bundle so it launches like any other Mac app (via Finder,
# Spotlight, `open TinyGPT.app`, dock pinning, etc.).
#
# SwiftPM only emits a raw Mach-O executable — perfectly runnable from
# the command line but not LaunchServices-friendly. This script copies
# the binary + its resource bundles + the MLX metallib into the right
# Contents/{MacOS,Resources} layout and writes an Info.plist that
# CFBundle/LaunchServices need.
#
# Usage:
#   ./scripts/build_macapp.sh                       # release build → ./build/TinyGPT.app
#   ./scripts/build_macapp.sh --debug               # debug build instead
#   ./scripts/build_macapp.sh --out /path/to/Apps   # custom output dir
#
# After running:
#   open ./build/TinyGPT.app                        # standard Mac launch
#   cp -r ./build/TinyGPT.app /Applications/        # install
#
# Not codesigned / notarized — Gatekeeper will warn on first launch.
# That's a separate, account-required step; for solo dev use, right-
# click → Open dismisses the warning permanently for this build.

set -euo pipefail

CONFIG="release"
OUT_DIR="$(pwd)/build"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)   CONFIG="debug"; shift ;;
        --release) CONFIG="release"; shift ;;
        --out)     OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$REPO_ROOT/native-mac"
BUILD_DIR="$PKG/.build/arm64-apple-macosx/$CONFIG"
APP="$OUT_DIR/TinyGPT.app"

echo "== build (swift build -c $CONFIG --product TinyGPTApp)"
( cd "$PKG" && swift build -c "$CONFIG" --product TinyGPTApp )

if [[ ! -x "$BUILD_DIR/TinyGPTApp" ]]; then
    echo "build did not produce $BUILD_DIR/TinyGPTApp" >&2
    exit 1
fi

echo "== assemble bundle → $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BUILD_DIR/TinyGPTApp" "$APP/Contents/MacOS/TinyGPT"
chmod +x "$APP/Contents/MacOS/TinyGPT"

# Also build + bundle the CLI binary. The Interp tab shells out to it
# for SAE / MEMIT / patch training so the app doesn't have to duplicate
# the CLI's training paths in-process.
( cd "$PKG" && swift build -c "$CONFIG" --product tinygpt )
if [[ -x "$BUILD_DIR/tinygpt" ]]; then
    cp "$BUILD_DIR/tinygpt" "$APP/Contents/MacOS/tinygpt-cli"
    chmod +x "$APP/Contents/MacOS/tinygpt-cli"
fi

# MLX needs its compiled Metal shader library at runtime. SwiftPM drops
# it next to the binary; the .app needs it in Resources so the binary's
# search path (which Foundation rewrites to the bundle when launched as
# an .app) finds it.
if [[ -f "$BUILD_DIR/mlx.metallib" ]]; then
    cp "$BUILD_DIR/mlx.metallib" "$APP/Contents/Resources/default.metallib"
    cp "$BUILD_DIR/mlx.metallib" "$APP/Contents/MacOS/mlx.metallib"
fi

# Resource bundles SwiftPM produces for swift-transformers + swift-crypto.
# Copy any *.bundle next to the binary into Resources/ so dynamic loader
# code finds them.
for b in "$BUILD_DIR"/*.bundle; do
    [[ -e "$b" ]] || continue
    cp -R "$b" "$APP/Contents/Resources/"
done

# App icon. Regenerated from browser/public/favicon.svg via
# scripts/make_icon.sh if missing.
ICON_SRC="$PKG/Resources/TinyGPT.icns"
if [[ ! -f "$ICON_SRC" ]]; then
    echo "== generating icon (scripts/make_icon.sh)"
    "$REPO_ROOT/scripts/make_icon.sh"
fi
if [[ -f "$ICON_SRC" ]]; then
    cp "$ICON_SRC" "$APP/Contents/Resources/TinyGPT.icns"
fi

# Info.plist — the minimum LaunchServices wants.
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>TinyGPT</string>
    <key>CFBundleDisplayName</key>
    <string>TinyGPT</string>
    <key>CFBundleIdentifier</key>
    <string>com.tinygpt.app</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1</string>
    <key>CFBundleExecutable</key>
    <string>TinyGPT</string>
    <key>CFBundleIconFile</key>
    <string>TinyGPT</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>TinyGPT — native macOS</string>
</dict>
</plist>
PLIST

# PkgInfo — legacy but some macOS code paths still check for it.
echo -n "APPL????" > "$APP/Contents/PkgInfo"

# Ad-hoc sign so the binary at least has a valid signature for Gatekeeper
# to evaluate. This is NOT a Developer ID signature; first launch still
# prompts the user to confirm. A real distribution build would replace
# this with `codesign --options runtime --sign "Developer ID Application: …"`.
echo "== ad-hoc codesign"
# Make every file in the bundle writable so codesign can write its
# extended-attribute signatures. SwiftPM hands the metallib over as
# read-only which trips codesign --force.
chmod -R u+w "$APP"
# Strip any inherited signatures on payload binaries before re-signing
# the whole bundle. Cleanest path.
codesign --remove-signature "$APP/Contents/MacOS/TinyGPT" 2>/dev/null || true
codesign --force --deep --sign - "$APP" 2>&1 | sed 's/^/  /' || \
    echo "  (codesign failed — app should still launch via right-click → Open)"

echo ""
echo "✓ wrote $APP"
echo "  size: $(du -sh "$APP" | cut -f1)"
echo ""
echo "launch with:  open \"$APP\""
echo "install with: cp -r \"$APP\" /Applications/"
