#!/bin/bash
# Rasterizes assets/icon.svg into assets/AppIcon.icns using only tools
# already present on macOS (qlmanage, sips, iconutil) - no new dependencies.
# Re-run this any time assets/icon.svg changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SVG_PATH="$PROJECT_ROOT/assets/icon.svg"
ICNS_PATH="$PROJECT_ROOT/assets/AppIcon.icns"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

ICONSET="$WORKDIR/AppIcon.iconset"
mkdir -p "$ICONSET"

qlmanage -t -s 1024 -o "$WORKDIR" "$SVG_PATH" >/dev/null
MASTER_PNG="$WORKDIR/icon.svg.png"

if [ ! -f "$MASTER_PNG" ]; then
    echo "qlmanage failed to produce a thumbnail from $SVG_PATH" >&2
    exit 1
fi

for size in 16 32 128 256 512; do
    double=$((size * 2))
    sips -z "$size" "$size" "$MASTER_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z "$double" "$double" "$MASTER_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$ICNS_PATH"
echo "Wrote $ICNS_PATH"
