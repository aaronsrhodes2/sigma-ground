#!/usr/bin/env bash
# Package the Mentat stage addon (stage/gmod/addon) as a .gma, the form a dedicated server loads from its addons/ folder.
#
#   stage/gmod/package_addon.sh [out_dir]        # default: stage/gmod/dist/mentat-stage.gma
#
# Uses gmad.exe from the game install (GMOD_DIR, default /d/Steam/steamapps/common/GarrysMod). The addon is Lua only, so the
# archive is small (about 0.5 MB) and has no content to license.
# Shell note (PLATINUM §20): bash first.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GMOD_DIR="${GMOD_DIR:-/d/Steam/steamapps/common/GarrysMod}"
GMAD="$GMOD_DIR/bin/gmad.exe"
OUT_DIR="${1:-$HERE/dist}"
[ -x "$GMAD" ] || { echo "gmad.exe not found at $GMAD (set GMOD_DIR)" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/mentat-stage" "$OUT_DIR"
cp -r "$HERE/addon/lua" "$STAGE/mentat-stage/lua"
cat > "$STAGE/mentat-stage/addon.json" <<'JSON'
{
  "title": "Mentat stage: holodeck and proximity voice",
  "type": "tool",
  "tags": ["fun", "roleplay"],
  "ignore": []
}
JSON
"$GMAD" create -folder "$(cygpath -w "$STAGE/mentat-stage")" -out "$(cygpath -w "$OUT_DIR/mentat-stage.gma")" >/dev/null
ls -l "$OUT_DIR/mentat-stage.gma"
echo "Place it in garrysmod/addons/ on the server (or serve it from a Workshop collection). Lua files are sent to players automatically."
