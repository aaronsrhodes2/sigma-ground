#!/usr/bin/env bash
# Package the Mentat stage addon (stage/gmod/addon) as a .gma, the form a dedicated server loads from its addons/ folder.
#
#   stage/gmod/package_addon.sh [out_dir]        # default: stage/gmod/dist/mentat-stage.gma
#
# Uses gmad from the game install (GMOD_DIR, default /d/Steam/steamapps/common/GarrysMod; GMAD names the program directly,
# for a machine where it lives elsewhere or has another name). The addon is Lua only, so the archive is small (about 0.5 MB)
# and has no content to license. gmad accepts only whitelisted file types: when it refuses a file, its own message (which
# names the file) is shown and nothing is written.
# Shell note (PLATINUM §20): bash first.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GMOD_DIR="${GMOD_DIR:-/d/Steam/steamapps/common/GarrysMod}"
GMAD="${GMAD:-$GMOD_DIR/bin/gmad.exe}"
OUT_DIR="${1:-$HERE/dist}"
[ -x "$GMAD" ] || { echo "gmad not found at $GMAD (set GMOD_DIR, or GMAD to the program)" >&2; exit 1; }

# gmad.exe wants Windows paths; under Git Bash cygpath makes them, anywhere else the path is already native
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s\n' "$1"; fi; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/mentat-stage" "$OUT_DIR"
cp -r "$HERE/addon/lua" "$STAGE/mentat-stage/lua"
cat > "$STAGE/mentat-stage/addon.json" <<'JSON'
{
  "title": "Mentat stage: holodeck and proximity voice",
  "type": "tool",
  "tags": ["fun", "roleplay"],
  "ignore": ["*.bak", "*~", "*.swp", ".DS_Store", "Thumbs.db"]
}
JSON
# a failed run must not leave the previous archive looking current
rm -f "$OUT_DIR/mentat-stage.gma"
if ! LOG="$("$GMAD" create -folder "$(native "$STAGE/mentat-stage")" -out "$(native "$OUT_DIR/mentat-stage.gma")" 2>&1)"; then
  printf '%s\n' "$LOG" | tail -12 >&2
  echo "package_addon: gmad refused the addon (see above); nothing was written" >&2
  exit 1
fi
[ -s "$OUT_DIR/mentat-stage.gma" ] || { printf '%s\n' "$LOG" | tail -12 >&2; echo "package_addon: gmad reported success but wrote no archive" >&2; exit 1; }
ls -l "$OUT_DIR/mentat-stage.gma"
echo "Place it in garrysmod/addons/ on the server (or serve it from a Workshop collection). Lua files are sent to players automatically."
