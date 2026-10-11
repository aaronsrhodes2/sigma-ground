#!/usr/bin/env bash
# Copy the PRIVATE Skippy body files (the print kit's six SVM2 parts + manifest.json, from stage/skippyverse/private/skippy/, gitignored,
# never committed) into the data folder(s) the game and the dedicated server read them from: garrysmod/data/mentat/skippy/.
# The server (avatar/body_net.lua, convar mentat_avatar_body_dir) reads them from there and sends them to the clients once.
#
#   stage/gmod/install_skippy_body.sh [source dir]        # default stage/skippyverse/private/skippy
#   stage/gmod/install_skippy_body.sh --remove            # take them out again (the avatar is the procedural can)
#
# Env: GMOD_GAME_DATA (default /d/Steam/steamapps/common/GarrysMod/garrysmod/data), GMOD_DS_DIR (default /d/Aaron/servers/gmod-ds).
# A cloud server: copy the same eight files by hand into <server>/garrysmod/data/mentat/skippy/ (outside git); `mentat_avatar_body_reload`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/../skippyverse/private/skippy}"
GAME="${GMOD_GAME_DATA:-/d/Steam/steamapps/common/GarrysMod/garrysmod/data}/mentat/skippy"
DS="${GMOD_DS_DIR:-/d/Aaron/servers/gmod-ds}/garrysmod/data/mentat/skippy"
FILES="manifest.json can.svm2 jacket.svm2 head.svm2 hat.svm2 hand_a.svm2 hand_b.svm2"
if [ "${1:-}" = "--remove" ]; then
  for d in "$GAME" "$DS"; do for f in $FILES; do rm -f "$d/$f"; done; rmdir "$d" 2>/dev/null; done
  echo "skippy body: removed from the data folders"; exit 0
fi
for f in $FILES; do [ -f "$SRC/$f" ] || { echo "skippy body: $SRC/$f is missing; the avatar stays the procedural can" >&2; exit 1; }; done
n=0
for d in "$GAME" "$DS"; do
  parent="$(dirname "$(dirname "$d")")"
  [ -d "$parent" ] || continue
  mkdir -p "$d" && for f in $FILES; do cp -f "$SRC/$f" "$d/$f"; done && { echo "skippy body: installed to $d"; n=$((n+1)); }
done
[ "$n" -gt 0 ] || { echo "skippy body: no game or server data folder found" >&2; exit 1; }
