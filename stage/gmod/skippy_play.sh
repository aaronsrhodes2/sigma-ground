#!/usr/bin/env bash
# One command to play with the holodeck by voice: the Skippy bridge, your microphone and the game, started in order and wired together.
#   bash stage/gmod/skippy_play.sh [--map gm_construct] [--key x] [--device NAME] [--text-only]
#   bash stage/gmod/skippy_play.sh --selftest-voice ["add two couches"] [--stop-after]
# See stage/gmod/holodeck/play.py for what it does and checks.
exec python "$(dirname "${BASH_SOURCE[0]}")/holodeck/play.py" "$@"
