#!/usr/bin/env bash
# Run the SkippyCraft client gametest headless: the real client in Docker (virtual display, software OpenGL), a scripted player
# in a private singleplayer world. Screenshots land in mod/build/run/clientGameTest/screenshots. Needs nothing on screen.
#
#   bash stage/skippycraft/mod/test.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export MSYS_NO_PATHCONV=1
win="$(cd "$here" && pwd -W 2>/dev/null || pwd)"
(cd "$here" && python gen_test_records.py >/dev/null)
docker build -q -t skippycraft-gametest -f "$win/test.Dockerfile" "$win" >/dev/null
docker volume create skippycraft-gradle >/dev/null
docker run --rm -v "$win:/mod" -v skippycraft-gradle:/root/.gradle -w /mod skippycraft-gametest \
  bash -c 'chmod +x gradlew && (Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &) && sleep 3 && export DISPLAY=:99 && ./gradlew runClientGameTest --no-daemon --console=plain -Dorg.gradle.configuration-cache=false'
find "$here/build/run" -name "*.png" -path "*screenshots*" 2>/dev/null
