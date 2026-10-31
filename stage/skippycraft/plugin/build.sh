#!/usr/bin/env bash
# Build the SkippyCraft plugin into dev-server/data/plugins/, compiled in Docker (eclipse-temurin:25-jdk) against the
# Paper API jars the server itself downloaded (dev-server/data/libraries): nothing to install on skippy-pc.
#
#   bash stage/skippycraft/plugin/build.sh      # then restart the server (docker compose restart skippycraft)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
data="$here/../dev-server/data"
[ -d "$data/libraries" ] || { echo "start the server once first: it downloads the Paper libraries" >&2; exit 1; }
export MSYS_NO_PATHCONV=1
docker run --rm -v "$here:/plugin" -v "$data/libraries:/libs:ro" -v "$data/plugins:/out" eclipse-temurin:25-jdk bash -c '
  set -e
  cp=$(find /libs -name "*.jar" | tr "\n" ":")
  rm -rf /tmp/build && mkdir -p /tmp/build
  javac -proc:none -d /tmp/build -cp "$cp" $(find /plugin/src -name "*.java")
  cp /plugin/plugin.yml /tmp/build/
  jar cf /out/skippycraft.jar -C /tmp/build .
  echo built /out/skippycraft.jar'
