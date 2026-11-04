#!/usr/bin/env bash
# Build the SkippyCraft Fabric mod in Docker (eclipse-temurin:25-jdk + a cached Gradle volume) and copy the jar to
# dev-server/data/mods. Fabric API goes there too (fabric-api-0.165.2+26.3.jar, from Modrinth). Then restart the server:
#
#   bash stage/skippycraft/mod/build.sh && (cd stage/skippycraft/dev-server && docker compose restart skippycraft)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mods="$here/../dev-server/data/mods"
mkdir -p "$mods"
export MSYS_NO_PATHCONV=1
docker volume create skippycraft-gradle >/dev/null
docker run --rm -v "$(cd "$here" && pwd -W 2>/dev/null || pwd):/mod" -v skippycraft-gradle:/root/.gradle -w /mod eclipse-temurin:25-jdk \
  bash -c 'chmod +x gradlew && ./gradlew build --no-daemon -q -Dorg.gradle.configuration-cache=false'
cp "$here/build/libs/skippycraft-1.0.0.jar" "$mods/skippycraft.jar"
echo "built $mods/skippycraft.jar"
