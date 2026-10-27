#!/usr/bin/env bash
# Is the live Mentat (skippy-core's, the only one) running master?
#
#   bash tools/check_mentat_deploy.sh            # exit 0: running origin/master; exit 1: drift; exit 2: cannot tell
#   MENTAT_URL=https://skippy-core.tailab2cc7.ts.net:8443 bash tools/check_mentat_deploy.sh
#
# Reads SIGMA_GROUND_REF from Mentat's /version (the commit the image was built from) and compares it with
# origin/master of this checkout. Findings F3 of misc/MENTAT_AUDIT_2026-10-13.md: the live copy had silently
# fallen behind master; nothing said so. Run it after every deploy and from the daily job.
# (Until 2026-10-27 this read the ref with `docker exec` from the PC's `skippytel-mentat` container; that container is gone.)
set -euo pipefail
URL="${MENTAT_URL:-http://skippy-core:7350}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

git fetch -q origin master 2>/dev/null || echo "warning: could not fetch origin (comparing with the last fetched origin/master)" >&2
master="$(git rev-parse origin/master)"

live="$(curl -s -m 15 "${URL%/}/version" | python -c 'import json,sys; print(json.load(sys.stdin).get("sigma_ground_ref",""))' 2>/dev/null || true)"
if [[ -z "$live" || "$live" == "dev" ]]; then
  echo "cannot tell: no SIGMA_GROUND_REF from $URL/version (is Mentat up, and is this PC on the tailnet?)" >&2
  exit 2
fi

if [[ "$live" == "$master" ]]; then
  echo "ok: Mentat at $URL runs origin/master (${master:0:12})"
  exit 0
fi

behind="$(git rev-list --count "$live..$master" 2>/dev/null || echo '?')"
echo "DRIFT: Mentat at $URL runs ${live:0:12}; origin/master is ${master:0:12} ($behind commits ahead of it)" >&2
echo "fix: bump pins.SIGMA_GROUND_REF in SkippyTel core/cd/deploy.json (and the ARG default in mentat/Dockerfile), push SkippyTel; the core deployer rebuilds Mentat" >&2
exit 1
