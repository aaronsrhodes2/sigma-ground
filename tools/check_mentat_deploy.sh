#!/usr/bin/env bash
# Is the live Mentat (SkippyTel's skippytel-mentat container) running master?
#
#   bash tools/check_mentat_deploy.sh            # exit 0: running origin/master; exit 1: drift; exit 2: cannot tell
#   MENTAT_CONTAINER=skippytel-mentat bash tools/check_mentat_deploy.sh
#
# Reads SIGMA_GROUND_REF from inside the container (the commit the image was built from) and compares it with
# origin/master of this checkout. Findings F3 of misc/MENTAT_AUDIT_2026-10-13.md: the live copy had silently
# fallen behind master; nothing said so. Run it after every deploy and from the daily job.
set -euo pipefail
CONTAINER="${MENTAT_CONTAINER:-skippytel-mentat}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

git fetch -q origin master 2>/dev/null || echo "warning: could not fetch origin (comparing with the last fetched origin/master)" >&2
master="$(git rev-parse origin/master)"

live="$(MSYS_NO_PATHCONV=1 docker exec "$CONTAINER" printenv SIGMA_GROUND_REF 2>/dev/null || true)"
if [[ -z "$live" ]]; then
  echo "cannot tell: no SIGMA_GROUND_REF in container '$CONTAINER' (is it running?)" >&2
  exit 2
fi

if [[ "$live" == "$master" ]]; then
  echo "ok: $CONTAINER runs origin/master (${master:0:12})"
  exit 0
fi

behind="$(git rev-list --count "$live..$master" 2>/dev/null || echo '?')"
echo "DRIFT: $CONTAINER runs ${live:0:12}; origin/master is ${master:0:12} ($behind commits ahead of the container)" >&2
echo "fix: bump SIGMA_GROUND_REF in SkippyTel/mentat/Dockerfile and .github/workflows/deploy.yml, then rebuild mentat-home" >&2
exit 1
