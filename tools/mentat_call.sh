#!/usr/bin/env bash
# Mentat does the physics (PLATINUM_RULES §21). Call one Mentat tool over the HTTP JSON API that runs
# inside SkippyTel's `skippytel-mentat` container (no published port — reached with `docker exec`).
#
#   tools/mentat_call.sh list                              # names + domains of the 200+ public tools
#   tools/mentat_call.sh hawking_temperature '{"mass_kg": 1.989e30}'
#   tools/mentat_call.sh ask 'What is the Schwarzschild radius of the Sun?'   # plain-English front door
#
# Prints the JSON reply ({"ok": true, "result": {...}}); exit 1 if the tool errored.
set -euo pipefail
CONTAINER="${MENTAT_CONTAINER:-skippytel-mentat}"
name="${1:?tool name, 'list' or 'ask'}"; shift || true
payload="${1:-{\}}"
MSYS_NO_PATHCONV=1 docker exec -i "$CONTAINER" python - "$name" "$payload" <<'PY'
import json, sys, urllib.request
name, payload = sys.argv[1], sys.argv[2]
base = "http://localhost:8080"
if name == "list":
    tools = json.load(urllib.request.urlopen(base + "/tools", timeout=30))
    for t in sorted(tools, key=lambda t: (t.get("domain", ""), t["name"])):
        print(f'{t.get("domain", ""):12s} {t["name"]:40s} {t.get("description", "").splitlines()[0][:90] if t.get("description") else ""}')
    sys.exit(0)
if name == "ask":
    body = json.dumps({"question": payload}).encode(); url = base + "/ask"
else:
    body = payload.encode(); url = base + f"/tools/{name}"
req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
try:
    out = urllib.request.urlopen(req, timeout=120).read().decode()
except urllib.error.HTTPError as e:
    out = e.read().decode()
print(out)
sys.exit(0 if json.loads(out).get("ok") else 1)
PY
