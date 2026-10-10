"""Mentat client (PLATINUM §21) — call sigma-ground's physics tools running inside SkippyTel's
`skippytel-mentat` container, from Python, via `docker exec` (the container publishes no port).

    from mentat_client import mentat
    value, source = mentat("hawking_temperature", {"mass_kg": 1.989e30})
"""
import json, os, subprocess

_INNER = r'''
import json, sys, urllib.request
name, payload = sys.argv[1], sys.argv[2]
url = "http://localhost:8080/ask" if name == "ask" else f"http://localhost:8080/tools/{name}"
body = (json.dumps({"question": payload}) if name == "ask" else payload).encode()
req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
try:
    print(urllib.request.urlopen(req, timeout=120).read().decode())
except urllib.error.HTTPError as e:
    print(e.read().decode())
'''

def mentat_raw(tool, args=None, container=None):
    container = container or os.environ.get("MENTAT_CONTAINER", "skippytel-mentat")
    payload = args if isinstance(args, str) else json.dumps(args or {})
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    out = subprocess.run(["docker", "exec", "-i", container, "python", "-c", _INNER, tool, payload],
                         capture_output=True, text=True, env=env)
    if not out.stdout.strip():
        raise RuntimeError(f"mentat {tool}: no reply; stderr={out.stderr[-400:]}")
    return json.loads(out.stdout)

def mentat(tool, args=None):
    """Return (value, source) for a Mentat tool; raises on {"ok": false}."""
    r = mentat_raw(tool, args)
    if not r.get("ok"):
        raise RuntimeError(f"mentat {tool}: {r.get('error')}")
    res = r["result"]
    return res.get("value"), res.get("source")

if __name__ == "__main__":
    import sys
    print(json.dumps(mentat_raw(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "{}"), indent=1))
