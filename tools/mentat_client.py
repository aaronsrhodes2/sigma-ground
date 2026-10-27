"""Mentat client (PLATINUM §21) — call sigma-ground's physics tools on Mentat, from Python, over HTTP.

There is ONE Mentat: skippy-core's (the chat model, the router, the DE440 data). From the house it is reached over the tailnet at http://skippy-core:7350; its public address (for
the Captain's friends) is https://skippy-core.tailab2cc7.ts.net:8443. MENTAT_URL overrides. (Until 2026-10-27 this ran `docker exec` into a `skippytel-mentat` container on the PC;
that container is gone.)

    from mentat_client import mentat
    value, source = mentat("hawking_temperature", {"mass_kg": 1.989e30})
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = "http://skippy-core:7350"


def mentat_raw(tool, args=None, container=None):
    """POST the tool (or `ask`) and return Mentat's reply as a dict. `container` is accepted for old callers and ignored."""
    base = os.environ.get("MENTAT_URL", DEFAULT_URL).rstrip("/")
    if tool == "ask":
        body = json.dumps({"question": args} if isinstance(args, str) else args or {}).encode()
        url = base + "/ask"
    else:
        body = (args if isinstance(args, str) else json.dumps(args or {})).encode()
        url = base + f"/tools/{tool}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json", "X-Skippy-Caller": "mentat_client.py"})
    try:
        out = urllib.request.urlopen(req, timeout=120).read().decode()
    except urllib.error.HTTPError as e:
        out = e.read().decode()
    except OSError as e:
        raise RuntimeError(f"mentat {tool}: {base} is unreachable ({e}); is skippy-core up, and is this PC on the tailnet?") from e
    if not out.strip():
        raise RuntimeError(f"mentat {tool}: empty reply from {base}")
    return json.loads(out)


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
