"""Fetch one object's observations from the Minor Planet Center get-obs API (GET with a JSON body), politely: one request per 20 s (the MPC robots.txt Crawl-Delay), a descriptive User-Agent.

usage: python scripts/fetch_mpc_obs.py 433 4 1   ->  D:/datasets/mpc/obs80_<designation>.txt (MPC 80-column format). Used for C24; the whole catalogue is not fetched this way (see C24).
"""
import json, sys, time, urllib.request, pathlib
out = pathlib.Path("D:/datasets/mpc")
UA = "sigma-ground research (personal; aaron.s.rhodes@gmail.com); one request per 20 s"
bodies = sys.argv[1:]
for i, b in enumerate(bodies):
    if i: time.sleep(20)
    req = urllib.request.Request("https://data.minorplanetcenter.net/api/get-obs",
        data=json.dumps({"desigs": [b], "output_format": ["OBS80"]}).encode(),
        method="GET", headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
    except Exception as e:
        print(b, "FAILED", e); continue
    d = json.loads(body)
    txt = d[0]["OBS80"] if isinstance(d, list) else d.get("OBS80") or json.dumps(d)[:200]
    (out / f"obs80_{b}.txt").write_text(txt, encoding="utf-8")
    n = txt.count("\n") + 1
    print(b, "bytes", len(txt.encode()), "lines", n, flush=True)
