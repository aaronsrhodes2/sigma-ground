"""Merge a second replay pass over the transport-failed rows into the first: python merge_replay.py FIRST.json SECOND.json -> FIRST.json updated."""
import json, sys
first, second = sys.argv[1], sys.argv[2]
A = json.load(open(first, encoding="utf-8")); B = {r["id"]: r for r in json.load(open(second, encoding="utf-8"))["rows"]}
n = 0
for i, r in enumerate(A["rows"]):
    if r["id"] in B:
        A["rows"][i] = B[r["id"]]; n += 1
rows = A["rows"]
A["summary"]["ok"] = sum(r["ok"] for r in rows); A["summary"]["n"] = len(rows)
A["summary"]["merged_rows"] = n
A["summary"]["undecided_left"] = sum(1 for r in rows if (r.get("router_exc") or "").split(":")[0] in ("ReadError", "ConnectError", "HTTPStatusError", "RemoteProtocolError", "ConnectTimeout", "ReadTimeout")
                                     or (r.get("route") is None and r.get("value") is None and r.get("router_error") is None and r.get("router_decline") is None and r.get("router_clarify") is None))
json.dump(A, open(first, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: A["summary"][k] for k in ("part", "n", "ok", "merged_rows", "undecided_left")}))
