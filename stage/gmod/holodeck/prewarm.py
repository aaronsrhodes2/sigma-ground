"""Pack the common found objects ahead of time, so the first player to ask for one gets it from the PC cache.

    python stage/gmod/holodeck/prewarm.py                    # print the plan and the download size; does nothing else
    python stage/gmod/holodeck/prewarm.py --go               # download and pack what is not cached yet
    python stage/gmod/holodeck/prewarm.py --refresh-sizes    # read the sources' own file listings (about 10 KB each, no model is downloaded)
                                                             # and rewrite approx_mb in prewarm_list.json

The list is `prewarm_list.json` (tracked): nouns the game's own library lacks (a couch, a lamp, a mug come from the library and need no
cache), each resolved by the SkippyVerse resolver to ONE card, with the download size the plan shows. Official public sources only
(the bridge's own download hosts: Poly Haven, Gazebo Fuel, NASA); every asset goes through the same code a player's request does
(licence gate, 40 MB per asset, one download at a time, textures once), so what lands in the cache is byte-for-byte what a live request would
have stored. The whole list must stay under TOTAL_CAP_MB; --go refuses a plan over it. Nothing already cached is touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))              # stage/
LIST = HERE / "prewarm_list.json"
TOTAL_CAP_MB = 150.0


def load_list(path: Path = LIST) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    items = doc["items"]
    for it in items:
        if not (isinstance(it.get("noun"), str) and isinstance(it.get("card"), str)):
            raise ValueError(f"bad prewarm entry {it!r}")
    return items


def plan(items, bridge, cache_dir=None) -> dict:
    """-> {"rows": [{noun, card, mb, status}], "to_fetch": n, "mb": total MB still to download}. Offline: reads the index and the caches."""
    acquire, _, _, _ = bridge._skippyverse()
    from skippyverse import meshpack
    ix = bridge.found_index()
    pc = bridge.pack_cache()
    rows, total = [], 0.0
    for it in items:
        row = {"noun": it["noun"], "card": it["card"], "mb": float(it.get("approx_mb", 0.0))}
        try:
            card = ix.card(it["card"])
        except Exception:
            card = None
        if not card:
            row["status"] = "not in the index"
        elif not card.get("redistributable"):
            row["status"] = "licence gate: not servable"
        elif pc is not None and pc.get(bridge.packcache_key(card["id"]), None, meshpack.PACK_VERSION) is not None:
            row["status"] = "cached"
        else:
            row["status"] = "to fetch"
            total += row["mb"]
        rows.append(row)
    return {"rows": rows, "to_fetch": sum(r["status"] == "to fetch" for r in rows), "mb": total}


def show(p: dict, out=print) -> None:
    out(f"{'noun':18} {'card':52} {'MB':>6}  status")
    for r in p["rows"]:
        out(f"{r['noun']:18} {r['card']:52} {r['mb']:6.1f}  {r['status']}")
    out(f"to fetch: {p['to_fetch']} of {len(p['rows'])}, about {p['mb']:.1f} MB to download (cap {TOTAL_CAP_MB:.0f} MB)")


def run(items, bridge, out=print, clock=time.monotonic) -> list[dict]:
    """Do the plan: for each card not yet cached, download, measure and pack it through the bridge's own /mesh code. Returns one result per item."""
    acquire, _, _, _ = bridge._skippyverse()
    ix = bridge.found_index()
    results = []
    for it in items:
        t0 = clock()
        res = {"noun": it["noun"], "card": it["card"]}
        try:
            card = ix.card(it["card"])
            if not card or not card.get("redistributable"):
                raise ValueError("not servable")
            pc = bridge.pack_cache()
            from skippyverse import meshpack
            if pc is not None and pc.get(bridge.packcache_key(card["id"]), None, meshpack.PACK_VERSION) is not None:
                res["status"] = "already cached"
            else:
                if (card.get("provenance") or {}).get("source") in ("Gazebo Fuel", "NASA 3D Resources") and not bridge.mesh_offer(card):
                    acquire.acquire_and_measure(card, fetch=bridge.bounded_fetch(time.monotonic() + 60), cache_dir=bridge.MESH_CACHE_DIR)   # (what /resolve does first)
                if not bridge.mesh_offer(card):
                    res["status"] = "box only (no drawable mesh)"
                else:
                    got = bridge.mesh_payload(card["id"], bridge.mesh_key(card["id"]), deadline=time.monotonic() + 120)
                    res.update(status="cached now", bytes=got["bytes"], tris=got["tris"])
        except Exception as e:
            res["status"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
        res["seconds"] = round(clock() - t0, 1)
        out(f"  {res['noun']:18} {res['status']}  ({res['seconds']} s)")
        results.append(res)
    return results


def refresh_sizes(items, bridge, fetch, path: Path = LIST) -> None:
    """Read each Poly Haven card's own file listing (the same one acquire.py reads) and store what its 1k glTF, .bin and used textures weigh."""
    from urllib.parse import quote
    ix = bridge.found_index()
    acquire, _, _, _ = bridge._skippyverse()
    for it in items:
        card = ix.card(it["card"])
        if (card.get("provenance") or {}).get("source") != "Poly Haven":
            continue
        raw = fetch(f"https://api.polyhaven.com/files/{quote(acquire.polyhaven_id(card))}", max_bytes=2_000_000)
        g = json.loads(raw.decode("utf-8"))["gltf"]
        entry = g["1k"]["gltf"] if "1k" in g else g[sorted(g)[0]]["gltf"]
        inc = entry.get("include") or {}
        n = int(entry.get("size") or 0) + sum(int(v.get("size") or 0) for v in inc.values() if int(v.get("size") or 0) <= acquire.TEXTURE_MAX)
        it["approx_mb"] = round(n / 1e6 + 0.1, 1)                    # an upper bound: every listed .bin and image under 8 MB (the look pass keeps only the colour and metal-rough maps)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    doc["items"] = items
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--go", action="store_true", help="really download and pack (without it, only the plan is printed)")
    ap.add_argument("--refresh-sizes", action="store_true", help="read the sources' file listings and update approx_mb (no model is downloaded)")
    a = ap.parse_args(argv)
    import skippy_bridge as bridge
    items = load_list()
    if a.refresh_sizes:
        refresh_sizes(items, bridge, bridge.bounded_fetch(time.monotonic() + 120))
        items = load_list()
    p = plan(items, bridge)
    show(p)
    if p["mb"] > TOTAL_CAP_MB:
        print(f"refusing: the plan is over {TOTAL_CAP_MB:.0f} MB; shorten prewarm_list.json")
        return 2
    if not a.go:
        print("nothing was downloaded. Re-run with --go to fetch and pack the rows marked 'to fetch'.")
        return 0
    if p["to_fetch"] == 0:
        print("everything is cached already.")
        return 0
    print("going:")
    results = run(items, bridge)
    bad = [r for r in results if r["status"].startswith("failed")]
    print(f"done: {sum(r['status'] == 'cached now' for r in results)} packed, {len(bad)} failed; cache {bridge.pack_cache().total_bytes() / 1e6:.1f} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
