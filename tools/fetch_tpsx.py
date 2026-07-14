"""Fetch + distill the NASA TPSX materials database (the Captain's source).

TPSX (tpsx.arc.nasa.gov) publishes thermal-protection material property sheets
as server-rendered pages, one per Material?id=N — each a proper cited table
(Property / Value / Units / Uncertainty / Source / Reference). US-government
work, freely retrievable. No list API exists, so we sweep ids politely
(1 request/s, raw HTML cached under D:/datasets/tpsx/ so a re-run costs
nothing) and stop after a run of consecutive misses.

Distilled output: sigma_ground/inventory/data/tpsx_materials.json — per
material the parsed property rows (value, units, uncertainty, source type,
reference note) with the page id as citation.

Run:  python tools/fetch_tpsx.py [max_id]        (default sweep cap 700)
"""
import json
import os
import re
import sys
import time
import urllib.request

_BASE = "https://tpsx.arc.nasa.gov/Material?id="
_CACHE = "D:/datasets/tpsx"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "tpsx_materials.json")
_MISS_RUN_STOP = 40                      # consecutive empty ids → end of catalog

_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r'<h2[^>]*class="material-name-font"[^>]*>(.*?)</h2>', re.S)


def _fetch(mid: int) -> str | None:
    os.makedirs(_CACHE, exist_ok=True)
    cache = os.path.join(_CACHE, f"material_{mid}.html")
    if os.path.exists(cache):
        return open(cache, encoding="utf-8", errors="replace").read()
    time.sleep(1.0)                                     # polite: 1 req/s
    req = urllib.request.Request(_BASE + str(mid), headers={
        "User-Agent": "sigma-ground-data-lane/1.0 (research; polite)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    open(cache, "w", encoding="utf-8").write(txt)
    return txt


def _clean(s: str) -> str:
    return _TAG.sub("", s).replace("&nbsp;", " ").replace("&amp;", "&").strip()


def _parse(html: str):
    """(material_name, [property rows]) or None if the page carries no table."""
    rows = []
    for row in _ROW.findall(html):
        cells = [_clean(c) for c in _CELL.findall(row)]
        if len(cells) >= 5 and cells[0] and cells[0] != "Property":
            rows.append({"property": cells[0], "value": cells[1],
                         "units": cells[2], "uncertainty": cells[3],
                         "source_type": cells[4],
                         "reference": cells[6] if len(cells) > 6 else ""})
    if not rows:
        return None
    m = _TITLE.search(html)
    name = _clean(m.group(1)) if m else ""
    return name or "(unnamed)", rows


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    out = {"_meta": {
        "source": "NASA TPSX Materials Database (tpsx.arc.nasa.gov) — "
                  "US government work, freely retrievable; page id cited "
                  "per material",
        "generated_by": "tools/fetch_tpsx.py"}}
    misses = 0
    for mid in range(1, cap + 1):
        html = _fetch(mid)
        got = _parse(html) if html else None
        if got is None:
            misses += 1
            if misses >= _MISS_RUN_STOP:
                print(f"  (stopping: {misses} consecutive misses at id {mid})")
                break
            continue
        misses = 0
        name, rows = got
        out[f"id_{mid}"] = {"name": name, "tpsx_id": mid, "properties": rows}
        if mid % 25 == 0 or mid < 4:
            print(f"  + id {mid}: {name[:50]} ({len(rows)} property rows)")
    dest = os.path.abspath(_OUT)
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    n = len(out) - 1
    nprops = sum(len(v["properties"]) for k, v in out.items() if k != "_meta")
    print(f"\nwrote {dest}: {n} materials, {nprops} property rows")


if __name__ == "__main__":
    main()
