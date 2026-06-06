"""Distill a canonical 2D outline per noun from Quick, Draw! (CC BY 4.0).

Quick Draw doodles are messy multi-stroke sketches; Deckard's Outline primitive
wants ONE clean, representative closed profile per noun. For a category this
tool, politely and once:

  1. streams the first N recognized doodles (a partial download, UA-identified),
  2. takes each doodle's LONGEST stroke -- the main contour,
  3. normalizes it: centre, PCA-rotate the principal axis to +x, scale to unit,
  4. resamples to a fixed point count, and
  5. picks the MEDOID -- the outline most typical of the set (reflection-aware) --
     so the result is a representative shape, not a random doodle (the Captain's note).

Writes ``sigma_ground/inventory/data/outlines/<slug>.json`` (a closed, normalized
polyline + CC-BY attribution). Run offline-ish; the result is shipped, not the dataset.

    python tools/distill_quickdraw.py feather leaf

Attribution: "Quick, Draw! by Google, Inc., CC BY 4.0".
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import urllib.parse
import urllib.request

_GCS = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{cat}.ndjson"
_UA = ("Deckard/0.2 (sigma-ground physics shape researcher; "
       "https://github.com/aaronsrhodes2/sigma-ground)")
_OUT_DIR = (pathlib.Path(__file__).resolve().parents[1]
            / "sigma_ground" / "inventory" / "data" / "outlines")
_N_SAMPLE = 150        # doodles to consider (partial stream)
_N_POINTS = 28         # points in the canonical outline


def _stream_drawings(category: str, n: int):
    """Stream the first n RECOGNIZED doodles' stroke lists (partial download)."""
    url = _GCS.format(cat=urllib.parse.quote(category))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    out = []
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if d.get("recognized") and d.get("drawing"):
                out.append(d["drawing"])
            if len(out) >= n:
                break
    return out


def _longest_stroke(drawing):
    """The doodle's longest stroke as [(x, y), ...] (Quick Draw: y is down)."""
    best = max(drawing, key=lambda s: len(s[0]))
    return [(float(x), -float(y)) for x, y in zip(best[0], best[1])]   # flip y -> up


def _normalize(pts):
    """Centre, PCA-rotate principal axis to +x, scale longest extent to 1."""
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    q = [(p[0] - cx, p[1] - cy) for p in pts]
    sxx = sum(u * u for u, v in q) / n
    syy = sum(v * v for u, v in q) / n
    sxy = sum(u * v for u, v in q) / n
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)      # principal-axis angle
    c, s = math.cos(-theta), math.sin(-theta)
    r = [(u * c - v * s, u * s + v * c) for u, v in q]
    ext = max(max(abs(u) for u, v in r), max(abs(v) for u, v in r)) or 1.0
    return [(u / ext, v / ext) for u, v in r]


def _resample(pts, m):
    """Resample a polyline to m points by arc length (closed loop)."""
    loop = pts + [pts[0]]
    seg = [math.dist(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]
    total = sum(seg) or 1.0
    cum = [0.0]
    for d in seg:
        cum.append(cum[-1] + d)
    out = []
    for k in range(m):
        target = total * k / m
        i = 0
        while i < len(seg) and cum[i + 1] < target:
            i += 1
        if i >= len(seg):
            out.append(loop[-1]); continue
        t = (target - cum[i]) / (seg[i] or 1.0)
        a, b = loop[i], loop[i + 1]
        out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _dist(a, b):
    """Min point-wise distance between two same-length outlines, reflection-aware."""
    d0 = sum(math.dist(a[i], b[i]) for i in range(len(a)))
    d1 = sum(math.dist(a[i], (b[i][0], -b[i][1])) for i in range(len(a)))   # mirror in x-axis
    return min(d0, d1)


def _medoid(outlines):
    """The outline minimizing total distance to all others."""
    best, best_cost = None, None
    for i, a in enumerate(outlines):
        cost = sum(_dist(a, b) for j, b in enumerate(outlines) if j != i)
        if best_cost is None or cost < best_cost:
            best, best_cost = a, cost
    return best


def distill(category: str) -> pathlib.Path:
    drawings = _stream_drawings(category, _N_SAMPLE)
    outlines = []
    for d in drawings:
        try:
            o = _resample(_normalize(_longest_stroke(d)), _N_POINTS)
            if len(o) == _N_POINTS:
                outlines.append(o)
        except Exception:
            continue
    if len(outlines) < 5:
        raise ValueError(f"too few usable outlines for '{category}' ({len(outlines)})")
    med = _medoid(outlines)
    slug = category.strip().lower().replace(" ", "_")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"{slug}.json"
    path.write_text(json.dumps({
        "object": category, "n_samples": len(outlines), "n_points": _N_POINTS,
        "profile": [[round(u, 4), round(v, 4)] for u, v in med],
        "source": "Quick, Draw! by Google, Inc.", "license": "CC BY 4.0",
    }, indent=1), encoding="utf-8")
    return path


def main(argv):
    cats = argv or ["feather"]
    ok, failed = [], []
    for i, cat in enumerate(cats):
        if i:
            time.sleep(1.0)                # polite: sequential, ~1 req/s
        try:
            p = distill(cat)
            print("wrote", p)
            ok.append(cat)
        except Exception as e:
            print(f"skip '{cat}': {type(e).__name__}: {e}")
            failed.append(cat)
    print(f"\n{len(ok)} written, {len(failed)} skipped" + (f": {failed}" if failed else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
