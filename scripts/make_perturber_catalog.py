"""Build the DE440 perturber catalogue: the 343 asteroids, the 30 Kuiper-belt objects and the Kuiper-belt ring, with DE440's own masses and the elements we hold.

Masses: the constants MAnnnn of the DE440 ASCII header (JPL, ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de440/header.440; au^3/day^2, converted with the header's AU).
  MA0001-MA1467 = the 343 asteroids, MA8001-MA8030 = the 30 Kuiper-belt objects, MA8201-MA8236 = the 36 equal point masses of the ring (Park et al. 2021, AJ 161:105, sec. 2.2).
Elements: JPL SBDB osculating elements, from fixtures/sbdb_asteroids_named_mba.json (main-belt asteroids with a name); an asteroid that is not in that file has elements = null and
is reported (it cannot be placed until its elements are added). Fallback: fixtures/sbdb_asteroids_54.json.

usage: python scripts/make_perturber_catalog.py [--header PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
HEADER = Path("D:/datasets/jpl/de440_ascii_header/header.440")


def parse_header(path: Path) -> dict:
    txt = path.read_text().splitlines()

    def group(n):
        i = next(k for k, l in enumerate(txt) if l.strip() == f"GROUP   {n}")
        j = next(k for k in range(i + 1, len(txt)) if txt[k].startswith("GROUP"))
        return txt[i + 1:j]
    names = []
    for l in group(1040):
        if not l.strip() or l.strip().isdigit():
            continue
        names += [l[k:k + 8].strip() for k in range(2, len(l), 8) if l[k:k + 8].strip()]
    vals = []
    for l in group(1041):
        if not l.strip() or re.fullmatch(r"\d+", l.strip()):
            continue
        vals += [float(x.replace("D", "E")) for x in l.split()]
    if len(names) != len(vals):
        raise ValueError(f"header: {len(names)} names but {len(vals)} values")
    return dict(zip(names, vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", type=Path, default=HEADER)
    ap.add_argument("--out", type=Path, default=FIX / "de440_perturbers_v2physics.json")
    a = ap.parse_args()
    c = parse_header(a.header)
    conv = c["AU"] ** 3 / 86400.0 ** 2                                   # au^3/day^2 -> km^3/s^2
    ma = {int(k[2:]): v * conv for k, v in c.items() if re.fullmatch(r"MA\d{4}", k)}
    cat = json.loads((FIX / "sbdb_asteroids_named_mba.json").read_text(encoding="utf-8"))
    rows = {}
    for r in cat["data"]:
        d = dict(zip(cat["fields"], r))
        m = re.match(r"\s*(\d+)", d["full_name"])
        if m:
            rows[int(m.group(1))] = d
    rows54 = {}
    for b in json.loads((FIX / "sbdb_asteroids_54.json").read_text(encoding="utf-8"))["bodies"]:      # fallback: the curated file also holds near-Earth objects such as Eros
        m = re.match(r"\s*(\d+)", str(b.get("full_name") or ""))
        if m:
            rows54[int(m.group(1))] = b
    asteroids, missing = [], []
    for num in sorted(n for n in ma if n < 8000):
        d = rows.get(num)
        el = None
        if d is not None:
            el = {"epoch_jd_tdb": float(d["epoch"]), "a_au": float(d["a"]), "e": float(d["e"]), "i_deg": float(d["i"]), "om_deg": float(d["om"]), "w_deg": float(d["w"]), "ma_deg": float(d["ma"])}
        elif num in rows54:
            b = rows54[num]
            d = {"name": b.get("name"), "diameter": b.get("diameter_km")}
            el = {"epoch_jd_tdb": float(b["epoch_jd_tdb"]), "a_au": float(b["a_au"]), "e": float(b["e"]), "i_deg": float(b["i_deg"]), "om_deg": float(b["om_deg"]), "w_deg": float(b["w_deg"]), "ma_deg": float(b["ma_deg"])}
        else:
            missing.append(num)
        asteroids.append({"number": num, "name": (d or {}).get("name"), "gm_km3_s2": ma[num], "diameter_km": float(d["diameter"]) if d and d.get("diameter") else None, "elements": el})
    kbo = [{"id": n, "gm_km3_s2": ma[n], "elements": None} for n in sorted(ma) if 8000 < n < 8100]
    ring_gm = [ma[n] for n in sorted(ma) if n >= 8200]
    doc = {"metadata": {"label": "v2physics_2026-10-14",
                        "masses": "JPL DE440 ASCII header header.440, constants MAnnnn (au^3/day^2 x AU^3/86400^2, AU = %.1f km)" % c["AU"],
                        "elements": "JPL SBDB osculating elements, fixtures/sbdb_asteroids_named_mba.json (heliocentric ecliptic J2000)",
                        "set": "Park et al. 2021 AJ 161:105 sec. 2.2: 343 asteroids (the DE430 set), 30 Kuiper-belt objects, a ring of 36 equal point masses in the ecliptic at a = 44 au",
                        "asteroids_without_elements": missing, "n_asteroids": len(asteroids), "n_kbo": len(kbo), "generator": "scripts/make_perturber_catalog.py"},
           "asteroids": asteroids, "kbo": kbo, "ring": {"n_points": len(ring_gm), "a_au": 44.0, "gm_each_km3_s2": ring_gm[0], "gm_total_km3_s2": sum(ring_gm), "all_equal": len(set(round(g, 9) for g in ring_gm)) == 1}}
    a.out.write_text(json.dumps(doc, indent=0))
    gm_all = sum(x["gm_km3_s2"] for x in asteroids)
    gm_have = sum(x["gm_km3_s2"] for x in asteroids if x["elements"])
    print(f"{len(asteroids)} asteroids, GM total {gm_all:.2f} km3/s2; with elements {sum(1 for x in asteroids if x['elements'])} ({100 * gm_have / gm_all:.1f} % of the mass); missing {len(missing)}: {missing}")
    print(f"{len(kbo)} Kuiper-belt objects (GM total {sum(x['gm_km3_s2'] for x in kbo):.1f}, no elements held); ring {doc['ring']['n_points']} points, GM total {doc['ring']['gm_total_km3_s2']:.1f}, equal: {doc['ring']['all_equal']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
