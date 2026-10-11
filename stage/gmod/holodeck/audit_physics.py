"""Audit the holodeck's props against Deckard's compiled objects: is what the engine gives each thing a plausible real object?

    python stage/gmod/holodeck/audit_physics.py [--kinds couch,chair] [--out misc/HOLODECK_PHYSICS_AUDIT.md]

For every kind in the library this asks Deckard (the stack's matter compiler: sigma_ground.deckard) to compile "a <kind>" and puts
its mass, size, materials and parts beside what the Source engine's model carries (mass, size, collision-body volume and number of
convex pieces, the surface material it selects). It judges nothing by itself: the numbers are Deckard's and the engine's, and the
flags are only ratios between them. Where Deckard's own result looks wrong (a chair at 123 kg) that is REPORTED, not patched
(sigma-ground CLAUDE.md: diagnostics flag, they never fix).

Needs no game and no network; Deckard runs without its language model (allow_llm=False), so a name it cannot ground says so.
Writes data/physics_audit.json and the markdown report. `--from-json` re-derives the medians and the flags from the stored numbers
(no Deckard run) and rewrites both files, so a change to the checks below reaches the published report without a new run.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
HU_M = 0.01905


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def deckard_record(names: list[str]) -> dict:
    """Ask Deckard for the first of the kind's names it can ground (its catalog says 'a bookcase', ours says 'bookshelf')."""
    from sigma_ground import deckard
    t = time.time()
    spec = None
    asked = names[0]
    for name in names:
        try:
            spec = deckard.research(article(name), allow_llm=False)
        except Exception as e:                                           # noqa: BLE001 - report, do not hide
            return {"ok": False, "error": f"research failed: {type(e).__name__}: {e}"}
        asked = name
        if spec.identified:
            break
    out = {"ok": True, "asked": article(asked), "tried": [article(n) for n in names], "identified": bool(spec.identified), "kind": getattr(spec, "kind", None),
           "parts": [p.name for p in getattr(spec, "parts", [])][:40],
           "materials": sorted({str(getattr(p, "material", "")) for p in getattr(spec, "parts", []) if getattr(p, "material", None)}),
           "sources": [str(getattr(f, "source", f))[:80] for f in (getattr(spec, "sources", None) or [])][:6]}
    if not spec.identified:
        out["note"] = "Deckard could not ground this name"
        return out
    try:
        c = deckard.compile(spec)
        bb = c.bbox
        # the densest material Deckard gave any part of this object (its own number; carved cavities have density 0)
        dens = [float(v) for v in (getattr(c, "density_by_label", None) or {}).values() if v and v > 0]
        out.update({
            "mass_kg": float(c.mass_kg),
            "part_density_max_kg_m3": max(dens) if dens else None,
            "size_m": sorted((abs(b[1] - b[0]) for b in bb), reverse=True),
            "com_m": [float(v) for v in c.com_m] if getattr(c, "com_m", None) is not None else None,
            "inertia_kgm2": [float(v) for v in c.inertia_kgm2] if getattr(c, "inertia_kgm2", None) is not None else None,
        })
    except Exception as e:                                               # noqa: BLE001
        out.update({"ok": False, "error": f"compile failed: {type(e).__name__}: {e}"})
    print(f"    {time.time() - t:.1f} s")                  # timing is not stored: it would make every re-run differ
    return out


def engine_record(kind: dict, measured: dict) -> dict:
    rows = []
    for v in kind["variants"]:
        m = measured.get(v["model"], {})
        ph = m.get("phys") or {}
        engine_mass = v.get("engine_mass", v["mass"])          # the catalog's "mass" is the real-world mass now (fidelity.py); this audit sets the ENGINE's beside Deckard
        size = sorted(((v["maxs"][i] - v["mins"][i]) * HU_M for i in range(3)), reverse=True)
        vol_m3 = ph.get("volume", 0) * HU_M ** 3
        rows.append({
            "model": v["model"].replace("models/", ""), "mass_kg": engine_mass, "size_m": [round(x, 3) for x in size],
            "hull_volume_m3": round(vol_m3, 4), "convexes": ph.get("convexes"), "material": v["material"],
            "surface_density": (ph.get("surface") or {}).get("density"), "friction": (ph.get("surface") or {}).get("friction"),
            "hull_density_kg_m3": round(engine_mass / vol_m3, 1) if vol_m3 > 0 else None,
        })
    return {"variants": rows, **engine_medians(rows)}


def engine_medians(rows: list) -> dict:
    """The kind's typical variant: the true median mass, and the median of each of the three (largest-first) dimensions."""
    return {"median_mass_kg": statistics.median(r["mass_kg"] for r in rows),
            "median_size_m": [round(statistics.median(r["size_m"][i] for r in rows), 3) for i in range(3)]}


def flags(d: dict, e: dict) -> list[str]:
    out = []
    if not d.get("ok", False):
        return [f"Deckard error: {d.get('error')}"]
    if not d.get("identified"):
        return ["Deckard cannot ground this name (no mass or size to compare)"]
    dm, em = d["mass_kg"], e["median_mass_kg"]
    r = em / dm if dm > 0 else math.inf
    if r > 4 or r < 0.25:
        out.append(f"engine mass {em:.0f} kg vs Deckard {dm:.1f} kg (x{r:.1f})")
    ds, es = d["size_m"], e["median_size_m"]
    rs = [a / b if b > 0 else math.inf for a, b in zip(es, ds)]
    if max(rs) > 2.5 or min(rs) < 0.4:
        out.append("size differs a lot: engine " + " x ".join(f"{x:.2f}" for x in es) + " m vs Deckard " + " x ".join(f"{x:.2f}" for x in ds) + " m")
    # a plausibility check that needs no reference: a thing heavier than a person per litre of its own hull is solid metal or stone
    for row in e["variants"]:
        if row["hull_density_kg_m3"] and row["hull_density_kg_m3"] > 1500 and ("locker" in row["model"] or "stove" in row["model"] or "chair" in row["model"]):
            out.append(f"{row['model']}: {row['mass_kg']:.0f} kg in a {row['hull_volume_m3']:.2f} m3 body is {row['hull_density_kg_m3']:.0f} kg/m3")
    # Deckard against itself: its mass over its own bounding box cannot exceed the densest material it gave any part (the box is
    # larger than the body, so the true mean density is lower still). Only rows compiled with that number can be checked.
    box = ds[0] * ds[1] * ds[2]
    ceiling = d.get("part_density_max_kg_m3")
    if dm > 0 and box > 0 and ceiling and dm / box > ceiling:
        out.append(f"Deckard's own mass is more than its own materials can fill its own box with: {dm:.0f} kg in {box:.2f} m3 is "
                   f"{dm / box:.0f} kg/m3, its densest part is {ceiling:.0f} kg/m3")
    return out


def rederive(row: dict) -> dict:
    """A stored row with its medians and flags recomputed from the stored numbers, and without the run-to-run timing."""
    row["engine"].update(engine_medians(row["engine"]["variants"]))
    row["deckard"].pop("seconds", None)
    row["flags"] = flags(row["deckard"], row["engine"])
    return row


def render(rows: list, out: str) -> None:
    lines = ["# Holodeck props against Deckard", "",
             "Generated by `stage/gmod/holodeck/audit_physics.py`. The numbers are Deckard's compiled objects and the Source engine's own "
             "(mass, size, collision body); the flags are only ratios between them. Nothing was corrected: where Deckard's result looks wrong it "
             "is listed as a Deckard finding for review, not patched. `est.` marks a Deckard result whose proportions came from a language model's "
             "own estimate (its catalog says so), not from measured data.", ""]
    ungrounded = [r for r in rows if r["deckard"].get("ok") and not r["deckard"].get("identified")]
    errors = [r for r in rows if not r["deckard"].get("ok")]
    lines += [f"{len(rows)} kinds: {len(rows) - len(ungrounded) - len(errors)} compiled by Deckard, {len(ungrounded)} it could not ground, "
              f"{len(errors)} errors; {sum(1 for r in rows if r['flags'])} carry a flag.", "",
              "| kind | engine mass kg (median) | Deckard mass kg | engine size m | Deckard size m | parts | materials | flags |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, e = r["deckard"], r["engine"]
        es = " x ".join(f"{x:.2f}" for x in e["median_size_m"])
        if d.get("mass_kg") is None:
            lines.append(f"| {r['kind']} | {e['median_mass_kg']:.0f} | - | {es} | - | - | - | {'; '.join(r['flags'])} |")
            continue
        ds = " x ".join(f"{x:.2f}" for x in d["size_m"])
        est = " est." if any("qwen" in str(x).lower() or "researched" in str(x).lower() for x in d.get("sources", [])) else ""
        lines.append(f"| {r['kind']}{' (as ' + d['asked'] + ')' if d.get('asked') and d['asked'] != article(r['kind']) else ''} | {e['median_mass_kg']:.0f} | {d['mass_kg']:.1f} | {es} | {ds}{est} | {len(d.get('parts') or [])} | "
                     f"{', '.join(d.get('materials') or []) or '-'} | {'; '.join(r['flags']) or ''} |")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="")
    ap.add_argument("--out", default=str(REPO / "misc" / "HOLODECK_PHYSICS_AUDIT.md"))
    ap.add_argument("--from-json", action="store_true", help="re-render the report from data/physics_audit.json without asking Deckard again")
    a = ap.parse_args(argv)
    path = HERE / "data" / "physics_audit.json"
    if a.from_json:
        rows = [rederive(r) for r in json.loads(path.read_text(encoding="utf-8"))]
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        render(rows, a.out)
        print(f"wrote {a.out} and {path}")
        return 0
    catalog = json.loads((HERE / "data" / "catalog.json").read_text(encoding="utf-8"))
    measured = {m["model"]: m for m in json.loads((HERE / "data" / "measured.json").read_text(encoding="utf-8"))["models"]}
    want = [k.strip() for k in a.kinds.split(",") if k.strip()]
    rows = []
    for kid in catalog["order"]:
        if want and kid not in want:
            continue
        k = catalog["kinds"][kid]
        print(f"[{len(rows) + 1}] {kid} ...", flush=True)
        names = [kid] + [n for n in k["names"] if n != kid] + [n for n in k.get("approx", [])]
        d = deckard_record(names[:8])
        e = engine_record(k, measured)
        rows.append({"kind": kid, "category": k["category"], "deckard": d, "engine": e, "flags": flags(d, e)})
    if want and path.exists():                       # a partial run updates just those kinds
        old = {r["kind"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        old.update({r["kind"]: r for r in rows})
        rows = [old[k] for k in catalog["order"] if k in old]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    render(rows, a.out)
    print(f"wrote {a.out} and {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
