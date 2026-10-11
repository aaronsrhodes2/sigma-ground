"""The fidelity layer: a real mass and a documented makeup for every holodeck prop, instead of the engine's gameplay masses (a
locker is 2000 kg in the engine, a stove 1500, a bathtub 900).

    python stage/gmod/holodeck/fidelity.py                      # writes data/fidelity.json and misc/HOLODECK_FIDELITY.md
    python stage/gmod/holodeck/fidelity.py --check              # exit 1 if either file is not what the inputs give
    python stage/gmod/holodeck/fidelity.py --verify-densities   # re-ask Mentat for the densities this file relies on (needs the stack up)

Inputs (all committed):
    data/fidelity_research.json   the sourced real-world reference table (one entry per real kind, each with its sources and caveats)
    data/fidelity_specs.json      which real family each model is (chosen from the photographs, data/sheets_final), the reference
                                  item that stands for it, its sourced mass range, and which models are left unaudited and why
    data/measured.json            the ENGINE's own geometry per model (bounds, hull volume and surface, collision pieces, mass)
    catalog_src.py                which models are in the library

For every catalog model with a family:

    mass = clamp(reference mass x s^k, low, high),   s = (box volume / reference box volume)^(1/3)

s is the linear scale factor, k its exponent: 3 for solids and frame-and-panel objects (mass follows volume), 2 for thin shells
(geometrically similar shells at constant wall thickness scale with surface area). low and high are the sourced range of the family.
Where the scaled figure lies more than `guard_factor` outside that range the sources do not describe the model, so it stays
"unaudited" with the engine's mass and the extrapolation is only recorded. A bottom-up estimate (Mentat's density x the engine hull's
surface or volume x a sourced thickness) is reported next to the sourced figure as a ratio; it NEVER sets the mass.

To add a model to the library: add it to catalog_src.py and give it a decision in data/fidelity_specs.json ("variants": a family and a fit,
or "unaudited" with the reason), then run this file and build_catalog.py. A new family needs a reference item from
data/fidelity_research.json (or new sourced research added to it), a sourced range, and either a cross-check or the reason there is none.

Mentat is the source of every density (data/fidelity_specs.json 'densities', checked with --verify-densities). It has none for steel,
foam, fabric, ceramic or cardboard: where the references describe steel the cross-check uses its iron and says so, and for the others
no cross-check is made. The masses of furniture themselves are not physics Mentat computes; they come from the research table.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import catalog_src  # noqa: E402

HU_M = 0.01905
DATA = HERE / "data"
SPECS_PATH = DATA / "fidelity_specs.json"
RESEARCH_PATH = DATA / "fidelity_research.json"
MEASURED_PATH = DATA / "measured.json"
OUT_JSON = DATA / "fidelity.json"
OUT_REPORT = REPO / "misc" / "HOLODECK_FIDELITY.md"
VERSION = 1

K_BY_TOPOLOGY = {"solid": 3, "frame-and-panel": 3, "shell": 2}
EVIDENCE_STEPS = {"opened": 0, "mixed": 1, "search-summary": 2, "single-source": 2}
TRIANGLES_PER_CONVEX_DENSE = 60


# ------------------------------------------------------------------------------------------------------------- inputs
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_specs() -> dict:
    return load_json(SPECS_PATH)


def load_research() -> dict:
    return load_json(RESEARCH_PATH)["items"]


def load_measured() -> dict:
    return {m["model"]: m for m in load_json(MEASURED_PATH)["models"]}


def library_models() -> list[tuple[str, str]]:
    """(kind, model path) for every curated variant, in catalog order."""
    return [(k["kind"], v if isinstance(v, str) else v["model"]) for k in catalog_src.KINDS for v in k["variants"]]


# ------------------------------------------------------------------------------------------------------------- numbers
def round_sig(x: float, n: int = 3) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, n - 1 - int(math.floor(math.log10(abs(x)))))


def r6(x):
    return None if x is None else round(float(x), 6)


def kg(x: float) -> str:
    """Kilograms to four significant figures, trailing zeros dropped (the sourced figures carry up to four)."""
    return f"{x:.4g}"


def box_volume(size) -> float:
    return size[0] * size[1] * size[2]


def scale_factor(vol: float, ref_vol: float, k: int) -> tuple[float, float]:
    """(s, s^k): the linear scale of a box of volume `vol` against one of `ref_vol`, and the mass factor it gives."""
    s = (vol / ref_vol) ** (1.0 / 3.0)
    return s, s ** k


def evaluate(fam: dict, vol: float, mode: str | None = None) -> dict:
    """The sourced mass of a model whose box has volume `vol` (m3), before and after the clamp to the family's sourced range."""
    ref = fam["reference"]
    k = K_BY_TOPOLOGY[fam["topology"]]
    mode = mode or fam["scale"]
    if mode == "fixed":
        s, factor, ref_vol = 1.0, 1.0, None
    else:
        ref_vol = box_volume(ref["size_m"])
        s, factor = scale_factor(vol, ref_vol, k)
    scaled = ref["mass_kg"] * factor
    low, high = fam["low_kg"], fam["high_kg"]
    clamped = "low" if scaled < low else "high" if scaled > high else None
    adopted = min(high, max(low, round_sig(min(high, max(low, scaled)))))
    if scaled > high:
        outside = scaled / high
    elif scaled < low:
        outside = low / scaled
    else:
        outside = 1.0
    return {"mode": mode, "k": k, "s": s, "factor": factor, "ref_volume_m3": ref_vol, "scaled_kg": scaled, "adopted_kg": adopted,
            "clamped": clamped, "outside_factor": outside, "low_kg": low, "high_kg": high}


def confidence(evidence: str, fit: str, clamped: bool, fixed: bool = False) -> str:
    """high: opened evidence, good fit, no clamp, the model's size used. Each of: mixed evidence, a partial fit, a clamp, a fixed scale
    (the size is not used) costs one step; search-summary or single-source evidence costs two. medium = one step, low = two or more."""
    steps = EVIDENCE_STEPS[evidence] + (1 if fit == "partial" else 0) + (1 if clamped else 0) + (1 if fixed else 0)
    return "high" if steps == 0 else "medium" if steps == 1 else "low"


def bottom_up(cc: dict, hull_surface_m2: float, hull_volume_m3: float, densities: dict) -> dict:
    """Density x the engine hull's surface area x a sourced thickness (shells), or density x the hull volume (solids).
    Reported next to the sourced mass; it never sets it."""
    layers, total = [], 0.0
    for layer in cc["layers"]:
        rho = densities[layer["material"]]["kg_m3"]
        if cc["mode"] == "shell":
            m = rho * hull_surface_m2 * layer["thickness_mm"] / 1000.0
        elif cc["mode"] == "solid":
            m = rho * hull_volume_m3 * layer.get("fill", 1.0)
        else:
            raise ValueError(f"unknown cross-check mode {cc['mode']!r}")
        total += m
        layers.append({"material": layer["material"], "density_kg_m3": rho, "thickness_mm": layer.get("thickness_mm"),
                       "thickness_source": layer.get("thickness_source"), "thickness_evidence": layer.get("thickness_evidence"),
                       "kg": r6(m)})
    return {"mode": cc["mode"], "bottom_up_kg": r6(total), "layers": layers, "excluded": cc.get("excluded"), "proxy": cc.get("proxy")}


def complexity(real: dict, convexes, triangles) -> dict:
    """The engine's collision pieces against the pieces a faithful rigid decomposition of the real object needs (an estimate drawn from
    the sourced construction text, not a measurement). Report only: nothing here changes the engine."""
    lo, hi = real["pieces"]                                  # lo: the major rigid bodies; hi: every distinct member counted on its own
    note = None
    if convexes is None:
        verdict = "unknown"
    elif real["interior"] == "open" and convexes == 1:
        verdict, note = "much simpler", "one convex hull fills an object whose interior or underside is open, so nothing can stand in or under it and it behaves as a solid block"
    elif convexes <= lo / 2:
        verdict, note = "much simpler", f"{convexes} piece{'s' if convexes != 1 else ''} where the real object needs at least {lo}"
    elif convexes < lo:
        verdict, note = "simpler", f"{convexes} pieces where the real object needs at least {lo}"
    elif convexes > 2 * hi:
        verdict, note = "much more complex", f"{convexes} pieces where the real object needs at most {hi}"
    elif convexes > hi:
        verdict, note = "more complex", f"{convexes} pieces where the real object needs at most {hi}"
    else:
        verdict = "comparable"
    if convexes and triangles and triangles / convexes > TRIANGLES_PER_CONVEX_DENSE:
        extra = f"dense collision mesh: {triangles} triangles in {convexes} piece{'s' if convexes != 1 else ''}"
        note = f"{note}; {extra}" if note else extra
    return {"engine_convexes": convexes, "engine_triangles": triangles, "real_pieces": [lo, hi], "interior": real["interior"],
            "verdict": verdict, "note": note}


# ------------------------------------------------------------------------------------------------------------- text
def size_text(size) -> str:
    return "x".join(f"{d:g}" for d in size) + " m" if size else "no size"


def source_text(fid: str, fam: dict, ev: dict, vol: float, conf: str, fit: str, fixed_short: str | None = None) -> str:
    """The one-paragraph provenance the catalog carries as mass_source (the full basis is in data/fidelity.json)."""
    ref = fam["reference"]
    head = f"fidelity/{fid} (research '{fam['research']}'): reference {ref['mass_kg']:g} kg"
    if ev["mode"] == "fixed":
        body = f"{head}, used as is ({fixed_short or fam.get('fixed_short') or 'scale fixed'}), sourced range {kg(ev['low_kg'])} to {kg(ev['high_kg'])} kg"
    else:
        body = (f"{head} at {size_text(ref['size_m'])}, box volume x{vol / ev['ref_volume_m3']:.2f} (linear s={ev['s']:.2f}, k={ev['k']}) = {kg(ev['scaled_kg'])} kg")
        if ev["clamped"]:
            body += f", {'below' if ev['clamped'] == 'low' else 'above'} the sourced {kg(ev['low_kg'])} to {kg(ev['high_kg'])} kg so held at {kg(ev['adopted_kg'])} kg"
        else:
            body += f", inside the sourced {kg(ev['low_kg'])} to {kg(ev['high_kg'])} kg"
    return f"{body}. Confidence {conf} ({fam['evidence']} evidence, {fit} fit). Detail: data/fidelity.json"


def makeup_text(fams: list[dict], research: dict) -> str:
    """A one-line makeup for the catalog: what each family is made of and how it is built."""
    out = []
    for fam in fams:
        r = research[fam["research"]]
        mats = fam.get("materials") or r["materials"]
        out.append(f"{fam['label']}: {', '.join(mats[:3])}; {fam['topology']}")
    return " + ".join(out)


# ------------------------------------------------------------------------------------------------------------- compute
def family_record(fam: dict, research: dict) -> dict:
    r = research[fam["research"]]
    return {
        "label": fam["label"], "research": fam["research"], "reference": fam["reference"],
        "low_kg": fam["low_kg"], "high_kg": fam["high_kg"], "bounds_basis": fam["bounds_basis"],
        "topology": fam["topology"], "k": K_BY_TOPOLOGY[fam["topology"]], "scale": fam["scale"], "fixed_reason": fam.get("fixed_reason"),
        "evidence": fam["evidence"], "research_confidence": r["confidence"],
        "makeup": {"materials": fam.get("materials") or r["materials"], "construction": r["construction"], "hollow": r["hollow"],
                   "wall_thickness_mm": r["wall_thickness_mm"]},
        "sources": r["sources"], "real_topology": fam["real"],
        "crosscheck": fam.get("crosscheck"), "crosscheck_unavailable": fam.get("crosscheck_unavailable"),
        "caveats": fam.get("caveats"),
    }


def geometry_of(m: dict) -> tuple[dict, float, float, float]:
    """The engine geometry of one model: the rounded record for the json, and the exact box volume, hull volume and hull surface."""
    mn, mx = m["mins"], m["maxs"]
    bbox = [(mx[i] - mn[i]) * HU_M for i in range(3)]
    vol = box_volume(bbox)
    ph = m.get("phys") or {}
    hull_vol = ph.get("volume", 0.0) * HU_M ** 3
    hull_sa = ph.get("surface_area", 0.0) * HU_M ** 2
    geo = {"box_m": [r6(d) for d in bbox], "box_volume_m3": r6(vol), "hull_volume_m3": r6(hull_vol), "hull_surface_m2": r6(hull_sa),
           "hull_fill": r6(hull_vol / vol) if vol else None, "engine_density_kg_m3": r6(m["mass"] / hull_vol) if hull_vol else None,
           "convexes": ph.get("convexes"), "triangles": ph.get("triangles"), "engine_material": m.get("material") or "default"}
    return geo, vol, hull_vol, hull_sa


def unaudited(rec: dict, engine: float, reason: str, **extra) -> dict:
    rec.update({"status": "unaudited", "mass_kg": engine, "mass_over_engine": 1.0, "mass_low_kg": None, "mass_high_kg": None, "confidence": None,
                "unaudited_reason": reason, "makeup": None, **extra})
    return rec


def compute(specs: dict | None = None, measured: dict | None = None, research: dict | None = None, models=None) -> dict:
    """Everything data/fidelity.json holds, from the committed inputs. Pure and deterministic."""
    specs = specs or load_specs()
    measured = measured or load_measured()
    research = research or load_research()
    models = models if models is not None else library_models()
    fams = specs["families"]
    guard = specs["method"]["guard_factor"]
    densities = specs["densities"]["values"]
    variants, used = {}, set()
    for kind, path in models:
        m = measured.get(path)
        if not m or m.get("missing") or m.get("no_physics"):
            continue                                        # build_catalog reports the unavailable ones
        geo, vol, hull_vol, hull_sa = geometry_of(m)
        engine = m["mass"]
        spec = specs["variants"].get(path)
        rec = {"kind": kind, "engine_mass_kg": engine, "geometry": geo, "family": None, "parts": None, "fit": None, "note": None,
               "cross_check": None, "complexity": None}
        variants[path] = rec
        if spec is None:
            reason = "no decision recorded for this model in data/fidelity_specs.json"
            unaudited(rec, engine, reason, mass_source=f"unaudited: {reason}; the engine mass {kg(engine)} kg is kept.")
            continue
        if "unaudited" in spec:
            unaudited(rec, engine, spec["unaudited"], hint=spec.get("hint"),
                      mass_source=f"unaudited: {spec['unaudited']}. The engine mass {kg(engine)} kg is kept: it is a gameplay value with no real-world source behind it.")
            continue
        fit = spec.get("fit", "good")
        parts = spec.get("parts") or [{"family": spec["family"], "scale": spec.get("scale")}]
        evals = []
        for p in parts:
            used.add(p["family"])
            evals.append((p["family"], fams[p["family"]], evaluate(fams[p["family"]], vol, p.get("scale"))))
        single = len(evals) == 1
        scaled = sum(e["scaled_kg"] for _, _, e in evals)
        adopted = sum(e["adopted_kg"] for _, _, e in evals)
        low = sum(e["low_kg"] for _, _, e in evals)
        high = sum(e["high_kg"] for _, _, e in evals)
        worst = max(e["outside_factor"] for _, _, e in evals)
        any_clamp = any(e["clamped"] for _, _, e in evals)
        evidence = max((fam["evidence"] for _, fam, _ in evals), key=lambda x: EVIDENCE_STEPS[x])
        conf = confidence(evidence, fit, any_clamp, any(e["mode"] == "fixed" for _, _, e in evals) and single)
        rec.update({"family": spec.get("family"), "parts": [p["family"] for p in parts] if "parts" in spec else None, "fit": fit,
                    "note": spec.get("note"), "evidence": evidence, "scaled_kg": r6(scaled), "outside_factor": r6(worst),
                    "scale_mode": evals[0][2]["mode"] if single else "composite", "scale_note": spec.get("scale_note"),
                    "linear_scale": r6(evals[0][2]["s"]) if single else None, "k": evals[0][2]["k"] if single else None,
                    "clamped": evals[0][2]["clamped"] if single else ("composite" if any_clamp else None),
                    "makeup": makeup_text([fam for _, fam, _ in evals], research)})
        if worst > guard:
            who = "; ".join(f"{f}: {kg(e['scaled_kg'])} kg against {kg(e['low_kg'])} to {kg(e['high_kg'])} kg" for f, _, e in evals if e["outside_factor"] > guard)
            reason = (f"the scaled mass lies {worst:.2f}x outside the sourced range ({who}), so the sources do not describe a model this far from their sizes; "
                      f"the extrapolated {kg(scaled)} kg is recorded, not adopted")
            unaudited(rec, engine, reason, extrapolated_kg=r6(scaled),
                      mass_source=f"unaudited: {reason}. The engine mass {kg(engine)} kg is kept.")
            rec["makeup"] = makeup_text([fam for _, fam, _ in evals], research)
        else:
            rec.update({"status": "audited", "mass_kg": adopted, "mass_over_engine": r6(adopted / engine), "confidence": conf,
                        "mass_low_kg": low, "mass_high_kg": high})
            if single:
                fid, fam, ev = evals[0]
                rec["mass_source"] = source_text(fid, fam, ev, vol, conf, fit, spec.get("scale_short"))
            else:
                bits = []
                for fid, fam, ev in evals:
                    how = "used as is" if ev["mode"] == "fixed" else f"scaled to {kg(ev['scaled_kg'])} kg"
                    if ev["clamped"]:
                        how += f", held at {kg(ev['adopted_kg'])} kg by the sourced {ev['clamped']}"
                    bits.append(f"{fid} ({fam['reference']['mass_kg']:g} kg reference) {how}")
                rec["mass_source"] = (f"fidelity/composite of {' and '.join(p['family'] for p in parts)}: {'; '.join(bits)}; "
                                      f"sum {kg(adopted)} kg inside {kg(low)} to {kg(high)} kg. Confidence {conf} ({evidence} evidence, {fit} fit). Detail: data/fidelity.json")
        # the cross-check and the complexity audit, for the audited and the extrapolated alike
        if single:
            fid, fam, ev = evals[0]
            cc = fam.get("crosscheck")
            if cc:
                compare = adopted if rec["status"] == "audited" else scaled
                bu = bottom_up(cc, hull_sa, hull_vol, densities)
                bu["compared_kg"] = r6(compare)
                bu["sourced_over_bottom_up"] = r6(compare / bu["bottom_up_kg"]) if bu["bottom_up_kg"] else None
                bu["note"] = "reported only; the sourced figure sets the mass"
                rec["cross_check"] = {"available": True, **bu}
            else:
                rec["cross_check"] = {"available": False, "reason": fam["crosscheck_unavailable"]}
            real = dict(fam["real"])
            if spec.get("real_pieces"):
                real["pieces"] = spec["real_pieces"]
        else:
            rec["cross_check"] = {"available": False, "reason": "composite of two sourced classes; no single thickness describes it"}
            pcs = [fam["real"]["pieces"] for _, fam, _ in evals]
            real = {"pieces": [max(p[0] for p in pcs), sum(p[1] for p in pcs)],
                    "interior": "open" if any(fam["real"]["interior"] == "open" for _, fam, _ in evals) else "closed"}
        rec["complexity"] = complexity(real, geo["convexes"], geo["triangles"])
        if spec.get("real_note"):
            rec["complexity"]["real_note"] = spec["real_note"]
    families = {fid: family_record(fams[fid], research) for fid in fams if fid in used}
    audited = [r for r in variants.values() if r["status"] == "audited"]
    summary = {
        "models": len(variants), "audited": len(audited), "unaudited": len(variants) - len(audited),
        "confidence": {c: sum(1 for r in audited if r["confidence"] == c) for c in ("high", "medium", "low")},
        "clamped": sum(1 for r in audited if r.get("clamped")),
        "engine_mass_over_fidelity_median": r6(_median([r["engine_mass_kg"] / r["mass_kg"] for r in audited])) if audited else None,
        "engine_mass_over_fidelity_max": r6(max(r["engine_mass_kg"] / r["mass_kg"] for r in audited)) if audited else None,
    }
    return {"version": VERSION, "hu_m": HU_M, "method": specs["method"], "densities": specs["densities"], "summary": summary,
            "families": families, "variants": variants}


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def load() -> dict:
    """The committed data/fidelity.json (what build_catalog reads)."""
    return load_json(OUT_JSON)


# ------------------------------------------------------------------------------------------------------------- report
def short_model(path: str) -> str:
    return path.replace("models/", "")


def render_report(fid: dict) -> str:
    V, F, S = fid["variants"], fid["families"], fid["summary"]
    L: list[str] = []
    w = L.append
    w("# Holodeck fidelity: real masses and makeup")
    w("")
    w("Generated by `stage/gmod/holodeck/fidelity.py` from `data/fidelity_research.json` (the sourced reference table), "
      "`data/fidelity_specs.json` (which real family each model is) and the engine's own geometry (`data/measured.json`); do not edit by hand. "
      "The numbers are written to `data/fidelity.json` and carried into the catalog as `mass`, `engine_mass`, `mass_source` by `build_catalog.py`. "
      "Nothing in the engine was changed: the complexity audit only reports.")
    w("")
    w("## What was done")
    w("")
    w("Half-Life 2 gave every prop a gameplay mass (a locker 2000 kg, a stove 1500 kg, a bathtub 900 kg, a mug 1 to 4 kg). Each of the "
      f"{S['models']} library models was looked at on its photograph (`data/sheets_final`), assigned the real-world family it is (an oak cask or "
      "a steel drum, a two-seat or three-seat sofa, a clawfoot tub or a steel-enamel one), and given the mass of a sourced reference item of that family, "
      "scaled to the model's size and held inside the family's sourced range. Where the sources do not describe a model, the engine mass is kept and the model is "
      "labelled unaudited; nothing is invented.")
    w("")
    w("- mass = clamp(reference mass x s^k, low, high), with s = (the model's bounding-box volume / the reference's)^(1/3) the linear scale factor and "
      "k its exponent: **3** for solids and frame-and-panel objects (mass follows volume) and **2** for thin shells (geometrically similar shells at constant "
      "wall thickness scale with surface area). That is mass proportional to V^1 and V^(2/3). low and high are the family's sourced range.")
    w("- A family or model marked *fixed* skips the scaling where the bounding box is not a fair measure (an umbrella canopy, a lamp arm, foliage, a high cistern, a flue).")
    w(f"- If the scaled mass is more than {fid['method']['guard_factor']:g}x outside the sourced range the model is left unaudited and the extrapolation only recorded.")
    w("- Weights are net or product weights where the sources give them. Package, shipping and unstated weights are marked as such in each reference "
      "(`weight_kind`) and were not preferred over a net figure.")
    w("- Confidence: " + fid["method"]["confidence"])
    w("")
    w("## Headline")
    w("")
    w(f"- {S['models']} models: **{S['audited']} audited** (mass replaced by a sourced figure), **{S['unaudited']} unaudited** (engine mass kept, reasons below).")
    w(f"- Confidence of the {S['audited']} audited: {S['confidence']['high']} high, {S['confidence']['medium']} medium, {S['confidence']['low']} low; "
      f"{S['clamped']} were held at a sourced bound because the model lies outside the sizes the sources describe.")
    w(f"- The engine mass is a median {S['engine_mass_over_fidelity_median']:.1f}x the sourced mass, at most {S['engine_mass_over_fidelity_max']:.0f}x.")
    w("")
    w("Largest corrections (engine mass over the adopted mass):")
    w("")
    w("| model | engine kg | adopted kg | engine / adopted |")
    w("|---|---:|---:|---:|")
    aud = [(p, r) for p, r in V.items() if r["status"] == "audited"]
    aud.sort(key=lambda pr: -pr[1]["engine_mass_kg"] / pr[1]["mass_kg"])
    for p, r in aud[:12]:
        w(f"| {short_model(p)} | {kg(r['engine_mass_kg'])} | {kg(r['mass_kg'])} | {r['engine_mass_kg'] / r['mass_kg']:.0f}x |")
    w("")
    under = [(p, r) for p, r in aud if r["mass_kg"] > r["engine_mass_kg"]]
    under.sort(key=lambda pr: -pr[1]["mass_kg"] / pr[1]["engine_mass_kg"])
    w("Where the sourced mass is HEAVIER than the engine's:")
    w("")
    w("| model | engine kg | adopted kg | adopted / engine |")
    w("|---|---:|---:|---:|")
    for p, r in under[:10]:
        w(f"| {short_model(p)} | {kg(r['engine_mass_kg'])} | {kg(r['mass_kg'])} | {r['mass_kg'] / r['engine_mass_kg']:.1f}x |")
    w("")
    w("## Every model")
    w("")
    w("`range` is the family's sourced range. `fit` is how well the family describes the photographed model (the last column says what the photograph shows); "
      "`clamp` marks a mass held at a bound.")
    w("")
    w("| kind | model | engine kg | adopted kg | range kg | engine / adopted | family | fit | confidence | clamp | photograph |")
    w("|---|---|---:|---:|---|---:|---|---|---|---|---|")
    for p, r in V.items():
        if r["status"] == "audited":
            fam = r["family"] or "+".join(r["parts"])
            w(f"| {r['kind']} | {short_model(p)} | {kg(r['engine_mass_kg'])} | {kg(r['mass_kg'])} | {kg(r['mass_low_kg'])} to {kg(r['mass_high_kg'])} | "
              f"{r['engine_mass_kg'] / r['mass_kg']:.2f} | {fam} | {r['fit']} | {r['confidence']} | {r['clamped'] or ''} | {r['note'] or ''} |")
        else:
            w(f"| {r['kind']} | {short_model(p)} | {kg(r['engine_mass_kg'])} | unaudited | | | {r.get('family') or ''} | {r.get('fit') or ''} | | | {r.get('note') or ''} |")
    w("")
    w("## Unaudited (engine mass kept)")
    w("")
    un = [(p, r) for p, r in V.items() if r["status"] == "unaudited"]
    if un:
        w("| model | engine kg | why | extrapolated or hinted figure |")
        w("|---|---:|---|---|")
        for p, r in un:
            extra = f"extrapolated {kg(r['extrapolated_kg'])} kg (not adopted)" if r.get("extrapolated_kg") is not None else (r.get("hint") or "")
            w(f"| {short_model(p)} | {kg(r['engine_mass_kg'])} | {r['unaudited_reason']} | {extra} |")
    else:
        w("None.")
    w("")
    w("## Held at a sourced bound")
    w("")
    w("These models are bigger or smaller than the sourced items of their family, so the scaled mass was held at the sourced low or high. "
      "The real mass of such a model is probably beyond the bound, in the direction named; `x outside` is how far.")
    w("")
    w("| model | scaled kg | held at kg | bound | x outside | family |")
    w("|---|---:|---:|---|---:|---|")
    for p, r in V.items():
        if r["status"] == "audited" and r.get("clamped"):
            what = r["family"] if r["clamped"] != "composite" else "+".join(r["parts"]) + " (one part)"
            w(f"| {short_model(p)} | {kg(r['scaled_kg'])} | {kg(r['mass_kg'])} | {r['clamped'] if r['clamped'] != 'composite' else 'a part'} | {r['outside_factor']:.2f} | {what} |")
    w("")
    w("## Bottom-up cross-check (reported, never used)")
    w("")
    w("For shells: Mentat's density x the engine hull's surface area x a sourced thickness. The hull surface of a body with several convex pieces "
      "counts the faces between the pieces, so it overstates a thin skin; what the references describe but the sum leaves out is named in the last column. "
      "Wherever the references say steel, Mentat's *iron* is used (Mentat has no steel density); Mentat has none for foam, fabric, ceramic or cardboard either, "
      "so those families have no cross-check. A ratio far above 1 means the sourced mass holds parts the sum leaves out (foam, compressor, doors); far below 1 "
      "means the hull surface overcounts.")
    w("")
    w("| model | adopted or extrapolated kg | bottom-up kg | ratio | layers | left out |")
    w("|---|---:|---:|---:|---|---|")
    for p, r in V.items():
        cc = r.get("cross_check")
        if cc and cc.get("available"):
            layers = "; ".join(f"{l['material']} {l['density_kg_m3']:g} kg/m3 x {l['thickness_mm']:g} mm ({l['thickness_evidence']})" for l in cc["layers"])
            w(f"| {short_model(p)} | {kg(cc['compared_kg'])} | {kg(cc['bottom_up_kg'])} | {cc['sourced_over_bottom_up']:.2f} | {layers} | {cc['excluded']} |")
    w("")
    w("Families with no cross-check, and why:")
    w("")
    for fid_, f in F.items():
        if not f.get("crosscheck") and f.get("crosscheck_unavailable"):
            w(f"- `{fid_}`: {f['crosscheck_unavailable']}")
    w("")
    w("## Complexity audit (report only)")
    w("")
    w("The engine's collision body (`convexes`, `triangles` from `measured.json`) against the rigid pieces a faithful decomposition of the real object needs, "
      "estimated from each family's sourced construction text (a chair has legs, a seat and a back; a bookcase is an open carcass with shelves): the low end is the major "
      "rigid bodies any faithful decomposition needs, the high end counts every distinct member (legs, rails, slats, shelves) on its own. "
      "A closed object (doors and drawers shut) is fine as one convex hull; an object with an open interior or underside is not, because a single hull makes it "
      "a solid block that nothing can stand in or under, and the engine derives its inertia from that hull as if it were solid. Nothing was changed.")
    w("")
    counts: dict[str, int] = {}
    for r in V.values():
        c = r.get("complexity")
        if c:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    w("Verdicts: " + ", ".join(f"{n} {v}" for v, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) + ".")
    w("")
    for verdict, gloss in (("much simpler", "Much simpler than the real object (a hollow or open thing collapsed into one or two hulls)"),
                           ("simpler", "Somewhat simpler"),
                           ("much more complex", "Much more complex than the real object needs"),
                           ("more complex", "Somewhat more complex")):
        rows = [(p, r["complexity"]) for p, r in V.items() if r.get("complexity") and r["complexity"]["verdict"] == verdict]
        if rows:
            w(f"- **{gloss}:** " + "; ".join(f"{short_model(p)} ({c['engine_convexes']} against {c['real_pieces'][0]} to {c['real_pieces'][1]})" for p, c in rows) + ".")
    dense = [(p, r["complexity"]) for p, r in V.items() if r.get("complexity") and "dense collision mesh" in (r["complexity"]["note"] or "")]
    if dense:
        w("- **Dense collision meshes:** " + "; ".join(f"{short_model(p)} ({c['engine_triangles']} triangles in {c['engine_convexes']} piece{'s' if c['engine_convexes'] != 1 else ''})" for p, c in dense) + ".")
    w("")
    w("| model | engine pieces | triangles | real pieces (est.) | interior | verdict | note |")
    w("|---|---:|---:|---|---|---|---|")
    for p, r in V.items():
        c = r.get("complexity")
        if c:
            note = "; ".join(x for x in (c["note"], c.get("real_note")) if x)
            w(f"| {short_model(p)} | {c['engine_convexes']} | {c['engine_triangles']} | {c['real_pieces'][0]} to {c['real_pieces'][1]} | {c['interior']} | {c['verdict']} | {note} |")
    w("")
    w("## Surface materials (reported, not corrected)")
    w("")
    w("The engine gives every model a surface material (`material` in the catalog: it sets friction, elasticity, impact sound). The research table has no friction or "
      "elasticity figures, so nothing here was corrected; the table sets the engine's material beside what the sources say each family is made of, so a mismatch "
      "(an upholstered sofa on `ceiling_tile` or `dirt`, a china basin on `concrete`) can be seen and decided by a person.")
    w("")
    w("| family | the sources say it is made of | engine surface materials of its models |")
    w("|---|---|---|")
    for fid_, f in F.items():
        mats: dict[str, int] = {}
        for p, r in V.items():
            if r.get("family") == fid_:
                mats[r["geometry"]["engine_material"]] = mats.get(r["geometry"]["engine_material"], 0) + 1
        if mats:
            w(f"| `{fid_}` | {', '.join(f['makeup']['materials'][:4])} | {', '.join(f'{m} ({n})' if n > 1 else m for m, n in sorted(mats.items()))} |")
    w("")
    w("## Families")
    w("")
    w("| family | reference | mass kg | what kind of weight | sourced range kg | topology (k) | scale | evidence | what the sources say it is made of |")
    w("|---|---|---:|---|---|---|---|---|---|")
    for fid_, f in F.items():
        mat = ", ".join(f["makeup"]["materials"])
        w(f"| `{fid_}` | {f['reference']['item']} | {kg(f['reference']['mass_kg'])} | {f['reference']['weight_kind']} | {kg(f['low_kg'])} to {kg(f['high_kg'])} | "
          f"{f['topology']} ({f['k']}) | {f['scale']} | {f['evidence']} | {mat} |")
    w("")
    w("Caveats the sources themselves record:")
    w("")
    for fid_, f in F.items():
        if f.get("caveats"):
            w(f"- `{fid_}`: {f['caveats']}")
    w("")
    w("## Honest limits")
    w("")
    w("- **Evidence.** `opened` means the reference figure was read on an opened page; `mixed` an opened reference with bounds from search summaries or a researcher's own pick; "
      "`search-summary` and `single-source` mean the figures could not be checked on a page or come from one maker. The confidence column follows from this, not from the fit alone.")
    w("- **Researcher's picks.** Several 'typical' masses are the researcher's own pick inside the sourced range (each says so in `mass_basis`); they are design inputs, not measurements.")
    w("- **Bounding boxes lie for open objects.** Scaling by box volume treats a chair or a lamp as if it filled its box; the clamp to the sourced range and the *fixed* exceptions "
      "are what keep it honest, and the held-at-a-bound list above shows where they bit.")
    w("- **Thin shells.** Wall thickness is the weak point of nearly every family (the sources rarely publish it); where it is sourced, the cross-check uses it.")
    w("- **What the models contain.** Masses are of the empty object: books, files and drink are not included, and a mattress is only included where a family says so.")
    w("- **Not covered.** Friction, elasticity, damping, the centre of mass and the inertia tensor are not sourced. The engine derives inertia from its collision hull "
      "and the mass, so a hollow object whose hull is a solid block has too small a moment of inertia for its mass (see the complexity audit).")
    w("- **Mentat.** Densities come from Mentat (`fidelity_specs.json`, re-checkable with `--verify-densities`). It has none for steel, foam, fabric, ceramic or cardboard; "
      "the masses of furniture are not physics Mentat computes, so they rest on the research table, not on Mentat.")
    w("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------- cli
def verify_densities(specs: dict) -> int:
    """Re-ask Mentat for every density the specs rely on (needs the SkippyTel stack up). Manual: the tests never call it."""
    bad = 0
    for name, row in specs["densities"]["values"].items():
        out = subprocess.run(["bash", str(REPO / "tools" / "mentat_call.sh"), "density", json.dumps({"material": name})],
                             capture_output=True, text=True, timeout=180)
        try:
            got = json.loads(out.stdout)["result"]["value"]
        except Exception:                                    # noqa: BLE001
            print(f"{name}: no answer from Mentat: {out.stdout[:200]!r} {out.stderr[:200]!r}")
            bad += 1
            continue
        ok = abs(got - row["kg_m3"]) < 1e-6
        print(f"{name}: Mentat {got} kg/m3, specs {row['kg_m3']} {'ok' if ok else 'MISMATCH'}")
        bad += 0 if ok else 1
    return 1 if bad else 0


def render_json(fid: dict) -> str:
    return json.dumps(fid, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if data/fidelity.json or the report differs from what the inputs give")
    ap.add_argument("--verify-densities", action="store_true", help="re-ask Mentat for the densities (needs the stack up)")
    a = ap.parse_args(argv)
    specs = load_specs()
    if a.verify_densities:
        return verify_densities(specs)
    fid = compute(specs)
    js, md = render_json(fid), render_report(fid)
    if a.check:
        bad = []
        if not OUT_JSON.exists() or json.loads(OUT_JSON.read_text(encoding="utf-8")) != json.loads(js):
            bad.append(OUT_JSON.name)
        if not OUT_REPORT.exists() or OUT_REPORT.read_text(encoding="utf-8") != md:
            bad.append(OUT_REPORT.name)
        if bad:
            print("stale: " + ", ".join(bad) + " (run python stage/gmod/holodeck/fidelity.py)")
            return 1
        print("fidelity.json and the report are current")
        return 0
    OUT_JSON.write_text(js, encoding="utf-8", newline="\n")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(md, encoding="utf-8", newline="\n")
    s = fid["summary"]
    print(f"{s['models']} models: {s['audited']} audited, {s['unaudited']} unaudited -> {OUT_JSON.name} and {OUT_REPORT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
