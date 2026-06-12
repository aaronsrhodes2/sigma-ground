"""The Mentat gauntlet — 20 minimal prompts, zero coaching, end to end.

The Captain's gate: a fully-LOCAL pipeline (qwen via ollama; no cloud) must turn
minimal natural-language prompts into rendered physics — 10 SIMULATIONS (the
model decides the sim from the question; Materia integrates; the viewer plays
frames) and 10 STATIC multi-part objects (Deckard researches the shape; the
viewer orbits it). We never help a prompt: the only permitted interventions are
SYSTEMIC fixes (parser tolerance, renderer coverage, physics capability), so a
failure here is a finding, not an embarrassment to hide.

Artifacts:  sigma_ground/radiance/web/data/g_<slug>.json   (viewer bundles)
Index:      merged into data/scenes.json (group "gauntlet")
Report:     misc/GAUNTLET_REPORT.json + console table

    python tools/mentat_gauntlet.py            # run everything not yet built
    python tools/mentat_gauntlet.py --force    # rebuild all
    python tools/mentat_gauntlet.py --only sim_cup_bounce obj_hammer
    python tools/mentat_gauntlet.py --statics | --sims
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

_DATA = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                     "radiance", "web", "data")
_REPORT = os.path.join(os.path.dirname(__file__), "..", "misc",
                       "GAUNTLET_REPORT.json")

# ── the 20 prompts (verbatim; the system sinks or swims) ────────────────────
STATICS = [
    ("obj_hammer",           "a hammer"),
    ("obj_chair",            "a wooden chair"),
    ("obj_flashlight",       "a flashlight"),
    ("obj_flashlight_parts", "a disassembled flashlight"),
    ("obj_car",              "a car"),
    ("obj_pushpin",          "a pushpin"),
    ("obj_wineglass",        "a wine glass"),
    ("obj_dumbbell",         "a dumbbell"),
    ("obj_guitar",           "an acoustic guitar"),
    ("obj_bookcase",         "a bookcase"),
]
SIMS = [
    ("sim_cup_bounce",    "Drop a coffee cup from 2 meters and show me the bounce and tumble."),
    ("sim_copper_ball",   "How fast is a 5 cm copper ball going when it falls from 2 km?"),
    ("sim_feather",       "Drop a feather from 8 feet."),
    ("sim_anvil",         "Drop an anvil from 50 meters."),
    ("sim_steel_heat",    "How hot does a steel ball get falling from 5 km?"),
    ("sim_skydiver",      "A skydiver jumps from 40 km up — does she break the sound barrier?"),
    ("sim_baseball_up",   "Throw a baseball straight up at 30 m/s — how high does it go and how fast is it moving when it lands?"),
    ("sim_mach2_slug",    "Fire a 2 cm tungsten slug at Mach 2 — how far does it travel before it slows below the speed of sound?"),
    ("sim_hammer_drop",   "Drop a hammer from 3 meters."),
    ("sim_wineglass_drop", "Drop a wine glass from a 75 cm table."),
]


def _save(slug: str, bundle: dict) -> None:
    os.makedirs(_DATA, exist_ok=True)
    with open(os.path.join(_DATA, f"{slug}.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=1)


# ── statics lane: prompt → Deckard research → static scene ──────────────────
def run_static(slug: str, prompt: str) -> dict:
    from sigma_ground import deckard
    from sigma_ground.radiance.scene_export import construct_to_scene, sdf_samples

    t0 = time.time()
    c = deckard.identify(prompt)                       # qwen researches; no coaching
    a = deckard.audit(deckard.research(prompt), c) if hasattr(deckard, "audit") else {}
    scene = construct_to_scene(c)
    scene["kind"] = "static"
    scene["sdf_samples"] = sdf_samples(c)
    scene["prompt"] = prompt
    _save(slug, scene)
    return {"status": "rendered", "kind": "static", "identified": c.identified,
            "parts": len(scene["csg_leaves"]), "mass_kg": round(c.mass_kg, 4),
            "verdict": a.get("verdict", "?"), "groundedness": a.get("groundedness"),
            "secs": round(time.time() - t0, 1)}


# ── sims lane: the question goes through Mentat's real front door ────────────
def run_sim(slug: str, question: str) -> dict:
    """The question goes through Mentat's REAL front door — the same
    `dispatch(text, mode="render")` the chat box calls. Mentat classifies,
    compiles, runs, gates, renders, and saves; the gauntlet only re-files the
    saved bundle under its own slug and records the verdict."""
    from sigma_ground.mcp.front_door import Session, dispatch

    t0 = time.time()
    try:
        env = dispatch(question, use_llm=True, session=Session(), mode="render")
    except Exception as e:
        return {"status": "error", "kind": "trajectory",
                "why": f"{type(e).__name__}: {e}", "secs": round(time.time() - t0, 1)}

    saved = env.get("saved") or {}
    if env.get("intent") == "render" and saved.get("path"):
        try:
            bundle = json.load(open(saved["path"], encoding="utf-8"))
        except Exception as e:
            return {"status": "error", "kind": "trajectory",
                    "why": f"saved bundle unreadable: {e}",
                    "secs": round(time.time() - t0, 1)}
        bundle.setdefault("scene", {})["prompt"] = question
        bundle["mentat_text"] = (env.get("text") or "")[:400]
        _save(slug, bundle)
        frames = len((bundle.get("trajectory") or {}).get("frames") or [])
        return {"status": "rendered", "kind": "trajectory",
                "verb": saved.get("title", "")[:40], "frames": frames,
                "secs": round(time.time() - t0, 1)}
    # honest non-render outcomes, verbatim from Mentat
    return {"status": {"simulate": "computed_not_rendered",
                       "clarify": "refused"}.get(env.get("intent"), "refused"),
            "kind": "trajectory", "why": (env.get("text") or "")[:160],
            "secs": round(time.time() - t0, 1)}


# ── index merge: the viewer's gallery lists every gauntlet artifact ──────────
def _merge_scenes(rows: dict) -> None:
    idx_path = os.path.join(_DATA, "scenes.json")
    try:
        idx = json.load(open(idx_path, encoding="utf-8"))
    except Exception:
        idx = []
    idx = [e for e in idx if e.get("group") != "gauntlet"]      # replace our group only
    prompts = dict(STATICS + SIMS)
    for slug, r in rows.items():
        if r.get("status") != "rendered":
            continue
        idx.append({"slug": slug, "title": prompts[slug][:64],
                    "question": prompts[slug], "verb": r.get("verb", "deckard"),
                    "group": "gauntlet", "kind": r["kind"],
                    **({"frames": r["frames"]} if "frames" in r else {})})
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=1)


def main(argv: list[str]) -> None:
    force = "--force" in argv
    only = [a for a in argv if not a.startswith("--")]
    lanes = []
    if "--sims" not in argv:
        lanes += [("static", s, p) for s, p in STATICS]
    if "--statics" not in argv:
        lanes += [("sim", s, q) for s, q in SIMS]
    if only:
        lanes = [L for L in lanes if L[1] in only]

    rows = {}
    if os.path.exists(_REPORT) and not force:
        try:
            rows = json.load(open(_REPORT, encoding="utf-8")).get("items", {})
        except Exception:
            rows = {}

    for kind, slug, prompt in lanes:
        if not force and rows.get(slug, {}).get("status") == "rendered" \
                and os.path.exists(os.path.join(_DATA, f"{slug}.json")):
            print(f"  = {slug:22s} (already rendered)")
            continue
        print(f"  > {slug:22s} {prompt[:60]}")
        try:
            rows[slug] = run_static(slug, prompt) if kind == "static" \
                else run_sim(slug, prompt)
        except Exception as e:
            rows[slug] = {"status": "error", "kind": kind,
                          "why": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc(limit=4)}
        r = rows[slug]
        print(f"    {r['status']:22s} {r.get('why', '')[:90]}")

    _merge_scenes(rows)
    ok = sum(1 for r in rows.values() if r.get("status") == "rendered")
    summary = {"rendered": ok, "total": len(STATICS) + len(SIMS), "items": rows}
    os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nGauntlet: {ok}/{len(STATICS) + len(SIMS)} rendered  "
          f"(report: misc/GAUNTLET_REPORT.json)")
    for slug, r in rows.items():
        mark = {"rendered": "+", "computed_not_rendered": "~"}.get(r["status"], "!")
        print(f"  {mark} {slug:22s} {r['status']:22s} {r.get('why', '')[:70]}")


if __name__ == "__main__":
    main(sys.argv[1:])
