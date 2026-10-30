"""Router recipes for the folded chains: a body's name goes straight into the formula tool (misc/FOLDED_CHAINS.md).

Measured 2026-10-29: the only tool chain real questions keep needing is "look the body up, then the formula": `solar_system_body > escape_velocity`
was 6 of the 12 chains in 521 distinct /ask questions (October research log), `> schwarzschild_radius` one more, and the benchmark runs hold 15 more
single calls with the body's mass typed in from memory. The v3 recipes taught exactly that chain (body_escape_velocity, body_schwarzschild,
body_hawking). escape_velocity and the GR mass tools now take `body_name` and look the body up themselves (mcp/tools/body_lookup.py), so these
recipes teach one call with the name, and `refold` relabels the old v3 seeds in place (same question, same index, so the teacher-rewrite cache
indexed by seed stays valid) so the next training set never teaches both.

    python -m sigma_ground.mcp.benchmark.router_recipes_fold --per-recipe 60 --out local-cache/router_fold
"""
from __future__ import annotations

from sigma_ground.mcp.benchmark import router_recipes as RR
from sigma_ground.mcp.benchmark.router_recipes import C, Q, S, U, single

FOLD_START = len(RR.RECIPES)

# names the body catalogue resolves (field/interface/body_data.py), the Sun and a few moons and a dwarf planet beyond the v3 list
BODY = C((("Earth", "earth"), ("Mars", "mars"), ("Venus", "venus"), ("Mercury", "mercury"), ("Jupiter", "jupiter"), ("Saturn", "saturn"),
          ("Uranus", "uranus"), ("Neptune", "neptune"), ("the Moon", "moon"), ("Pluto", "pluto"), ("the Sun", "sun"), ("Io", "io"),
          ("Europa", "europa"), ("Titan", "titan"), ("Ganymede", "ganymede"), ("Ceres", "ceres")))
ALT = U(100, 4e4, ("km", 1), ("m", 1e-3))                  # base: kilometres (altitude_km)

single("fold_escape_body", "fold", {"body": BODY}, [
    "What is the escape velocity from {body}?",
    "What's the escape velocity from the surface of {body}?",
    "How fast must a rocket go to leave {body} for good?",
    "Escape speed at the surface of {body}?",
    "What speed does it take to break free of {body}'s gravity?",
    "How fast would you need to launch something to escape {body}?",
], "escape_velocity", {"body_name": S("body")}, per=70, bounds={1: (100, 1e6)})

single("fold_escape_altitude", "fold", {"body": BODY, "h": Q(ALT)}, [
    "What is the escape velocity {h} above {body}'s surface?",
    "A probe sits {h} above {body}. How fast must it go to escape?",
    "Escape speed at an altitude of {h} over {body}?",
], "escape_velocity", {"body_name": S("body"), "altitude_km": S("h")}, per=50, bounds={1: (100, 1e6)})

single("fold_schwarzschild_body", "fold", {"body": BODY}, [
    "What would the Schwarzschild radius of {body} be?",
    "If {body} collapsed into a black hole, how big would its event horizon be?",
    "How small would {body} have to be squeezed to become a black hole?",
    "Event horizon radius of a black hole with the mass of {body}?",
], "schwarzschild_radius", {"body_name": S("body")}, per=60)

single("fold_hawking_body", "fold", {"body": BODY}, [
    "What would the Hawking temperature of {body} be if it were a black hole?",
    "If {body} became a black hole, how hot would it glow from Hawking radiation?",
    "Hawking temperature of a black hole with the mass of {body}?",
], "hawking_temperature", {"body_name": S("body")}, per=40)

single("fold_photon_sphere_body", "fold", {"body": BODY}, [
    "Where would the photon sphere be if {body} were a black hole?",
    "Photon sphere radius for a black hole of {body}'s mass?",
], "photon_sphere_radius", {"body_name": S("body")}, per=30)

single("fold_time_dilation_surface", "fold", {"body": BODY}, [
    "How much slower does a clock tick on the surface of {body} than far away in space?",
    "Gravitational time dilation factor at the surface of {body}?",
    "What is the clock rate at {body}'s surface relative to infinity?",
], "gravitational_time_dilation", {"body_name": S("body")}, per=40)

FOLD_RECIPES = RR.RECIPES[FOLD_START:]

# old v3 recipe id -> the folded tool that replaces its two steps (the body is the first step's body_name)
FOLDED = {"body_escape_velocity": "escape_velocity", "body_schwarzschild": "schwarzschild_radius", "body_hawking": "hawking_temperature"}


def refold(seed: dict) -> dict:
    """A seed of a superseded chain recipe relabelled as the one folded call (question, group and index unchanged); any other seed as it is.

    The verified value is recomputed with the folded tool, so the label carries the number the tool now gives (JPL GM, not the Fact-Sheet mass).
    """
    tool = FOLDED.get(seed.get("meta", {}).get("recipe"))
    if tool is None:
        return seed
    body = seed["plan"]["steps"][0]["tool_args"]["body_name"]
    from sigma_ground.mcp.tools import gr, kinematics
    from sigma_ground.mcp.tools.body_lookup import with_body
    fn = {"escape_velocity": kinematics.escape_velocity_classical, "schwarzschild_radius": gr.schwarzschild_radius,
          "hawking_temperature": gr.hawking_temperature}[tool]
    res = with_body(fn, body, None, needs_radius=tool == "escape_velocity")
    if res.value is None:
        raise ValueError(f"refold: {tool}(body_name={body!r}) gave no value: {res.notes}")
    meta = {**seed["meta"], "family": "fold", "steps": 1, "refolded_from": seed["meta"]["recipe"],
            "verified": [{"tool": tool, "value": res.value, "units": res.units}]}
    return {**seed, "plan": {"tool": tool, "tool_args": {"body_name": body}}, "meta": meta}


def main() -> int:
    import argparse
    import asyncio
    from pathlib import Path
    from sigma_ground.mcp.benchmark import router_dataset as D
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_fold")
    p.add_argument("--per-recipe", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    a = p.parse_args()
    a.only = ",".join(r.id for r in FOLD_RECIPES)
    return asyncio.run(RR._amain(a))


if __name__ == "__main__":
    import sys
    sys.exit(main())
