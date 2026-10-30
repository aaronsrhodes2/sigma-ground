"""Router recipes for the sky: "where is <body> at <time>" -> body_position.

body_position arrived 2026-10-26, after mentat-router-v31 was trained, and the router sent "Where will Mars be on March 1, 2027, as seen
from Earth?" to solar_system_body (Mars's fact sheet). Mentat's /ask now has a fixed rule for it (SkippyTel mentat/position_stage.py); these
recipes teach the next router the same thing, so the rule and the model agree. Contrast pairs ride along: the same body with no time and a
size word ("how far is Mars from the Sun?") stays with solar_system_body, which the existing astro recipes already teach.

    python -m sigma_ground.mcp.benchmark.router_recipes_sky --per-recipe 60 --out local-cache/router_sky
"""
from __future__ import annotations

from sigma_ground.mcp.benchmark import router_recipes as RR
from sigma_ground.mcp.benchmark.router_recipes import C, L, S, single

SKY_START = len(RR.RECIPES)

BODY = C((("Mars", "mars"), ("Venus", "venus"), ("Jupiter", "jupiter"), ("Saturn", "saturn"), ("Mercury", "mercury"), ("Uranus", "uranus"),
          ("Neptune", "neptune"), ("Pluto", "pluto"), ("the Moon", "moon")))
NOT_EARTH = C(BODY.options + (("the Sun", "sun"),))
DATE = C((("March 1, 2027", "2027-03-01"), ("July 20, 1969", "1969-07-20"), ("2031-06-15", "2031-06-15"), ("1 January 2000", "2000-01-01"),
          ("December 25, 2026", "2026-12-25"), ("4 July 1976", "1976-07-04"), ("2040-08-12", "2040-08-12"), ("October 3, 1957", "1957-10-03"),
          ("May 5, 2035", "2035-05-05"), ("2024-04-08", "2024-04-08")))

single("sky_from_earth", "astro", {"b": NOT_EARTH, "d": DATE}, [
    "Where will {b} be on {d}, as seen from Earth?",
    "Where was {b} in the sky on {d}?",
    "What are the right ascension and declination of {b} on {d}?",
    "Give me the position of {b} on {d}.",
    "How far was {b} from Earth on {d}?",
    "Where is {b} on {d}?",
], "body_position", {"body_name": S("b"), "when": S("d"), "central_body": L("earth")})

single("sky_from_sun", "astro", {"b": BODY, "d": DATE}, [
    "How far is {b} from the Sun on {d}?",
    "Where is {b} in its orbit around the Sun on {d}?",
    "Heliocentric position of {b} on {d}.",
    "What was the distance between {b} and the Sun on {d}?",
], "body_position", {"body_name": S("b"), "when": S("d"), "central_body": L("sun")})

single("sky_earth_from_sun", "astro", {"d": DATE}, [
    "Where is the Earth on {d}?",
    "How far was the Earth from the Sun on {d}?",
    "Where will Earth be in its orbit on {d}?",
], "body_position", {"body_name": L("earth"), "when": S("d"), "central_body": L("sun")})

single("sky_now", "astro", {"b": NOT_EARTH}, [
    "Where is {b} right now?",
    "Where is {b} tonight?",
    "Where's {b} in the sky at the moment?",
    "How far away is {b} from us today?",
], "body_position", {"body_name": S("b"), "when": L("now"), "central_body": L("earth")})

SKY_RECIPES = RR.RECIPES[SKY_START:]


def main() -> int:
    import argparse
    import asyncio
    from pathlib import Path
    from sigma_ground.mcp.benchmark import router_dataset as D
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_sky")
    p.add_argument("--per-recipe", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    a = p.parse_args()
    a.only = ",".join(r.id for r in SKY_RECIPES)
    return asyncio.run(RR._amain(a))


if __name__ == "__main__":
    import sys
    sys.exit(main())
