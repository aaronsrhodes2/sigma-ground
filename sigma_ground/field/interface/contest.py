"""Our prediction against JPL's prediction at the time, both against reality: the lookup over fixtures/vintages/FAIR_CONTEST_v2.json (built by scripts/fair_contest_report.py
from the fair contest, C26 M-B).

Every run of the contest starts our integrator from a past JPL ephemeris's OWN state at a date after that ephemeris was issued, so the comparison is dynamics against dynamics
with the same starting knowledge: `ours_km` is our position error against reality (JPL's latest fit, DE442) at the lead, `jpl_at_the_time_km` is that old ephemeris's own
continuation against the same reality, `dynamics_gap_km` the distance between the two, `reality_spread_km` how far JPL's previous reality (DE440) sits from today's (below it
nothing can be told from the truth), and `ours_closer` the count of runs where ours was nearer, with the sign test's p. Nothing is computed here but the lookup."""
from __future__ import annotations

import functools
import json
from pathlib import Path

_FILE = Path(__file__).parent / "fixtures" / "vintages" / "FAIR_CONTEST_v2.json"
BODIES = ("Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
_ALIASES = {"luna": "Moon", "the moon": "Moon", "earth-moon barycentre": "Earth", "emb": "Earth"}


class ContestError(ValueError):
    """No contest table, or a body or lead it does not hold; the message says what it does hold."""


@functools.lru_cache(maxsize=1)
def _table(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(body_name: str) -> str:
    key = body_name.strip().lower()
    key = _ALIASES.get(key, key).lower()
    for b in BODIES:
        if key == b.lower():
            return b
    raise ContestError(f"{body_name!r} is not in the contest; it scores {', '.join(BODIES)} (Mars to Pluto as their system barycentres)")


def prediction_contest(body_name: str, lead_years: int = 5, path: Path | str | None = None) -> dict:
    p = Path(path) if path else _FILE
    if not p.is_file():
        raise ContestError("the fair contest has not been run on this checkout (fixtures/vintages/FAIR_CONTEST_v2.json is missing; scripts/fair_jpl_contest.py --predictor ... then scripts/fair_contest_report.py build it)")
    t = _table(str(p))
    body = _canonical(body_name)
    rows = t["table"].get(body, {})
    lead = str(int(lead_years))
    if lead not in rows:
        raise ContestError(f"no runs of {body} reach +{lead} yr; the leads held are {', '.join(sorted(rows, key=int))} yr")
    c = rows[lead]
    meta = t["metadata"]
    out = {"body": body, "lead_years": int(lead), "ours_km": c["eO_med"], "jpl_at_the_time_km": c["eV_med"], "dynamics_gap_km": c["dyn_med"], "start_offset_km": c["e0_med"],
           "reality_spread_km": c.get("eT_med"), "ours_closer": f"{c['ours_wins']} of {c['n']}", "p_sign_test": c["p_sign"], "runs": c["n"], "reality": meta["truth"], "predictor": meta["predictor"],
           "ephemerides": meta["vintages"], "what": "every run starts from a past JPL ephemeris's own state; medians over the runs; heliocentric km"}
    gap_word = "indistinguishable from JPL's own dynamics (sign test p > 0.05)" if c["p_sign"] > 0.05 else ("closer to reality than JPL's prediction at the time" if c["ours_wins"] > c["n"] / 2 else "further from reality than JPL's prediction at the time")
    floor = f"; reality itself is only known to {c['eT_med']:.3f} km there" if c.get("eT_med") is not None else ""
    out["summary"] = (f"{body}, {lead} years ahead, {c['n']} runs: ours {c['eO_med']:.3f} km from reality ({meta['truth']}), JPL's prediction at the time {c['eV_med']:.3f} km, "
                      f"the two dynamics {c['dyn_med']:.3f} km apart; ours was closer in {c['ours_wins']} of {c['n']} runs, so it is {gap_word}{floor}.")
    return out
