"""How well this model predicts a body: the lookup over fixtures/body_skill_table.json (built by scripts/make_body_skill_table.py from the runs against JPL DE440), with the class and a plain sentence.

The class decides what may be claimed: a body that loses its orbital phase within a year gets no position claim at all; a body whose two measurements disagree by an order of magnitude gets neither number as its skill."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from . import body_data

_FIX = Path(__file__).parent / "fixtures"
CLASSES = ("predictable", "loses_orbital_phase", "measurements_disagree", "not_scored_at_a_converged_step")


class SkillError(ValueError):
    """The body is not one of those scored; the message says which are."""


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads((_FIX / "body_skill_table.json").read_text(encoding="utf-8"))


def stale_sources() -> list[str]:
    """Source files whose bytes no longer match the hash the table was built from (a table is a cache of a calculation; this says when it is out of date)."""
    out = []
    for rel, h in _table()["metadata"]["sources"].items():
        p = _FIX / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            out.append(rel)
    return out


def scored_bodies() -> list[str]:
    return sorted(_table()["bodies"])


def _km(x: float) -> str:
    if x >= 100:
        return f"{x:,.0f} km"
    if x >= 1:
        return f"{x:.3g} km"
    return f"{x * 1000:.3g} m" if x >= 0.001 else f"{x * 1e6:.3g} mm"


def body_skill(name: str) -> dict:
    tab = _table()
    entry, close = body_data.resolve(name)
    key = entry["name"] if entry else name.strip()
    if key not in tab["bodies"]:
        raise SkillError(f"'{name}' is not scored: the table holds {len(tab['bodies'])} bodies (the Moon, the planets to Pluto and 43 moons), not the Sun, asteroids, comets or Kuiper-belt objects"
                         + (f"; close names: {', '.join(close)}" if close else ""))
    b = json.loads(json.dumps(tab["bodies"][key]))                      # a copy
    cls = b["class"]
    fs = b["forecast_step_0.025d"]
    lead_text = ", ".join(f"+{k} yr {_km(v['median_km'])}" for k, v in fs.items())
    if cls == "predictable":
        s = (f"{key} is predictable: about {_km(b['best_estimate_5yr_km'])} from JPL DE440 after 5 years ({b['best_estimate_basis']}).")
    elif cls == "loses_orbital_phase":
        c = b["converged"]
        s = (f"{key} loses its orbital phase within a year: even at the converged step its position is {_km(c['error_km']['1'])} off after 1 year, {c['error_at_1yr_over_orbital_radius']:.1f} times its orbital radius about {c['relative_to']}. "
             "This model cannot say where it is; give no position. The cause is not known.")
    elif cls == "measurements_disagree":
        s = f"{key}: {b['note']}."
    else:
        s = f"{key} is not scored at a converged step: {b['note']}."
    if key in ("Mercury", "Moon"):
        from . import where_when
        s += " " + where_when._STEP_CAVEATS[key]
    b.update({"body": key, "summary": s, "forecast_step_leads_text": f"at the forecast step of 0.025 d, median error: {lead_text}",
              "class_definitions": {"predictable": "measured error against DE440 is bounded at the size given",
                                    "loses_orbital_phase": "converged +1 yr error is at least a tenth of the orbital radius; no position claim",
                                    "measurements_disagree": "the converged moon model and the full-system run differ by more than 10 times where the step is adequate in both; cause not established",
                                    "not_scored_at_a_converged_step": "only scored at a step that cannot integrate it"},
              "table_sources_changed_since_build": stale_sources() or None})
    return b
