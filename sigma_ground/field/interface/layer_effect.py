"""How much each physics layer of the forecast model matters for a body: the lookup over fixtures/layer_effect_table.json (built by scripts/make_layer_effect_table.py from the C23b ablation runs).

The number is a ratio: the median error against JPL DE440 with that one layer removed, over the error with every layer on. It measures how far the layer moves the agreement with DE440, not an absolute accuracy,
and DE440 itself contains every one of these effects, so a layer that DE440 includes and the model lacks shows up as a large ratio."""
from __future__ import annotations

import functools
import json
from pathlib import Path

from . import body_data

_FIX = Path(__file__).parent / "fixtures"
LAYER_NAMES = {
    "eih": "general relativity (Einstein-Infeld-Hoffmann cross terms between bodies)",
    "zonals": "the planets' J2, J3, J4 gravity-field terms",
    "tides": "the tidal force between bodies",
    "srp": "solar radiation pressure",
    "asteroids": "the 343 largest asteroids",
    "ring": "the main-belt ring that stands for the many small asteroids",
    "kbo": "the Kuiper-belt objects",
    "c22": "the Moon's C22 gravity term",
}
UNDETECTABLE_BELOW = 1.05                      # a ratio under this is inside the run-to-run scatter of the four starts


class LayerEffectError(ValueError):
    """The body is not one of those ablated; the message says which are."""


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads((_FIX / "layer_effect_table.json").read_text(encoding="utf-8"))


def model_changed_since_build() -> bool | None:
    """True when the forecast model's fingerprint no longer equals the one the table was measured with; None when it cannot be checked."""
    try:
        from . import reduced_model as rm
        return rm.model_fingerprint("restated_zon") != _table()["model_fingerprint"]
    except Exception:
        return None


def ablated_bodies() -> list[str]:
    return list(_table()["control_km"])


def layer_effect(name: str) -> dict:
    t = _table()
    entry, close = body_data.resolve(name)
    key = entry["name"] if entry else name.strip()
    if key not in t["control_km"]:
        raise LayerEffectError(f"'{name}' was not ablated: the table holds {', '.join(t['control_km'])}" + (f"; close names: {', '.join(close)}" if close else ""))
    control = t["control_km"][key]
    leads = [str(x) for x in t["leads_years"]]
    layers: dict[str, dict] = {}
    for k, per_body in t["without_layer_km"].items():
        row = {"description": LAYER_NAMES[k], "without_layer_km": {}, "ratio": {}}
        for lead in leads:
            w, c = per_body[key][lead]["km"], control[lead]["km"]
            row["without_layer_km"][lead] = w
            row["ratio"][lead] = w / c
        layers[k] = row
    ranked = sorted(layers, key=lambda k: layers[k]["ratio"]["5"], reverse=True)
    big = [k for k in ranked if layers[k]["ratio"]["5"] >= UNDETECTABLE_BELOW]
    helped_by_removal = [k for k in ranked if layers[k]["ratio"]["5"] < 1 / UNDETECTABLE_BELOW]
    nothing = [k for k in ranked if k not in big and k not in helped_by_removal]
    c5 = control["5"]["km"]
    parts = [f"For {key}, the error against JPL DE440 at +5 yr is {_fmt(c5)} with every layer on."]
    if big:
        parts.append("Removing a layer makes it worse by: " + "; ".join(f"{LAYER_NAMES[k]} x{layers[k]['ratio']['5']:,.3g}" for k in big[:4]) + (f" (and {len(big) - 4} smaller)" if len(big) > 4 else "") + ".")
    if nothing:
        parts.append("No detectable effect (within 5 %): " + ", ".join(LAYER_NAMES[k] for k in nothing) + ".")
    if helped_by_removal:
        parts.append("Removing these made the agreement BETTER: " + "; ".join(f"{LAYER_NAMES[k]} x{layers[k]['ratio']['5']:.2f}" for k in helped_by_removal)
                     + ". That is unexplained (a wrong coefficient or a compensating error); it is recorded, not resolved.")
    if key == "Moon":
        parts.append("The Moon's error with every layer on is the floor of this comparison, so a layer whose effect is smaller than that (the tides) cannot show.")
    return {"body": key, "error_with_all_layers_km": {l: control[l]["km"] for l in leads}, "layers": layers, "ranked_by_5yr_ratio": ranked, "summary": " ".join(parts),
            "how_measured": ("each layer removed alone from the forecast's own model, hindcast starts 2000, 2005, 2010, 2015, step 0.0125 d, medians, truth JPL DE440; ratio = error without the layer over error with every layer; "
                             "misc/CANDIDATE_LEDGER.md C23b"),
            "model_changed_since_build": model_changed_since_build()}


def _fmt(x: float) -> str:
    return f"{x:,.0f} km" if x >= 100 else (f"{x:.3g} km" if x >= 1 else f"{x * 1000:.3g} m")
