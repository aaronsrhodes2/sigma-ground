"""A body's name in place of its mass (and radius) on the formula tools that need them.

Mentat's router kept answering "escape velocity from Mars" in two steps, solar_system_body then escape_velocity, or in one step with the mass typed in
from memory (benchmark runs and the October 2026 research log; misc/FOLDED_CHAINS.md). Both put a lookup in the model's hands. With `body_name` the
tool looks the body up itself, in the same catalogue body_constants reads (field/interface/body_data.py: JPL's GM, exact to its published digits, and
the radius with its own source), so the router names the body and nothing else.

The mass used is GM / G. Every tool here multiplies it by G again, so a GM-only formula (r_s = 2 GM / c^2, v_esc = sqrt(2 GM / r)) keeps the GM's
exactness; the cited sources are the body catalogue's own, per field.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from sigma_ground.mcp.provenance import ToolResult


def _lookup(body_name: str) -> tuple[dict | None, ToolResult | None]:
    from sigma_ground.field.interface import body_data
    entry, close = body_data.resolve(body_name)
    if entry is None or "gm_km3_s2" not in entry:
        why = (f"'{body_name}' is not in the body catalogue." + (f" Close names: {', '.join(close)}." if close else "")) if entry is None \
            else f"{entry.get('name', body_name)} has no published mass (GM) in the catalogue."
        return None, ToolResult(value=None, source="sigma-ground body catalogue (JPL DE440 header, satellite ephemerides, Horizons headers)",
                                notes=why + " Give mass_kg (and radius_m) instead.", inputs={"body_name": body_name})
    d = body_data.describe(entry)
    src = d.get("sources", {})
    return {"name": d.get("name", body_name), "gm_km3_s2": d["gm_km3_s2"], "mass_kg": d["mass_kg"]["value"],
            "radius_m": d["radius_km"] * 1e3 if isinstance(d.get("radius_km"), (int, float)) else None,
            "gm_source": src.get("gm_km3_s2", "body catalogue"), "radius_source": src.get("radius_km", "body catalogue")}, None


def with_body(compute: Callable[..., ToolResult], body_name: str | None, mass_kg: float | None, *,
              radius_m: float | None = None, altitude_km: float | None = None, needs_radius: bool = False) -> ToolResult:
    """compute(mass_kg) or compute(mass_kg, radius_m), with the mass (and, when not given, the radius) of `body_name` looked up.

    Without body_name this is the tool as it always was. With it the body's GM wins over a typed mass (the typed one is the model's memory);
    the radius is radius_m (from the centre) if given, else the body's radius plus altitude_km, else the body's surface.
    """
    if not body_name:
        # the contract's own missing_argument error (structured, with an example), as when mass_kg was a required parameter
        from sigma_ground.mcp.contract import ContractError
        if mass_kg is None:
            raise ContractError("missing_argument", "needs 'mass_kg' (the mass, kg) or 'body_name' (a body to look up, e.g. 'mars')", param="mass_kg",
                                expected="mass_kg (a mass) or body_name (a body in the catalogue)",
                                example={"mass_kg": "1.989e30 kg", **({"radius_m": "6.957e8 m"} if needs_radius else {})})
        if not needs_radius:
            return compute(mass_kg)
        if radius_m is None:
            raise ContractError("missing_argument", "needs 'radius_m' (distance from the centre, m), or 'body_name' to use that body's surface",
                                param="radius_m", expected="radius_m (a length from the centre)", example={"mass_kg": mass_kg, "radius_m": "6.371e6 m"})
        return compute(mass_kg, radius_m)
    body, err = _lookup(body_name)
    if err:
        return err
    notes = [f"{body['name']}: GM {body['gm_km3_s2']:.12g} km^3/s^2 ({body['gm_source']}); mass = GM / G = {body['mass_kg']:.6g} kg."]
    if mass_kg is not None:
        notes.append(f"The mass_kg given ({mass_kg:g}) was not used: the catalogue's value for {body['name']} was.")
    inputs = {"body_name": body["name"], "mass_kg": body["mass_kg"]}
    used_radius = False
    if needs_radius:
        if radius_m is not None:
            r = float(radius_m)
            where = "radius_m given (from the centre)"
        else:
            if body["radius_m"] is None:
                return ToolResult(value=None, source="sigma-ground body catalogue", inputs={"body_name": body_name},
                                  notes=f"{body['name']} has no radius in the catalogue; give radius_m (from the centre).")
            used_radius = True
            r = body["radius_m"] + (float(altitude_km) * 1e3 if altitude_km else 0.0)
            where = f"radius {body['radius_m'] / 1e3:g} km ({body['radius_source']})" + (f" + altitude {altitude_km:g} km" if altitude_km else ", at the surface")
        notes.append(f"r = {r:g} m: {where}.")
        inputs.update({"radius_m": r} if altitude_km is None else {"radius_m": r, "altitude_km": altitude_km})
        res = compute(body["mass_kg"], r)
    else:
        res = compute(body["mass_kg"])
    if res.value is None:
        return res
    return dataclasses.replace(res, inputs=inputs, provenance_tag="DERIVED", notes=" ".join([*notes, res.notes or ""]).strip(),
                               source=f"{res.source}; {body['name']}'s GM from {body['gm_source']}" + (f", radius from {body['radius_source']}" if used_radius else ""))
