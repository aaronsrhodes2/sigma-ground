"""MCP exposure of the Materia simulator — the Mentat↔Materia seam.

Mentat (this MCP server) is a *consumer* of the Materia track's public API:
`translate`, `run_spec`, `synthesize_chain`, `SCENARIOS`, `VERB_MANIFEST`. We
import them; we never edit Materia's package. Materia owns routing + chaining;
Mentat owns exposing the capability to MCP clients and the qwen switchboard.

Three tools:
  simulate(scenario)            — natural-language front door (Materia routes +
                                  chains internally; single OR multi-verb).
  run_simulation(verb, params)  — structured, deterministic, no LLM.
  list_simulation_scenarios()   — discovery (the verb manifest).

Materia's MateriaResult.to_dict() has no top-level `value`, so the switchboard's
answer-extractor can't read it. We wrap every result in a ToolResult: the
chain's primary output as `value`, the worked detail in `inputs.chain`, and the
self-check carried into `notes`/`provenance_tag` so groundedness survives the
MCP boundary (a failed sub-check is flagged, never hidden).
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult


def _units_for_output(key: str | None) -> str:
    """Best-effort SI units from an output field's name suffix."""
    if not key:
        return ""
    if key.endswith("_m_s"):
        return "m/s"
    if key.endswith("_km_h"):
        return "km/h"
    if key.endswith("_K"):
        return "K"
    if key.endswith("_J"):
        return "J"
    if key.endswith("_pct"):
        return "%"
    if "mach" in key:
        return ""
    if key.endswith("_s"):
        return "s"
    if key.endswith("_m"):
        return "m"
    return ""


def _wrap(results: list, *, spec=None, materia=None) -> ToolResult:
    """Turn a list of MateriaResults (a 1+ step chain) into one ToolResult."""
    if materia is None:
        from sigma_ground import materia as materia
    final = results[-1]
    primary_key = next(iter(final.outputs), None)
    primary_val = final.outputs.get(primary_key) if primary_key else None
    verbs = [r.name for r in results]
    all_passed = all(r.validation.get("passed", True) for r in results)

    if spec is not None and len(results) > 1:
        note = materia.synthesize_chain(spec, results)
    else:
        note = final.summary
    if not all_passed:
        failed = ", ".join(r.name for r in results
                           if not r.validation.get("passed", True))
        note = f"[self-check FAILED: {failed}] {note}"

    # Flag drops above the modeled atmosphere — the isothermal / constant-g model
    # is only valid to ~100 km; beyond that an "impact speed" may be a t_max-
    # truncated mid-fall value, so label it rather than assert it.
    alt = max((r.inputs.get("drop_altitude_m") or r.inputs.get("start_altitude_m")
               or 0.0) for r in results)
    if alt > 150_000:
        note = (f"[out of modeled regime: {alt/1000:.0f} km is above the "
                f"isothermal-atmosphere / constant-g model (~100 km); result "
                f"approximate] {note}")

    return ToolResult(
        value=primary_val,
        units=_units_for_output(primary_key),
        source=f"sigma_ground.materia ({' → '.join(verbs)})",
        provenance_tag="DERIVED",
        notes=note,
        inputs={"primary_output": primary_key,
                "chain": [r.to_dict() for r in results]},
    )


def simulate(scenario: str, use_llm: bool = True) -> ToolResult:
    """Run a natural-language what-if through the Materia simulator.

    Materia translates the scenario to a (possibly multi-step, chained)
    Simulation Spec and runs it. If it isn't a scenario Materia can model, the
    clarification is returned with value None — never a fabricated answer.

    Args:
        scenario: plain-English physics what-if (e.g. "how fast does a 5 cm
                  copper ball hit the ground from 10 km?").
        use_llm:  allow the qwen residual for phrasings the keywords miss
                  (degrades gracefully if ollama is unreachable).
    """
    from sigma_ground import materia
    spec = materia.translate(scenario, use_qwen=use_llm)
    if not spec.is_runnable():
        return ToolResult(
            value=None, source="sigma_ground.materia.translator",
            provenance_tag="SPECULATIVE-PENDING", notes=spec.note,
            inputs={"scenario": scenario, "route": spec.source})
    try:
        results = materia.run_spec(spec)
    except Exception as e:   # boundary: a degenerate/out-of-range scenario must not crash
        return ToolResult(
            value=None, source="sigma_ground.materia",
            provenance_tag="SPECULATIVE-PENDING",
            notes=(f"Recognized the scenario but could not simulate it "
                   f"({type(e).__name__}: {e}) — likely degenerate or "
                   f"out-of-range inputs."),
            inputs={"scenario": scenario, "route": spec.source})
    return _wrap(results, spec=spec, materia=materia)


# Slots that name a physical size/mass — a zero or negative value is
# unphysical and would otherwise divide-by-zero deep in the engine.
_POSITIVE_SLOTS = ("radius_m", "diameter_m", "payload_mass_kg", "mass_kg",
                   "drag_area_m2")


def _validate_params(params: dict[str, Any]) -> str | None:
    """Reject obviously-unphysical inputs with a clear message up front
    (better than a cryptic downstream crash)."""
    for k in _POSITIVE_SLOTS:
        if k in params:
            try:
                v = float(params[k])
            except (TypeError, ValueError):
                return f"{k} must be a number (got {params[k]!r})"
            if v <= 0:
                return f"{k} must be positive (got {v})"
    return None


def _decline(notes: str, **inputs) -> ToolResult:
    return ToolResult(value=None, source="sigma_ground.materia",
                      provenance_tag="SPECULATIVE-PENDING", notes=notes,
                      inputs=inputs)


def run_simulation(verb: str, params: dict[str, Any] | None = None) -> ToolResult:
    """Run one Materia verb directly with explicit parameters (no LLM).

    Args:
        verb:   a name from list_simulation_scenarios() (e.g.
                "terminal_velocity_drop").
        params: keyword arguments for that verb's slots.
    """
    from sigma_ground import materia
    params = params or {}
    if verb not in materia.SCENARIOS:
        return _decline(f"Unknown simulation verb {verb!r}. "
                        f"Known: {sorted(materia.SCENARIOS)}",
                        verb=verb, params=params)
    bad = _validate_params(params)
    if bad:
        return _decline(f"Cannot simulate {verb!r}: {bad}", verb=verb, params=params)
    try:
        result = materia.SCENARIOS[verb](**params)
    except Exception as e:   # boundary: an engine error must never crash the tool
        return _decline(f"Could not simulate {verb!r} "
                        f"({type(e).__name__}: {e})", verb=verb, params=params)
    return _wrap([result], materia=materia)


def list_simulation_scenarios() -> ToolResult:
    """List the Materia simulation verbs, with their inputs and outputs."""
    from sigma_ground import materia
    catalog = {
        verb: {"description": m.get("description", ""),
               "slots": list(m.get("slots", {})),
               "outputs": list(m.get("outputs", []))}
        for verb, m in materia.VERB_MANIFEST.items()
    }
    return ToolResult(
        value=list(catalog),
        source="sigma_ground.materia.VERB_MANIFEST",
        provenance_tag="DERIVED",
        notes=("Materia simulation verbs reachable via simulate() (NL) or "
               "run_simulation(verb, params) (structured)."),
        inputs={"catalog": catalog})
