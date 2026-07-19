"""Simulation Spec — the compiled plan the engine executes.

compile-then-run: the translator emits a Spec (one or more verb calls) up
front, and run_spec executes it deterministically. A single verb today; a
chain tomorrow (a later step may read earlier results). The LLM produces this
Spec and nothing else — it never executes a step or does the combining
arithmetic. That separation is what keeps a small model safe to use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .scenarios import SCENARIOS


@dataclass
class SpecStep:
    verb: str
    params: dict = field(default_factory=dict)
    # Data flow: {slot_name: (from_step_index, output_field)} — bind an input
    # slot to a PRIOR step's named output. This is what makes a chain a
    # decomposition (sub-solve → feed forward) rather than N independent runs.
    bindings: dict = field(default_factory=dict)


@dataclass
class SimulationSpec:
    question: str
    steps: list                       # list[SpecStep]
    source: str = "deterministic"     # "deterministic" | "qwen" | "clarify"
    note: str = ""
    # Set only when source == "clarify" AND the decline is a specific,
    # nameable missing variable (e.g. a material_required verb whose trigger
    # matched but no material was named) rather than a generic "I don't
    # understand this at all". Lets the caller ask a targeted follow-up
    # instead of a generic capability blurb, and re-resolve against the
    # SAME verb once the answer comes back (see front_door.Session
    # .pending_clarification).
    missing_slot: str = ""
    missing_verb: str = ""

    def is_runnable(self) -> bool:
        return bool(self.steps) and all(s.verb in SCENARIOS for s in self.steps)


def run_spec(spec: SimulationSpec) -> list:
    """Execute the chain, threading each step's bindings from prior results.

    A binding {slot: (i, field)} sets the step's `slot` to results[i].outputs
    [field]. A binding to a missing step or output raises loudly — a chain is
    only trustworthy if every hand-off is real (no silent defaulting).
    """
    results = []
    for n, step in enumerate(spec.steps):
        if step.verb not in SCENARIOS:
            raise KeyError(f"Unknown Materia verb: {step.verb!r}")
        params = dict(step.params)
        for slot, (from_step, field_name) in step.bindings.items():
            if not 0 <= from_step < n:
                raise IndexError(
                    f"step {n} ({step.verb}) binds {slot} to step {from_step}, "
                    f"which is not a prior step")
            src = results[from_step]
            if field_name not in src.outputs:
                raise KeyError(
                    f"step {n} ({step.verb}) binds {slot} to "
                    f"{src.name}.{field_name}, but that output does not exist; "
                    f"available: {sorted(src.outputs)}")
            params[slot] = src.outputs[field_name]
        results.append(SCENARIOS[step.verb](**params))
    return results


def synthesize_chain(spec: SimulationSpec, results: list) -> str:
    """Combine a chain's sub-results into one narration + an overall verdict.

    The chain is trustworthy only if EVERY sub-step passed its own self-check —
    weakest-link. We never average away a failed step.
    """
    lines = [f"━━ Materia chain · {len(results)} steps ━━", f"Q: {spec.question}", ""]
    for i, r in enumerate(results):
        lines.append(f"  [step {i}] {r.name}: {r.summary}")
    all_pass = all(r.validation.get("passed", True) for r in results)
    mark = ("✓ every step validated" if all_pass
            else "✗ a step failed its self-check")
    lines += ["", f"  chain self-check: {mark}"]
    return "\n".join(lines)
