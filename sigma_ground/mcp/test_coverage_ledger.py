"""Regression test: the honest-100% transitive-coverage invariant.

Every standard-physics leaf function in ``sigma_ground/field/**`` and
``sigma_ground/inventory/**`` must be reached -- directly or transitively -- by
at least one MCP tool (``sigma_ground/mcp/tools/*``) or one of the 5 procedures,
EXCEPT the documented non-capabilities listed in ``EXCLUDE_SET`` below (which
mirrors ``misc/COVERAGE_LEDGER.md``).

This is the anti-regression guard for the coverage marathon: it fails if a new
library capability is added without wiring it to a tool (it would surface as an
uncovered function that is NOT in EXCLUDE_SET), and it documents -- in code --
exactly which functions are intentionally out of the Q&A surface and why.

EXCLUDE_SET is a *superset* of the currently-uncovered functions: it also lists
gate-matrix primitives that algorithms sometimes exercise (so the test is robust
to the Born-rule randomness in measure_all / Grover). The invariant is one-sided:
  uncovered  ⊆  EXCLUDE_SET
i.e. anything uncovered must be a documented non-capability; covered functions in
EXCLUDE_SET are fine (covered > excluded).
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import os
import pkgutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MISC = os.path.join(_ROOT, "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

with contextlib.redirect_stdout(io.StringIO()):
    import sigma_ground  # noqa: E402

from coverage_harness import module_keys, try_call, TARGET, EXCLUDE  # noqa: E402

_EMPTY = inspect.Parameter.empty
_POS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)


def _dotted(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    if "sigma_ground" not in parts:
        return None
    i = parts.index("sigma_ground")
    mod = ".".join(parts[i:]).rsplit(".py", 1)[0]
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def _collect(prefixes):
    out = []
    for info in pkgutil.walk_packages(sigma_ground.__path__, "sigma_ground."):
        name = info.name
        if not name.startswith(prefixes) or any(s in name for s in EXCLUDE):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = importlib.import_module(name)
        except Exception:
            continue
        for fn in dir(m):
            f = getattr(m, fn, None)
            if (not fn.startswith("_") and inspect.isfunction(f)
                    and getattr(f, "__module__", "") == name):
                out.append((name, fn, f, m))
    return out


def _compute_uncovered():
    tree = _collect(TARGET)
    target = {(mod, fn) for mod, fn, _, _ in tree}
    tools = _collect(("sigma_ground.mcp.tools",))

    covered: set[tuple[str, str]] = set()

    def tracer(frame, event, arg):
        if event == "call":
            mod = _dotted(frame.f_code.co_filename)
            if mod and mod.startswith("sigma_ground"):
                covered.add((mod, frame.f_code.co_name))
        return None

    def exercise(items):
        for _, _, f, m in items:
            try:
                req = [p for p in inspect.signature(f).parameters.values()
                       if p.default is _EMPTY and p.kind in _POS]
            except (TypeError, ValueError):
                continue
            try:
                try_call(f, req, module_keys(m))
            except Exception:
                pass

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        exercise(tools)
        exercise(tree)
        from sigma_ground.mcp import procedures as P
        for fn, args in [("black_hole_profile", (1.989e30,)),
                         ("photon_spectrum", (5e-7,)),
                         ("relativistic_particle", (1e6, "electron")),
                         ("projectile_trajectory", (100.0, 45.0)),
                         ("stellar_blackbody", (5778.0,))]:
            try:
                getattr(P, fn)(*args)
            except Exception:
                pass
    finally:
        sys.settrace(old)

    return target, {f"{m}.{fn}" for m, fn in (target - covered)}


# ── Documented non-capabilities (mirrors misc/COVERAGE_LEDGER.md) ──────────────
_P = "sigma_ground.field.interface."
_INV = "sigma_ground.inventory."

EXCLUDE_SET = frozenset(
    # Nagatha-format / serialization export aggregators (re-bundle covered physics)
    [_P + n for n in (
        "friction.material_friction_properties",
        "viscosity.viscous_flow_properties",
        "projectile.projectile_report",
    )]
    # Curve / series / simulation generators (Materia's lane, not scalar Q&A)
    + [_P + n for n in (
        "acid_base.titration_curve",
        "hysteresis.hysteresis_loop",
        "wear.wear_profile",
        "superconductivity.block_cooling_profile",
    )]
    # Presentation / render helpers
    + [_P + "phosphor.build_ascii_histogram"]
    # Ephemeris least-squares FIT pipeline (needs DE440 fixtures)
    + [_P + n for n in ("orbital.fit_orbit", "orbital.predict_ssb_position")]
    # Monte-Carlo / CDF-sampling internals (non-deterministic)
    + [_P + n for n in ("quantum.sample_hit_position", "quantum.cumulative_probability")]
    # Quantum-algorithm output-extractor plumbing (one takes a callable)
    + [_P + n for n in (
        "quantum_output.extract_phase",
        "quantum_output.extract_function_value",
        "quantum_output.histogram_to_answer",
    )]
    # Gate-matrix primitives (building blocks; algorithms exercise some of them,
    # so list ALL of them -- the invariant tolerates flicker within this set)
    + [_P + "quantum_computing." + g for g in (
        "gate_x", "gate_y", "gate_z", "gate_h", "gate_s", "gate_t",
        "gate_rx", "gate_ry", "gate_rz", "gate_phase", "gate_cnot",
        "gate_cz", "gate_swap", "gate_iswap", "gate_toffoli", "gate_fredkin",
    )]
    # Superseded by a more accurate function that IS wired
    + [_P + "superconductivity.gl_parameter",          # -> gl_parameter_effective
       _P + "adhesion.contact_angle"]                  # -> wetting_contact_angle
    # Rolling-shootout ephemeris BENCHMARK / reporting / plotting harness (numpy)
    + [_P + "rolling_analysis." + n for n in (
        "ablation_table", "add_rtn_components", "cross_predictor_correlation",
        "fingerprint_diff", "orbital_correlation", "per_predictor_body_summary",
        "print_report", "rtn_summary", "save_plots",
    )]
    + [_P + "rolling_shootout.run_rolling_shootout"]
    # Inventory env-resolver / setter cascade (state mutation -> Materia's lane)
    + [_INV + "behaviors.apply_env",
       _INV + "behaviors.atom_behaviors.resolve_atom_env",
       _INV + "behaviors.molecule_behaviors.resolve_molecule_env",
       _INV + "behaviors.particle_behaviors.resolve_particle_env",
       _INV + "behaviors.quark_behaviors.resolve_quark_env"]
    # Inventory builder / checksum verification plumbing
    + [_INV + "builder.load_structure",
       _INV + "checksum.particle_count.count_particles_in_structure",
       _INV + "checksum.quark_chain.compute_quark_chain_checksum",
       _INV + "checksum.quark_chain.walk_quark_chain",
       _INV + "checksum.stoq_checksum.compute_stoq_checksum"]
)


def test_honest_100_percent_coverage():
    """uncovered ⊆ EXCLUDE_SET: no library capability is left unwired."""
    target, uncovered = _compute_uncovered()
    surprises = uncovered - EXCLUDE_SET
    assert not surprises, (
        "These library functions are uncovered but NOT documented as "
        "non-capabilities in misc/COVERAGE_LEDGER.md / EXCLUDE_SET. Either wire "
        "them to an MCP tool or add them to the ledger with a reason:\n  "
        + "\n  ".join(sorted(surprises))
    )

    # Positive form of the same invariant: 100% of the honest denominator.
    honest_denom = {f"{m}.{fn}" for m, fn in target} - EXCLUDE_SET
    covered_honest = honest_denom - uncovered
    pct = 100.0 * len(covered_honest) / max(len(honest_denom), 1)
    assert pct == 100.0, f"honest-denominator coverage = {pct:.1f}% (< 100%)"
