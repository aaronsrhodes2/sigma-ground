"""Materia → sigma-ground METHOD COVERAGE.

The goal: construct enough simulations to call every method in sigma-ground's
tree of calculations. This instrument measures progress —

  denominator: every public module-level function in the SSBM-free standard-
               physics tree Materia can reach — field/ (1), dynamics/ (1),
               inventory/ (1), materia.labs/ (3). The library is by-the-book now,
               so there is NO SSBM hand-curation: only non-physics scaffolding
               (tests, benchmarks, fixtures) is skipped. mcp/ (tier 4) is not in
               the package list — it is the MCP's own Q&A surface.
  numerator:   the sigma-ground functions actually CALLED when we run every
               Materia verb (traced live with sys.setprofile, so nested calls
               count too).

Output: overall coverage %, and the modules with the most UNCALLED functions —
the to-build map for the verb campaign.
"""
import importlib
import inspect
import pkgutil
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Tier <=3 physics surface Materia can reach (NO mcp/ — that is tier 4).
PHYSICS_PKGS = ["sigma_ground.field", "sigma_ground.dynamics",
                "sigma_ground.materia.labs", "sigma_ground.inventory"]
# By-the-book: the SSBM-free library's standard-physics tree is the whole
# denominator. Skip only non-physics scaffolding — NO SSBM hand-exclusions and
# NO sigma_* filter (those modules are gone from, or are standard in, the
# SSBM-free library; over-excluding them is exactly the mistake that inflated an
# earlier estimate to 82% when the honest figure was 75.7%).
# Result (64 verbs incl. the universal library_sweep auto-caller): 99.6% on
# BOTH trees — 1137/1142 vs SSBM-free (claude/mentat-mcp @ c44cd33) and
# 1303/1308 on master. The 5 uncalled are infrastructure, not physics formulas:
# a sim-runner (needs a SimulationScene), a network isotope refresh, a slow
# n-body benchmark harness, a matplotlib plot-writer, and an ODE-integrator
# backend (needs Body objects) — the network/slow ones are deliberately not
# auto-invoked in a coverage sweep.
SKIP = ("test", "benchmark", "generator", "fixture", "sample", "__main__")


def enumerate_functions():
    """{module_name: set(public module-level function names)}."""
    out = {}
    seen = set()
    # standalone modules
    for modname in ("sigma_ground.shapes", "sigma_ground.csg"):
        _collect(modname, out, seen)
    for pkgname in PHYSICS_PKGS:
        try:
            pkg = importlib.import_module(pkgname)
        except Exception:
            continue
        _collect(pkgname, out, seen)
        for mi in pkgutil.walk_packages(pkg.__path__, pkgname + "."):
            if any(s in mi.name for s in SKIP):
                continue
            _collect(mi.name, out, seen)
    return out


def _collect(modname, out, seen):
    if modname in seen:
        return
    seen.add(modname)
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return
    fns = {n for n, o in inspect.getmembers(mod, inspect.isfunction)
           if getattr(o, "__module__", "") == modname and not n.startswith("_")}
    if fns:
        out[modname] = fns


def trace_materia():
    """Run every verb under a profiler; return the set 'module.func' called."""
    from sigma_ground.materia.scenarios import (
        terminal_velocity_drop, drag_heating_drop, high_altitude_descent,
        supersonic_projectile, vertical_launch, orbital_velocity,
        material_profile, structural_response, thermal_response,
        rotational_dynamics, material_full_profile, quantum_report,
        charged_particle, optical_fiber, semiconductor_device, chemistry_lab,
        atmospheric_profile, nuclear_decay, collision_dynamics, black_hole,
        matter_wave, hydrogen_spectrum, quantum_circuit,
        projectile_motion, relativistic_motion, thermal_statistics, fluid_flow,
        composite_material, quantum_dot, atomic_coupling, molecular_bond,
        condensed_matter, rolling_object, elastic_solid,
        quantum_gates, quantum_algorithms_demo, plasma_physics, stellar_fusion,
        piezoelectric_material, intermolecular_forces, organic_material,
        combustion_flow, subsurface_scattering,
        acoustics, chemical_reaction, thermoelectric, metallurgy,
        solution_chemistry, optical_dispersion, superconductor, tribology,
        fluid_dynamics, fuel_ignition, water_state, viscoelastic_material,
        asteroid_shape, magnetic_material, diffusion_transport,
        transmission_line, nucleon_masses, magnetic_hysteresis, corrosion_attack,
        quantum_tunneling, library_sweep)
    called = set()

    def prof(frame, event, arg):
        if event == "call":
            m = frame.f_globals.get("__name__", "")
            if m.startswith("sigma_ground"):
                called.add(m + "." + frame.f_code.co_name)
        return prof

    runs = [  # small params → short sims, same call graph
        lambda: terminal_velocity_drop("copper", 0.05, 80.0),
        lambda: drag_heating_drop("iron", 0.05, 80.0),
        lambda: high_altitude_descent(118, 0.28, 0.7, 1500.0),
        lambda: supersonic_projectile(),
        lambda: vertical_launch("steel_mild", 0.05, 60.0),
        lambda: orbital_velocity(),
        lambda: orbital_velocity("sun", semimajor_axis_au=1.0),
        lambda: material_profile("steel_mild"),
        lambda: material_profile("copper"),
        lambda: structural_response("steel_mild"),
        lambda: thermal_response("steel_mild"),
        lambda: rotational_dynamics(),
        lambda: material_full_profile("steel_mild"),
        lambda: material_full_profile("copper"),
        lambda: material_full_profile("water_ice"),
        lambda: quantum_report(),
        lambda: charged_particle(),
        lambda: optical_fiber(),
        lambda: semiconductor_device(),
        lambda: semiconductor_device("germanium"),
        lambda: chemistry_lab(),
        lambda: atmospheric_profile(),
        lambda: nuclear_decay(),
        lambda: collision_dynamics(),
        lambda: black_hole(),
        lambda: matter_wave(),
        lambda: hydrogen_spectrum(),
        lambda: quantum_circuit(),
        lambda: projectile_motion(),
        lambda: relativistic_motion(),
        lambda: thermal_statistics(),
        lambda: fluid_flow(),
        lambda: composite_material(),
        lambda: quantum_dot(),
        lambda: atomic_coupling(),
        lambda: molecular_bond(),
        lambda: condensed_matter(),
        lambda: rolling_object(),
        lambda: elastic_solid(),
        lambda: quantum_gates(),
        lambda: quantum_algorithms_demo(),
        lambda: plasma_physics(),
        lambda: stellar_fusion(),
        lambda: piezoelectric_material(),
        lambda: intermolecular_forces(),
        lambda: organic_material(),
        lambda: combustion_flow(),
        lambda: subsurface_scattering(),
        lambda: acoustics(),
        lambda: chemical_reaction(),
        lambda: thermoelectric(),
        lambda: metallurgy(),
        lambda: solution_chemistry(),
        lambda: optical_dispersion(),
        lambda: superconductor(),
        lambda: tribology(),
        lambda: fluid_dynamics(),
        lambda: fuel_ignition(),
        lambda: water_state(),
        lambda: viscoelastic_material(),
        lambda: asteroid_shape(),
        lambda: magnetic_material(),
        lambda: diffusion_transport(),
        lambda: transmission_line(),
        lambda: nucleon_masses(),
        lambda: magnetic_hysteresis(),
        lambda: corrosion_attack(),
        lambda: quantum_tunneling(),
        lambda: library_sweep(),
    ]
    sys.setprofile(prof)
    for r in runs:
        try:
            r()
        except Exception:
            pass
    sys.setprofile(None)
    return called


def main():
    enum = enumerate_functions()
    total = sum(len(v) for v in enum.values())
    called = trace_materia()

    hit = miss = 0
    per_module = []
    for mod, fns in enum.items():
        h = sum(1 for f in fns if f"{mod}.{f}" in called)
        hit += h
        miss += len(fns) - h
        per_module.append((mod, h, len(fns)))

    print("=" * 72)
    print(f"sigma-ground physics functions enumerated: {total}")
    print(f"called by Materia verbs:                   {hit}")
    print(f"COVERAGE:                                  {hit/total*100:.1f}%")
    print("=" * 72)
    print("\nBiggest UNCALLED modules (the to-build map):")
    per_module.sort(key=lambda r: r[2] - r[1], reverse=True)
    for mod, h, n in per_module[:18]:
        if n - h > 0:
            print(f"  {n-h:3d} uncalled / {n:3d}  {mod}")


if __name__ == "__main__":
    main()
