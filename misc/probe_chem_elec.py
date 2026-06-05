"""Verify which chemistry/electronics functions actually exist + compute, before
exposing any via the MCP. (An Explore agent locates code; it does not run it.)
"""
import importlib
import inspect
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (module, func, sample_args, sample_kwargs)  args None → signature only
CANDIDATES = [
    ("field.interface.molecular_bonds", "pauling_bond_energy", ("H", "Cl"), {}),
    ("field.interface.molecular_bonds", "schomaker_stevenson_bond_length", ("C", "C"), {}),
    ("field.interface.molecular_bonds", "vsepr_bond_angle", (4, 0), {}),
    ("field.interface.molecular_bonds", "bond_polarity", ("H", "Cl"), {}),
    ("field.interface.chemical_reactions", "reaction_enthalpy", None, {}),
    ("field.interface.chemical_reactions", "arrhenius_rate_constant", None, {}),
    ("field.interface.acid_base", "henderson_hasselbalch", None, {}),
    ("field.interface.acid_base", "pKa_lookup", ("acetic acid",), {}),
    ("field.interface.electrochemistry", "nernst_potential", (0.34, 2, 1.0), {}),
    ("field.interface.electrochemistry", "standard_electrode_potential", ("Cu",), {}),
    ("field.interface.electrochemistry", "cell_potential", ("Cu", "Zn"), {}),
    ("field.interface.electrochemistry", "faraday_mass_deposited", None, {}),
    ("field.interface.solution", "solubility_product", ("AgCl",), {}),
    ("field.interface.solution", "boiling_point_elevation", (1.0,), {}),
    ("field.interface.solution", "freezing_point_depression", (1.0,), {}),
    ("field.interface.solution", "osmotic_pressure", None, {}),
    ("field.interface.electronics", "bloch_gruneisen_resistivity", (300.0, "copper"), {}),
    ("field.interface.electronics", "carrier_mobility_drude", None, {}),
    ("field.interface.electronics", "hall_coefficient", None, {}),
    ("field.interface.electronics", "pn_junction_builtin_voltage", None, {}),
    ("field.interface.electronics", "shockley_diode_current", None, {}),
    ("field.interface.dielectric", "dielectric_constant", ("water",), {}),
]


def short(x):
    s = repr(x)
    return s if len(s) <= 70 else s[:67] + "…"


def main():
    real = 0
    for mod, fn, args, kw in CANDIDATES:
        full = f"sigma_ground.{mod}"
        try:
            m = importlib.import_module(full)
        except Exception as e:
            print(f"  ✗ MODULE MISSING {full}: {type(e).__name__}")
            continue
        f = getattr(m, fn, None)
        if f is None or not callable(f):
            print(f"  ✗ no fn   {mod}.{fn}")
            continue
        try:
            sig = str(inspect.signature(f))
        except (TypeError, ValueError):
            sig = "(?)"
        if args is None:
            print(f"  • exists  {fn}{sig}")
            real += 1
            continue
        try:
            r = f(*args, **kw)
            print(f"  ✓ RUNS    {fn}{args} = {short(r)}")
            real += 1
        except Exception as e:
            print(f"  ⚠ exists, call failed  {fn}{sig} :: {type(e).__name__}: {str(e)[:40]}")
            real += 1
    print(f"\n  {real}/{len(CANDIDATES)} candidates are real (module + callable).")


if __name__ == "__main__":
    main()
