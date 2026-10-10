"""U5 step 0: fetch every physical leaf from Mentat (PLATINUM 21) and store value + source.

Run from the sigma-ground repo root:  python -X utf8 <this> <outdir>
"""
import sys, json, os
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

out = sys.argv[1]
L = {}
def get(key, tool, args):
    v, s = mentat(tool, args)
    L[key] = {"value": v, "source": s, "tool": tool, "args": args}
    return v

sun = get("sun", "solar_system_body", {"body_name": "sun"})
mer = get("mercury", "solar_system_body", {"body_name": "mercury"})
G = get("G", "lookup_constant", {"name": "G"})
c = get("c", "lookup_constant", {"name": "c"})
au = get("au", "lookup_constant", {"name": "au"})
rs = get("r_s_sun", "schwarzschild_radius", {"mass_kg": sun["mass_kg"]})

# e = 0.2056 [LIT] NASA NSSDC Mercury fact sheet (verified by web search 2026-10-10); Mentat has no eccentricity field.
E_LIT = 0.2056
L["e_mercury"] = {"value": E_LIT, "source": "[LIT] NASA NSSDC Mercury Fact Sheet, nssdc.gsfc.nasa.gov/planetary/factsheet/mercuryfact.html (orbital eccentricity 0.2056)"}

from sigma_ground.inventory.core.sigma import XI
L["XI"] = {"value": XI, "source": "sigma_ground.inventory.core.sigma.XI (repo constant, Omega_b/(Omega_b+Omega_c) Planck 2018)"}

a = mer["semimajor_axis_au"] * au
L["a_mercury_m"] = {"value": a, "source": "Mentat semimajor_axis_au x Mentat au"}

# symbolic legs
get("xi_over_3", "simplify_expr", {"expression": f"{XI}/3"})
get("xi_over_6", "simplify_expr", {"expression": f"{XI}/6"})
get("dV_dr", "differentiate_expr", {"expression": "-G*M/r - G*M*xi*r_s/(2*r**2)", "variable": "r"})
get("exp_sigma_series", "series_expansion", {"expression": "exp(xi*r_s/r)", "variable": "xi", "point": "0", "order": 2})
# Binet: u'' + (1 - 2b/L^2) u = GM/L^2  ->  advance per orbit 2 pi (1/sqrt(1-2b/L^2) - 1), to first order in b
get("binet_advance_series", "series_expansion",
    {"expression": "2*pi*(1/sqrt(1 - 2*b/L2) - 1)", "variable": "b", "point": "0", "order": 2})
# ratio of sigma-hook advance (b = G M xi r_s/2, L2 = G M p) to GR advance 6 pi G M/(c^2 p), with r_s = 2 G M/c^2
get("ratio_symbolic", "simplify_expr",
    {"expression": "(2*pi*(G*M*xi*(2*G*M/c**2)/2)/(G*M*p)) / (6*pi*G*M/(c**2*p))"})
# GR advance per orbit (rad) for the nominal orbit, evaluated by Mentat
p_expr = f"{a}*(1-{E_LIT}**2)"
get("dw_GR_rad_per_orbit", "simplify_expr",
    {"expression": f"6*pi*{G}*{sun['mass_kg']}/({c}**2*{p_expr})"})
get("dw_sigma_rad_per_orbit_pred", "simplify_expr",
    {"expression": f"pi*{XI}*{rs}/({p_expr})"})
# orbits per Julian century (36525 d is a definition) from Mentat's period
get("orbits_per_century", "simplify_expr", {"expression": f"36525/{mer['orbital_period_d']}"})
get("rad_to_arcsec", "simplify_expr", {"expression": "180*3600/pi"})
json.dump(L, open(os.path.join(out, "leaves.json"), "w"), indent=1)
print(json.dumps({k: v["value"] for k, v in L.items()}, indent=1, default=str))
