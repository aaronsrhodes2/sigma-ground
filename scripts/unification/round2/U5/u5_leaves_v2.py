"""U5 v2 step 0: fetch every physical leaf from Mentat (PLATINUM 21) and store value + source.

v2 = v1 leaves (fresh calls) + the legs the skeptics asked for:
  * XI provenance check: the sigma.py docstring's own Planck numbers, evaluated by Mentat
  * harmonic-gauge static Schwarzschild expansion (sign of the engine's 2PN leading term)
  * literature bounds turned into allowed extra advances (Will 2006, Will 2014, Park 2017)

Run from the sigma-ground repo root:  python -X utf8 <this> <outdir>
Writes <outdir>/leaves_v2.json
"""
import sys, json, os
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from mentat_client import mentat

out = sys.argv[1]
L = {}


def get(key, tool, args):
    v, s = mentat(tool, args)
    L[key] = {"value": v, "source": s, "tool": tool, "args": args}
    return v


def lit(key, value, source):
    L[key] = {"value": value, "source": source}
    return value


# ---------------- v1 leaves, fresh calls ----------------
sun = get("sun", "solar_system_body", {"body_name": "sun"})
mer = get("mercury", "solar_system_body", {"body_name": "mercury"})
G = get("G", "lookup_constant", {"name": "G"})
c = get("c", "lookup_constant", {"name": "c"})
au = get("au", "lookup_constant", {"name": "au"})
rs = get("r_s_sun", "schwarzschild_radius", {"mass_kg": sun["mass_kg"]})

E_LIT = lit("e_mercury", 0.2056,
            "[LIT] NASA NSSDC Mercury Fact Sheet, nssdc.gsfc.nasa.gov/planetary/factsheet/mercuryfact.html "
            "(orbital eccentricity 0.2056; re-fetched 2026-10-10 in the v2 repair, also a = 57.909e6 km, P = 87.969 d)")

from sigma_ground.inventory.core.sigma import XI
L["XI"] = {"value": XI,
           "source": "sigma_ground.inventory.core.sigma.XI = 0.1582 (repo constant). NOTE: the docstring at sigma.py:40 "
                     "attributes it to Planck 2018 Omega_b h^2 = 0.02237, Omega_c h^2 = 0.1200, but those numbers give "
                     "0.1571 (see leaf xi_docstring_check); 0.1582 is used here as the repo constant, not as a Planck value."}

a = mer["semimajor_axis_au"] * au
L["a_mercury_m"] = {"value": a, "source": "Mentat semimajor_axis_au x Mentat au"}

get("xi_over_3", "simplify_expr", {"expression": f"{XI}/3"})
get("xi_over_6", "simplify_expr", {"expression": f"{XI}/6"})
get("dV_dr", "differentiate_expr", {"expression": "-G*M/r - G*M*xi*r_s/(2*r**2)", "variable": "r"})
get("exp_sigma_series", "series_expansion", {"expression": "exp(xi*r_s/r)", "variable": "xi", "point": "0", "order": 2})
get("binet_advance_series", "series_expansion",
    {"expression": "2*pi*(1/sqrt(1 - 2*b/L2) - 1)", "variable": "b", "point": "0", "order": 2})
get("ratio_symbolic", "simplify_expr",
    {"expression": "(2*pi*(G*M*xi*(2*G*M/c**2)/2)/(G*M*p)) / (6*pi*G*M/(c**2*p))"})
p_expr = f"{a}*(1-{E_LIT}**2)"
get("dw_GR_rad_per_orbit", "simplify_expr", {"expression": f"6*pi*{G}*{sun['mass_kg']}/({c}**2*{p_expr})"})
get("dw_sigma_rad_per_orbit_pred", "simplify_expr", {"expression": f"pi*{XI}*{rs}/({p_expr})"})
get("orbits_per_century", "simplify_expr", {"expression": f"36525/{mer['orbital_period_d']}"})
get("rad_to_arcsec", "simplify_expr", {"expression": "180*3600/pi"})

# ---------------- v2 additions ----------------
# (1) XI provenance: the docstring's own Planck 2018 numbers (sigma.py:40)
get("xi_docstring_check", "simplify_expr", {"expression": "0.02237/(0.02237+0.1200)"})

# (2) Gauss perturbation bookkeeping for a = (mu/(c^2 r^2)) [ (a1 mu/r + a2 v^2) n + a3 (n.v) v ]:
#     orbit-averaged d(omega) = (pi mu/(c^2 p)) (-a1 - 2 a2 + 2 a3); ratio to GR (a1, a2, a3) = (4, -1, 4) -> 6
get("gauss_ratio_iers", "simplify_expr", {"expression": "(-4 - 2*(-1) + 2*4)/6"})
get("gauss_ratio_engine_gr1pn", "simplify_expr", {"expression": "(-(-4) - 2*(1) + 2*(-4))/6"})
get("gauss_ratio_engine_eih", "simplify_expr", {"expression": "(-4 - 2*(-1) + 2*(-4))/6"})

# (3) harmonic-gauge static Schwarzschild: harmonic radius r = R - m (R areal, m = GM/c^2) gives
#     g00 = -(r-m)/(r+m), g_rr = (r+m)/(r-m).  A particle at rest has coordinate acceleration
#     d^2r/dt^2 = -c^2 Gamma^r_00 = (c^2/2) g^rr d_r g00.  In units of GM/r^2 = c^2 m/r^2:
dg00 = get("harm_dg00_dr", "differentiate_expr", {"expression": "-(r-m)/(r+m)", "variable": "r"})
get("harm_static_accel_over_GM_r2", "simplify_expr",
    {"expression": f"(1/2)*((r-m)/(r+m))*({dg00})*r**2/m"})
get("harm_static_accel_in_x_check", "simplify_expr",
    {"expression": "(-(m/x)**2*((m/x)-m)/((m/x)+m)**3) + (1-x)/(1+x)**3"})
get("harm_static_accel_series", "series_expansion",
    {"expression": "-(1-x)/(1+x)**3", "variable": "x", "point": "0", "order": 3})

# (4) literature bounds on beta, as allowed extra advance in units of GR: delta(omega)/omega_GR = (2+2g-b)/3 - 1
lit("will2014_beta_bound", 8e-5,
    "[LIT] Will 2014, Living Rev. Relativ. 17, 4 (arXiv:1403.7377), Table 4: beta-1 < 8e-5 (perihelion shift, "
    "J2 = (2.2 +/- 0.1)e-7); text after eq. 65: beta-1 = (-4.1 +/- 7.8)e-5 (MESSENGER fits, Cassini gamma adopted). "
    "Read from the arXiv PDF in the v2 repair.")
lit("will2014_beta_central", -4.1e-5, "[LIT] Will 2014 §4.2, beta-1 = (-4.1 +/- 7.8)e-5")
lit("will2014_beta_sigma", 7.8e-5, "[LIT] Will 2014 §4.2, beta-1 = (-4.1 +/- 7.8)e-5")
lit("will2006_2g_b_1_bound", 3e-3,
    "[LIT] Will 2006, Living Rev. Relativ. 9, 3 (gr-qc/0510072), text after eq. 50: excess perihelion shift known to "
    "about 0.1 percent from radar observations of Mercury 1966-1990, giving |2 gamma - beta - 1| < 3e-3. "
    "Read from the arXiv PDF in the v2 repair.")
lit("park2017_beta_central", -2.7e-5,
    "[LIT] Park, Folkner, Konopliv, Williams, Smith, Zuber 2017, AJ 153, 121, doi:10.3847/1538-3881/aa5be2: "
    "beta-1 = (-2.7 +/- 3.9)e-5 (verified by the v1 runner via MIT DSpace and by two skeptics; the v2 repair could not "
    "re-fetch it: publisher bot-check, ADS 405, web-search budget exhausted)")
lit("park2017_beta_sigma", 3.9e-5, "[LIT] Park+2017, as above")
lit("will_GR_arcsec_cy", 42.98, "[LIT] Will 2014 eq. 65 / Will 2006 eq. 50: 42.98 arcsec/cy for gamma = beta = 1")
lit("design_budget_arcsec_cy", 0.05,
    "design's own conservative budget (U5 design entry); ~0.1 % of 42.98 arcsec/cy, i.e. the Will 2006 radar level")

get("will2014_allowed_arcsec_cy", "simplify_expr", {"expression": "42.98*8e-5/3"})
get("park2017_1sigma_arcsec_cy", "simplify_expr", {"expression": "42.98*3.9e-5/3"})
get("will2006_allowed_arcsec_cy", "simplify_expr", {"expression": "42.98*3e-3/3"})
get("design_budget_fraction_of_GR", "simplify_expr", {"expression": "0.05/42.98"})

json.dump(L, open(os.path.join(out, "leaves_v2.json"), "w"), indent=1)
print(json.dumps({k: v["value"] for k, v in L.items()}, indent=1, default=str))
