"""U9 repair (v2) — why the GR formula with Mentat leaves gives 42.997"/cy and not the literature 42.98"/cy.

Flag raised by two reviewers: Mentat solar_system_body('sun').mass_kg = 1.98892e30 is labelled 'IAU 2015 nominal',
but IAU 2015 B3 defines only the nominal mass PARAMETER (GM)_sun^N = 1.3271244e20 m^3 s^-2 [LIT, arXiv:1510.07674,
fetched and read this session] and says SI masses must be (GM)/G with the G stated.  Hypothesis: 1.98892e30 =
(GM)_sun^N / 6.67259e-11 (an obsolete G), so with Mentat's G = 6.6743e-11 the product G*M_sun is too high.
Expectation (written before running): G_implied ~ 6.6726e-11; G*M_sun/(GM)^N - 1 ~ +2.6e-4; GR advance with the IAU
GM ~ 42.98"/cy.  This is a Mentat data-label issue, not an engine-physics defect; it cannot change any U9 verdict,
because every engine precession is reported as a ratio to u = pi G M/(c^2 p) computed from the same leaves.
All arithmetic is done by Mentat simplify_expr (sympy N()).
Usage: python -X utf8 u9_anchor_v2.py OUT.json
"""
import json, sys
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.field.interface.rolling_analysis import _ORBITAL

OUT = sys.argv[1]
SRC = {}


def M(tool, args, key):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


G = M("lookup_constant", {"name": "G"}, "G")
C = M("lookup_constant", {"name": "c"}, "c")
AU = M("lookup_constant", {"name": "au"}, "au")
sun = M("solar_system_body", {"body_name": "sun"}, "sun")
mer = M("solar_system_body", {"body_name": "mercury"}, "mercury")
E = _ORBITAL["Mercury"][1]
SRC["e_mercury"] = {"value": E, "source": "sigma_ground.field.interface.rolling_analysis._ORBITAL['Mercury'] (JPL fact sheet)"}
GM_IAU = "1.3271244e20"     # [LIT] IAU 2015 Resolution B3 nominal (GM)_sun^N, arXiv:1510.07674 (exact by definition)
Ms, Mm = sun["mass_kg"], mer["mass_kg"]
a = f"({mer['semimajor_axis_au']}*{AU})"
P_fs = f"({mer['orbital_period_d']}*86400)"
res = {}
res["G_implied_by_Msun_and_IAU_GM"] = M("simplify_expr", {"expression": f"N({GM_IAU}/{Ms}, 8)"}, "G_implied")
res["GMsun_mentat_over_IAU_minus_1"] = M("simplify_expr", {"expression": f"N({G}*{Ms}/{GM_IAU} - 1, 6)"}, "GM_ratio")


def adv(gm, period):
    expr = f"N(6*pi*{gm}/({C}**2*{a}*(1-{E}**2)) * (180*3600/pi) * (36525*86400)/{period}, 10)"
    return expr


gm_m = f"({G}*({Ms}+{Mm}))"
gm_i = f"({GM_IAU}+{G}*{Mm})"
kep = lambda gm: f"(2*pi*sqrt({a}**3/{gm}))"
res["adv_mentatGM_keplerP"] = M("simplify_expr", {"expression": adv(gm_m, kep(gm_m))}, "adv_mentatGM_keplerP")
res["adv_mentatGM_factsheetP"] = M("simplify_expr", {"expression": adv(gm_m, P_fs)}, "adv_mentatGM_factsheetP")
res["adv_IAUGM_keplerP"] = M("simplify_expr", {"expression": adv(gm_i, kep(gm_i))}, "adv_IAUGM_keplerP")
res["adv_IAUGM_factsheetP"] = M("simplify_expr", {"expression": adv(gm_i, P_fs)}, "adv_IAUGM_factsheetP")
res["kepler_P_days_mentatGM"] = M("simplify_expr", {"expression": f"N({kep(gm_m)}/86400, 8)"}, "P_kep_mentat")
res["kepler_P_days_IAUGM"] = M("simplify_expr", {"expression": f"N({kep(gm_i)}/86400, 8)"}, "P_kep_IAU")
res["LIT_Will2014"] = 42.98
for k, v in res.items():
    print(f"{k:34s} {v}")
json.dump({"results": res, "sources": SRC,
           "LIT": {"(GM)_sun^N = 1.3271244e20 m^3 s^-2": "IAU 2015 Resolution B3, Mamajek et al., arXiv:1510.07674 "
                   "(PDF fetched 2026-10-10, copy in lit_v2/); the resolution says SI masses = (GM)/G with G stated, "
                   "e.g. CODATA 2014 G = 6.67408e-11",
                   "42.98 arcsec/cy": "C. M. Will, Living Rev. Relativ. 17, 4 (2014), eq. (65), arXiv:1403.7377"}},
          open(OUT, "w", encoding="utf-8"), indent=1)
