"""U1 follow-up diagnostic (run after the pre-registered engine check fired its "> 5 % = nbody bug" branch).

Question: after the exponent defect (r**4 for r**5) is removed, can the engine's per-step velocity-rescale damping
represent PSR J1738+0333's GW loss in float64 at all?  The counterfactual r**5 copy in u1_kill_test.py measured
8.9x Peters instead of 1x, which suggests the per-step loss is below double-precision resolution.

Everything here is scratch: the engine is NOT patched. Subclasses live in this file only and exist to show what an
engine change would have to do. Leaves via Mentat (tools/mentat_client.py), [LIT] as in u1_kill_test.py.
"""
import json, math, os, sys
import numpy as np

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO)
from mentat_client import mentat_raw  # noqa: E402
import sigma_ground.field.interface.nbody as nb  # noqa: E402
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles  # noqa: E402

LOG = []


def M(tool, args, label):
    r = mentat_raw(tool, args)
    if not r.get("ok"):
        raise RuntimeError(r)
    LOG.append(dict(label=label, tool=tool, args=args, value=r["result"].get("value"), source=r["result"].get("source")))
    return r["result"]["value"]


def calc(expr, label):
    return float(str(M("simplify_expr", {"expression": f"N(({expr}), 17)"}, label)))


G = float(M("lookup_constant", {"name": "G"}, "G")); C = float(M("lookup_constant", {"name": "c"}, "c"))
MS = float(M("lookup_constant", {"name": "M_sun"}, "M_sun"))
DAY = float(M("convert_units", {"value": 1, "from_units": "day", "to_units": "s"}, "day"))
Pb = calc(f"0.3547907398724*{DAY}", "Pb [s] (Freire+2012 [LIT])")
m1 = calc(f"1.46*{MS}", "m_p"); m2 = calc(f"0.181*{MS}", "m_c"); Mt = calc(f"{m1}+{m2}", "M")
a = calc(f"({G}*{Mt}*{Pb}**2/(4*pi**2))**(1/3)", "a")
v = calc(f"sqrt({G}*{Mt}/{a})", "v_rel")
peters = calc(f"-(192*pi/5)*(2*pi*{G}/({C}**3*{Pb}))**(5/3)*{m1}*{m2}*{Mt}**(-1/3)", "Pbdot Peters")
L_true = calc(f"(32/5)*{G}**4*{m1}**2*{m2}**2*{Mt}/({C}**5*{a}**5)", "L_GW Peters at a [W]")
mu = calc(f"{m1}*{m2}/{Mt}", "reduced mass")
KE_rel = calc(f"{mu}*{v}**2/2", "KE_rel [J]")

# 1. per-step fractional loss at the pre-registered dt and its float64 fate
out = dict(sources=LOG)
rows = []
for spo in (2000, 200, 20, 1):
    dt = Pb / spo
    eps = calc(f"{L_true}*{dt}/{KE_rel}", f"eps = L dt / KE_rel at Pb/{spo}")
    damp = 1.0 + (-eps)                       # exactly the engine's expression: damp = 1.0 + dE_dt*dt/ke_rel
    dv_needed = v * (1 - math.sqrt(1 - eps)) if eps > 1e-12 else v * eps / 2
    rows.append(dict(steps_per_orbit=spo, dt_s=dt, eps_per_step=eps, damp_float64=damp, damp_is_exactly_1=(damp == 1.0),
                     realised_eps=1.0 - damp, realised_over_needed=(1.0 - damp) / eps,
                     dv_needed_m_s=dv_needed, ulp_of_v_m_s=math.ulp(v), dv_over_ulp=dv_needed / math.ulp(v)))
out["float64_resolution"] = rows
out["machine"] = dict(ulp_below_1=1.0 - np.nextafter(1.0, 0.0), half_ulp=(1.0 - np.nextafter(1.0, 0.0)) / 2)


# 2. counterfactual engines (scratch subclasses)
class _Base(NBodySystem):
    exponent = 4
    mode = "engine"          # "engine": damp each step as coded; "unity": force damp = 1 (pure recomputation);
                             # "batched": accumulate the loss, apply when it exceeds 2e-14 of KE_rel

    def __init__(self, *a_, **k_):
        super().__init__(*a_, **k_)
        self._acc = 0.0

    def _apply_gw_damping(self, vel_new, dt):
        b0, b1 = self.bodies[0], self.bodies[1]
        r = float(np.linalg.norm(b1.position_m - b0.position_m))
        mu_red = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg)
        M_tot = b0.mass_kg + b1.mass_kg
        dE_dt = -(32.0 / 5.0) * nb._G ** 4 * b0.mass_kg ** 2 * b1.mass_kg ** 2 * M_tot / (nb._c ** 5 * r ** self.exponent)
        v_rel = vel_new[1] - vel_new[0]
        ke_rel = 0.5 * mu_red * float(np.dot(v_rel, v_rel))
        if self.mode == "unity":
            damp = 1.0
        elif self.mode == "batched":
            self._acc += dE_dt * dt
            if abs(self._acc) < 2e-14 * ke_rel:
                return vel_new                        # untouched: no recomputation round-off either
            damp = 1.0 + self._acc / ke_rel
            self._acc = 0.0
        else:
            damp = max(0.0, 1.0 + dE_dt * dt / ke_rel)
        v_cm = (b0.mass_kg * vel_new[0] + b1.mass_kg * vel_new[1]) / M_tot
        vel_new[0] = v_cm - math.sqrt(damp) * b1.mass_kg / M_tot * v_rel
        vel_new[1] = v_cm + math.sqrt(damp) * b0.mass_kg / M_tot * v_rel
        return vel_new


def make(exponent, mode):
    return type(f"NB_r{exponent}_{mode}", (_Base,), dict(exponent=exponent, mode=mode))


def bodies():
    return [CelestialBody(m1, np.array([-m2 / Mt * a, 0.0, 0.0]), np.array([0.0, -m2 / Mt * v, 0.0]), 1.2e4, 0.0),
            CelestialBody(m2, np.array([m1 / Mt * a, 0.0, 0.0]), np.array([0.0, m1 / Mt * v, 0.0]), 1.0e7, 0.0)]


def energy_series(cls, gw, spo, n_orb):
    s = cls(bodies(), toggles=PhysicsToggles(gw_damping=gw))
    dt = Pb / spo; n = spo * n_orb
    t = np.empty(n + 1); E = np.empty(n + 1); ph = np.empty(n + 1)

    def phase():
        r = s.bodies[1].position_m - s.bodies[0].position_m
        return math.atan2(r[1], r[0])
    t[0] = 0; E[0] = s.total_energy(); ph[0] = phase()
    for k in range(1, n + 1):
        s.step(dt); t[k] = s.time; E[k] = s.total_energy(); ph[k] = phase()
    return t, E, np.unwrap(ph)


def pbdot(cls, spo=2000, n_orb=20):
    t, Eoff, phoff = energy_series(NBodySystem, False, spo, n_orb)
    _, Eon, phon = energy_series(cls, True, spo, n_orb)
    Edot = np.polyfit(t, Eon - Eoff, 1)[0]
    pb = Pb * (-1.5 / Eoff[0]) * Edot
    dph = phon - phoff
    A = np.vstack([t ** 2, t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, dph, rcond=None)
    pb_tim = -coef[0] * Pb ** 2 / math.pi
    # round-off noise floor of the energy estimator: rms of (E_on - E_off) about its linear fit, in units of one orbit's loss
    lin = np.polyval(np.polyfit(t, Eon - Eoff, 1), t)
    noise = float(np.sqrt(np.mean((Eon - Eoff - lin) ** 2)))
    return dict(Pbdot=pb, over_peters=pb / peters, Pbdot_timing=pb_tim, timing_over_peters=pb_tim / peters,
                energy_resid_rms_J=noise, t=t[::100].tolist(), dE_over_E=((Eon - Eoff) / abs(Eoff[0]))[::100].tolist(),
                dphi=dph[::100].tolist())


cases = {
    "engine_as_coded_r4": NBodySystem,
    "r5_per_step_as_coded": make(5, "engine"),
    "r5_damp_forced_1_recompute_only": make(5, "unity"),
    "r5_batched_accumulator": make(5, "batched"),
}
res = {k: pbdot(c) for k, c in cases.items()}
res["r5_batched_accumulator_100orb"] = pbdot(make(5, "batched"), 2000, 100)
for k in res:
    res[k]["over_peters_mentat"] = calc(f"({res[k]['Pbdot']})/({peters})", f"{k}: Pbdot/Peters")
    res[k]["timing_over_peters_mentat"] = calc(f"({res[k]['Pbdot_timing']})/({peters})", f"{k}: timing Pbdot/Peters")
out["peters"] = peters
out["one_orbit_loss_J"] = calc(f"{L_true}*{Pb}", "Peters energy loss per orbit [J]")
out["cases"] = res
json.dump(out, open(os.path.join(OUT, "u1_gw_precision_diag.json"), "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(dict(rows=rows, cases={k: (v["over_peters_mentat"], v["timing_over_peters_mentat"], v["energy_resid_rms_J"]) for k, v in res.items()}), indent=1, default=str))
