"""U1 follow-up: where does float64 round-off take over the engine's GW-damping path?

Scratch counterfactual (engine NOT patched): the engine's own velocity-rescale damping with Peters' r**5 and the
loss multiplied by a gain g. Measured Pbdot/(g * Peters) vs g, per-step (as coded) and batched (accumulate, apply
when > 2e-14 of KE_rel). If the mechanism is exact and only resolution limits it, the ratio -> 1 as g grows.
The as-coded r**4 engine is the point g = a/1 m = 1.73e9.
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


class Gain(NBodySystem):
    gain = 1.0
    batched = False

    def __init__(self, *a_, **k_):
        super().__init__(*a_, **k_)
        self._acc = 0.0

    def _apply_gw_damping(self, vel_new, dt):
        b0, b1 = self.bodies[0], self.bodies[1]
        r = float(np.linalg.norm(b1.position_m - b0.position_m))
        mu_red = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg)
        M_tot = b0.mass_kg + b1.mass_kg
        dE_dt = -self.gain * (32.0 / 5.0) * nb._G ** 4 * b0.mass_kg ** 2 * b1.mass_kg ** 2 * M_tot / (nb._c ** 5 * r ** 5)
        v_rel = vel_new[1] - vel_new[0]
        ke_rel = 0.5 * mu_red * float(np.dot(v_rel, v_rel))
        if self.batched:
            self._acc += dE_dt * dt
            if abs(self._acc) < 2e-14 * ke_rel:
                return vel_new
            damp = 1.0 + self._acc / ke_rel
            self._acc = 0.0
        else:
            damp = max(0.0, 1.0 + dE_dt * dt / ke_rel)
        v_cm = (b0.mass_kg * vel_new[0] + b1.mass_kg * vel_new[1]) / M_tot
        vel_new[0] = v_cm - math.sqrt(damp) * b1.mass_kg / M_tot * v_rel
        vel_new[1] = v_cm + math.sqrt(damp) * b0.mass_kg / M_tot * v_rel
        return vel_new


def bodies():
    return [CelestialBody(m1, np.array([-m2 / Mt * a, 0.0, 0.0]), np.array([0.0, -m2 / Mt * v, 0.0]), 1.2e4, 0.0),
            CelestialBody(m2, np.array([m1 / Mt * a, 0.0, 0.0]), np.array([0.0, m1 / Mt * v, 0.0]), 1.0e7, 0.0)]


SPO, NORB = 2000, 20


def series(sysobj):
    dt = Pb / SPO; n = SPO * NORB
    t = np.empty(n + 1); E = np.empty(n + 1); ph = np.empty(n + 1)

    def phase():
        r = sysobj.bodies[1].position_m - sysobj.bodies[0].position_m
        return math.atan2(r[1], r[0])
    t[0] = 0; E[0] = sysobj.total_energy(); ph[0] = phase()
    for k in range(1, n + 1):
        sysobj.step(dt); t[k] = sysobj.time; E[k] = sysobj.total_energy(); ph[k] = phase()
    return t, E, np.unwrap(ph)


t, Eoff, phoff = series(NBodySystem(bodies(), toggles=PhysicsToggles(gw_damping=False)))
out = dict(peters=peters, steps_per_orbit=SPO, n_orbits=NORB, rows=[], sources=LOG)
for batched in (False, True):
    for g in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5):
        cls = type("G", (Gain,), dict(gain=g, batched=batched))
        _, Eon, phon = series(cls(bodies(), toggles=PhysicsToggles(gw_damping=True)))
        Edot = np.polyfit(t, Eon - Eoff, 1)[0]
        pb_e = Pb * (-1.5 / Eoff[0]) * Edot
        A = np.vstack([t ** 2, t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, phon - phoff, rcond=None)
        pb_t = -coef[0] * Pb ** 2 / math.pi
        row = dict(batched=batched, gain=g, energy_ratio=pb_e / (g * peters), timing_ratio=pb_t / (g * peters))
        out["rows"].append(row)
        print(row, flush=True)
json.dump(out, open(os.path.join(OUT, "u1_gain_scan.json"), "w", encoding="utf-8"), indent=1, default=str)
