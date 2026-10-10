"""U1 v2 — evidence for engine defect 2 (per-step multiplicative GW damping drops sub-half-ulp losses; the CM
recompute then adds a systematic spurious decay).  Repair requested by the could-it-fail skeptic.

(a) The shipped repo test test_physics_stability.TestGWDampingOnPlanetaryOrbits
    .test_gw_damping_does_not_add_measurable_drift_beyond_integrator steps Earth-Sun with include_gw_loss=True at
    dt = 3600 s.  Claim: even with the engine's inflated r^-4 law, eps = |dE/dt| dt / KE_rel = 3.99e-17 < 2^-54, so
    damp == 1.0 at every step and the toggle is a pure velocity recompute.  Check: (i) Mentat eps; (ii) the test's own
    system, stepped for the test's N_STEPS with the ENGINE's _apply_gw_damping (unmodified; an instance-level recorder
    only reads the same expression before delegating to the engine method); (iii) the final state equals, bit for
    bit, a scratch run in which damp is forced to 1.
(b) J1738+0333 recompute-only (damp forced to 1, scratch subclass) apparent Pbdot / Peters at five initial orbital
    phases and two step sizes (10 orbits) -> is the spurious decay systematic and does it scale with step count?

Nothing in the repo is modified.  Leaves via Mentat (tools/mentat_client.py).  [LIT] J1738: Freire et al. 2012.
"""
import json, math, os, sys, time
import numpy as np

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO)
from mentat_client import mentat_raw  # noqa: E402
import sigma_ground.field.interface.nbody as nb  # noqa: E402
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles  # noqa: E402
from sigma_ground.field.interface import test_physics_stability as TPS  # noqa: E402  (import only; no test is run)

LOG = []


def M(tool, args, label):
    r = mentat_raw(tool, args)
    if not r.get("ok"):
        raise RuntimeError(r)
    LOG.append(dict(label=label, tool=tool, args=args, value=r["result"].get("value"), source=r["result"].get("source")))
    return r["result"]["value"]


def calc(expr, label):
    return float(str(M("simplify_expr", {"expression": f"N(({expr}), 17)"}, label)))


t0 = time.time()
G = float(M("lookup_constant", {"name": "G"}, "G")); C = float(M("lookup_constant", {"name": "c"}, "c"))
MS = float(M("lookup_constant", {"name": "M_sun"}, "M_sun")); AU = float(M("lookup_constant", {"name": "au"}, "au"))
earth = M("solar_system_body", {"body_name": "earth"}, "earth record")
HALF_ULP = (1.0 - float(np.nextafter(1.0, 0.0))) / 2

# ── (a) the shipped Earth-Sun GW test ─────────────────────────────────────────
DT = TPS.TestGWDampingOnPlanetaryOrbits.DT_S
N = TPS.TestGWDampingOnPlanetaryOrbits.N_STEPS
eps_mentat = calc(f"(32/5)*{G}**4*{MS}**2*{earth['mass_kg']}**2*({MS}+{earth['mass_kg']})/({C}**5*{AU}**4)*{DT}"
                  f"/(0.5*{MS}*{earth['mass_kg']}/({MS}+{earth['mass_kg']})*{G}*{MS}/{AU})",
                  "Earth-Sun eps per 1 h step with the engine's r^-4 law (Mentat masses, v = sqrt(GM/AU))")


def attach_recorder(s, rec):
    """Instance-level wrapper: evaluates the engine's own damp expression (read-only) and delegates to the
    unmodified engine method. The engine file is not touched."""
    orig = s._apply_gw_damping

    def wrapped(vel_new, dt):
        b0, b1 = s.bodies[0], s.bodies[1]
        r = float(np.linalg.norm(b1.position_m - b0.position_m))
        mu = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg); Mt = b0.mass_kg + b1.mass_kg
        dE_dt = -(32.0 / 5.0) * nb._G ** 4 * b0.mass_kg ** 2 * b1.mass_kg ** 2 * Mt / (nb._c ** 5 * r ** 4)
        ke = 0.5 * mu * float(np.linalg.norm(vel_new[1] - vel_new[0])) ** 2
        damp = max(0.0, 1.0 + dE_dt * dt / ke)
        rec["n"] += 1; rec["n_not_1"] += int(damp != 1.0)
        rec["eps_min"] = min(rec["eps_min"], -dE_dt * dt / ke); rec["eps_max"] = max(rec["eps_max"], -dE_dt * dt / ke)
        return orig(vel_new, dt)
    s._apply_gw_damping = wrapped


class UnityES(NBodySystem):
    """scratch: damp forced to 1 (pure CM recompute, same float operations as the engine otherwise)"""
    def _apply_gw_damping(self, vel_new, dt):
        b0, b1 = self.bodies[0], self.bodies[1]; Mt = b0.mass_kg + b1.mass_kg
        v_rel = vel_new[1] - vel_new[0]
        v_cm = (b0.mass_kg * vel_new[0] + b1.mass_kg * vel_new[1]) / Mt
        vel_new[0] = v_cm - math.sqrt(1.0) * b1.mass_kg / Mt * v_rel
        vel_new[1] = v_cm + math.sqrt(1.0) * b0.mass_kg / Mt * v_rel
        return vel_new


rec = dict(n=0, n_not_1=0, eps_min=1.0, eps_max=0.0)
s_plain = TPS._earth_sun_system()
s_gw = TPS._earth_sun_system(); attach_recorder(s_gw, rec)
b = TPS._earth_sun_system().bodies
s_unity = UnityES([CelestialBody(x.mass_kg, x.position_m.copy(), x.velocity_m_s.copy(), x.radius_m, x.love_number_k2) for x in b],
                  include_gr=False)
for _ in range(N):
    s_plain.step(DT, include_gw_loss=False)
    s_gw.step(DT, include_gw_loss=True)
    s_unity.step(DT, include_gw_loss=True)
r_plain = float(np.linalg.norm(s_plain.bodies[1].position_m)); r_gw = float(np.linalg.norm(s_gw.bodies[1].position_m))
bit_identical = all(np.array_equal(a_.position_m, u.position_m) and np.array_equal(a_.velocity_m_s, u.velocity_m_s)
                    for a_, u in zip(s_gw.bodies, s_unity.bodies))
part_a = dict(test="test_physics_stability.TestGWDampingOnPlanetaryOrbits.test_gw_damping_does_not_add_measurable_drift_beyond_integrator",
              dt_s=DT, n_steps=N, eps_mentat=eps_mentat, half_ulp_below_1=HALF_ULP, eps_below_half_ulp=eps_mentat < HALF_ULP,
              recorder=rec, damp_equals_1_at_every_step=(rec["n_not_1"] == 0 and rec["n"] == N),
              gw_run_bit_identical_to_damp_forced_1=bit_identical,
              gw_additional_drift_km=abs(r_gw - r_plain) / 1000.0,
              reading="the shipped test's GW toggle never applies any GW loss; its 'GW additional drift' is CM-recompute round-off")
print(json.dumps(part_a, indent=1), flush=True)

# ── (b) J1738 recompute-only bias vs phase and step count ─────────────────────
DAY = float(M("convert_units", {"value": 1, "from_units": "day", "to_units": "s"}, "day"))
Pb = calc(f"0.3547907398724*{DAY}", "Pb [s] (Freire+2012 [LIT])")
m1 = calc(f"1.46*{MS}", "m_p"); m2 = calc(f"0.181*{MS}", "m_c"); Mt = calc(f"{m1}+{m2}", "M")
a = calc(f"({G}*{Mt}*{Pb}**2/(4*pi**2))**(1/3)", "a"); v = calc(f"sqrt({G}*{Mt}/{a})", "v_rel")
peters = calc(f"-(192*pi/5)*(2*pi*{G}/({C}**3*{Pb}))**(5/3)*{m1}*{m2}*{Mt}**(-1/3)", "Pbdot Peters")
dlnP = M("differentiate_expr", {"expression": "log(2*pi*sqrt((-G*m1*m2/(2*En))**3/(G*M)))", "variable": "En"}, "dlnPb/dE")
KFAC = calc(f"({dlnP})*En", "Kepler factor E dlnPb/dE (Mentat)")


def bodies(th):
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    return [CelestialBody(m1, R @ np.array([-m2 / Mt * a, 0, 0]), R @ np.array([0, -m2 / Mt * v, 0]), 1.2e4, 0.0),
            CelestialBody(m2, R @ np.array([m1 / Mt * a, 0, 0]), R @ np.array([0, m1 / Mt * v, 0]), 1.0e7, 0.0)]


def series(cls, gw, th, spo, n_orb):
    s = cls(bodies(th), toggles=PhysicsToggles(gw_damping=gw)); dt = Pb / spo; n = spo * n_orb
    t = np.empty(n + 1); E = np.empty(n + 1); t[0] = 0; E[0] = s.total_energy()
    for k in range(1, n + 1):
        s.step(dt); t[k] = s.time; E[k] = s.total_energy()
    return t, E


rows = []
for spo in (2000, 1000):
    for th in (0.0, 0.7, 1.9, 3.1, 4.4):
        t, Eoff = series(NBodySystem, False, th, spo, 10)
        _, Eon = series(UnityES, True, th, spo, 10)
        Edot_fit = np.polyfit(t, Eon - Eoff, 1)[0]
        Edot_end = (Eon[-1] - Eon[0] - (Eoff[-1] - Eoff[0])) / t[-1]
        rows.append(dict(steps_per_orbit=spo, theta_rad=th, n_orbits=10,
                         fit_over_peters=Pb * (KFAC / Eoff[0]) * Edot_fit / peters,
                         endpoint_over_peters=Pb * (KFAC / Eoff[0]) * Edot_end / peters))
        print(rows[-1], flush=True)
summ = {}
for spo in (2000, 1000):
    f = [r["fit_over_peters"] for r in rows if r["steps_per_orbit"] == spo]
    summ[str(spo)] = dict(min=min(f), max=max(f), mean=float(np.mean(f)), all_positive_decay=all(x > 0 for x in f))
summ["ratio_mean_2000_over_1000"] = calc(f"{summ['2000']['mean']}/{summ['1000']['mean']}", "mean bias Pb/2000 over Pb/1000")
out = dict(id="U1-v2 engine defect 2", earth_sun_shipped_test=part_a,
           j1738_recompute_only=dict(rows=rows, summary=summ, Pbdot_peters=peters, kepler_factor=KFAC,
                                     note="damp forced to 1 (scratch subclass): any apparent decay is CM-recompute round-off"),
           sources=LOG, runtime_s=time.time() - t0)
json.dump(out, open(os.path.join(OUT, "u1_engine_defect2_v2.json"), "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(summ, indent=1), "runtime", time.time() - t0)
