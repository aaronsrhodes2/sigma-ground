"""U7 — N-body leg: does a moved source reach test masses at all distances in one force evaluation?

Repo engine sigma_ground.field.interface.nbody (unmodified). Source = Sun mass (Mentat solar_system_body),
5 test masses (1 kg) at d = 1, 2, 4, 8, 16 au (Mentat lookup_constant au) on the +x axis, all at rest.
The source is displaced by delta = 1e-3 au (longitudinal, then transverse) at t = 0.
  (1) compute_accelerations before/after: dA(d) at every d in the first evaluation; slope of ln|dA| vs ln d = -3.
  (2) integrate both systems (moved / unmoved) with NBodySystem.step at dt and dt/2: arrival = first step whose
      test-mass position differs by more than 1e-3 of the step-1 difference... i.e. first nonzero response;
      t_arr = n_first * dt for all d -> p = 0 and t_arr scales with dt (no propagation speed).
  (3) the same with PhysicsToggles(gr_1pn=True): the engine's 1PN layer is also instantaneous.
"""
import json, math, sys
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
sun, sun_src = mentat("solar_system_body", {"body_name": "Sun"})
au, au_src = mentat("lookup_constant", {"name": "au"})
G, G_src = mentat("lookup_constant", {"name": "G"})
c, c_src = mentat("lookup_constant", {"name": "c"})
M = sun["mass_kg"]
ds = np.array([1, 2, 4, 8, 16], float)
delta = 1e-3 * au

def bodies(src_offset):
    b = [CelestialBody(mass_kg=M, position_m=np.array(src_offset, float), velocity_m_s=np.zeros(3),
                       radius_m=sun["radius_m"], love_number_k2=0.0)]
    for d in ds:
        b.append(CelestialBody(mass_kg=1.0, position_m=np.array([d * au, 0.0, 0.0]), velocity_m_s=np.zeros(3),
                               radius_m=1.0, love_number_k2=0.0))
    return b

def ols(x, y):
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, np.log(y), rcond=None)[0][0])

res = {}
for tog_name, tog in (("newton", PhysicsToggles()), ("gr_1pn", PhysicsToggles(gr_1pn=True))):
    out = {}
    for mode, off in (("longitudinal", [delta, 0, 0]), ("transverse", [0, delta, 0])):
        a0 = NBodySystem(bodies([0, 0, 0]), toggles=tog).compute_accelerations()[1:]
        a1 = NBodySystem(bodies(off), toggles=tog).compute_accelerations()[1:]
        dA = np.linalg.norm(a1 - a0, axis=1)
        # closed form (Newtonian) for the control [DERIVED]: longitudinal GM[1/(d-δ)^2 - 1/d^2]; transverse |GM r/|r|^3 diff|
        if mode == "longitudinal":
            expect = G * M * (1 / (ds * au - delta) ** 2 - 1 / (ds * au) ** 2)
        else:
            r1 = np.stack([ds * au, -delta * np.ones_like(ds), 0 * ds], 1)
            r0 = np.stack([ds * au, 0 * ds, 0 * ds], 1)
            expect = np.linalg.norm(-G * M * (r1 / np.linalg.norm(r1, axis=1)[:, None] ** 3
                                             - r0 / np.linalg.norm(r0, axis=1)[:, None] ** 3), axis=1)
        out[mode] = dict(dA=dA.tolist(), slope=ols(ds, dA), expect=expect.tolist(),
                         ratio_to_closed_form=(dA / expect).tolist())
    # arrival by integration: velocity response of each test mass, moved vs unmoved source, for several dt
    arr = {}
    dA_long = np.array(out["longitudinal"]["dA"])
    for dt in (86400.0, 43200.0, 600.0, 60.0):
        sA = NBodySystem(bodies([0, 0, 0]), toggles=tog); sB = NBodySystem(bodies([delta, 0, 0]), toggles=tog)
        first = {}
        for n in range(1, 4):
            sA.step(dt); sB.step(dt)
            for k, d in enumerate(ds):
                dv = np.linalg.norm(sB.bodies[k + 1].velocity_m_s - sA.bodies[k + 1].velocity_m_s)
                if int(d) not in first and dv > 0:
                    # fraction of the full static field change delivered in this first response: dv / (dA dt n)
                    first[int(d)] = dict(step=n, t_s=n * dt, dv_m_s=float(dv), fraction_of_static=float(dv / (dA_long[k] * dt * n)))
        ts = np.array([first[int(d)]["t_s"] for d in ds])
        arr[f"{dt:g}s"] = dict(first=first, p=ols(ds, ts), t_arr_s=ts.tolist())
    out["arrival"] = arr
    res[tog_name] = out

summary = dict(p_nbody=res["newton"]["arrival"]["86400s"]["p"],
               t_arr_by_dt={k: v["t_arr_s"] for k, v in res["newton"]["arrival"].items()},
               slope_long=res["newton"]["longitudinal"]["slope"], slope_trans=res["newton"]["transverse"]["slope"],
               slope_long_1pn=res["gr_1pn"]["longitudinal"]["slope"], p_1pn=res["gr_1pn"]["arrival"]["86400s"]["p"], p_by_dt={k: v["p"] for k, v in res["newton"]["arrival"].items()},
               fraction_of_static_60s={d: v["fraction_of_static"] for d, v in res["newton"]["arrival"]["60s"]["first"].items()},
               light_travel_time_16au_s=16 * au / c)
json.dump(dict(mentat=dict(M_sun=dict(value=M, source=sun_src), au=dict(value=au, source=au_src),
                           G=dict(value=G, source=G_src), c=dict(value=c, source=c_src)),
               distances_au=ds.tolist(), delta_m=delta, results=res, summary=summary),
          open(f"{OUT}/nbody_results.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(json.dumps({k: res["newton"][k]["ratio_to_closed_form"] for k in ("longitudinal", "transverse")}, indent=1))
