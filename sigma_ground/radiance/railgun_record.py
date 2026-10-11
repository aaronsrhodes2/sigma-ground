"""Railgun recorder -- the millisecond shot as a frame bundle a player replays.

Sibling of ``thermal_record.record_fall_thermal`` and ``trajectory.record_*``:
it runs the SAME solver the ``rail_shot`` Materia verb runs
(``materia.railgun.simulate_rail_shot``), cuts the integrated history into
frames, and REFUSES to emit a bundle that disagrees with the verb's scalar
answer or with itself.

FRAME CONTRACT (additive to trajectory.py; every new key is optional, an old
player ignores them):

    {"t_sim": t,                              # seconds of simulated time
     "bodies": [{"pos": [x,y,z], "quat": [x,y,z,w],
                 "temperature_k": T,          # armature bulk temperature
                 "rgb": [r,g,b],              # thermal_emission_rgb(T), 0..1,
                                              # (0,0,0) below the Draper point
                 "emit": e}],                 # 0..1 glow intensity
     "events":   [{"kind": ..., ...}]?,       # discrete moments on this frame
     "readouts": {"I_A": .., "V_cap_V": .., "v_m_s": .., "F_N": .., "T_K": ..}}

``trajectory.readout_spec`` gives each readout's label, unit and source once
(not per frame).  ``emit`` is the Stefan-Boltzmann ratio (T / T_melt)^4 while
the body is above the Draper point and 0 below it: a relative brightness cue
with the melting point as 1.0, NOT a calibrated radiance (the thermal_emission
docstring says absolute intensity scales as T^4 and is left to the renderer).

Honest limits: frames are the solver's own states, picked evenly by step
index (no reconstruction; the player lerps between them); the armature is drawn as a box
whose cross-section is derived from its mass (render-only); everything the
verb does not model (skin effect, arcing, erosion, ...) is absent here too.
"""
from __future__ import annotations

import math

import numpy as np

from .scene_export import _bake_material

_TOL_SCALAR = 0.02      # frames vs the verb's scalar answer (as thermal_record)
_TOL_FRAME = 0.01       # frames vs their own integrals (x = int v dt, dQ = int I dt)

READOUT_SPEC = {
    "I_A": {"label": "Rail current", "unit": "A",
            "source": "materia.railgun state I (RLC with back-EMF), sg_solve_ivp rk4"},
    "V_cap_V": {"label": "Bank voltage", "unit": "V",
                "source": "materia.railgun state V_c; C dV/dt = -I"},
    "v_m_s": {"label": "Armature speed", "unit": "m/s",
              "source": "materia.railgun state v; m dv/dt = L' I^2 / 2"},
    "F_N": {"label": "Armature force", "unit": "N",
            "source": "F = L' I^2 / 2 (checked vs electromagnetic_force_analysis)"},
    "T_K": {"label": "Armature temperature", "unit": "K",
            "source": "materia.railgun: m c_p(T) dT = I^2 R_arm(T) dt, library "
                      "resistivity(T) and specific heat"},
}

_PARAM_KEYS = ("cap_F", "volts", "rail_len_m", "rail_radius_m", "rail_spacing_m",
               "armature_material", "armature_mass_kg", "rail_material",
               "circuit_R_ohm", "circuit_L0_H", "L_per_m", "T0_K", "tolerance")


def _glow(material_key, T):
    """rgb from the library's thermal emission, or [0,0,0] below the Draper point."""
    from ..field.interface import thermal_emission as te
    if not te.is_visibly_glowing(T):
        return [0.0, 0.0, 0.0]
    key = material_key if material_key in te.THERMAL_EMISSION_MATERIALS else "blackbody"
    return [float(c) for c in te.thermal_emission_rgb(key, T)]


def record_rail_shot(*, n_frames: int = 240, target_watch_s: float = 15.0,
                     expected: dict | None = None, **params) -> dict:
    """Run the shot and record it.  ``params`` are the ``rail_shot`` slots.

    ``expected`` is the verb's scalar outputs (``rail_shot(**params).outputs``);
    when omitted the recorder asks the verb itself.  Raises ValueError on any
    self-check failure, on a failed or stalled shot, or if the frames disagree
    with the scalar answer by more than 2 % (or with their own integrals by
    more than 1 %): a refusal, never a quiet render.
    """
    from ..materia import railgun as RG
    from ..materia.scenarios import rail_shot as verb

    bad = [k for k in params if k not in _PARAM_KEYS]
    if bad:
        raise TypeError(f"unknown rail_shot parameter(s): {bad}")
    sim = RG.simulate_rail_shot(**params)
    val = RG.check_rail_shot(sim)
    if not val["passed"]:
        raise ValueError("rail shot failed its self-checks, refusing to record: "
                         + ", ".join(val["failed_checks"]))
    sc, mo = sim["scalars"], sim["model"]

    if expected is None:
        res = verb(**params)
        if not res.validation.get("passed"):
            raise ValueError("rail_shot verb failed its self-checks, refusing to "
                             "record: " + ", ".join(res.validation["failed_checks"]))
        expected = res.outputs
    exp_v = expected.get("muzzle_speed_m_s", expected.get("speed_at_failure_m_s"))
    pairs = [("end speed", sc["speed_end_m_s"], exp_v),
             ("peak current", sc["peak_current_A"], expected.get("peak_current_A")),
             ("armature dT", sc["T_end_K"] - mo["T0_K"], expected.get("delta_T_K")),
             ("shot time", sc["t_end_s"], expected.get("shot_time_s"))]
    scalar_residual = {}
    for name, got, want in pairs:
        if want is None:
            raise ValueError(f"expected scalar answer lacks {name!r}: the recorder "
                             f"refuses to render without the verb's number")
        r = abs(got - want) / max(abs(want), 1e-300)
        scalar_residual[name] = r
        if r > _TOL_SCALAR:
            raise ValueError(
                f"scalar cross-check failed: recorder {name}={got:.6g} vs verb "
                f"{want:.6g} ({r * 100:.1f}% > {_TOL_SCALAR * 100:g}%) -- refusing "
                f"to render physics that disagrees with the cited answer")

    # ---- frames: uniform in sim time, linear between the solver's own steps ----
    t_arr = sim["t"]
    t_end = float(t_arr[-1])
    # frames ARE the solver's own states, picked evenly by step index (a coasting
    # tail, if any, becomes one exact constant-velocity interval)
    idx = np.unique(np.linspace(0, len(t_arr) - 1, int(n_frames)).astype(int))
    times = t_arr[idx]

    def at(arr):
        return arr[idx]

    X, I, Vc, V, F, T = (at(sim[k]) for k in ("x", "I", "Vc", "v", "F", "T"))
    y0 = mo["rail_radius_m"]
    mat = mo["armature_material"]
    frames = []
    T_melt = mo["T_melt_K"]
    for k, tk in enumerate(times):
        rgb = _glow(mat, float(T[k]))
        glowing = any(c > 0.0 for c in rgb)
        emit = min(1.0, (float(T[k]) / T_melt) ** 4) if glowing else 0.0
        frames.append({
            "t_sim": float(tk),
            "bodies": [{"pos": [float(X[k]), y0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0],
                        "temperature_k": float(T[k]), "rgb": rgb, "emit": emit}],
            "readouts": {"I_A": float(I[k]), "V_cap_V": float(Vc[k]),
                         "v_m_s": float(V[k]), "F_N": float(F[k]),
                         "T_K": float(T[k])}})

    # ---- events (discrete moments, attached to the frame they fall in) ---------
    ev = [(0.0, {"kind": "discharge_start", "volts": mo["volts"]})]
    k_i = int(np.argmax(np.abs(sim["I"])))
    ev.append((float(t_arr[k_i]), {"kind": "peak_current",
                                    "I_A": float(abs(sim["I"][k_i]))}))
    neg = np.nonzero(sim["Vc"] < 0.0)[0]
    if neg.size:
        ev.append((float(t_arr[neg[0]]), {"kind": "bank_reversal",
                                           "note": "no crowbar modelled"}))
    hot = np.nonzero(sim["T"] >= 700.0)[0]       # thermal_emission Draper point
    if hot.size:
        ev.append((float(t_arr[hot[0]]), {"kind": "visible_glow", "T_K": 700.0}))
    ev.append((t_end, {"kind": "muzzle_exit" if sc["exited_rails"] else "armature_melt",
                       "speed_m_s": sc["speed_end_m_s"], "x_m": sc["x_end_m"],
                       "T_K": sc["T_end_K"]}))
    traj_events = []
    for te_, e in ev:
        j = int(np.argmin(np.abs(times - te_)))
        frames[j].setdefault("events", []).append(e)
        traj_events.append(dict(e, t_sim=te_))

    # ---- frames checked against their own integrals (independent of the scalars)
    x_int = float(np.sum(0.5 * (V[1:] + V[:-1]) * np.diff(times)))
    q_int = float(np.sum(0.5 * (I[1:] + I[:-1]) * np.diff(times)))
    q_bank = mo["cap_F"] * (mo["volts"] - float(Vc[-1]))
    frame_checks = {
        "x_equals_integral_of_v": abs(x_int - float(X[-1])) / max(float(X[-1]), 1e-300),
        "charge_equals_integral_of_I": abs(q_int - q_bank) / max(abs(q_bank), 1e-300),
    }
    for name, r in frame_checks.items():
        if r > _TOL_FRAME:
            raise ValueError(f"frame cross-check failed: {name} off by "
                             f"{r * 100:.2f}% (> {_TOL_FRAME * 100:g}%) -- the "
                             f"recorded frames do not obey the solver's own "
                             f"kinematics/charge balance; refusing to render")
    if sc["exited_rails"] and abs(float(X[-1]) - mo["rail_len_m"]) > 1e-9 * mo["rail_len_m"]:
        raise ValueError("last frame is not at the muzzle")

    # ---- scene: two rail boxes + an armature box (render-only geometry) --------
    rho_d = mo["armature_density_kg_m3"]
    gap = mo["gap_m"]
    vol = mo["armature_mass_kg"] / rho_d
    side = math.sqrt(vol / gap)                       # x and y extent, z = gap
    L = mo["rail_len_m"]
    r = mo["rail_radius_m"]
    zc = 0.5 * mo["rail_spacing_m"]
    rail_leaf = lambda z: {"op": "add", "material": mo["rail_material"],
                           "shape": {"type": "Box", "center": [L / 2.0, y0, z],
                                     "x": L + side, "y": 2 * r, "z": 2 * r}}
    scene = {
        "name": (f"railgun - {mo['armature_mass_kg'] * 1e3:.3g} g {mat} armature, "
                 f"{L:g} m of {mo['rail_material']} rails, "
                 f"{mo['cap_F'] * 1e3:.3g} mF at {mo['volts']:g} V"),
        "bodies": [{"pivot": [0, 0, 0], "label": "armature"}],
        "csg_leaves": [
            rail_leaf(zc), rail_leaf(-zc),
            {"op": "add", "material": mat, "body": 0,
             "temperature_k": mo["T0_K"],            # the pre-play still; frames override
             "shape": {"type": "Box", "center": [0, 0, 0],
                       "x": side, "y": side, "z": gap}}],
        "materials": {mat: _bake_material(mat, rho_d),
                      mo["rail_material"]: _bake_material(mo["rail_material"])},
        "physics": {"mass_kg": mo["armature_mass_kg"], "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-L * 0.1, L * 1.1], [0.0, 4 * r + side], [-zc - r, zc + r]],
        "camera": {"target": [L / 2.0, y0, 0.0], "orbit_radius": max(0.8 * L, 6 * zc),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0], "az0": 0.6, "el0": 0.25},
        "identified": True,
        "source": "railgun shot (materia.railgun): RLC + back-EMF + F = L' I^2/2 "
                  "+ Joule heating, library resistivity(T)/c_p/melting point",
    }
    validation = {
        "self_checks": {k: v["passed"] for k, v in val["checks"].items()},
        "scalar_residual": {k: round(v, 8) for k, v in scalar_residual.items()},
        "frame_checks": {k: round(v, 8) for k, v in frame_checks.items()},
        "ledger_worst_residual_over_E0": sc["ledger_rel"],
        "reason": sc["reason"], "not_modelled": list(RG.NOT_MODELLED),
        "L_prime_source": mo["L_prime_source"],
    }
    return {
        "scene": scene,
        "trajectory": {
            "frames": frames, "t_end_s": t_end, "natural_timescale_s": t_end,
            "suggested_rate": max(1e-6, t_end / target_watch_s),
            "body_labels": ["armature"], "events": traj_events,
            "readout_spec": READOUT_SPEC, "validation": validation,
        },
        "kind": "trajectory", "thermal": True, "railgun": True,
    }
