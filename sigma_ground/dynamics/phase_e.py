"""Phase E4 — the fluid-vs-gravity comparison runner.

One seed (dynamics.seeding), two engines, one report:

  n-body side : field.interface.nbody — pure pairwise gravity, symplectic.
  fluid side  : dynamics.continuum with self_gravity_G and the isothermal
                velocity-dispersion EOS — gravity + pressure + viscosity.

The comparison is at the level of ENSEMBLE STATISTICS, per the Phase E
doctrine: pointwise trajectory subtraction between two chaotic N-body
systems is meaningless past a few dynamical times (exponential
decorrelation), so the residual is measured on statistical structure —
energy budget, radial distribution, velocity dispersion, clustering —
where the physics difference (pressure + viscosity vs. none) lives.

The residual IS the experiment: where the two descriptions diverge is
exactly where the fluid closure stops describing a collisional swarm, and
that divergence point (shell-crossing, kernel-contact clustering) is real
physics, not error. Every configured number rides in the report as a
disclosed Choice; nothing is normalized away silently.
"""
from __future__ import annotations

import math

from .continuum import continuum_step
from .seeding import G_CODATA, to_continuum, to_nbody


# ── engine-neutral state extraction ─────────────────────────────────────────

def _nbody_state(system):
    ms, ps, vs = [], [], []
    for b in system.bodies:
        ms.append(float(b.mass_kg))
        ps.append((float(b.position_m[0]), float(b.position_m[1]),
                   float(b.position_m[2])))
        vs.append((float(b.velocity_m_s[0]), float(b.velocity_m_s[1]),
                   float(b.velocity_m_s[2])))
    return ms, ps, vs


def _sph_state(scene):
    ms, ps, vs = [], [], []
    for p in scene.parcels:
        ms.append(float(p.mass))
        ps.append((p.position.x, p.position.y, p.position.z))
        vs.append((p.velocity.x, p.velocity.y, p.velocity.z))
    return ms, ps, vs


# ── ensemble diagnostics (identical math for both engines) ──────────────────

def diagnostics(masses, positions, velocities, G):
    """Statistical snapshot of one engine's state.

    ke_j / pe_j / e_total_j : the energy budget (PE from the same pairwise
        Newtonian sum both engines gravitate by).
    r_* : radial distribution about the barycenter — min / median / max.
    speed_std_m_s : velocity dispersion (std of |v|) — the fluid side's
        c_s should track this if the closure is honest.
    nn_mean_m : mean nearest-neighbor distance — the clustering /
        self-gravity-wake statistic; drops as clumps form.
    """
    n = len(masses)
    total_m = sum(masses)
    cx = sum(m * p[0] for m, p in zip(masses, positions)) / total_m
    cy = sum(m * p[1] for m, p in zip(masses, positions)) / total_m
    cz = sum(m * p[2] for m, p in zip(masses, positions)) / total_m

    ke = sum(0.5 * m * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
             for m, v in zip(masses, velocities))
    pe = 0.0
    nn = [math.inf] * n
    for i in range(n):
        xi, yi, zi = positions[i]
        for j in range(i + 1, n):
            dx = xi - positions[j][0]
            dy = yi - positions[j][1]
            dz = zi - positions[j][2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r > 0.0:
                pe -= G * masses[i] * masses[j] / r
            nn[i] = min(nn[i], r)
            nn[j] = min(nn[j], r)

    radii = sorted(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                             + (p[2] - cz) ** 2) for p in positions)
    speeds = [math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
              for v in velocities]
    mean_s = sum(speeds) / n
    var_s = sum((s - mean_s) ** 2 for s in speeds) / n
    finite_nn = [d for d in nn if math.isfinite(d)]

    return {
        "ke_j": ke,
        "pe_j": pe,
        "e_total_j": ke + pe,
        "r_min_m": radii[0],
        "r_med_m": radii[len(radii) // 2],
        "r_max_m": radii[-1],
        "speed_std_m_s": math.sqrt(var_s),
        "nn_mean_m": (sum(finite_nn) / len(finite_nn)) if finite_nn else 0.0,
    }


def _residual(a, b):
    """b relative to a, per diagnostic — fractional where the scale is
    meaningful, absolute where it is not. NOT error: physics difference."""
    out = {}
    for k in a:
        av, bv = a[k], b[k]
        out["d_" + k] = bv - av
        if abs(av) > 1e-300:
            out["dfrac_" + k] = (bv - av) / abs(av)
    return out


# ── the runner ──────────────────────────────────────────────────────────────

def run_comparison(bodies, *, dt, n_steps, sample_every=10,
                   material_key, sound_speed_m_s,
                   softening_m, G=G_CODATA, label=""):
    """Run the SAME seed through both engines, sampling ensemble
    diagnostics as they go. Returns the report dict (JSON-ready).

    Choices disclosed in the report: dt, n_steps, softening, c_s (the
    dispersion closure), material_key (sets SPH packing radii — masses are
    identical either way). Facts: G (CODATA), the seed bodies' masses.
    """
    sysnb = to_nbody(bodies, softening_m=softening_m)
    scene = to_continuum(bodies, material_key=material_key, G=G,
                         gravity_softening_m=softening_m,
                         eos="isothermal", sound_speed_m_s=sound_speed_m_s)

    samples = []

    def sample(t):
        d_nb = diagnostics(*_nbody_state(sysnb), G)
        d_sp = diagnostics(*_sph_state(scene), G)
        samples.append({"t_s": t, "nbody": d_nb, "sph": d_sp,
                        "residual": _residual(d_nb, d_sp)})

    sample(0.0)
    for k in range(1, n_steps + 1):
        sysnb.step(dt)
        continuum_step(scene, dt)
        if k % sample_every == 0 or k == n_steps:
            sample(k * dt)

    n = len(bodies)
    return {
        "kind": "phase_e_comparison",
        "label": label,
        "engines": {
            "nbody": "field.interface.nbody (velocity-Verlet, pure gravity)",
            "sph": ("dynamics.continuum (self-gravity + isothermal EOS "
                    "+ Monaghan viscosity)"),
        },
        "choices": [
            {"description": "timestep", "value": dt, "unit": "s"},
            {"description": "steps", "value": n_steps},
            {"description": "Plummer softening (both engines)",
             "value": softening_m, "unit": "m"},
            {"description": ("isothermal sound speed = velocity-dispersion "
                             "closure (Goldreich-Tremaine)"),
             "value": sound_speed_m_s, "unit": "m/s"},
            {"description": "SPH packing material", "value": material_key},
        ],
        "facts": [
            {"description": "G (CODATA 2018)", "value": G,
             "unit": "m^3 kg^-1 s^-2"},
            {"description": "bodies", "value": n},
            {"description": "total mass", "unit": "kg",
             "value": sum(b.mass_kg for b in bodies)},
        ],
        "notes": ("Residuals compare ENSEMBLE statistics, never pointwise "
                  "trajectories (chaotic decorrelation makes those "
                  "meaningless). The residual is the physics: pressure + "
                  "viscosity vs. pure gravity."),
        "samples": samples,
    }


__all__ = ["diagnostics", "run_comparison"]
