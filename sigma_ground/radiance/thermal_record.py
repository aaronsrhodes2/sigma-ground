"""Thermal recorders — Materia heat physics → watchable, temperature-carrying
bundles.

``record_fall_thermal`` is ``trajectory.record_fall``'s thermal sibling: the
same drag-integrated sphere fall, but every recorded frame's body also carries
its BODY TEMPERATURE — the running drag-work integral translated through
ΔT = f·Q/(m·c_p), the exact physics ``materia.scenarios.drag_heating_drop``
cites as a scalar, cut into moments. The viewer feeds that per-frame T to the
Planck × Kirchhoff incandescence, so the ball glows where (and WHEN) the
physics says it glows.

Cross-validation across the seam: the caller (the front door's
``sphere_thermal`` handle) passes the scenario's own force-integral ΔT as
``expected_delta_T_K``; this recorder RE-integrates the fall independently and
refuses to emit a bundle whose final ΔT disagrees by more than 2% — Materia's
two-independent-measures discipline, extended to the render.

Frame contract (see trajectory.py): bodies gain an optional ``temperature_k``.
Precedence in the renderers: per-cell field > frame body T > leaf static T >
physics_env.temperature_k > 293.15.

Honest limits (flagged): drag work is the ONLY heat source — contact/bounce
dissipation is recorded in the energy ledger but NOT converted to body heat;
no radiative/convective cooling during the fall (both second-order over these
timescales at these temperatures); the body is isothermal (the per-cell
windward gradient is the Phase-4 field upgrade).
"""
from __future__ import annotations

from .scene_export import _bake_material, _suggest_camera  # noqa: F401  (camera helper kept for parity with trajectory.py)


def _windward_deposit_field(material_key, radius_m, frames, *, mcp_inv, T0,
                            T_ambient, n=24, n_keyframes=8, sub_dt=2.0):
    """The flagship's per-cell field: each sub-interval's drag dissipation
    ΔQ = f·Δq_drag is DEPOSITED on the windward surface cells (Newtonian cosθ
    weighting — the same law as materia's windward_heating_field) and Fourier-
    conducted inward by ``diffuse_fvm`` with the passing air as the ambient
    bath. Energy-conserving by construction: total deposited == f·q_drag_total
    (exact bookkeeping), unlike a held-temperature boundary.

    HONEST FLAGS: f rides in via mcp_inv (the same adiabatic-upper-bound flag
    as the bulk ΔT); ALL of each interval's dissipation lands on the leading
    face (a second ceiling — real flows heat the whole boundary layer); the
    deposit cadence (sub_dt) is a discretization choice, fine enough that
    conduction spreads each parcel ~a cell before the next lands.

    Returns (fspec, field_samples, deposited_J).
    """
    import numpy as np
    from ..materia.thermal_field import material_grids
    from ..dynamics.fields.heat import diffuse_fvm
    from .scene_export import field_spec_from_grid, field_samples
    from .trajectory import poses_at

    R = float(radius_m)
    half = R * 1.18
    dx = 2.0 * half / n
    axis = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    rr = np.sqrt(X * X + Y * Y + Z * Z)
    label = (rr <= R).astype(np.int8)
    solid = label == 1
    k, rho_cp, _ = material_grids(label, ["air", material_key], T_ref=T_ambient)
    # ADIABATIC void (no bath): the bulk model this field must agree with is
    # f=1 "all drag heat enters and STAYS in the body" — so the field carries
    # the same energy, spatially resolved (face hot early, equilibrating by
    # impact), and its mean tracks the frames' bulk T. Adding an air bath here
    # would silently contradict the cited scalar. k_void=0 already insulates
    # every face via the harmonic mean.
    bc = {}
    # windward (falling → −y) surface cells, cosθ-weighted from the stagnation point
    from scipy import ndimage as ndi
    surface = solid & ~ndi.binary_erosion(solid)
    safe = np.where(rr > 0, rr, 1.0)
    cos = -(Y / safe)                                # v̂ = −y → leading face is −y
    w = np.where(surface & (cos > 0.0), cos, 0.0)
    w_sum = float(w.sum()) or 1.0
    cell_cap = rho_cp * dx ** 3                      # J/K per cell

    traj = {"frames": frames}
    t_end = frames[-1]["t_sim"]

    def bulk_T(t):
        return poses_at(traj, t)[0].get("temperature_k", T0)

    def q_at(t):                                     # invert T(t)=T0+q·mcp_inv
        return (bulk_T(t) - T0) / mcp_inv if mcp_inv > 0 else 0.0

    key_times = [t_end * i / (n_keyframes - 1) for i in range(n_keyframes)]
    T_grid = np.full(label.shape, float(T_ambient))
    T_grid[~solid] = T_ambient
    kf_grids = [T_grid.astype(np.float32).copy()]
    deposited = 0.0
    t = 0.0
    for kt in key_times[1:]:
        while t < kt - 1e-9:
            t1 = min(t + sub_dt, kt)
            dQ = max(0.0, q_at(t1) - q_at(t))        # f is already inside mcp_inv
            if dQ > 0.0:
                dT = dQ * w / (w_sum * np.where(solid, cell_cap, 1.0))
                T_grid = T_grid + np.where(solid, dT, 0.0)
                deposited += dQ
            T_grid, _ = diffuse_fvm(T_grid, k, rho_cp, dx,
                                    total_time=t1 - t, **bc)
            t = t1
        kf_grids.append(T_grid.astype(np.float32).copy())
    fspec = field_spec_from_grid(
        kf_grids, key_times, voxel_size=dx, outside_value=T_ambient,
        source=("per-interval drag dissipation deposited on the windward face "
                "(Newtonian cosθ) + Fourier conduction (diffuse_fvm) + ambient "
                "air bath — same f flag as the bulk ΔT; all-on-face is a "
                "second ceiling"))
    return fspec, field_samples(0, fspec), deposited


def record_fall_thermal(material_key: str = "iron", radius_m: float = 0.05,
                        start_altitude_m: float = 30_000.0, *,
                        body_fraction: float = 1.0, T0: float = 288.15,
                        expected_delta_T_K: float | None = None,
                        T: float = 288.15, dt_max: float = 0.02,
                        t_max: float = 600.0, frame_dt: float = 0.2,
                        windward_field: bool = False,
                        target_watch_s: float = 15.0) -> dict:
    """Drop a sphere through the atmosphere, recording pose AND temperature.

    The motion loop is ``trajectory.record_fall``'s (drag integration, emergent
    Johnson–Thornton rebound, settle); the thermal channel is the trapezoidal
    ∫|F_drag|·v dt accumulator from ``materia.engine.simulate_fall``, applied
    per frame as T(t) = T0 + f·q_drag(t)/(m·c_p).
    """
    from ..materia.engine import (make_atmospheric_drag, _material_density,
                                  _DensityMaterial, _air_viscosity)
    from ..materia.labs.forces import drag_force
    from ..field.interface.atmosphere import density_at_altitude
    from ..field.interface.thermal import specific_heat_j_kg_K
    from ..shapes import Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step

    density, mat_name = _material_density(material_key, T)
    shape = Sphere(radius_m)
    parcel = PhysicsParcel(shape, _DensityMaterial(density),
                           position=Vec3(0.0, max(start_altitude_m, radius_m), 0.0),
                           velocity=Vec3(0.0, 0.0, 0.0), label=material_key)
    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -9.80665, 0.0), ground=False)
    cb = make_atmospheric_drag(scene.gravity, T)
    mu = _air_viscosity(T)
    c_p = specific_heat_j_kg_K(material_key, T0)
    mcp_inv = body_fraction / (parcel.mass * c_p)

    q_drag, prev_power = 0.0, 0.0                 # trapezoidal drag-work integral

    def _body():
        p = parcel.position
        return {"pos": [p.x, p.y, p.z], "quat": [0.0, 0.0, 0.0, 1.0],
                "temperature_k": round(T0 + q_drag * mcp_inv, 3)}

    try:
        from ..field.interface.impact import coefficient_of_restitution as _cor
    except Exception:
        _cor = None

    frames = [{"t_sim": 0.0, "bodies": [_body()]}]
    next_rec = frame_dt
    contacts = []
    apex_m = parcel.position.y
    while scene.time < t_max:
        actual_dt = step(scene, dt=dt_max, sub_steps=4, external_forces=cb)
        spd = parcel.velocity.length()
        rho = density_at_altitude(max(0.0, parcel.position.y), T)
        power = drag_force(parcel, rho, mu).length() * spd
        q_drag += 0.5 * (prev_power + power) * actual_dt
        prev_power = power
        apex_m = max(apex_m, parcel.position.y)
        if parcel.position.y <= radius_m and parcel.velocity.y < 0.0:
            v_in = abs(parcel.velocity.y)          # floor contact: rebound
            e = 0.5
            if _cor is not None:
                try:
                    e = float(_cor(material_key, velocity=max(v_in, 1e-3),
                                   radius_m=radius_m))
                except Exception:
                    pass
            e = min(max(e, 0.02), 0.98)
            contacts.append({"t": round(scene.time, 4),
                             "v_impact": round(v_in, 4), "e": round(e, 4),
                             "E_before": round(0.5 * parcel.mass * v_in ** 2, 6),
                             "E_after": round(0.5 * parcel.mass * (e * v_in) ** 2, 6)})
            parcel.position = Vec3(parcel.position.x, radius_m, parcel.position.z)
            parcel.velocity = Vec3(parcel.velocity.x, e * v_in, parcel.velocity.z)
            if e * v_in < 0.04:                    # settled
                parcel.velocity = Vec3(0.0, 0.0, 0.0)
                break
        if scene.time >= next_rec:
            frames.append({"t_sim": scene.time, "bodies": [_body()]})
            next_rec += frame_dt
    frames.append({"t_sim": scene.time, "bodies": [_body()]})
    t_end = scene.time

    # ── the cross-check: refuse a render whose thermodynamics drifted ──
    delta_T_final = q_drag * mcp_inv
    thermal_residual = None
    if expected_delta_T_K is not None and expected_delta_T_K > 0:
        thermal_residual = abs(delta_T_final - expected_delta_T_K) / expected_delta_T_K
        if thermal_residual > 0.02:
            raise ValueError(
                f"thermal cross-check failed: recorder ΔT={delta_T_final:.1f} K vs "
                f"scenario ΔT={expected_delta_T_K:.1f} K "
                f"({thermal_residual * 100:.1f}% > 2%) — refusing to render "
                f"physics that disagrees with the cited answer")

    r = radius_m
    top = max(start_altitude_m, apex_m) + 2 * r
    wide = max(4 * r, 0.5)
    follow = top > 40 * r
    bbox = ([[-4 * r, 4 * r], [-4 * r, 4 * r], [-4 * r, 4 * r]] if follow
            else [[-wide, wide], [-0.1, top], [-wide, wide]])
    scene_spec = {
        "name": f"{mat_name.lower()} sphere — drag heating "
                f"(f={body_fraction:g} adiabatic upper bound)",
        "bodies": [{"pivot": [0, 0, 0], "label": material_key}],
        "csg_leaves": [{"op": "add", "material": material_key, "body": 0,
                        "temperature_k": T0,       # the pre-play still; frames override
                        "shape": {"type": "Sphere", "center": [0, 0, 0],
                                  "radius": radius_m}}],
        "materials": {material_key: _bake_material(material_key, density)},
        "physics": {"mass_kg": parcel.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": bbox,
        "camera": {"target": ([0.0, start_altitude_m, 0.0] if follow else
                              [0.0, min(top, max(2 * r, top * 0.5)), 0.0]),
                   "orbit_radius": 8 * r if follow else max(6 * r, 0.4 * top),
                   "fov_deg": 45.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.4, "el0": 0.15,
                   **({"follow": True} if follow else {})},
        "identified": True,
        "source": ("falling sphere (Materia drag integration) + body temperature "
                   "from the drag-work integral, ΔT = f·Q/(m·c_p)"),
    }

    validation = {
        "contacts": contacts,
        "apex_m": round(apex_m, 3),
        "energy_monotone": all(c["E_after"] < c["E_before"]
                               for c in contacts),
        "delta_T_final_K": round(delta_T_final, 3),
        "expected_delta_T_K": (round(expected_delta_T_K, 3)
                               if expected_delta_T_K is not None else None),
        "thermal_residual": (round(thermal_residual, 6)
                             if thermal_residual is not None else None),
        "body_fraction_flag": (f"f={body_fraction:g} — fraction of drag "
                               f"heat entering the body; f=1 is the "
                               f"adiabatic upper bound"),
    }
    if windward_field:
        # the flagship per-cell channel: the SAME f·q_drag energy, spatially
        # resolved — deposited on the leading face, conducted inward
        fspec, fsamples, deposited = _windward_deposit_field(
            material_key, radius_m, frames, mcp_inv=mcp_inv, T0=T0, T_ambient=T)
        scene_spec["csg_leaves"][0]["fields"] = {"temperature_k": fspec}
        scene_spec["field_samples"] = fsamples
        validation["windward_deposited_J"] = round(deposited, 1)
        validation["windward_model"] = (
            "per-interval drag dissipation → windward cosθ deposit → Fourier "
            "conduction; deposited == f·q_drag to bookkeeping exactness")

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": max(1e-6, t_end / target_watch_s),
            "body_labels": [material_key],
            "validation": validation,
        },
        "kind": "trajectory",
        "thermal": True,
    }


def record_thermal_field(handle: dict) -> dict:
    """A `conduction_field` render-handle → a watchable field bundle.

    The physics is ALREADY FROZEN in the handle (keyframe T grids from
    ``materia.thermal_field.evolve_contact_field``); this recorder samples and
    repackages — it computes nothing. The scene stages the two solids in the
    theater matching the environment preset (IRT → room, ISM → deep_space), so
    the physics_env that bounded the solver is the physics_env the render
    carries — the loop the preset table exists to close. In deep_space the
    scene has NO lights: the hot cube is visible only by its own Planck glow
    and fades as it cools past the Draper point — the honest picture.
    """
    from .scene_export import (_bake_material, field_spec_from_grid,
                               field_samples)
    from .theaters import stage

    hot, cold = handle["hot_material"], handle["cold_material"]
    cube, slab = handle["cube_box"], handle["slab_box"]
    env_name = handle.get("environment", "IRT")
    ambient = handle.get("ambient_T", 293.15)
    leaves = [
        {"op": "add", "material": hot, "temperature_k": handle.get("T_hot", 900.0),
         "shape": {"type": "Box", "center": cube["center"],
                   "x": cube["x"], "y": cube["y"], "z": cube["z"]}},
        {"op": "add", "material": cold, "temperature_k": ambient,
         "shape": {"type": "Box", "center": slab["center"],
                   "x": slab["x"], "y": slab["y"], "z": slab["z"]}},
    ]
    fspec = field_spec_from_grid(
        handle["T_frames"], handle["times_s"],
        voxel_size=handle["dx"], center=handle["grid_center"],
        outside_value=ambient,
        source=("materia.contact_conduction — diffuse_fvm keyframes "
                f"(harmonic-mean interface flux), environment {env_name}"))
    for leaf in leaves:                        # one shared grid covers both solids
        leaf["fields"] = {"temperature_k": fspec}
    hx = slab["x"] * 0.5
    top = cube["center"][2] + cube["z"] * 0.5
    content = {
        "name": handle.get("label", "contact conduction"),
        "csg_leaves": leaves,
        "materials": {hot: _bake_material(hot), cold: _bake_material(cold)},
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-hx, hx], [-hx, hx], [0.0, top]],
        "identified": True,
        "source": ("contact conduction — per-cell temperature field frozen by "
                   "Materia; Radiance displays it (never integrates)"),
    }
    scene = stage(content, "deep_space" if env_name.upper() == "ISM" else "room")
    scene["field_samples"] = (field_samples(0, fspec) + field_samples(1, fspec))
    scene["kind"] = "trajectory"
    times = handle["times_s"]
    frames = [{"t_sim": round(float(t), 4), "bodies": []} for t in times]
    t_end = float(times[-1])
    return {"scene": scene,
            "trajectory": {"frames": frames, "t_end_s": t_end,
                           "natural_timescale_s": t_end,
                           "suggested_rate": max(1e-6, t_end / 12.0),
                           "body_labels": [],
                           "validation": {
                               "keyframes": len(times),
                               "environment": env_name,
                               "note": "static solids; playback time drives the "
                                       "field keyframe mix (uFldU)"}},
            "kind": "trajectory", "thermal": True}


__all__ = ["record_fall_thermal", "record_thermal_field"]
