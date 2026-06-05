"""Trajectory recording — Materia motion → a serializable per-frame time series.

The Materia verbs compute a trajectory and keep only the scalar answer. The
viewer needs the *frames*. Rather than edit the (in-flux) materia engine, we
reuse its validated pieces — `make_atmospheric_drag`, `_material_density`, the
density-wrapper — and the dynamics stepper, recording (t, position) along the
way. Output is the contract bundle: a one-body SceneSpec + a Trajectory of
poses, with an auto-suggested playback rate from the phenomenon's own timescale.

Quaternions are identity for now (a sphere has no visible orientation); rotation
arrives with Materia's rigid-body stage and slots straight into `quat`.
"""
from __future__ import annotations

from .scene_export import _bake_material, _suggest_camera


def record_fall(material_key: str = "copper", radius_m: float = 0.05,
                start_altitude_m: float = 10_000.0, T: float = 288.15,
                dt_max: float = 0.02, t_max: float = 600.0,
                frame_dt: float = 0.2, target_watch_s: float = 15.0) -> dict:
    """Drop a sphere through the atmosphere, recording a pose every `frame_dt`."""
    from ..materia.engine import (make_atmospheric_drag, _material_density,
                                  _DensityMaterial)
    from ..shapes import Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step

    density, mat_name = _material_density(material_key, T)
    shape = Sphere(radius_m)
    parcel = PhysicsParcel(shape, _DensityMaterial(density),
                           position=Vec3(0.0, start_altitude_m, 0.0),
                           velocity=Vec3(0.0, 0.0, 0.0), label=material_key)
    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -9.80665, 0.0), ground=False)
    cb = make_atmospheric_drag(scene.gravity, T)

    def _pose():
        p = parcel.position
        return {"pos": [p.x, p.y, p.z], "quat": [0.0, 0.0, 0.0, 1.0]}

    frames = [{"t_sim": 0.0, "bodies": [_pose()]}]
    next_rec = frame_dt
    while parcel.position.y > 0.0 and scene.time < t_max:
        step(scene, dt=dt_max, sub_steps=4, external_forces=cb)
        if scene.time >= next_rec:
            frames.append({"t_sim": scene.time, "bodies": [_pose()]})
            next_rec += frame_dt
    frames.append({"t_sim": scene.time, "bodies": [_pose()]})
    t_end = scene.time

    r = radius_m
    bbox = ((-r, r), (-r, r), (-r, r))
    scene_spec = {
        "name": f"{mat_name.lower()} sphere",
        # body 0 = the sphere; pivot is its rest center, so the frame `pos`
        # tracks the centre directly (a sphere needs no rotation, but the
        # plumbing carries `quat` for the day Materia's rigid stage rotates it).
        "bodies": [{"pivot": [0, 0, 0], "label": material_key}],
        "csg_leaves": [{"op": "add", "material": material_key, "body": 0,
                        "shape": {"type": "Sphere", "center": [0, 0, 0],
                                  "radius": radius_m}}],
        "materials": {material_key: _bake_material(material_key, density)},
        "physics": {"mass_kg": parcel.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [list(b) for b in bbox],
        "camera": _suggest_camera(bbox),
        "identified": True,
        "source": "falling sphere (Materia drag integration)",
    }
    # sim-seconds per wall-second: short events → slow-mo (<1), long falls →
    # gentle time-lapse (>1) so the whole thing plays in ~target_watch_s.
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": [material_key],
        },
    }


def bounce_heights(material_key, radius_m=0.07, drop_height_m=0.45, floor_y=0.0,
                   dt=0.02, t_total=3.0, g=9.80665):
    """Centre-height of a dropped sphere per frame — a KINEMATIC drop+bounce whose
    rebound is the material's EMERGENT clatter.

    Each impact's rebound velocity = v_impact x coefficient_of_restitution(material,
    v_impact, r) — the velocity-dependent Hertz/Johnson COR from field.interface.
    impact (consumed, not re-derived). A bouncer (rubber, COR~1) climbs back to
    near its drop height again and again; a thudder (lead/copper, COR~0.1) dies on
    the first contact. The envelope is set by what the material IS.
    """
    from ..field.interface.impact import coefficient_of_restitution
    rest_y = floor_y + radius_m
    y, vy, ys = drop_height_m + rest_y, 0.0, []
    for _ in range(int(t_total / dt) + 1):
        ys.append(round(y, 5))
        vy -= g * dt
        y += vy * dt
        if y <= rest_y and vy < 0.0:                  # ground contact
            v_impact = max(abs(vy), 1e-3)
            try:
                e = coefficient_of_restitution(material_key, velocity=v_impact,
                                               radius_m=radius_m)
            except Exception:
                e = 0.5
            y, vy = rest_y, e * v_impact               # emergent rebound
            if vy < 0.04:
                vy = 0.0                               # settle
    return ys
