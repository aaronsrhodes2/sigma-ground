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
        "csg_leaves": [{"op": "add", "material": material_key,
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
