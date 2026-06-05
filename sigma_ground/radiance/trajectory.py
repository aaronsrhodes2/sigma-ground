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


def _lay_flat_quat(dx, dy, dz):
    """A CONSTANT orientation laying the object's thinnest axis vertical (+y) —
    the rest pose of a flat plate falling broadside, so the broad face we weigh
    for drag is the face we render horizontal. NOT flutter: it never varies in
    time (real tumbling is chaotic rigid-body aerodynamics Materia doesn't model).
    """
    s = 2 ** -0.5                                  # sin/cos of 45°
    thin = min((dx, "x"), (dy, "y"), (dz, "z"))[1]
    if thin == "x":
        return [0.0, 0.0, s, s]                    # +x → +y  (90° about z)
    if thin == "z":
        return [-s, 0.0, 0.0, s]                   # +z → +y  (−90° about x)
    return [0.0, 0.0, 0.0, 1.0]                    # thinnest already vertical


def record_object_fall(construct, start_altitude_m: float = 2.4384, *,
                       cd: float = 1.0, area_m2: float | None = None,
                       T: float = 288.15, dt: float = 0.01, t_max: float = 60.0,
                       frame_dt: float = 0.05, target_watch_s: float = 5.0,
                       floor: bool = True) -> dict:
    """Drop a Deckard Construct — its REAL compiled shape, mass and area — through
    the atmosphere, integrating Materia's drag, and emit the viewer's trajectory
    bundle. The object's actual geometry (a feather's cone+ellipsoid, say) is the
    moving body, not a sphere.

    Drag acts on the BROAD face: a thin object dropped flat presents its largest
    face, so unless `area_m2` is given the area is the largest bbox face — never
    the (edge-on) footprint, which would fall far too fast. The body is laid flat
    by a constant orientation; a straight-down terminal-velocity fall, no flutter.
    """
    from ..materia.engine import simulate_drag_run   # lazy: tier-3 sibling, no cycle
    from .scene_export import construct_to_scene, _bake_material

    scene = construct_to_scene(construct)
    mass_kg = scene["physics"]["mass_kg"]
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
    if area_m2 is None:
        area_m2 = max(dx * dy, dy * dz, dx * dz, 1e-9)    # broad face, not footprint

    # Materia integrates the fall (gravity + altitude-varying atmospheric drag).
    run = simulate_drag_run(mass_kg, area_m2, start_altitude_m=start_altitude_m,
                            cd_value=cd, T=T, dt=dt, t_max=t_max,
                            sample_every=max(1, round(frame_dt / dt)))
    hist = run.get("history") or []
    fall_t = run.get("fall_time_s") or (hist[-1]["t"] if hist else 0.0)

    quat = _lay_flat_quat(dx, dy, dz)
    frames = [{"t_sim": round(s["t"], 4),
               "bodies": [{"pos": [0.0, round(s["altitude_m"], 5), 0.0],
                           "quat": quat}]}
              for s in hist]
    if not frames:                                   # degenerate: emit one pose
        frames = [{"t_sim": 0.0,
                   "bodies": [{"pos": [0.0, start_altitude_m, 0.0], "quat": quat}]}]

    # The construct's leaves become ONE dynamic body (body 0), rotated about its
    # centre of mass and translated by the per-frame pose.
    for leaf in scene["csg_leaves"]:
        leaf["body"] = 0
    scene["bodies"] = [{"pivot": list(construct.com_m), "label": construct.name}]

    if floor:
        # No grounded wood reflectance, so rather than fake a colour we use an
        # honest neutral dark matte stage (a backdrop, not a material claim) — a
        # pale object reads against it.
        stage = _bake_material("wood_oak", 700.0)
        stage["color_rgb"] = [0.07, 0.07, 0.08]
        stage["mechanism"] = ("neutral render stage (dark matte backdrop, not a "
                              "grounded material colour)")
        stage["emergent"] = False
        scene["materials"]["stage_dark"] = stage
        half = max(0.6, 1.5 * max(dx, dz))
        scene["csg_leaves"].append({                 # static floor (no body)
            "op": "add", "material": "stage_dark",
            "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
                      "x": 2 * half, "y": 0.1, "z": 2 * half}})

    # Frame the whole fall column, floor → just above the release height.
    top = start_altitude_m + max(dx, dy, dz)
    wide = max(1.0, 1.5 * max(dx, dz))
    scene["bbox"] = [[-wide, wide], [-0.1, top + 0.1], [-wide, wide]]
    scene["camera"] = {"target": [0.0, start_altitude_m * 0.5, 0.0],
                       "orbit_radius": max(3.0, 1.6 * start_altitude_m),
                       "fov_deg": 46.0, "up": [0.0, 1.0, 0.0],
                       "az0": 0.35, "el0": 0.22}
    scene["kind"] = "trajectory"

    return {
        "scene": scene,
        "trajectory": {
            "frames": frames,
            "t_end_s": round(frames[-1]["t_sim"], 4),
            "natural_timescale_s": fall_t,
            "suggested_rate": max(1e-6, (fall_t or 1.5) / target_watch_s),
            "body_labels": [construct.name],
        },
        "kind": "trajectory",
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
