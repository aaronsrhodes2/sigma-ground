"""Trajectory recording — Materia motion → a serializable per-frame time series.

The Materia verbs compute a trajectory and keep only the scalar answer. The
viewer needs the *frames*. Rather than edit the (in-flux) materia engine, we
reuse its validated pieces — `make_atmospheric_drag`, `_material_density`, the
density-wrapper — and the dynamics stepper, recording (t, position) along the
way. Output is the contract bundle: a one-body SceneSpec + a Trajectory of
poses, with an auto-suggested playback rate from the phenomenon's own timescale.

THE FRAME CONTRACT (consumed by viewer.js posesAt and poses_at below):

    {"t_sim": t, "bodies": [{"pos": [x,y,z], "quat": [x,y,z,w],
                             "temperature_k": T?}]}

``temperature_k`` is OPTIONAL per body per frame — the body's bulk temperature
at that moment, frozen sim output from a thermal recorder (thermal_record.py).
Renderer precedence (replacement, not addition — these are the same physical
quantity at different times):

    per-cell field  >  frame body temperature_k  >  leaf static temperature_k
                    >  physics_env.temperature_k  >  293.15

and the viewer's heat slider adds a user-probe delta on top of whichever won.
Between frames every channel (pos, quat, temperature_k) is interpolated —
NON-DERIVED (audit): inter-frame lerp is playback reconstruction of frozen sim
states, never the heat equation (the renderer displays fields, it does not
integrate them).

Quaternions are identity for now (a sphere has no visible orientation); rotation
arrives with Materia's rigid-body stage and slots straight into `quat`.
"""
from __future__ import annotations

from .scene_export import _bake_material, _suggest_camera


def poses_at(trajectory: dict, t: float) -> list:
    """The interpolated body states at sim-time ``t`` — the Python twin of
    viewer.js ``posesAt`` (binary search; lerp pos; nlerp quat; lerp
    temperature_k when both endpoints carry it, hold when one does).

    Returns [{"pos", "quat", "temperature_k"?}, ...] per body.
    """
    frames = (trajectory or {}).get("frames") or []
    if not frames:
        return []

    def _pack(bodies):
        out = []
        for b in bodies:
            d = {"pos": list(b["pos"]), "quat": list(b.get("quat") or [0, 0, 0, 1])}
            if b.get("temperature_k") is not None:
                d["temperature_k"] = float(b["temperature_k"])
            out.append(d)
        return out

    if t <= frames[0]["t_sim"]:
        return _pack(frames[0]["bodies"])
    if t >= frames[-1]["t_sim"]:
        return _pack(frames[-1]["bodies"])
    lo, hi = 0, len(frames) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if frames[mid]["t_sim"] <= t:
            lo = mid
        else:
            hi = mid
    fa, fb = frames[lo], frames[hi]
    u = (t - fa["t_sim"]) / max(1e-9, fb["t_sim"] - fa["t_sim"])
    out = []
    for k, ba in enumerate(fa["bodies"]):
        bb = fb["bodies"][k] if k < len(fb["bodies"]) else ba
        pa, pb = ba["pos"], bb["pos"]
        pos = [pa[i] + (pb[i] - pa[i]) * u for i in range(3)]
        qa = ba.get("quat") or [0, 0, 0, 1]
        qb = bb.get("quat") or [0, 0, 0, 1]
        d = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
        s = -1.0 if d < 0 else 1.0                    # shortest-path nlerp
        q = [qa[i] + (qb[i] * s - qa[i]) * u for i in range(4)]
        n = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5 or 1.0
        body = {"pos": pos, "quat": [v / n for v in q]}
        Ta, Tb = ba.get("temperature_k"), bb.get("temperature_k")
        if Ta is not None and Tb is not None:
            body["temperature_k"] = Ta + (Tb - Ta) * u     # playback lerp, NOT physics
        elif Ta is not None or Tb is not None:
            body["temperature_k"] = float(Ta if Ta is not None else Tb)
        out.append(body)
    return out


def bake_frame_temperatures(bundle: dict, t_sim: float) -> dict:
    """A deep-copied SceneSpec with each body-bound leaf's ``temperature_k``
    overridden by the trajectory's interpolated body temperature at ``t_sim`` —
    the Python renderer's ground-truth still for any scrub position. (Pose is
    deliberately NOT baked: incandescence doesn't depend on pose, and geometric
    pose parity is the viewer's job.)"""
    import copy
    scene = copy.deepcopy(bundle["scene"])
    bodies = poses_at(bundle.get("trajectory") or {}, t_sim)
    temps = {k: b["temperature_k"] for k, b in enumerate(bodies)
             if b.get("temperature_k") is not None}
    if temps:
        for leaf in scene.get("csg_leaves", []):
            if leaf.get("body") in temps:
                leaf["temperature_k"] = temps[leaf["body"]]
    return scene


def record_fall(material_key: str = "copper", radius_m: float = 0.05,
                start_altitude_m: float = 10_000.0, T: float = 288.15,
                dt_max: float = 0.02, t_max: float = 600.0,
                frame_dt: float = 0.2, target_watch_s: float = 15.0,
                v0_m_s: float = 0.0) -> dict:
    """Drop (or launch: ``v0_m_s`` upward) a sphere through the atmosphere,
    recording a pose every `frame_dt`. At the floor the sphere REBOUNDS with the
    material's emergent Johnson–Thornton restitution (same model bounce_heights
    demos) and settles; the quat stays identity — a sphere has no visible spin.
    """
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
                           position=Vec3(0.0, max(start_altitude_m, radius_m), 0.0),
                           velocity=Vec3(0.0, float(v0_m_s), 0.0), label=material_key)
    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -9.80665, 0.0), ground=False)
    cb = make_atmospheric_drag(scene.gravity, T)

    def _pose():
        p = parcel.position
        return {"pos": [p.x, p.y, p.z], "quat": [0.0, 0.0, 0.0, 1.0]}

    try:
        from ..field.interface.impact import coefficient_of_restitution as _cor
    except Exception:
        _cor = None

    frames = [{"t_sim": 0.0, "bodies": [_pose()]}]
    next_rec = frame_dt
    contacts = []
    apex_m = parcel.position.y
    while scene.time < t_max:
        step(scene, dt=dt_max, sub_steps=4, external_forces=cb)
        apex_m = max(apex_m, parcel.position.y)
        if parcel.position.y <= radius_m and parcel.velocity.y < 0.0:
            v_in = abs(parcel.velocity.y)               # floor contact: rebound
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
            if e * v_in < 0.04:                          # settled
                parcel.velocity = Vec3(0.0, 0.0, 0.0)
                break
        if scene.time >= next_rec:
            frames.append({"t_sim": scene.time, "bodies": [_pose()]})
            next_rec += frame_dt
    frames.append({"t_sim": scene.time, "bodies": [_pose()]})
    t_end = scene.time

    r = radius_m
    top = max(start_altitude_m, apex_m) + 2 * r
    wide = max(4 * r, 0.5)
    follow = top > 40 * r
    # The viewer scales its ray/shadow epsilons from the bbox DIAGONAL. A
    # follow-camera scene must therefore carry an OBJECT-sized bbox — a 40 km
    # fall-column bbox makes the hit epsilon ~60 m and a 5 cm ball can never
    # register a hit. Only short, whole-column views keep the wide bbox.
    bbox = ([[-4 * r, 4 * r], [-4 * r, 4 * r], [-4 * r, 4 * r]] if follow
            else [[-wide, wide], [-0.1, top], [-wide, wide]])
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
        "bbox": bbox,
        "camera": {"target": ([0.0, start_altitude_m, 0.0] if follow else
                              [0.0, min(top, max(2 * r, top * 0.5)), 0.0]),
                   "orbit_radius": 8 * r if follow else max(6 * r, 0.4 * top),
                   "fov_deg": 45.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.4, "el0": 0.15,
                   **({"follow": True} if follow else {})},
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
            "validation": {"contacts": contacts,
                           "apex_m": round(apex_m, 3),
                           "energy_monotone": all(c["E_after"] < c["E_before"]
                                                  for c in contacts)},
        },
    }


def _radius_arrow_leaves(r, height_m, body, material="concrete"):
    """A radius line + tip crossbar painted on a disc's face, on the given
    ``body`` index. A plain cylinder is rotationally symmetric about its own
    spin axis — it renders IDENTICALLY at every angle, so no small marker
    bump reads clearly from a face-on camera (confirmed the hard way:
    an earlier single off-axis sphere marker was nearly invisible in
    practice). A radius line reads like a clock hand instead: the long line
    shows the radius, the crossbar near the outer end marks "this end
    points outward", so both rotation direction and rate are legible at a
    glance, from directly above or at an angle."""
    z = height_m / 2.0
    # Sized generously on purpose: a first attempt at ~0.16r thickness / 0.18r
    # width rendered essentially invisibly at this demo's actual disc scale
    # (confirmed with a deliberately 3x-oversized debug render before tuning
    # back down — the geometry/material/shader were always correct, the
    # feature was just too small to read at normal camera distance).
    thick = 0.30 * r
    return [
        {"op": "add", "material": material, "body": body,
         "shape": {"type": "Box", "center": [0.5 * r, 0.0, z],
                   "x": 0.80 * r, "y": 0.22 * r, "z": thick}},
        {"op": "add", "material": material, "body": body,
         "shape": {"type": "Box", "center": [0.85 * r, 0.0, z],
                   "x": 0.10 * r, "y": 0.55 * r, "z": thick}},
    ]


def record_motor_spin(radius_m: float = 0.05, height_m: float = 0.01,
                      material_key: str = "copper",
                      motor_speed_rad_s: float = -3.0,
                      motor_max_torque: float = 2.0, T: float = 288.15,
                      dt: float = 1.0 / 960.0, t_max: float = 4.0,
                      frame_dt: float = 0.02, target_watch_s: float = 8.0) -> dict:
    """Spin a disc about its own axis under a torque-capped RevoluteJoint
    motor, recording the SOLVED orientation into the trajectory frame
    contract's ``quat`` channel.

    dynamics/joints.py's RevoluteJoint motor and the leapfrog stepper are
    independently validated (tests/test_joints.py::
    test_motor_ledger_bounds_energy_gain), and the frame contract already
    carries ``quat`` per body, but until now nothing converted a solved
    joint angle into played-back rotation — build_demo.py's "chair tip" is
    explicitly labeled a SCRIPTED kinematic preview awaiting exactly this.
    The disc here is a bare placeholder cylinder: no gear-tooth geometry, no
    blueprint-extracted data — this function proves only the render/dynamics
    bridge, ahead of any real mechanism.
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.shapes import Cylinder
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import RevoluteJoint
    from ..dynamics.stepper import step

    from ..dynamics.mechanisms.plug import Plug
    from ..dynamics.mechanisms.choice import Choice

    material_choice = Choice(
        "placeholder disc material — no blueprint mechanism pins a metal yet",
        material_key, checked="sigma_ground.field.interface.surface.MATERIALS "
        "(25 entries)",
        alternatives=["steel_mild", "aluminum", "tungsten", "titanium"],
        reason="mid-range generic metal; kept absent a specific machine to "
              "match, not compared against the table before picking")
    size_choice = Choice(
        "placeholder disc radius/height", [radius_m, height_m],
        checked="sigma_ground/blueprint/catalog/ (empty — no MechanismSpec on "
        "file)", reason="arbitrary demo scale chosen for a legible render; "
        "supersede once real rotor geometry is blueprint-extracted")
    density, mat_name = _material_density(material_key, T)
    shape = Cylinder(radius_m, height_m)
    disc = PhysicsParcel(shape, _DensityMaterial(density),
                         position=Vec3(0.0, 2.0 * radius_m, 0.0), label=material_key)
    axis = Vec3(0.0, 0.0, 1.0)                    # Cylinder's own axis (kernel/shapes.py)
    joint = RevoluteJoint(disc, None, disc.position, axis)
    # nothing in this scene turns the disc — the motor is a declared plug —
    # and nothing HOLDS it either: the world-anchored hinge is a plugged
    # SUPPORT (the disc would fall under gravity without it)
    plug = Plug(joint, "artificial rotation from external source",
                motor_speed_rad_s=motor_speed_rad_s,
                motor_max_torque_nm=motor_max_torque,
                equivalent="a motor that is not modeled")
    support = Plug.support("disc held in place as if by a shaft mount, "
                           "without the mount")
    scene = PhysicsScene([disc], ground=False, constraints=[joint])

    def _pose():
        p = disc.position
        return {"pos": [p.x, p.y, p.z], "quat": [round(x, 6) for x in disc.orientation]}

    frames = [{"t_sim": 0.0, "bodies": [_pose()]}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    while scene.time < t_max:
        step(scene, dt=dt, sub_steps=1)
        if scene.time >= next_rec:
            frames.append({"t_sim": round(scene.time, 4), "bodies": [_pose()]})
            next_rec += frame_dt
    frames.append({"t_sim": round(scene.time, 4), "bodies": [_pose()]})
    t_end = scene.time
    de = scene.total_kinetic_energy() - e0

    r = radius_m
    scene_spec = {
        "name": f"{mat_name.lower()} disc — RevoluteJoint motor (placeholder gear, "
                f"no blueprint geometry yet)",
        "bodies": [{"pivot": [0, 0, 0], "label": material_key}],
        "csg_leaves": [{"op": "add", "material": material_key, "body": 0,
                        "shape": {"type": "Cylinder", "center": [0, 0, 0],
                                  "radius": radius_m, "height": height_m}},
                       *_radius_arrow_leaves(r, height_m, body=0)],
        "materials": {material_key: _bake_material(material_key, density),
                      "concrete": _bake_material("concrete")},
        "physics": {"mass_kg": disc.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": list(disc.inertia_body)},
        "bbox": [[-4 * r, 4 * r], [-4 * r, 4 * r], [-4 * r, 4 * r]],
        "camera": {"target": [0.0, disc.position.y, 0.0],
                   "orbit_radius": max(8 * r, 0.3), "fov_deg": 45.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.5, "el0": 1.0},
        "identified": True,
        "plugs": [plug.to_dict(), support.to_dict()],
        "choices": [material_choice.to_dict(), size_choice.to_dict()],
        "source": "motor-driven RevoluteJoint (dynamics/joints.py, SOLVED not scripted) — "
                  "placeholder disc; real gear geometry arrives with blueprint "
                  "extraction. " + plug.cite() + ". " + support.cite() + ". "
                  + material_choice.cite() + ". " + size_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": [material_key],
            "validation": {
                "motor_work_j": joint.motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= joint.motor_work_j + 1e-6,
                "final_angle_rad": joint.angle(),
            },
        },
        "kind": "trajectory",
    }


def record_gear_train_spin(radii_m=(0.05, 0.09, 0.045),
                           height_m: float = 0.01,
                           material_keys=("copper", "gold", "iron"),
                           motor_speed_rad_s: float = -3.0,
                           motor_max_torque: float = 2.0,
                           ratios=(1.8, -1.5), T: float = 288.15,
                           dt: float = 1.0 / 960.0, t_max: float = 5.0,
                           frame_dt: float = 0.02,
                           target_watch_s: float = 8.0) -> dict:
    """A chain of N wheels: wheel 0 is motor-driven (dynamics/joints.py's
    RevoluteJoint), each later wheel is held at ``ratios[i-1]`` relative to
    the previous one via GearCouplingJoint — a KINEMATIC ratio constraint,
    not simulated tooth contact (real involute geometry, and real ratios,
    arrive with blueprint extraction). Placeholder wheel sizing only:
    radii scale with the visual convention "more teeth = bigger wheel",
    cosmetic until real tooth counts exist.

    This is the first recorder to drive MORE THAN ONE body through a
    SOLVED constraint (the render contract's `bodies` list already
    supported >1 body — build_demo.py's "chair tip" proved multi-body
    playback, but scripted; this is multi-body playback of something the
    constraint solver actually solved, per-body).
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.shapes import Cylinder
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import RevoluteJoint, GearCouplingJoint
    from ..dynamics.stepper import step

    n = len(radii_m)
    if len(material_keys) != n or len(ratios) != n - 1:
        raise ValueError("material_keys must match radii_m; ratios must have "
                         "one fewer entry than radii_m")

    axis = Vec3(0.0, 0.0, 1.0)
    rmax = max(radii_m)
    y0 = 2.0 * rmax
    wheels, densities, mat_names, xs = [], [], [], []
    x = 0.0
    for i, r in enumerate(radii_m):
        if i > 0:
            x += radii_m[i - 1] + r + 0.4 * min(radii_m[i - 1], r)   # clear gap
        xs.append(x)
        density, mat_name = _material_density(material_keys[i], T)
        densities.append(density)
        mat_names.append(mat_name)
        shape = Cylinder(r, height_m)
        w = PhysicsParcel(shape, _DensityMaterial(density),
                          position=Vec3(x, y0, 0.0), label=material_keys[i])
        wheels.append(w)

    from ..dynamics.mechanisms.plug import Plug
    from ..dynamics.mechanisms.choice import Choice

    joints = [RevoluteJoint(wheels[0], None, wheels[0].position, axis)]
    # nothing in this scene turns wheel 0 — the drive motor is a declared
    # plug — and the wheels hang on world-anchored hinges: plugged supports
    plug = Plug(joints[0], "artificial rotation from external source",
                motor_speed_rad_s=motor_speed_rad_s,
                motor_max_torque_nm=motor_max_torque,
                equivalent="a motor that is not modeled")
    support = Plug.support("wheels held in place as if mounted on a frame, "
                           "without the frame")
    material_choice = Choice(
        "gear-wheel materials per body", list(material_keys),
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild", "iron", "nickel", "titanium", "tungsten"],
        reason="gold/copper are soft/decorative, not real load-bearing gear "
              "materials; kept for visual per-wheel contrast rather than "
              "mechanical realism — steel_mild (the table's actual machine "
              "alloy) was never compared before picking")
    ratio_choice = Choice(
        "gear-train stage ratios", list(ratios),
        checked="sigma_ground/blueprint/catalog/kelly_1944_watch_going_train_"
        "18000bph.md (cited ratios run 6-10:1)",
        reason="this demo defines no teeth_a/teeth_b at all, so the ratio "
              "has no gear-count backing; picked only to look like a "
              "plausible train, not cross-checked against the one cited "
              "blueprint on file for this mechanism class")
    size_choice = Choice(
        "wheel radii/height (placeholder gear sizing)",
        [list(radii_m), height_m],
        checked="blueprint/catalog kelly_1944 (module_mm/face_width_mm null "
        "for this train — GEOMETRY_GAP)",
        reason="no cited absolute module/pitch exists for this mechanism "
              "class; radii picked only to satisfy a more-teeth-bigger-"
              "wheel visual convention")
    couplings = []
    for i in range(1, n):
        ji = RevoluteJoint(wheels[i], None, wheels[i].position, axis)
        joints.append(ji)
        couplings.append(GearCouplingJoint(joints[i - 1], ji, ratios[i - 1]))

    scene = PhysicsScene(wheels, ground=False, constraints=joints + couplings)

    def _poses():
        return [{"pos": [w.position.x, w.position.y, w.position.z],
                 "quat": [round(v, 6) for v in w.orientation]} for w in wheels]

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    while scene.time < t_max:
        step(scene, dt=dt, sub_steps=1)
        if scene.time >= next_rec:
            frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
            next_rec += frame_dt
    frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
    t_end = scene.time
    de = scene.total_kinetic_energy() - e0

    bodies_spec = [{"pivot": [0, 0, 0], "label": material_keys[i]} for i in range(n)]
    materials = {}
    leaves = []
    for i, r in enumerate(radii_m):
        mk = material_keys[i]
        if mk not in materials:
            materials[mk] = _bake_material(mk, densities[i])
        leaves.append({"op": "add", "material": mk, "body": i,
                       "shape": {"type": "Cylinder", "center": [0, 0, 0],
                                 "radius": r, "height": height_m}})
        leaves.extend(_radius_arrow_leaves(r, height_m, body=i))
    materials.setdefault("concrete", _bake_material("concrete"))

    x_lo = xs[0] - radii_m[0] - 0.02
    x_hi = xs[-1] + radii_m[-1] + 0.02
    scene_spec = {
        "name": "gear train — RevoluteJoint motor + GearCouplingJoint ratios "
                "(placeholder wheels, no blueprint geometry yet)",
        "bodies": bodies_spec,
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": sum(w.mass for w in wheels), "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[x_lo, x_hi], [y0 - 2 * rmax, y0 + 2 * rmax], [-2 * rmax, 2 * rmax]],
        "camera": {"target": [(x_lo + x_hi) / 2.0, y0, 0.0],
                   "orbit_radius": max(6 * rmax, 0.6 * (x_hi - x_lo), 0.3),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0], "az0": 0.5, "el0": 1.0},
        "identified": True,
        "plugs": [plug.to_dict(), support.to_dict()],
        "choices": [material_choice.to_dict(), ratio_choice.to_dict(),
                   size_choice.to_dict()],
        "source": "motor-driven RevoluteJoint + GearCouplingJoint rate constraints "
                  "(dynamics/joints.py, SOLVED not scripted) — placeholder wheels; "
                  "real gear geometry and ratios arrive with blueprint extraction. "
                  + plug.cite() + ". " + support.cite() + ". "
                  + material_choice.cite() + ". " + ratio_choice.cite() + ". "
                  + size_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": list(material_keys),
            "validation": {
                "motor_work_j": joints[0].motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= joints[0].motor_work_j + 1e-6,
                "ratios_commanded": list(ratios),
                "final_angles_rad": [j.angle() for j in joints],
            },
        },
        "kind": "trajectory",
    }


def record_gear_mesh_spin(module_m: float = 0.004, teeth_a: int = 20,
                          teeth_b: int = 31,
                          pressure_angle_deg: float = 20.0,
                          face_width_m: float = 0.01,
                          motor_speed_rad_s: float = -1.5,
                          motor_max_torque: float = 0.5, T: float = 288.15,
                          dt: float = 1.0 / 960.0, t_max: float = 12.0,
                          frame_dt: float = 0.02,
                          target_watch_s: float = 12.0) -> dict:
    """Two REAL involute gears meshing — the Phase 4 payoff: actual
    kernel/gear.py InvoluteGear tooth geometry (adversarially verified
    against independent brute-force ground truth), placed at the standard
    center distance m*(Na+Nb)/2 and counter-rotating at the exact tooth
    ratio via Phase 1's GearCouplingJoint. Because the involute profile is
    THE conjugate-action profile, correct ratio + correct center distance +
    correct initial phasing means the rendered teeth interleave through the
    whole rotation without scripting — the mesh works because the geometry
    is right, not because anyone animated it.

    Honesty notes:
      - module is [ESTIMATED]: the Phase 2 Kelly (1944) source cites tooth
        counts and ratios but NO module/pitch anywhere near its worked
        example (a flagged GEOMETRY_GAP in the blueprint catalog) — so the
        physical size here is an explicit estimate, stated in the scene's
        source string, never laundered into a citation.
      - the coupling is still Phase 1's KINEMATIC ratio constraint, not
        simulated tooth-contact force; the teeth are real geometry riding
        on a solved ratio. teeth_b defaults to an ODD count so the mesh
        point (angle pi in B's local frame) lands exactly on a tooth GAP
        with no initial rotation: gap centers sit at odd multiples of
        beta_b = pi/N_b, and pi = N_b*beta_b is an odd multiple iff N_b is
        odd (an even N_b needs a beta_b initial phase twist instead).
      - the parcels carry the gears' REAL mass/inertia (grid quadrature on
        the actual tooth geometry) but a deliberately REDUCED collision
        radius: meshing gears' bounding spheres overlap BY DESIGN, and the
        stepper's honest-but-coarse sphere-sphere narrow phase would
        otherwise emit phantom contacts fighting the coupling constraint.
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.gear import InvoluteGear
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import RevoluteJoint, GearCouplingJoint
    from ..dynamics.stepper import step
    from ..dynamics.mechanisms.choice import Choice
    import math

    teeth_choice = Choice(
        "gear_a/gear_b tooth counts (gear_b must be odd — see docstring)",
        [teeth_a, teeth_b],
        checked="sigma_ground/blueprint/catalog/"
        "kelly_1944_watch_going_train_18000bph.md",
        alternatives=["escape_pinion 8t / escape_wheel 15t (Kelly-cited, "
                     "real odd-toothed mesh pair)"],
        reason="picked for a legible 2-gear mesh demo; the odd-parity "
              "constraint on teeth_b is derived, but the specific counts "
              "were never cross-checked against the cited real pair on file")
    pressure_angle_choice = Choice(
        "pressure angle (deg)", pressure_angle_deg,
        checked="blueprint kelly_1944 (pressure_angle_deg: null — same "
        "GEOMETRY_GAP as module)", alternatives=[14.5, 25.0],
        reason="20 deg is the common modern ISO/AGMA involute standard, "
              "picked as a plausible default rather than a cited source")

    alpha = math.radians(pressure_angle_deg)
    axis = Vec3(0.0, 0.0, 1.0)
    gear_a = InvoluteGear(module=module_m, teeth=teeth_a, pressure_angle=alpha,
                          face_width=face_width_m, grid_resolution=48)
    gear_b = InvoluteGear(module=module_m, teeth=teeth_b, pressure_angle=alpha,
                          face_width=face_width_m, grid_resolution=48)
    center_distance = module_m * (teeth_a + teeth_b) / 2.0   # standard mesh distance

    y0 = 0.15
    pos_a = Vec3(0.0, y0, 0.0)
    pos_b = Vec3(center_distance, y0, 0.0)

    mats = ("copper", "iron")
    mats_choice = Choice(
        "gear material pair", list(mats),
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild", "titanium", "tungsten", "nickel"],
        reason="reused this file's clock-train color-contrast convention; "
              "copper is mechanically a poor choice for load-bearing teeth "
              "(soft, low wear resistance) versus steel_mild, never checked")
    densities = {}
    for mk in mats:
        densities[mk], _ = _material_density(mk, T)

    def _parcel(gear, pos, mk):
        vol = gear.volume()
        mass = densities[mk] * vol
        inertia = tuple(mass * gear.inertia_factor(ax) for ax in ("x", "y", "z"))
        # collision radius shrunk below half the center distance so the two
        # meshing gears' bounding spheres can NEVER phantom-contact (see
        # docstring); mass/inertia stay the real quadrature values.
        return PhysicsParcel(0.45 * center_distance, _DensityMaterial(densities[mk]),
                             mass=mass, position=pos, inertia_body=inertia,
                             label=mk)

    wheel_a = _parcel(gear_a, pos_a, mats[0])
    wheel_b = _parcel(gear_b, pos_b, mats[1])

    from ..dynamics.mechanisms.plug import Plug
    ja = RevoluteJoint(wheel_a, None, pos_a, axis)
    # nothing in this scene turns gear A — the drive motor is a declared
    # plug — and both gears hang on world-anchored hinges: plugged supports
    plug = Plug(ja, "artificial rotation from external source",
                motor_speed_rad_s=motor_speed_rad_s,
                motor_max_torque_nm=motor_max_torque,
                equivalent="a motor that is not modeled")
    support = Plug.support("gears held at their mesh distance as if mounted "
                           "on a frame, without the frame")
    jb = RevoluteJoint(wheel_b, None, pos_b, axis)
    # omega_b = -(Na/Nb)*omega_a (external mesh counter-rotation);
    # GearCouplingJoint enforces s_a + ratio*s_b = 0, i.e.
    # omega_b = -omega_a/ratio (sign convention pinned empirically by
    # tests/test_joints.py::test_gear_coupling_holds_the_commanded_ratio),
    # so ratio = Nb/Na.
    coupling = GearCouplingJoint(ja, jb, teeth_b / teeth_a)

    scene = PhysicsScene([wheel_a, wheel_b], ground=False,
                         constraints=[ja, jb, coupling])

    def _poses():
        return [{"pos": [w.position.x, w.position.y, w.position.z],
                 "quat": [round(v, 6) for v in w.orientation]}
                for w in (wheel_a, wheel_b)]

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    while scene.time < t_max:
        step(scene, dt=dt, sub_steps=1)
        if scene.time >= next_rec:
            frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
            next_rec += frame_dt
    frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
    t_end = scene.time
    de = scene.total_kinetic_energy() - e0

    def _gear_leaf(gear, body, mk):
        return {"op": "add", "material": mk, "body": body,
                "shape": {"type": "Gear", "module": gear.module,
                          "teeth": gear.teeth,
                          "pressure_angle": gear.pressure_angle,
                          "addendum_coeff": gear.addendum_coeff,
                          "dedendum_coeff": gear.dedendum_coeff,
                          "fillet_coeff": gear.fillet_coeff,
                          "face_width": gear.face_width,
                          "center": [0.0, 0.0, 0.0],
                          "source": "involute profile (kernel/gear.py, verified); "
                                    "module [estimated] — no module cited in the "
                                    "Kelly 1944 source (GEOMETRY_GAP)"}}

    r_max = max(gear_a.r_a, gear_b.r_a)
    x_lo, x_hi = -gear_a.r_a - 0.01, center_distance + gear_b.r_a + 0.01
    scene_spec = {
        "name": "meshing involute gears — real tooth geometry (kernel/gear.py), "
                "kinematic ratio coupling (module [estimated]: Kelly cites no module)",
        "bodies": [{"pivot": [0, 0, 0], "label": mats[0]},
                  {"pivot": [0, 0, 0], "label": mats[1]}],
        "csg_leaves": [_gear_leaf(gear_a, 0, mats[0]),
                       _gear_leaf(gear_b, 1, mats[1])],
        "materials": {mk: _bake_material(mk, densities[mk]) for mk in mats},
        "physics": {"mass_kg": wheel_a.mass + wheel_b.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[x_lo, x_hi], [y0 - r_max - 0.01, y0 + r_max + 0.01],
                 [-face_width_m, face_width_m]],
        "camera": {"target": [center_distance / 2.0, y0, 0.0],
                   "orbit_radius": max(1.6 * (x_hi - x_lo), 0.25),
                   "fov_deg": 40.0, "up": [0.0, 1.0, 0.0], "az0": 0.0, "el0": 0.12},
        "identified": True,
        "plugs": [plug.to_dict(), support.to_dict()],
        "choices": [teeth_choice.to_dict(), pressure_angle_choice.to_dict(),
                   mats_choice.to_dict()],
        "source": "InvoluteGear SDF (kernel/gear.py, adversarially verified vs "
                  "brute-force ground truth) + GearCouplingJoint ratio "
                  "(dynamics/joints.py, SOLVED not scripted). Tooth counts/ratio "
                  "conventions per blueprint doctrine; module_m is [ESTIMATED] — "
                  "the Kelly (1944) catalog source cites tooth counts but no "
                  "module (flagged GEOMETRY_GAP in blueprint/catalog). "
                  + plug.cite() + ". " + support.cite() + ". "
                  + teeth_choice.cite() + ". " + pressure_angle_choice.cite() + ". "
                  + mats_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": list(mats),
            "validation": {
                "motor_work_j": ja.motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= ja.motor_work_j + 1e-6,
                "ratio_commanded": teeth_b / teeth_a,
                "center_distance_m": center_distance,
                "final_angles_rad": [ja.angle(), jb.angle()],
            },
        },
        "kind": "trajectory",
    }


def record_windmill_spinup(wind_speed_m_s: float = 10.0,
                           n_blades: int = 4,
                           blade_length_m: float = 0.5,
                           blade_chord_m: float = 0.15,
                           blade_thickness_m: float = 0.004,
                           blade_pitch_deg: float = 45.0,
                           radius_centroid_m: float = 0.6,
                           blade_material: str = "aluminum",
                           dt: float = 1.0 / 960.0, t_max: float = 40.0,
                           frame_dt: float = 0.05,
                           target_watch_s: float = 15.0) -> dict:
    """A windmill rotor spun by WIND — the merged lane's first fully NATURAL
    drive (plugs: [] and it means it): simulated air pushes on real pitched
    blade area (dynamics/mechanisms/wind.py's flat-plate model, flagged
    SIMPLIFIED there), torque emerges from the blade geometry (zero pitch =
    zero torque — gated), and the rotor self-limits at the closed-form
    terminal tip speed where blade motion cancels the inflow. Nothing
    prescribes the final speed; the physics finds it.

    Mounted on a RigidBearing, not a RevoluteJoint — see KNOWN_GAPS.md
    (PHYSICS_GAP: the joint's swing loop is unstable under orientation-
    coupled external torque; the bearing is both the workaround and the
    physically better abstraction for a stiff nacelle).

    Blade/render geometry share ONE construction: the same azimuth+pitch
    rotation that builds each physics Blade (centroid + normal) also builds
    its Rotated-Box render leaf, so the picture and the forces cannot
    disagree about where the blades are.
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step
    from ..dynamics.quat import quat_from_axis_angle, quat_mul
    from ..dynamics.mechanisms.wind import (RotorWind, build_rotor_blades,
                                            terminal_omega)
    from ..dynamics.mechanisms.bearing import RigidBearing
    from ..dynamics.mechanisms.choice import Choice
    import math

    blade_material_choice = Choice(
        "rotor blade material", blade_material,
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["carbon_fiber — real wind-turbine blade composite",
                     "steel_mild — traditional farm-windmill sail material"],
        reason="aluminum sheet is plausible for this flat-plate (non-"
              "airfoil) blade model, but no comparison against the "
              "composite/steel alternatives was recorded before picking it")
    structure_material_choice = Choice(
        "hub and mast/tower material", "iron",
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild — the real material wind-turbine towers "
                     "and hub castings are built from"],
        reason="iron is plausible for a cast hub but not realistic for a "
              "load-bearing tower; steel_mild was never checked before "
              "iron was carried over from the hub pick to the mast too")
    geometry_choice = Choice(
        "rotor/blade geometry (count, length, chord, thickness, pitch, "
        "centroid radius) and hub cylinder size",
        {"n_blades": n_blades, "blade_length_m": blade_length_m,
         "blade_chord_m": blade_chord_m, "blade_thickness_m": blade_thickness_m,
         "blade_pitch_deg": blade_pitch_deg,
         "radius_centroid_m": radius_centroid_m,
         "hub_r_m": 0.08, "hub_h_m": 0.1},
        checked="sigma_ground/blueprint/catalog/ (only the clock's Kelly "
        "1944 entry exists — no wind-rotor design catalog in this "
        "codebase)",
        reason="no blueprint for wind-rotor geometry exists; every value "
              "picked to be a plausible small-turbine demo scale — pitch "
              "in particular sets terminal_omega directly and is not "
              "derived from any target tip-speed ratio")
    wind_choice = Choice(
        "ambient wind speed (m/s)", wind_speed_m_s, alternatives=[5.0],
        reason="round demo scenario input, not a looked-up quantity; also "
              "offered as a labeled 5 m/s scene variant")

    beta = math.radians(blade_pitch_deg)
    dens_blade, _ = _material_density(blade_material, 288.15)
    dens_hub, _ = _material_density("iron", 288.15)
    blades, area = build_rotor_blades(n_blades, radius_centroid_m,
                                      blade_length_m, blade_chord_m, beta)

    # rotor mass/inertia from the blade boxes + hub cylinder (deterministic
    # arithmetic: box principal inertias, pitched about the radial axis,
    # plus parallel-axis m*R_c^2; azimuth placement doesn't change I about
    # the spin axis). Transverse inertia set equal to the axial value —
    # those DOF are held by the bearing, flagged approximation.
    m_blade = dens_blade * blade_length_m * blade_chord_m * blade_thickness_m
    t2, L2, c2 = blade_thickness_m ** 2, blade_length_m ** 2, blade_chord_m ** 2
    I_own_x = m_blade * (L2 + c2) / 12.0          # about the blade normal axis
    I_own_z = m_blade * (t2 + L2) / 12.0          # about the chord axis
    I_own_pitched = (math.cos(beta) ** 2 * I_own_x
                     + math.sin(beta) ** 2 * I_own_z)
    hub_r, hub_h = 0.08, 0.1
    m_hub = dens_hub * math.pi * hub_r ** 2 * hub_h
    I_axis = (n_blades * (m_blade * radius_centroid_m ** 2 + I_own_pitched)
              + 0.5 * m_hub * hub_r ** 2)
    mass = n_blades * m_blade + m_hub

    hub_pos = Vec3(0.0, 0.9, 0.0)                  # atop a mast, rotor axis +x
    rotor = PhysicsParcel(0.05, _DensityMaterial(dens_blade), mass=mass,
                          position=hub_pos,
                          inertia_body=(I_axis, I_axis, I_axis),
                          label="rotor")
    bearing = RigidBearing(rotor, hub_pos, Vec3(1.0, 0.0, 0.0))
    from ..dynamics.mechanisms.plug import Plug
    support = Plug.support("rotor held on an ideal rigid bearing atop the "
                           "mast (mast rendered, not structurally simulated)")
    scene = PhysicsScene([rotor], gravity=Vec3(0.0, -9.80665, 0.0), ground=False)
    wind = RotorWind(rotor, blades, Vec3(wind_speed_m_s, 0.0, 0.0))
    w_star = terminal_omega(wind_speed_m_s, beta, radius_centroid_m)

    def _pose():
        return [{"pos": [rotor.position.x, rotor.position.y, rotor.position.z],
                 "quat": [round(v, 6) for v in rotor.orientation]}]

    frames = [{"t_sim": 0.0, "bodies": _pose()}]
    next_rec = frame_dt
    t = 0.0
    omega_trace = [0.0]
    while t < t_max:
        step(scene, dt=dt, sub_steps=1, external_forces=wind)
        bearing.project()
        t += dt
        if t >= next_rec:
            frames.append({"t_sim": round(t, 4), "bodies": _pose()})
            next_rec += frame_dt
            omega_trace.append(round(bearing.omega(), 5))
    frames.append({"t_sim": round(t, 4), "bodies": _pose()})
    t_end = t

    # ── render: blades from the SAME azimuth+pitch construction ──
    leaves = []
    for b in range(n_blades):
        phi = 2.0 * math.pi * b / n_blades
        q_b = quat_mul(quat_from_axis_angle((1.0, 0.0, 0.0), phi),
                      quat_from_axis_angle((0.0, 1.0, 0.0), -beta))
        leaves.append({"op": "add", "material": blade_material, "body": 0,
                       "shape": {"type": "Rotated", "center": [0.0, 0.0, 0.0],
                                 "rot": [round(v, 8) for v in q_b],
                                 "shape": {"type": "Box",
                                           "center": [0.0, radius_centroid_m, 0.0],
                                           "x": blade_thickness_m,
                                           "y": blade_length_m,
                                           "z": blade_chord_m}}})
    # hub: cylinder along the rotor axis (+x — the canonical Cylinder is
    # z-axis, so wrap it in the same Rotated node the blades use)
    q_hub = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
    leaves.append({"op": "add", "material": "iron", "body": 0,
                   "shape": {"type": "Rotated", "center": [0.0, 0.0, 0.0],
                             "rot": [round(v, 8) for v in q_hub],
                             "shape": {"type": "Cylinder", "center": [0, 0, 0],
                                       "radius": hub_r, "height": hub_h}}})
    # mast: STATIC world geometry (no body), from ground to the hub
    leaves.append({"op": "add", "material": "iron",
                   "shape": {"type": "Box",
                             "center": [-0.08, hub_pos.y / 2.0, 0.0],
                             "x": 0.06, "y": hub_pos.y, "z": 0.06}})

    r_tip = radius_centroid_m + blade_length_m / 2.0
    materials = {blade_material: _bake_material(blade_material, dens_blade),
                "iron": _bake_material("iron", dens_hub)}
    scene_spec = {
        "name": f"windmill — NATURAL drive: {wind_speed_m_s:g} m/s wind on "
                f"{n_blades} pitched blades (nothing plugged)",
        "bodies": [{"pivot": [0, 0, 0], "label": "rotor"}],
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [I_axis, I_axis, I_axis]},
        "bbox": [[-0.3, 0.3], [-0.05, hub_pos.y + r_tip + 0.05],
                 [-r_tip - 0.05, r_tip + 0.05]],
        "camera": {"target": [0.0, hub_pos.y, 0.0],
                   "orbit_radius": max(3.0 * r_tip, 0.6),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0],
                   "az0": 1.2, "el0": 0.1},
        "identified": True,
        # NATURAL drive (the wind is the mover, computed by the physics) —
        # but the HOLDER is still artificial: the mast is rendered, not
        # structurally simulated, so the bearing is a plugged support
        "plugs": [support.to_dict()],
        "choices": [blade_material_choice.to_dict(),
                   structure_material_choice.to_dict(),
                   geometry_choice.to_dict(), wind_choice.to_dict()],
        "variants": [
            {"label": "wind 5 m/s", "slug": "windmill_5ms", "value": 5.0},
            {"label": "wind 10 m/s", "slug": "windmill_10ms", "value": 10.0},
        ],
        "variant_current": f"windmill_{wind_speed_m_s:g}ms",
        "source": f"NATURAL drive — flat-plate wind force on pitched blades "
                  f"(dynamics/mechanisms/wind.py, SIMPLIFIED_MODEL flagged "
                  f"there; closed-form static torque and terminal tip speed "
                  f"gated in tests/test_mechanism_wind.py). Wind "
                  f"{wind_speed_m_s:g} m/s along +x; blade pitch "
                  f"{blade_pitch_deg:g} deg; terminal omega* = "
                  f"{w_star:.2f} rad/s emerges, not prescribed. Mounted on "
                  f"RigidBearing (see KNOWN_GAPS.md). Nothing plugged. "
                  + blade_material_choice.cite() + " "
                  + structure_material_choice.cite() + " "
                  + geometry_choice.cite() + " " + wind_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": ["rotor"],
            "validation": {
                "wind_speed_m_s": wind_speed_m_s,
                "blade_pitch_deg": blade_pitch_deg,
                "terminal_omega_expected_rad_s": w_star,
                "omega_trace_rad_s": omega_trace,
                "final_omega_rad_s": bearing.omega(),
                "bearing_absorbed_energy_j": bearing.absorbed_energy_j,
                "spin_energy_j": 0.5 * I_axis * bearing.omega() ** 2,
            },
        },
        "kind": "trajectory",
    }


def record_windmill_theater(wind_speed_m_s: float = 10.0,
                            n_blades: int = 4,
                            blade_length_m: float = 0.5,
                            blade_chord_m: float = 0.15,
                            blade_thickness_m: float = 0.004,
                            blade_pitch_deg: float = 45.0,
                            radius_centroid_m: float = 0.6,
                            blade_material: str = "aluminum",
                            r_crank_m: float = 0.03,
                            l_rod_m: float = 0.09,
                            piston_radius_m: float = 0.01,
                            dt: float = 1.0 / 960.0, t_max: float = 40.0,
                            frame_dt: float = 0.05,
                            target_watch_s: float = 15.0) -> dict:
    """THE WINDMILL THEATER — the vision statement's second worked example,
    finished: "a metal windmill turning a gearset and reciprocating a water
    pump connected to a reservoir of water." Wind (NATURAL, record_
    windmill_spinup's aero model) spins a rotor on a RigidBearing; a
    BearingGearCoupling (Arc A Phase 1, load-blind — flagged in
    KNOWN_GAPS.md) hands that rate to a real 2-stage InvoluteGear spur
    train (Phase 2); the final arbor IS a crank (its own gear-derived mass,
    CoM offset by r_crank_m from its own hinge) driving a real four-bar
    slider-crank (Phase 0: RevoluteJoint -> RevoluteJoint -> RevoluteJoint
    + unmotored PrismaticJoint, gated against the closed form s(theta) =
    r*cos(theta) + sqrt(l^2 - r^2*sin(theta)^2)); a ReciprocatingPumpState
    (Phase 4, SIMPLIFIED_MODEL — pure volume bookkeeping, no fluid
    dynamics) watches the piston's stroke reversals and fills a reservoir.

    The reservoir's water level is COSMETIC: a thin slab rides a hand-
    computed (not PhysicsParcel, not solver-tracked) body whose Y position
    tracks pump.volume_m3 / tank_footprint_area each frame — reusing the
    existing per-body pos/quat animation channel rather than adding a new
    per-leaf-parameter one (see KNOWN_GAPS.md and dynamics/mechanisms/
    pump.py's own docstring for why).
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step
    from ..dynamics.quat import quat_from_axis_angle, quat_mul, twist_angle
    from ..dynamics.joints import RevoluteJoint, GearCouplingJoint, PrismaticJoint
    from ..dynamics.mechanisms.wind import (RotorWind, build_rotor_blades,
                                            terminal_omega)
    from ..dynamics.mechanisms.bearing import RigidBearing
    from ..dynamics.mechanisms.bearing_gear_coupling import BearingGearCoupling
    from ..dynamics.mechanisms.pump import ReciprocatingPumpState
    from ..dynamics.mechanisms.plug import Plug
    from ..kernel.gear import InvoluteGear
    from ..dynamics.mechanisms.choice import Choice
    import math

    AXIS = Vec3(1.0, 0.0, 0.0)
    SLIDE_AXIS = Vec3(0.0, 1.0, 0.0)
    MODULE_M = 0.002
    PRESSURE_ANGLE = math.radians(20.0)
    FACE_WIDTH_M = 0.006
    TEETH_P0 = 12
    TEETH_W1, TEETH_P1 = 60, 15
    TEETH_W2 = 45
    COUPLING_RATIO = 4.0

    structure_material_choice = Choice(
        "hub/mast/gearset/crank/rod/piston/tank material", "iron",
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild", "nickel", "titanium"],
        reason="machined mechanical hardware is conventionally mild steel, "
              "not unalloyed iron; steel_mild was never checked before "
              "defaulting to iron across every structural part in this "
              "scene")
    blade_material_choice = Choice(
        "windmill blade material", blade_material,
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["titanium", "steel_mild", "carbon_fiber", "wood_oak"],
        reason="lightest structural metal in the table, consistent with "
              "the docstring's stated \"metal windmill\" design intent; "
              "lighter composites were passed over only because the brief "
              "specifies metal")
    drivetrain_geometry_choice = Choice(
        "gearset/crank/rod/piston geometry (crank radius, rod length, "
        "piston radius, gear face width, tooth counts P0/W1/P1/W2, "
        "bearing-to-gearset coupling ratio)",
        {"r_crank_m": r_crank_m, "l_rod_m": l_rod_m,
         "piston_radius_m": piston_radius_m, "face_width_m": FACE_WIDTH_M,
         "teeth": [TEETH_P0, TEETH_W1, TEETH_P1, TEETH_W2],
         "coupling_ratio": COUPLING_RATIO},
        checked="sigma_ground/blueprint/catalog/ (no windmill/pump "
        "mechanism entry exists, unlike the clock's Kelly-1944 catalog "
        "entry)",
        reason="genuinely novel drivetrain with no blueprint to ground "
              "it; l_rod_m/r_crank_m=3.0 sits near the conventional "
              "minimum ratio for limiting slider-crank angularity, but "
              "that reasoning was never stated in code before this flag")
    motor_torque_choice = Choice(
        "follower motor torque cap (N*m) on the load-blind "
        "BearingGearCoupling", 50.0,
        reason="arbitrary cap on the coupling's ability to track its "
              "commanded rate (see KNOWN_GAPS.md's load-blind entry); an "
              "under/over-generous cap could mask or exaggerate that "
              "coupling's own physics gap")

    def _gear(teeth):
        return InvoluteGear(module=MODULE_M, teeth=teeth,
                            pressure_angle=PRESSURE_ANGLE,
                            face_width=FACE_WIDTH_M, grid_resolution=36)

    def _cd(na, nb):
        return MODULE_M * (na + nb) / 2.0

    # ── rotor: identical construction to record_windmill_spinup ──
    beta = math.radians(blade_pitch_deg)
    dens_blade, _ = _material_density(blade_material, 288.15)
    dens_hub, _ = _material_density("iron", 288.15)
    blades, area = build_rotor_blades(n_blades, radius_centroid_m,
                                      blade_length_m, blade_chord_m, beta)
    m_blade = dens_blade * blade_length_m * blade_chord_m * blade_thickness_m
    t2, L2, c2 = blade_thickness_m ** 2, blade_length_m ** 2, blade_chord_m ** 2
    I_own_x = m_blade * (L2 + c2) / 12.0
    I_own_z = m_blade * (t2 + L2) / 12.0
    I_own_pitched = (math.cos(beta) ** 2 * I_own_x
                     + math.sin(beta) ** 2 * I_own_z)
    hub_r, hub_h = 0.08, 0.1
    blade_geometry_choice = Choice(
        "rotor/blade geometry (count, length, chord, thickness, pitch, "
        "centroid radius) and hub cylinder size — shared with "
        "record_windmill_spinup",
        {"n_blades": n_blades, "blade_length_m": blade_length_m,
         "blade_chord_m": blade_chord_m,
         "blade_thickness_m": blade_thickness_m,
         "blade_pitch_deg": blade_pitch_deg,
         "radius_centroid_m": radius_centroid_m,
         "hub_r_m": hub_r, "hub_h_m": hub_h},
        checked="sigma_ground/blueprint/catalog/ (no wind-rotor design "
        "catalog in this codebase)",
        reason="drives the real aerodynamic torque model and rotor "
              "mass/inertia directly; no rotor-design catalog exists to "
              "check against")
    m_hub = dens_hub * math.pi * hub_r ** 2 * hub_h
    I_axis = (n_blades * (m_blade * radius_centroid_m ** 2 + I_own_pitched)
              + 0.5 * m_hub * hub_r ** 2)
    rotor_mass = n_blades * m_blade + m_hub
    hub_pos = Vec3(0.0, 0.9, 0.0)
    rotor = PhysicsParcel(0.05, _DensityMaterial(dens_blade), mass=rotor_mass,
                          position=hub_pos,
                          inertia_body=(I_axis, I_axis, I_axis), label="rotor")
    bearing = RigidBearing(rotor, hub_pos, AXIS)
    wind = RotorWind(rotor, blades, Vec3(wind_speed_m_s, 0.0, 0.0))
    w_star = terminal_omega(wind_speed_m_s, beta, radius_centroid_m)
    support_rotor = Plug.support("rotor held on an ideal rigid bearing atop "
                                 "the mast (mast rendered, not structurally "
                                 "simulated)")

    # ── gearset: Arc A Phase 2 layout, axis parallel to the rotor ──
    dens_iron, _ = _material_density("iron", 288.15)
    g_p0, g_w1, g_p1, g_w2 = (_gear(TEETH_P0), _gear(TEETH_W1),
                              _gear(TEETH_P1), _gear(TEETH_W2))
    y0 = -0.6
    y1 = y0 + _cd(TEETH_P0, TEETH_W1)
    y2 = y1 + _cd(TEETH_P1, TEETH_W2)
    P = Vec3(0.0, y2, 0.0)                       # crank pivot

    def _arbor(y, gears, label):
        m = sum(dens_iron * g.volume() for g in gears)
        iz = sum(dens_iron * g.volume() * g.inertia_factor("z") for g in gears)
        pos = Vec3(0.0, y, 0.0)
        return PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=m,
                             position=pos, inertia_body=(iz, iz, iz), label=label)

    arbor0 = _arbor(y0, [g_p0], "arbor0")
    arbor1 = _arbor(y1, [g_w1, g_p1], "arbor1")

    # arbor2 IS the crank: same gear-derived mass/inertia, CoM offset by
    # r_crank_m from its own hinge anchor P (Arc A Phase 4)
    crank_tip0 = P + Vec3(0.0, r_crank_m, 0.0)
    m_w2 = dens_iron * g_w2.volume()
    iz_w2 = dens_iron * g_w2.volume() * g_w2.inertia_factor("z")
    arbor2 = PhysicsParcel(0.01, _DensityMaterial(dens_iron), mass=m_w2,
                           position=crank_tip0,
                           inertia_body=(iz_w2, iz_w2, iz_w2), label="crank")

    j0 = RevoluteJoint(arbor0, None, arbor0.position, AXIS)
    j1 = RevoluteJoint(arbor1, None, arbor1.position, AXIS)
    j2 = RevoluteJoint(arbor2, None, P, AXIS)
    ratio01 = TEETH_W1 / TEETH_P0
    ratio12 = TEETH_W2 / TEETH_P1
    c01 = GearCouplingJoint(j0, j1, ratio01)
    c12 = GearCouplingJoint(j1, j2, ratio12)
    coupling = BearingGearCoupling(bearing, j0, COUPLING_RATIO,
                                   motor_max_torque=motor_torque_choice.value)
    support_gearset = Plug.support(
        "gearset arbors, crank and piston slide held as if by a mounted "
        "frame/cylinder guide, without modeling the frame")

    # ── slider-crank + pump: Arc A Phase 0 / 4 ──
    rod_piston_mass_choice = Choice(
        "rod and piston mass (kg)", [0.01, 0.02],
        checked="none — unlike every other body in this scene, these are "
        "picked directly rather than derived from material density x "
        "geometry",
        reason="picked to give plausible slider-crank inertia at demo "
              "scale; affects real dynamics (piston motion, stroke timing)")
    piston0 = P + Vec3(0.0, r_crank_m + l_rod_m, 0.0)
    rod_mid0 = P + Vec3(0.0, r_crank_m + 0.5 * l_rod_m, 0.0)
    rod = PhysicsParcel(0.005, _DensityMaterial(dens_iron),
                        mass=rod_piston_mass_choice.value[0],
                        position=rod_mid0, label="rod")
    piston = PhysicsParcel(0.005, _DensityMaterial(dens_iron),
                           mass=rod_piston_mass_choice.value[1],
                           position=piston0, label="piston")
    j_rod_crank = RevoluteJoint(rod, arbor2, crank_tip0, AXIS)
    j_rod_piston = RevoluteJoint(rod, piston, piston0, AXIS)
    j_slide = PrismaticJoint(piston, None, piston0, SLIDE_AXIS)

    piston_area = math.pi * piston_radius_m ** 2
    stroke_length = 2.0 * r_crank_m
    pump = ReciprocatingPumpState(j_slide, piston_area, stroke_length)

    # ── reservoir: fixed tank footprint, water level tracked from volume ──
    tank_choice = Choice(
        "reservoir tank dimensions (width, depth, height, wall thickness, "
        "x-position)", [0.08, 0.08, 0.06, 0.005, 0.15],
        reason="sized to look plausible next to the mechanism and give a "
              "visible fill over the simulated run length; feeds real "
              "physics (tank_footprint_area used in the water-level "
              "calculation), so not purely cosmetic; no catalog for "
              "reservoir sizing exists")
    tank_w, tank_d, tank_h, wall_t, tank_x = tank_choice.value
    tank_footprint_area = (tank_w - 2.0 * wall_t) * (tank_d - 2.0 * wall_t)

    all_parcels = [rotor, arbor0, arbor1, arbor2, rod, piston]
    all_constraints = [j0, j1, j2, c01, c12, j_rod_crank, j_rod_piston, j_slide]
    scene = PhysicsScene(all_parcels, gravity=Vec3(0.0, -9.80665, 0.0),
                         ground=False, constraints=all_constraints)
    scene.solver_iterations = 20

    def _poses():
        bodies = [{"pos": [p.position.x, p.position.y, p.position.z],
                  "quat": [round(v, 6) for v in p.orientation]}
                 for p in all_parcels]
        depth = min(tank_h, pump.volume_m3 / tank_footprint_area)
        bodies.append({"pos": [tank_x, wall_t + depth, 0.0],
                       "quat": [0.0, 0.0, 0.0, 1.0]})
        return bodies

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    t = 0.0
    omega_trace = [0.0]
    while t < t_max:
        coupling.step()
        step(scene, dt=dt, sub_steps=1, external_forces=wind)
        bearing.project()
        pump.step()
        t += dt
        if t >= next_rec:
            frames.append({"t_sim": round(t, 4), "bodies": _poses()})
            next_rec += frame_dt
            omega_trace.append(round(bearing.omega(), 5))
    frames.append({"t_sim": round(t, 4), "bodies": _poses()})
    t_end = t
    water_body = len(all_parcels)                # index of the cosmetic body

    # ── render ──
    leaves = []
    for b in range(n_blades):
        phi = 2.0 * math.pi * b / n_blades
        q_b = quat_mul(quat_from_axis_angle((1.0, 0.0, 0.0), phi),
                      quat_from_axis_angle((0.0, 1.0, 0.0), -beta))
        leaves.append({"op": "add", "material": blade_material, "body": 0,
                       "shape": {"type": "Rotated", "center": [0.0, 0.0, 0.0],
                                 "rot": [round(v, 8) for v in q_b],
                                 "shape": {"type": "Box",
                                           "center": [0.0, radius_centroid_m, 0.0],
                                           "x": blade_thickness_m,
                                           "y": blade_length_m,
                                           "z": blade_chord_m}}})
    q_hub = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
    leaves.append({"op": "add", "material": "iron", "body": 0,
                   "shape": {"type": "Rotated", "center": [0.0, 0.0, 0.0],
                             "rot": [round(v, 8) for v in q_hub],
                             "shape": {"type": "Cylinder", "center": [0, 0, 0],
                                       "radius": hub_r, "height": hub_h}}})
    leaves.append({"op": "add", "material": "iron",
                   "shape": {"type": "Box",
                             "center": [-0.08, hub_pos.y / 2.0, 0.0],
                             "x": 0.06, "y": hub_pos.y, "z": 0.06}})

    # gears — InvoluteGear's native axis is local z; wrap in the SAME
    # Rotated node the hub above uses to reorient onto the shared spin axis
    q_gear = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
    z_plane = 2.0 * FACE_WIDTH_M

    def _gear_leaf(gear, body, plane, mk):
        return {"op": "add", "material": mk, "body": body,
                "shape": {"type": "Rotated",
                          "center": [plane * z_plane, 0.0, 0.0],
                          "rot": [round(v, 8) for v in q_gear],
                          "shape": {"type": "Gear", "module": gear.module,
                                    "teeth": gear.teeth,
                                    "pressure_angle": gear.pressure_angle,
                                    "addendum_coeff": gear.addendum_coeff,
                                    "dedendum_coeff": gear.dedendum_coeff,
                                    "fillet_coeff": gear.fillet_coeff,
                                    "face_width": gear.face_width,
                                    "center": [0.0, 0.0, 0.0],
                                    "source": f"teeth={gear.teeth} "
                                              "[estimated] (windmill "
                                              "drivetrain, no cited "
                                              "blueprint source)"}}}

    leaves += [_gear_leaf(g_p0, 1, 0, "iron"),
              _gear_leaf(g_w1, 2, 0, "iron"),
              _gear_leaf(g_p1, 2, 1, "iron"),
              _gear_leaf(g_w2, 3, 1, "iron")]
    # crank arm: box from the crank body's own CoM (at the tip) back to its
    # local anchor offset (the pivot direction is FIXED in local frame —
    # always (0,-r_crank_m,0) — so this box always spans pivot-to-tip)
    leaves.append({"op": "add", "material": "iron", "body": 3,
                   "shape": {"type": "Box",
                             "center": [0.0, -0.5 * r_crank_m, 0.0],
                             "x": 0.01, "y": r_crank_m, "z": 0.01}})
    # rod: symmetric about its own CoM by construction (rod_mid0 is exactly
    # the midpoint of crank_tip0/piston0), so a plain axis-aligned box
    # spanning +/- l_rod_m/2 along local y tracks the rod's real motion
    leaves.append({"op": "add", "material": "iron", "body": 4,
                   "shape": {"type": "Box", "center": [0.0, 0.0, 0.0],
                             "x": 0.008, "y": l_rod_m, "z": 0.008}})
    # piston: centered on its own CoM, never rotates (PrismaticJoint locks
    # angular velocity to zero relative to world)
    leaves.append({"op": "add", "material": "iron", "body": 5,
                   "shape": {"type": "Box", "center": [0.0, 0.0, 0.0],
                             "x": 0.03, "y": 2.0 * piston_radius_m,
                             "z": 2.0 * piston_radius_m}})

    # tank: open-top container (solid box minus an inset cavity), STATIC
    # world geometry (no body key)
    leaves.append({"op": "add", "material": "iron",
                   "shape": {"type": "Box",
                             "center": [tank_x, 0.5 * tank_h, 0.0],
                             "x": tank_w, "y": tank_h, "z": tank_d}})
    leaves.append({"op": "subtract", "material": "air",
                   "shape": {"type": "Box",
                             "center": [tank_x, wall_t + 0.5 * tank_h, 0.0],
                             "x": tank_w - 2.0 * wall_t, "y": tank_h,
                             "z": tank_d - 2.0 * wall_t}})
    # water: thin slab riding the cosmetic body, sized to the tank's cavity
    leaves.append({"op": "add", "material": "water", "body": water_body,
                   "shape": {"type": "Box", "center": [0.0, 0.0, 0.0],
                             "x": tank_w - 2.0 * wall_t - 0.002,
                             "y": 0.004,
                             "z": tank_d - 2.0 * wall_t - 0.002}})

    materials = {blade_material: _bake_material(blade_material, dens_blade),
                "iron": _bake_material("iron", dens_iron),
                "water": _bake_material("water", 1000.0),
                "air": _bake_material("air", 1.225)}

    r_tip = radius_centroid_m + blade_length_m / 2.0
    x_lo = min(-0.3, -r_crank_m - l_rod_m - 0.05)
    x_hi = max(0.3, tank_x + 0.5 * tank_w + 0.05)
    y_lo = min(-0.05, y0 - 0.1)
    y_hi = max(hub_pos.y + r_tip + 0.05, P.y + r_crank_m + l_rod_m + 0.05)
    scene_spec = {
        "name": f"windmill theater — {wind_speed_m_s:g} m/s wind turns a "
                "gearset, reciprocates a pump, fills a reservoir",
        "bodies": ([{"pivot": [0, 0, 0], "label": "rotor"},
                    {"pivot": [0, 0, 0], "label": "arbor0"},
                    {"pivot": [0, 0, 0], "label": "arbor1"},
                    {"pivot": [0, 0, 0], "label": "crank"},
                    {"pivot": [0, 0, 0], "label": "rod"},
                    {"pivot": [0, 0, 0], "label": "piston"},
                    {"pivot": [0, 0, 0], "label": "water"}]),
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": sum(p.mass for p in all_parcels),
                    "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[x_lo, x_hi], [y_lo, y_hi], [-r_tip - 0.05, r_tip + 0.05]],
        "camera": {"target": [0.0, 0.5 * (y_lo + y_hi), 0.0],
                   "orbit_radius": max(1.2 * (x_hi - x_lo), 1.0 * (y_hi - y_lo), 0.6),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0],
                   "az0": 0.9, "el0": 0.15},
        "identified": True,
        # NATURAL drive (wind, computed by the physics) — but every HOLDER
        # is artificial: the mast/frame/guide aren't structurally modeled
        "plugs": [support_rotor.to_dict(), support_gearset.to_dict()],
        "choices": [structure_material_choice.to_dict(),
                   blade_material_choice.to_dict(),
                   blade_geometry_choice.to_dict(),
                   drivetrain_geometry_choice.to_dict(),
                   motor_torque_choice.to_dict(),
                   rod_piston_mass_choice.to_dict(), tank_choice.to_dict()],
        "variants": [
            {"label": "wind 5 m/s", "slug": "windmill_theater_5ms", "value": 5.0},
            {"label": "wind 10 m/s", "slug": "windmill_theater_10ms", "value": 10.0},
        ],
        "variant_current": f"windmill_theater_{wind_speed_m_s:g}ms",
        "source": f"NATURAL drive — flat-plate wind (dynamics/mechanisms/"
                  f"wind.py, SIMPLIFIED_MODEL flagged there) on {n_blades} "
                  f"pitched blades at {wind_speed_m_s:g} m/s; terminal "
                  f"omega* = {w_star:.2f} rad/s emerges, not prescribed. "
                  "Gearset teeth [estimated] (no cited blueprint source, "
                  "unlike the clock's Kelly-1944 data). PHYSICS_GAP: "
                  "bearing-to-gearset coupling is load-blind (KNOWN_GAPS.md)"
                  " — the rotor spins up as if driving nothing. "
                  "SIMPLIFIED_MODEL: the pump is pure positive-displacement "
                  "volume bookkeeping (dynamics/mechanisms/pump.py) — no "
                  "fluid inertia, viscosity, or back-pressure, and it "
                  "cannot load the crank. " + support_rotor.cite() + " "
                  + support_gearset.cite() + " "
                  + structure_material_choice.cite() + " "
                  + blade_material_choice.cite() + " "
                  + blade_geometry_choice.cite() + " "
                  + drivetrain_geometry_choice.cite() + " "
                  + motor_torque_choice.cite() + " "
                  + rod_piston_mass_choice.cite() + " " + tank_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": ["rotor", "arbor0", "arbor1", "crank", "rod",
                           "piston", "water"],
            "validation": {
                "wind_speed_m_s": wind_speed_m_s,
                "terminal_omega_expected_rad_s": w_star,
                "omega_trace_rad_s": omega_trace,
                "final_omega_rad_s": bearing.omega(),
                "bearing_absorbed_energy_j": bearing.absorbed_energy_j,
                "pump_strokes": pump.strokes,
                "pump_volume_m3": pump.volume_m3,
                "tank_capacity_m3": tank_footprint_area * tank_h,
                "cumulative_gear_ratio": COUPLING_RATIO / (ratio01 * ratio12),
            },
        },
        "kind": "trajectory",
    }


def record_escapement_clock(pendulum_length_m: float = 0.4,
                            pendulum_mass_kg: float = 0.05,
                            release_deg: float = 6.0,
                            escape_teeth: int = 30,
                            escape_wheel_radius_m: float = 0.03,
                            escape_wheel_mass_kg: float = 0.01,
                            escape_wheel_height_m: float = 0.008,
                            spring_k: float = 6.0e-4,
                            spring_theta_wound0: float = 5.0,
                            dt: float = 1.0 / 1920.0, t_max: float = 12.0,
                            frame_dt: float = 0.02,
                            target_watch_s: float = 12.0) -> dict:
    """The Phase 3 capstone: a spring-driven escape wheel gated by a real
    pendulum — dynamics/mechanisms/spring.py's MainspringState (a Hooke's-
    law torque cap, reusing RevoluteJoint's motor as-is) driving an escape
    wheel through dynamics/mechanisms/escapement.py's Escapement (one
    tooth pitch released per pendulum half-swing). Both pieces are
    independently gated against closed-form solutions
    (tests/test_mechanism_spring.py, tests/test_mechanism_escapement.py)
    BEFORE being wired together here, per the plan's own de-risking order
    for this phase — the highest-risk phase, with no reference
    implementation to lean on.

    Still placeholder shapes (a bare disc for the escape wheel, a box+ball
    for the pendulum) and NOT yet coupled to a real gear train (Phase 1's
    GearCouplingJoint) or real tooth geometry (Phase 4) — this proves the
    spring+escapement physics alone, watchably: the pendulum swings, the
    escape wheel ticks forward exactly one tooth per swing, and (given
    enough sim time relative to spring_theta_wound0) the spring runs down
    and the ticking visibly slows and stops, same as winding down a real
    clock.

    spring_theta_wound0 defaults to 5.0 rad: a PROPERLY GATED escapement
    unwinds only one tooth pitch (2pi/30 here) per tick, ~0.5 rad/s of
    sim time — an earlier 60-rad default only appeared to run down in 12 s
    because the escape wheel was silently free-running past its stop (the
    angle-wrap brake bug fixed in mechanisms/escapement.py); with the brake
    actually holding, 5 rad exhausts in ~10 s and the wind-down story
    honestly completes inside the window.
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.shapes import Cylinder
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import RevoluteJoint
    from ..dynamics.stepper import step
    from ..dynamics.quat import qrot, quat_from_axis_angle
    from ..dynamics.mechanisms.spring import MainspringState
    from ..dynamics.mechanisms.escapement import Escapement
    from ..dynamics.mechanisms.choice import Choice
    import math

    axis = Vec3(0.0, 0.0, 1.0)
    L = pendulum_length_m

    escape_teeth_choice = Choice(
        "escape wheel tooth count", escape_teeth,
        checked="sigma_ground/blueprint/catalog/"
        "kelly_1944_watch_going_train_18000bph.md",
        alternatives=[15],
        reason="the catalog cites 15 (Kelly 1944 p.18, conf 0.95) for a real "
              "escapement, used later by record_clock — this standalone "
              "phase kept 30 for its own tick pacing within the demo window "
              "rather than the cited value")
    pendulum_choice = Choice(
        "pendulum length/mass and release angle",
        [pendulum_length_m, pendulum_mass_kg, release_deg],
        checked="none — this phase proves spring+escapement physics only, "
        "not real timekeeping (record_clock later derives a cited length "
        "from a real cadence)",
        reason="round demo numbers picked to swing plausibly and stay in "
              "the small-angle regime; no timing target applies here")
    spring_choice = Choice(
        "mainspring stiffness k and initial wound angle (rad)",
        [spring_k, spring_theta_wound0],
        checked="dynamics/mechanisms/spring.py (MVP linear Hooke's-law "
        "model, no cited real mainspring curve)",
        reason="tuned so the spring visibly winds down within the demo "
              "window given escape_teeth's tooth pitch; not itself sourced")

    # ── pendulum: built AT VERTICAL (angle()==0 is TRUE vertical, the
    # escapement's zero-crossing trigger) then displaced to the release
    # angle — angle() reads 0 at CONSTRUCTION, so building it already
    # tilted would put "zero" at the release point, never crossed again.
    pivot = Vec3(0.0, 0.55, 0.0)
    mat_pend = "iron"
    mat_pend_choice = Choice(
        "pendulum rod material", mat_pend,
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild", "wood_oak"],
        reason="iron is a reasonable generic metal and nearly identical to "
              "steel_mild in this table; wood_oak is the more period-"
              "accurate material for a real precision pendulum rod and was "
              "not checked")
    density_pend, _ = _material_density(mat_pend, 288.15)
    q0 = list(quat_from_axis_angle((0.0, 0.0, 1.0), 0.0))
    off0 = qrot(q0, (0.0, -0.5 * L, 0.0))          # pivot -> CoM, world (matches
    rod = PhysicsParcel(0.5 * L, _DensityMaterial(density_pend),           # test_joints.py's _rod helper)
                        mass=pendulum_mass_kg,
                        position=pivot + Vec3(off0[0], off0[1], off0[2]),
                        orientation=q0,
                        inertia_body=(pendulum_mass_kg * L * L / 12.0,
                                     1.0e-3 * pendulum_mass_kg,
                                     pendulum_mass_kg * L * L / 12.0),
                        label="pendulum")
    pend_joint = RevoluteJoint(rod, None, pivot, axis)
    qr = list(quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(release_deg)))
    off_r = qrot(qr, (0.0, -0.5 * L, 0.0))
    rod.position = pivot + Vec3(off_r[0], off_r[1], off_r[2])
    rod.orientation = qr

    # ── escape wheel: a placeholder disc, driven by the spring's motor ──
    r = escape_wheel_radius_m
    mat_esc = "copper"
    mat_esc_choice = Choice(
        "escape wheel material", mat_esc,
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["iron", "steel_mild", "nickel", "titanium"],
        reason="copper is soft and not structurally realistic for a real "
              "wear-bearing escapement wheel (brass/hardened steel are "
              "standard); steel_mild is the closer real-world fit and was "
              "not compared before picking")
    escape_wheel_size_choice = Choice(
        "escape wheel radius/mass/height (dynamical mass is derived from "
        "this mass, NOT from mat_esc's density — the material choice only "
        "affects the rendered color)",
        [escape_wheel_radius_m, escape_wheel_mass_kg, escape_wheel_height_m],
        checked="blueprint/catalog kelly_1944 (module_mm/face_width_mm null "
        "for this train — GEOMETRY_GAP)",
        reason="no cited module or absolute size exists for this gear "
              "train; picked to look plausible at demo scale")
    density_esc, _ = _material_density(mat_esc, 288.15)
    esc_shape = Cylinder(r, escape_wheel_height_m)
    esc_pos = Vec3(0.35, 0.55, 0.0)
    wheel = PhysicsParcel(esc_shape, _DensityMaterial(
        escape_wheel_mass_kg / esc_shape.volume()), position=esc_pos,
        label="escape_wheel")
    esc_joint = RevoluteJoint(wheel, None, wheel.position, axis,
                              motor_speed=0.0, motor_max_torque=0.0)

    scene = PhysicsScene([rod, wheel], ground=False,
                         constraints=[pend_joint, esc_joint])
    spring = MainspringState(esc_joint, k=spring_k, theta_wound0=spring_theta_wound0)
    escapement = Escapement(esc_joint, pend_joint, teeth=escape_teeth, direction=-1.0)
    from ..dynamics.mechanisms.plug import Plug
    support = Plug.support("escape-wheel arbor and pendulum pivot held as if "
                           "by a frame, without the frame")

    def _poses():
        return [{"pos": [rod.position.x, rod.position.y, rod.position.z],
                 "quat": [round(v, 6) for v in rod.orientation]},
                {"pos": [wheel.position.x, wheel.position.y, wheel.position.z],
                 "quat": [round(v, 6) for v in wheel.orientation]}]

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    t = 0.0
    tick_frames = []
    while t < t_max:
        spring.drive(direction=-1.0)
        step(scene, dt=dt, sub_steps=1)
        t += dt
        if escapement.step(t):
            tick_frames.append(round(t, 4))
        if t >= next_rec:
            frames.append({"t_sim": round(t, 4), "bodies": _poses()})
            next_rec += frame_dt
    frames.append({"t_sim": round(t, 4), "bodies": _poses()})
    t_end = t
    de = scene.total_kinetic_energy() - e0

    bodies_spec = [{"pivot": [0, 0, 0], "label": "pendulum"},
                  {"pivot": [0, 0, 0], "label": "escape_wheel"}]
    bob_r = 0.12 * L
    leaves = [
        {"op": "add", "material": mat_pend, "body": 0,
         "shape": {"type": "Box", "center": [0.0, 0.0, 0.0],
                   "x": 0.06 * L, "y": L, "z": 0.06 * L}},
        {"op": "add", "material": mat_pend, "body": 0,
         "shape": {"type": "Sphere", "center": [0.0, -0.5 * L, 0.0],
                   "radius": bob_r}},
        {"op": "add", "material": mat_esc, "body": 1,
         "shape": {"type": "Cylinder", "center": [0, 0, 0],
                   "radius": r, "height": escape_wheel_height_m}},
        *_radius_arrow_leaves(r, escape_wheel_height_m, body=1),
    ]
    materials = {
        mat_pend: _bake_material(mat_pend, density_pend),
        mat_esc: _bake_material(mat_esc, density_esc),
        "concrete": _bake_material("concrete"),
    }

    # The pendulum's CoM (the tracked frame position) swings near
    # pivot.y - 0.5*L; the rendered box+bob extend a further 0.5*L (box) and
    # bob_r (bob) below THAT — the full assembly spans roughly pivot.y (rod
    # top, at the pivot) down to pivot.y - L - bob_r (bob's lower edge), not
    # just a thin band at pivot height.
    x_lo, x_hi = -0.5 * L - bob_r - 0.02, esc_pos.x + r + 0.02
    y_lo, y_hi = pivot.y - L - bob_r - 0.02, pivot.y + 0.02
    scene_spec = {
        "name": "escapement clock — MainspringState + Escapement gating a "
                "pendulum-driven escape wheel (placeholder shapes, no real "
                "gear train or tooth geometry yet)",
        "bodies": bodies_spec,
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": rod.mass + wheel.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[x_lo, x_hi], [y_lo, y_hi], [-2 * bob_r, 2 * bob_r]],
        # A near-front view (small az) separates the pendulum and escape
        # wheel side-by-side instead of foreshortening one behind the
        # other; target sits above the assembly's vertical midpoint since
        # the pivot end (near-empty rod) needs less headroom than the
        # bob+wheel end does.
        "camera": {"target": [0.5 * (x_lo + x_hi), y_lo + 0.65 * (y_hi - y_lo), 0.0],
                   "orbit_radius": max(1.4 * (x_hi - x_lo), 1.1 * (y_hi - y_lo), 0.3),
                   "fov_deg": 55.0, "up": [0.0, 1.0, 0.0], "az0": 0.21, "el0": 0.17},
        "identified": True,
        # no DRIVE plugged (the mainspring is PRESENT, MODELED machinery —
        # an energy store that winds down), but the HOLDERS are artificial:
        # arbor and pendulum pivot hang on world anchors, no frame modeled
        "plugs": [support.to_dict()],
        "choices": [mat_esc_choice.to_dict(), mat_pend_choice.to_dict(),
                   escape_teeth_choice.to_dict(),
                   escape_wheel_size_choice.to_dict(),
                   spring_choice.to_dict(), pendulum_choice.to_dict()],
        "source": "MainspringState + Escapement (dynamics/mechanisms/, SOLVED not "
                  "scripted, gated independently against closed forms before being "
                  "wired together) — placeholder shapes; real gear train and tooth "
                  "geometry arrive with blueprint extraction + Phase 4. "
                  + mat_esc_choice.cite() + ". " + mat_pend_choice.cite() + ". "
                  + escape_teeth_choice.cite() + ". "
                  + escape_wheel_size_choice.cite() + ". "
                  + spring_choice.cite() + ". " + pendulum_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": ["pendulum", "escape_wheel"],
            "validation": {
                "ticks": escapement.ticks,
                "tick_times_s": tick_frames,
                "motor_work_j": esc_joint.motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= esc_joint.motor_work_j + 1e-6,
                "spring_wound_out": spring.wound_out,
                "theta_wound_remaining_rad": spring.theta_wound(),
            },
        },
        "kind": "trajectory",
    }


def record_clock(module_m: float = 0.002, face_width_m: float = 0.006,
                 release_deg: float = 6.0, spring_k: float = 0.12,
                 spring_theta_wound0: float = 30.0,
                 pendulum_mass_kg: float = 0.05,
                 dt: float = 1.0 / 960.0, t_max: float = 60.0,
                 frame_dt: float = 0.05,
                 target_watch_s: float = 15.0) -> dict:
    """THE CLOCK — Phase 5's final assembly, and the merged lane's flagship:
    every prior phase composed into one recorded, watchable movement.

      Phase 2 (cited data): the going train's tooth counts come from the
        blueprint catalog's Kelly (1944) entry, loaded and consumed as
        BlueprintFacts — barrel 72 : center 80/12 : third 75/10 :
        fourth 80/10 : escape 15/8 — never re-typed constants.
      Phase 1 (kinematics): arbors chained by GearCouplingJoint at those
        cited ratios (empirically-pinned sign convention: ratio =
        pinion_leaves/wheel_teeth, positive = counter-rotation).
      Phase 3 (energy + gating): MainspringState drives the barrel;
        Escapement gates the escape arbor against a real pendulum.
      Phase 4 (geometry): every wheel and pinion is a real InvoluteGear.

    Timekeeping is DERIVED, not tuned: the cited train gives 600 escape-
    wheel turns per center turn (cumulative_ratio over the spec's own mesh
    list) and a 15-tooth escape wheel; our escapement's anchor cadence
    (one tooth per pendulum HALF-period — deliberately NOT Kelly's lever
    double-impulse, which would give his 18,000 beats/hour; anchor cadence
    gives exactly half, 9,000/hour) then requires a 0.8 s pendulum, whose
    end-pivoted-rod length L = (3g/2)(T/2pi)^2 = 0.2385 m follows from the
    closed form already gated in tests/test_joints.py. With that L, the
    center arbor — the minute hand — turns at exactly 2pi/3600 rad/s by
    construction, accurate to the pendulum-period gate (~1%).

    Honesty flags: module (physical size) is [ESTIMATED] — Kelly cites no
    module (GEOMETRY_GAP); the pendulum length is DERIVED (cited ratios +
    closed-form pendulum), not itself cited; wheels are solid discs (no
    spoking), several times heavier than a real movement's crossed-out
    wheels, so the spring constants are sized to this build [estimated];
    the hour hand's 12:1 motion work is definitional, coupled directly
    (ratio -12: two meshes in real motion work = net co-rotation) rather
    than modeled as its own gear pair.
    """
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.gear import InvoluteGear
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import RevoluteJoint, GearCouplingJoint
    from ..dynamics.stepper import step
    from ..dynamics.quat import qrot, quat_from_axis_angle
    from ..dynamics.mechanisms.spring import MainspringState
    from ..dynamics.mechanisms.escapement import Escapement
    from ..blueprint import catalog as bp_catalog, cumulative_ratio
    import math

    spec = bp_catalog.load("kelly_1944_watch_going_train_18000bph")
    t_barrel = spec.gear("barrel").teeth.value
    t_cp = spec.gear("center_pinion").teeth.value
    t_cw = spec.gear("center_wheel").teeth.value
    t_tp = spec.gear("third_pinion").teeth.value
    t_tw = spec.gear("third_wheel").teeth.value
    t_fp = spec.gear("fourth_pinion").teeth.value
    t_fw = spec.gear("fourth_wheel").teeth.value
    t_ep = spec.gear("escape_pinion").teeth.value
    t_ew = spec.escapement.escape_wheel_teeth.value
    esc_per_center = cumulative_ratio(spec, [("center_wheel", "third_pinion"),
                                             ("third_wheel", "fourth_pinion"),
                                             ("fourth_wheel", "escape_pinion")])

    # derived pendulum (see docstring): anchor cadence, cited ratios
    g = 9.80665
    ticks_per_hour = esc_per_center * t_ew
    T_pend = 2.0 * 3600.0 / ticks_per_hour
    L = 1.5 * g * (T_pend / (2.0 * math.pi)) ** 2

    from ..dynamics.mechanisms.choice import Choice

    alpha = math.radians(20.0)          # pressure angle [estimated]: not cited
    geometry_choice = Choice(
        "gear pressure angle (deg) and face width (m)",
        [20.0, face_width_m],
        checked="blueprint kelly_1944 spec's pressure_angle_deg and "
        "face_width_mm fields (both null — same GEOMETRY_GAP class as "
        "module)", alternatives=[14.5],
        reason="20 deg is the modern ISO/AGMA standard, picked over the "
              "historically-plausible 14.5 deg alternative for a 1940s "
              "train; face width sized to look proportionate at the "
              "estimated module")
    axis = Vec3(0.0, 0.0, 1.0)
    m = module_m

    def _gear(teeth):
        # grid_resolution 36: inertia feeds the solver only weakly here (the
        # escapement's effective inertia is escape-arbor-dominated; upstream
        # arbors reflect down by ratio^2), so ~% accuracy is plenty
        return InvoluteGear(module=m, teeth=teeth, pressure_angle=alpha,
                            face_width=face_width_m, grid_resolution=36)

    # arbor layout: standard mesh center distances along x, meshes stacked
    # in alternating z-planes so wheel k and pinion k+1 share a plane
    def _cd(na, nb):
        return m * (na + nb) / 2.0

    x_barrel = 0.0
    x_center = x_barrel + _cd(t_barrel, t_cp)
    x_third = x_center + _cd(t_cw, t_tp)
    x_fourth = x_third + _cd(t_tw, t_fp)
    x_escape = x_fourth + _cd(t_fw, t_ep)
    z_plane = 2.0 * face_width_m                 # plane pitch
    y0 = 0.0

    dens_iron, _ = _material_density("iron", 288.15)
    dens_cu, _ = _material_density("copper", 288.15)
    material_choice = Choice(
        "wheel material (iron) / pinion material (copper)", ["iron", "copper"],
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["steel_mild", "nickel", "titanium"],
        reason="no brass entry exists in the table; real clock wheels are "
              "brass and pinions hardened steel (the opposite hardness "
              "split from what's used here) — steel_mild is the closer "
              "real-world match for both roles and was never compared "
              "before picking iron/copper for color contrast alone")

    # (arbor label, x, [(gear, z_plane_index, material), ...])
    arbors = [
        ("barrel", x_barrel, [(_gear(t_barrel), 0, "iron")]),
        ("center", x_center, [(_gear(t_cp), 0, "copper"), (_gear(t_cw), 1, "iron")]),
        ("third", x_third, [(_gear(t_tp), 1, "copper"), (_gear(t_tw), 2, "iron")]),
        ("fourth", x_fourth, [(_gear(t_fp), 2, "copper"), (_gear(t_fw), 3, "iron")]),
        ("escape", x_escape, [(_gear(t_ep), 3, "copper"), (_gear(t_ew), 4, "iron")]),
    ]
    dens = {"iron": dens_iron, "copper": dens_cu}

    parcels, joints = [], []
    min_dx = min(b[1] - a[1] for a, b in zip(arbors, arbors[1:]))
    for label, x, gears in arbors:
        mass = sum(dens[mk] * gr.volume() for gr, _, mk in gears)
        iz = sum(dens[mk] * gr.volume() * gr.inertia_factor("z") for gr, _, mk in gears)
        # transverse inertia approximated as the z value — those DOF are
        # locked by the revolute joint, so only the spin axis is dynamic
        pos = Vec3(x, y0, 0.0)
        # collision radius shrunk WELL below half the closest arbor spacing:
        # meshing gears overlap bounding spheres BY DESIGN (see
        # record_gear_mesh_spin docstring), and the pendulum swings close
        # behind the escape arbor — every parcel here is fully jointed, so
        # the sphere narrow phase has no legitimate job in this scene and
        # phantom contacts would both fight the constraints AND trip the
        # contact path's project() sweep
        p = PhysicsParcel(0.25 * min_dx, _DensityMaterial(dens["iron"]),
                          mass=mass, position=pos,
                          inertia_body=(iz, iz, iz), label=label)
        parcels.append(p)
        joints.append(RevoluteJoint(p, None, pos, axis))

    # spring motor on the barrel joint (drive() sets speed/torque each step)
    spring_choice = Choice(
        "mainspring stiffness k and initial wound angle (rad), pendulum "
        "bob mass (kg)",
        [spring_k, spring_theta_wound0, pendulum_mass_kg],
        checked="dynamics/mechanisms/spring.py (MVP linear Hooke's-law "
        "model, no cited real mainspring curve — solid unspoked wheels "
        "here are also heavier than a real movement's, so the constant is "
        "sized to THIS build, not a real spring)",
        reason="tuned empirically for a ~60s watchable run duration; "
              "pendulum mass is a plausible small-clock figure with "
              "nothing in the catalog to check it against")
    spring = MainspringState(joints[0], k=spring_k,
                             theta_wound0=spring_theta_wound0)

    # couplings at the CITED ratios (pinion/wheel per the pinned convention)
    ratios = [t_cp / t_barrel, t_tp / t_cw, t_fp / t_tw, t_ep / t_fw]
    couplings = [GearCouplingJoint(joints[k], joints[k + 1], ratios[k])
                for k in range(4)]

    # pendulum: built at vertical (angle()==0 IS vertical, the escapement's
    # trigger), then displaced to the release angle — same construction the
    # escapement-clock recorder and test_joints.py use
    pivot = Vec3(x_escape, 0.12, -3.5 * z_plane)
    layout_choice = Choice(
        "pendulum pivot placement (height above arbor row, z offset in "
        "mesh-plane pitches) and release angle (deg)",
        [0.12, -3.5, release_deg],
        reason="pure layout geometry clearing the escape arbor and gear "
              "planes; a real movement's back-cock position isn't modeled "
              "or cited. Release angle is a small-angle demo pick with no "
              "cited amplitude")
    q0 = list(quat_from_axis_angle((0.0, 0.0, 1.0), 0.0))
    off0 = qrot(q0, (0.0, -0.5 * L, 0.0))
    # collision radius 0.01, NOT the rod's half-length: the pendulum CoM
    # passes within ~4 cm of the escape arbor (in z) and a half-length
    # bounding sphere would phantom-contact it every step
    rod = PhysicsParcel(0.01, _DensityMaterial(dens_iron),
                        mass=pendulum_mass_kg,
                        position=pivot + Vec3(off0[0], off0[1], off0[2]),
                        orientation=q0,
                        inertia_body=(pendulum_mass_kg * L * L / 12.0,
                                     1.0e-3 * pendulum_mass_kg,
                                     pendulum_mass_kg * L * L / 12.0),
                        label="pendulum")
    pend_joint = RevoluteJoint(rod, None, pivot, axis)
    qr = list(quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(release_deg)))
    off_r = qrot(qr, (0.0, -0.5 * L, 0.0))
    rod.position = pivot + Vec3(off_r[0], off_r[1], off_r[2])
    rod.orientation = qr

    # hour hand: its own body on the center arbor's AXIS but displaced to
    # its own z-plane (the hands' plane) — co-axial parcels at the same
    # point would phantom-contact forever; coupled at the definitional
    # 12:1 (ratio -12: co-rotation at 1/12 the rate)
    hand_m = 0.01
    hour_len = 0.035
    z_hands = 5.5 * (2.0 * face_width_m)
    hour = PhysicsParcel(0.005, _DensityMaterial(dens_iron), mass=hand_m,
                         position=Vec3(x_center, y0, z_hands - 0.006),
                         inertia_body=(hand_m * hour_len ** 2 / 12.0,) * 3,
                         label="hour_hand")
    hour_joint = RevoluteJoint(hour, None, hour.position, axis)
    couplings.append(GearCouplingJoint(joints[1], hour_joint, -12.0))

    all_parcels = parcels + [rod, hour]
    all_joints = joints + [pend_joint, hour_joint]
    scene = PhysicsScene(all_parcels, ground=False,
                         constraints=all_joints + couplings)
    # a 5-deep coupling chain plus escapement braking wants more Gauss-
    # Seidel sweeps than the 10-iteration default to stay crisp
    scene.solver_iterations = 20

    # spring drives the barrel +z world (direction=-1, Phase 0/3 sign
    # convention); the four counter-rotating meshes leave the escape arbor
    # ALSO +z world (even count), so its joint s runs negative and the
    # escapement stop must walk negative: direction=-1
    escapement = Escapement(joints[4], pend_joint, teeth=t_ew, direction=-1.0)
    from ..dynamics.mechanisms.plug import Plug
    from ..dynamics.mechanisms.choice import Choice
    support = Plug.support("arbors, pendulum pivot and hour-hand axis held "
                           "as if by clock plates (plates/pillars/dial "
                           "rendered below, not structurally simulated)")

    def _poses():
        return [{"pos": [p.position.x, p.position.y, p.position.z],
                 "quat": [round(v, 6) for v in p.orientation]}
                for p in all_parcels]

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    t = 0.0
    winding_trace = [spring.theta_wound()]
    while t < t_max:
        spring.drive(direction=-1.0)
        step(scene, dt=dt, sub_steps=1)
        t += dt
        escapement.step(t)
        if t >= next_rec:
            frames.append({"t_sim": round(t, 4), "bodies": _poses()})
            next_rec += frame_dt
            winding_trace.append(spring.theta_wound())
    frames.append({"t_sim": round(t, 4), "bodies": _poses()})
    t_end = t
    de = scene.total_kinetic_energy() - e0

    # ── scene spec: real gear leaves per arbor + hands + pendulum ──
    def _gear_leaf(gear, body, plane, mk):
        return {"op": "add", "material": mk, "body": body,
                "shape": {"type": "Gear", "module": gear.module,
                          "teeth": gear.teeth,
                          "pressure_angle": gear.pressure_angle,
                          "addendum_coeff": gear.addendum_coeff,
                          "dedendum_coeff": gear.dedendum_coeff,
                          "fillet_coeff": gear.fillet_coeff,
                          "face_width": gear.face_width,
                          "center": [0.0, 0.0, plane * z_plane],
                          "source": f"teeth={gear.teeth} (Kelly 1944, cited — "
                                    "blueprint/catalog); module [estimated]"}}

    leaves = []
    for body, (label, x, gears) in enumerate(arbors):
        for gr, plane, mk in gears:
            leaves.append(_gear_leaf(gr, body, plane, mk))
    # pendulum rod + bob (body 5)
    bob_r = 0.12 * L
    leaves += [
        {"op": "add", "material": "iron", "body": 5,
         "shape": {"type": "Box", "center": [0.0, 0.0, 0.0],
                   "x": 0.05 * L, "y": L, "z": 0.05 * L}},
        {"op": "add", "material": "iron", "body": 5,
         "shape": {"type": "Sphere", "center": [0.0, -0.5 * L, 0.0],
                   "radius": bob_r}},
    ]
    # minute hand: a leaf ON the center arbor (body 1), front plane
    minute_len = 0.055
    leaves += [
        {"op": "add", "material": "copper", "body": 1,
         "shape": {"type": "Box", "center": [minute_len / 2.0, 0.0, z_hands],
                   "x": minute_len, "y": 0.004, "z": 0.004}},
        # hour hand: its own body (body 6) whose parcel already sits AT the
        # hands' plane, so its leaf is centered in its own local frame
        {"op": "add", "material": "iron", "body": 6,
         "shape": {"type": "Box", "center": [hour_len / 2.0, 0.0, 0.0],
                   "x": hour_len, "y": 0.006, "z": 0.004}},
    ]

    # ── clock case: back/front plates + 4 pillars + dial, RENDERED (not
    # structurally simulated — same "mast rendered, not simulated" precedent
    # as record_windmill_spinup) as STATIC world geometry (no "body" key).
    # This is what the support plug below cites: the arbors still hang on
    # world anchors, the plates don't actually hold anything up in the
    # physics, but the honest visual is now present rather than implied. ──
    r_barrel = m * t_barrel / 2.0 + m
    r_case = r_barrel + 0.005                     # clears every gear on the train
    case_x_lo = x_barrel - r_case
    case_x_hi = x_escape + m * t_ew / 2.0 + m + 0.005
    case_y_lo = y0 - r_case
    case_y_hi = y0 + r_case
    plate_t = 0.25 * z_plane
    z_back = -0.5 * z_plane                       # clears the barrel gear (plane 0)
    z_front = 4.5 * z_plane                       # clears the escape wheel (plane 4)
    case_cx, case_cy = 0.5 * (case_x_lo + case_x_hi), 0.5 * (case_y_lo + case_y_hi)
    leaves += [
        {"op": "add", "material": "iron",
         "shape": {"type": "Box", "center": [case_cx, case_cy, z_back],
                   "x": case_x_hi - case_x_lo, "y": case_y_hi - case_y_lo,
                   "z": plate_t}},
        {"op": "add", "material": "iron",
         "shape": {"type": "Box", "center": [case_cx, case_cy, z_front],
                   "x": case_x_hi - case_x_lo, "y": case_y_hi - case_y_lo,
                   "z": plate_t}},
    ]
    pillar_r = 0.5 * z_plane
    pillar_inset = z_plane
    for px in (case_x_lo + pillar_inset, case_x_hi - pillar_inset):
        for py in (case_y_lo + pillar_inset, case_y_hi - pillar_inset):
            leaves.append({"op": "add", "material": "iron",
                           "shape": {"type": "Cylinder",
                                     "center": [px, py, 0.5 * (z_back + z_front)],
                                     "radius": pillar_r, "height": z_front - z_back}})
    # dial: thin disc between the front plate and the hands, sized purely
    # from the gap actually available (robust to module_m/face_width_m)
    front_plate_face = z_front + 0.5 * plate_t
    hour_back_face = (z_hands - 0.5 * face_width_m) - 0.002
    z_dial = 0.5 * (front_plate_face + hour_back_face)
    dial_t = max(1e-4, min(0.0015, 0.4 * (hour_back_face - front_plate_face)))
    dial_r = minute_len + 0.01
    dial_material = "ceramic_alumina"
    dial_choice = Choice(
        "clock dial material", dial_material,
        checked="sigma_ground.field.interface.surface.MATERIALS (25 entries)",
        alternatives=["aluminum", "iron", "copper", "bone", "glass"],
        reason="closest available match to a real dial's ivory/enamel face; "
              "no dedicated enamel entry exists in the materials table")
    leaves.append({"op": "add", "material": dial_material,
                   "shape": {"type": "Cylinder",
                             "center": [x_center, y0, z_dial],
                             "radius": dial_r, "height": dial_t}})
    for i in range(12):                            # hour-tick marks at the rim
        ang = i * (math.pi / 6.0)
        leaves.append({"op": "add", "material": "iron",
                       "shape": {"type": "Box",
                                 "center": [x_center + 0.88 * dial_r * math.sin(ang),
                                            y0 + 0.88 * dial_r * math.cos(ang),
                                            z_dial + 0.5 * dial_t + 0.001],
                                 "x": 0.006, "y": 0.006, "z": 0.002}})

    dens_dial, _ = _material_density(dial_material, 288.15)
    materials = {"iron": _bake_material("iron", dens_iron),
                "copper": _bake_material("copper", dens_cu),
                dial_material: _bake_material(dial_material, dens_dial)}
    x_lo = x_barrel - r_barrel - 0.01
    x_hi = x_escape + m * t_ew / 2.0 + 0.02
    y_lo = -(L + bob_r + 0.02) + pivot.y - 0.12   # pendulum reach below its pivot
    y_hi = pivot.y + 0.03
    hand_choice = Choice(
        "hand mass (kg) and lengths (m): hour/minute",
        [hand_m, hour_len, minute_len],
        checked="none exists — no blueprint/catalog covers clock-hand "
        "dimensions; hand_m also breaks this function's own mass-from-"
        "density*volume pattern used by every other body",
        reason="cosmetic proportions relative to the dial that also set "
              "the hour hand's own moment of inertia; not sourced")
    scene_spec = {
        "name": "THE CLOCK — Kelly (1944) cited going train, spring-driven, "
                "pendulum-gated, real involute teeth (module [estimated])",
        "bodies": ([{"pivot": [0, 0, 0], "label": a[0]} for a in arbors]
                  + [{"pivot": [0, 0, 0], "label": "pendulum"},
                     {"pivot": [0, 0, 0], "label": "hour_hand"}]),
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": sum(p.mass for p in all_parcels),
                    "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[x_lo, x_hi], [y_lo, y_hi], [-3 * z_plane, 7 * z_plane]],
        "camera": {"target": [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.0],
                   "orbit_radius": max(1.2 * (x_hi - x_lo), 1.0 * (y_hi - y_lo), 0.3),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0], "az0": 0.15, "el0": 0.12},
        "identified": True,
        # no DRIVE plugged (the mainspring is PRESENT, MODELED machinery;
        # the pendulum is natural physics) — but the HOLDERS are artificial:
        # no clock plates are modeled, the arbors hang on world anchors
        "plugs": [support.to_dict()],
        "choices": [material_choice.to_dict(), geometry_choice.to_dict(),
                   spring_choice.to_dict(), layout_choice.to_dict(),
                   hand_choice.to_dict(), dial_choice.to_dict()],
        "source": "Going train teeth CITED: Kelly, A Practical Course in Horology "
                  "(1944) pp.16-18 via blueprint/catalog (each gear leaf carries "
                  "its own citation). DERIVED: pendulum L=%.4f m from the cited "
                  "600:1 escape:center ratio + anchor cadence (NOT Kelly's lever "
                  "double-impulse). ESTIMATED: module (no module cited — "
                  "GEOMETRY_GAP), pressure angle, spring constants, solid "
                  "(unspoked) wheels. No drive plugged (the spring is present, "
                  "modeled machinery). " % L + support.cite() + " "
                  + material_choice.cite() + " " + geometry_choice.cite() + " "
                  + spring_choice.cite() + " " + layout_choice.cite() + " "
                  + hand_choice.cite() + " " + dial_choice.cite(),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": [a[0] for a in arbors] + ["pendulum", "hour_hand"],
            "validation": {
                "ticks": escapement.ticks,
                "tick_times_s": [round(x, 4) for x in escapement.tick_times],
                "pendulum_period_expected_s": T_pend,
                "pendulum_length_m": L,
                "esc_per_center_ratio_cited": esc_per_center,
                "escape_teeth_cited": t_ew,
                "minute_rate_expected_rad_s": 2.0 * math.pi / 3600.0,
                "motor_work_j": joints[0].motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= joints[0].motor_work_j + 1e-6,
                "winding_trace_rad": [round(w, 6) for w in winding_trace],
                "final_angles_rad": [j.angle() for j in all_joints],
            },
        },
        "kind": "trajectory",
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


# ── rigid-contact helpers (bounce + tumble) ────────────────────────────────
# The quaternion math was LIFTED to dynamics/quat.py (actuation epic M0) —
# these aliases keep every call site here byte-identical.
from ..dynamics.quat import (qrot as _qrot, qrot_inv as _qrot_inv,  # noqa: E402
                             quat_step as _quat_step)


def _dominant_material(construct):
    """The densest non-air part's SUBSTANCE (a leaf label may be a part name;
    the layer table maps it to what it is made of) — a proxy for the surface
    that hits the floor."""
    label_material = {L.name: L.material for L in getattr(construct, "layers", [])}
    cands = sorted(((rho, lbl) for lbl, rho in
                    (construct.density_by_label or {}).items()
                    if lbl != "air" and rho), reverse=True)
    if not cands:
        return "steel_mild"
    lbl = cands[0][1]
    return label_material.get(lbl, lbl)


def _surface_offsets(construct, n: int = 240):
    """Surface points as offsets from the CoM (LOCAL frame) — the candidate
    contact points. Falls back to the bbox corners if sampling comes up dry."""
    import random
    random.seed(8128)                              # deterministic contact set
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    try:
        pts = construct.composed.sample_surface(n, bounds=((x0, y0, z0),
                                                           (x1, y1, z1)))
    except Exception:
        pts = []
    cx, cy, cz = construct.com_m
    out = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in pts]
    if len(out) < 8:
        out = [(x - cx, y - cy, z - cz)
               for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    return out


def _support(quat, offsets):
    """(support height, contact offset): the lowest WORLD-y surface point under
    orientation ``quat``. The body rests with its CoM at y = support height."""
    best_y, best_r = 1e30, offsets[0]
    for r in offsets:
        wy = _qrot(quat, r)[1]
        if wy < best_y:
            best_y, best_r = wy, r
    return -best_y, best_r


def record_object_fall(construct, start_altitude_m: float = 2.4384, *,
                       cd: float = 1.0, area_m2: float | None = None,
                       T: float = 288.15, dt: float = 0.01, t_max: float = 60.0,
                       frame_dt: float = 0.05, target_watch_s: float = 5.0) -> dict:
    """Drop a Deckard Construct — its REAL compiled shape, mass and area — through
    the atmosphere, integrating Materia's drag, and emit the viewer's trajectory
    bundle. The object's actual geometry (a feather's cone+ellipsoid, say) is the
    moving body, not a sphere.

    Drag acts on the BROAD face: a thin object dropped flat presents its largest
    face, so unless `area_m2` is given the area is the largest bbox face — never
    the (edge-on) footprint, which would fall far too fast. The body falls flat
    (constant orientation, no aerodynamic flutter — not modeled), then BOUNCES:
    ground contact uses the material's emergent Johnson–Thornton restitution,
    and the off-CoM contact point spins it up through the construct's real
    inertia — the tumble is a consequence of the shape, never a script.

    SHAPES COME ONLY FROM DECKARD. We render the construct and nothing else — no
    hand-authored floor or scenery. Black is empty space (and on the passthrough
    glasses the real floor shows through, where the object comes to rest at y=0);
    if a scene ever needs a floor it must be a Deckard-grounded construct, never a
    faked box — Deckard returns identified=False for "floor", so we add none.
    """
    from ..materia.engine import simulate_drag_run   # lazy: tier-3 sibling, no cycle
    from .scene_export import construct_to_scene

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

    # ── ground contact: bounce + tumble (emergent restitution) ────────────
    # The descent is Materia's drag integration; at the floor the EXISTING
    # Johnson–Thornton restitution model (field.interface.impact) sets each
    # rebound, and the off-CoM contact point applies a torque impulse through
    # the construct's real inertia — a cup tumbles BECAUSE its handle offsets
    # the contact, not because anyone scripted it.   Honest-physics envelope:
    # a single frictionless contact point per impact (no horizontal skitter,
    # no rolling), torque-free rotation between contacts (no gyroscopic
    # precession), drag neglected at post-bounce speeds, and the object
    # freezes in its final contact orientation (no roll-to-stable-face).
    # Per-contact energy is recorded and must strictly decrease.
    import math
    offsets = _surface_offsets(construct)
    mat = _dominant_material(construct)
    q = list(quat)
    support0, _ = _support(q, offsets)
    g = 9.80665
    m = max(mass_kg, 1e-12)
    Il = [max(v, 1e-12) for v in construct.inertia_kgm2]
    r_char = max(0.5 * min(dx, dy, dz), 1e-3)

    frames = [{"t_sim": round(s["t"], 4),
               "bodies": [{"pos": [0.0, round(s["altitude_m"], 5), 0.0],
                           "quat": list(q)}]}
              for s in hist if s["altitude_m"] > support0]
    if frames:
        t = frames[-1]["t_sim"]
        v_imp = (hist[-1].get("speed_m_s")
                 or math.sqrt(max(0.0, 2.0 * g * start_altitude_m)))
    else:
        t, v_imp = 0.0, math.sqrt(max(0.0, 2.0 * g * max(start_altitude_m, 0.0)))
        frames = [{"t_sim": 0.0,
                   "bodies": [{"pos": [0.0, max(start_altitude_m, support0), 0.0],
                               "quat": list(q)}]}]

    try:
        from ..field.interface.impact import coefficient_of_restitution
        _cor, cor_fallback = coefficient_of_restitution, False
    except Exception:
        _cor, cor_fallback = None, True

    def _erot(w):
        wl = _qrot_inv(q, w)
        return 0.5 * (Il[0] * wl[0] ** 2 + Il[1] * wl[1] ** 2 + Il[2] * wl[2] ** 2)

    contacts = []
    y, v, w = support0, -abs(v_imp), [0.0, 0.0, 0.0]
    dt2 = 0.005
    next_rec = t + frame_dt
    settle_v = max(0.05, math.sqrt(2.0 * g * 0.005 * max(dx, dy, dz)))
    t_stop, settled = t + 12.0, False
    while t < t_stop and not settled:
        support, r_contact = _support(q, offsets)
        if y <= support and v < 0.0:
            v_in = abs(v)
            e = 0.5
            if _cor is not None:
                try:
                    e = float(_cor(mat, velocity=max(v_in, 1e-3),
                                   radius_m=r_char))
                except Exception:
                    cor_fallback = True
            e = min(max(e, 0.02), 0.98)
            E_before = 0.5 * m * v_in * v_in + _erot(w)
            y, v = support, e * v_in
            rc = _qrot(q, r_contact)                       # world contact offset
            J = m * (1.0 + e) * v_in                       # normal impulse (+y)
            dL = (-rc[2] * J, 0.0, rc[0] * J)              # r × (0, J, 0)
            Ll = _qrot_inv(q, dL)                          # to the body frame
            dwl = (Ll[0] / Il[0], Ll[1] / Il[1], Ll[2] / Il[2])
            dw = _qrot(q, dwl)                             # back to world
            w = [e * w[i] + dw[i] for i in range(3)]
            E_rot = _erot(w)
            budget = 0.3 * (1.0 - e * e) * 0.5 * m * v_in * v_in
            if E_rot > budget > 0.0:                       # rotation pays from the
                s = math.sqrt(budget / E_rot)              # dissipated budget only
                w = [x * s for x in w]
                E_rot = budget
            E_after = 0.5 * m * v * v + E_rot
            contacts.append({"t": round(t, 4), "v_impact": round(v_in, 4),
                             "e": round(e, 4), "E_before": round(E_before, 6),
                             "E_after": round(E_after, 6)})
            if v < settle_v:
                v, w, settled = 0.0, [0.0, 0.0, 0.0], True
                y, _ = _support(q, offsets)
                break
        v -= g * dt2
        y += v * dt2
        if any(w):
            q = _quat_step(q, w, dt2)
        t += dt2
        if t >= next_rec:
            frames.append({"t_sim": round(t, 4),
                           "bodies": [{"pos": [0.0, round(y, 5), 0.0],
                                       "quat": [round(x, 6) for x in q]}]})
            next_rec += frame_dt
    rest_y, _ = _support(q, offsets)
    for _k in range(3):                                    # a beat at rest
        t += frame_dt
        frames.append({"t_sim": round(t, 4),
                       "bodies": [{"pos": [0.0, round(rest_y, 5), 0.0],
                                   "quat": [round(x, 6) for x in q]}]})

    # The construct's leaves become ONE dynamic body (body 0), rotated about its
    # centre of mass and translated by the per-frame pose.
    for leaf in scene["csg_leaves"]:
        leaf["body"] = 0
    scene["bodies"] = [{"pivot": list(construct.com_m), "label": construct.name}]

    # Frame the whole fall column, ground → just above the release height.
    top = start_altitude_m + max(dx, dy, dz)
    wide = max(1.0, 1.5 * max(dx, dz))
    scene["bbox"] = [[-wide, wide], [-0.1, top + 0.1], [-wide, wide]]
    scene["camera"] = {"target": [0.0, start_altitude_m * 0.5, 0.0],
                       "orbit_radius": max(3.0, 1.6 * start_altitude_m),
                       "fov_deg": 46.0, "up": [0.0, 1.0, 0.0],
                       "az0": 0.35, "el0": 0.22}
    if start_altitude_m > 20.0 * max(dx, dy, dz):
        scene["camera"]["follow"] = True               # long fall: keep it on screen
        scene["camera"]["orbit_radius"] = max(0.5, 6.0 * max(dx, dy, dz))
        scene["camera"]["target"] = [0.0, start_altitude_m, 0.0]
        # object-sized bbox: the viewer's hit/shadow epsilons scale from the
        # bbox diagonal — a km-tall column bbox would swallow the object
        s = 2.0 * max(dx, dy, dz)
        scene["bbox"] = [[-s, s], [-s, s], [-s, s]]
    scene["kind"] = "trajectory"
    try:
        from .scene_export import sdf_samples           # in-page self-check data
        scene["sdf_samples"] = sdf_samples(construct)
    except Exception:
        pass

    t_total = frames[-1]["t_sim"]
    return {
        "scene": scene,
        "trajectory": {
            "frames": frames,
            "t_end_s": round(t_total, 4),
            "natural_timescale_s": t_total,
            "suggested_rate": max(1e-6, (t_total or 1.5) / target_watch_s),
            "body_labels": [construct.name],
            "validation": {
                "contacts": contacts,
                "restitution_model": ("fallback e=0.5 (material unknown)"
                                      if cor_fallback else
                                      f"Johnson–Thornton COR for {mat!r}"),
                "energy_monotone": all(c["E_after"] < c["E_before"]
                                       for c in contacts),
            },
        },
        "kind": "trajectory",
    }


def record_descent(payload_mass_kg: float = 118.0, drag_area_m2: float = 0.28,
                   cd: float = 0.70, start_altitude_m: float = 35_000.0, *,
                   T: float = 288.15, frame_dt: float = 0.5,
                   target_watch_s: float = 18.0) -> dict:
    """A high-altitude payload descent (skydiver verb) as viewer frames.

    The rendered body is a sphere of radius sqrt(A/pi) — NOT a faked figure:
    `simulate_drag_run` literally integrates `Sphere(r_eff)` as the parcel, so
    the shape on screen IS the body the physics computed, and it is named so.
    """
    import math
    from ..materia.engine import simulate_drag_run

    run = simulate_drag_run(payload_mass_kg, drag_area_m2,
                            start_altitude_m=start_altitude_m, cd_mode="fixed",
                            cd_value=cd, T=T, dt=0.02, t_max=3000.0,
                            sample_every=max(1, round(frame_dt / 0.02)))
    hist = run.get("history") or []
    r_eff = math.sqrt(drag_area_m2 / math.pi)
    frames = [{"t_sim": round(s["t"], 3),
               "bodies": [{"pos": [0.0, round(s["altitude_m"], 3), 0.0],
                           "quat": [0.0, 0.0, 0.0, 1.0]}]} for s in hist] \
        or [{"t_sim": 0.0, "bodies": [{"pos": [0.0, start_altitude_m, 0.0],
                                       "quat": [0.0, 0.0, 0.0, 1.0]}]}]
    t_end = frames[-1]["t_sim"]
    scene_spec = {
        "name": f"payload — drag-equivalent sphere (A={drag_area_m2:g} m², "
                f"m={payload_mass_kg:g} kg)",
        "bodies": [{"pivot": [0, 0, 0], "label": "payload"}],
        "csg_leaves": [{"op": "add", "material": "payload", "body": 0,
                        "shape": {"type": "Sphere", "center": [0, 0, 0],
                                  "radius": r_eff}}],
        "materials": {"payload": _bake_material("payload")},   # honest grey: no model
        "physics": {"mass_kg": payload_mass_kg, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        # object-sized bbox: the follow camera tracks the body, and the
        # viewer's precision constants scale from this diagonal
        "bbox": [[-2 * r_eff, 2 * r_eff], [-2 * r_eff, 2 * r_eff],
                 [-2 * r_eff, 2 * r_eff]],
        "camera": {"target": [0.0, start_altitude_m, 0.0],
                   "orbit_radius": max(8 * r_eff, 3.0), "fov_deg": 45.0,
                   "up": [0.0, 1.0, 0.0], "follow": True},
        "identified": True,
        "source": "high-altitude descent (Materia drag integration)",
        "kind": "trajectory",
    }
    return {"scene": scene_spec,
            "trajectory": {"frames": frames, "t_end_s": t_end,
                           "natural_timescale_s": t_end,
                           "suggested_rate": max(1e-6, t_end / target_watch_s),
                           "body_labels": ["payload"]},
            "kind": "trajectory"}


def record_horizontal_run(mass_kg: float = 0.02, diameter_m: float = 0.01,
                          launch_mach: float = 2.5, *, T: float = 288.15,
                          frame_dt: float = 0.02,
                          target_watch_s: float = 10.0) -> dict:
    """A supersonic projectile decelerating along +x (1-D Materia model: no
    gravity drop in this verb — rendered at skim height y=r, stated plainly)."""
    import math
    from ..materia.engine import simulate_drag_run
    from ..field.interface.atmosphere import speed_of_sound

    v0 = launch_mach * speed_of_sound(T)
    area = math.pi * (diameter_m / 2.0) ** 2
    run = simulate_drag_run(mass_kg, area, v0_mps=v0, orientation="horizontal",
                            cd_mode="mach", T=T, dt=5e-4, t_max=60.0,
                            sample_every=max(1, round(frame_dt / 5e-4)))
    hist = run.get("history") or []
    r = diameter_m / 2.0
    frames = [{"t_sim": round(s["t"], 4),
               "bodies": [{"pos": [round(s.get("distance_m") or 0.0, 3), r, 0.0],
                           "quat": [0.0, 0.0, 0.0, 1.0]}]} for s in hist] \
        or [{"t_sim": 0.0, "bodies": [{"pos": [0.0, r, 0.0],
                                       "quat": [0.0, 0.0, 0.0, 1.0]}]}]
    t_end = frames[-1]["t_sim"]
    x_end = frames[-1]["bodies"][0]["pos"][0]
    scene_spec = {
        "name": f"supersonic slug — Mach {launch_mach:g}, ⌀{diameter_m * 1000:g} mm",
        "bodies": [{"pivot": [0, 0, 0], "label": "slug"}],
        "csg_leaves": [{"op": "add", "material": "tungsten", "body": 0,
                        "shape": {"type": "Sphere", "center": [0, 0, 0],
                                  "radius": r}}],
        "materials": {"tungsten": _bake_material("tungsten")},
        "physics": {"mass_kg": mass_kg, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        # object-sized bbox (the slug travels ~hundreds of metres; the follow
        # camera rides along, so precision stays at slug scale)
        "bbox": [[-4 * r, 4 * r], [-4 * r, 4 * r], [-4 * r, 4 * r]],
        "camera": {"target": [0.0, r, 0.0], "orbit_radius": max(20 * r, 0.3),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0], "follow": True},
        "identified": True,
        "source": "supersonic projectile (Materia transonic-drag integration)",
        "kind": "trajectory",
    }
    return {"scene": scene_spec,
            "trajectory": {"frames": frames, "t_end_s": t_end,
                           "natural_timescale_s": t_end,
                           "suggested_rate": max(1e-6, t_end / target_watch_s),
                           "body_labels": ["slug"]},
            "kind": "trajectory"}


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


def record_bell_strike(bell_material: str = "iron", bell_diameter_m: float = 0.15,
                       stone_mass_kg: float = 0.2, observer_distance_m: float = 50.0,
                       stone_material: str = "granite",
                       decay_tau_s: float | None = None, T: float = 288.15,
                       dt: float = 1.0 / 960.0, frame_dt: float = 0.002,
                       target_watch_s: float = 6.0) -> dict:
    """A hanging bell struck by a thrown stone. The bell (a solid Cone, apex
    at the crown where it hangs, rim at the base where it's struck) is
    world-welded in place ("hung", without a real chain modeled — a Plug
    support). A stone free-falls under gravity onto the rim; the ENGINE's
    own pairwise contact solver (dynamics/collision.py, already exercised
    by every bounce in this file) resolves the impact -- this function only
    DETECTS it (for the v_impact report) and does not hand-roll the
    response. On impact: ring_frequency (field.interface.acoustics -- the
    SAME formula the acoustics() Materia verb cites for bells) gives the
    sustained tone; the ALREADY-EXISTING Hertzian impact machinery
    (field.interface.impact -- impact_energy_partition/
    impact_sound_frequency/hertz_contact_duration, the same first-principles
    contact model record_fall's own bounce restitution already uses) gives
    the impact's REAL dissipated energy and its distinct high-frequency
    "click" transient, reported alongside the sustained ring rather than
    invented from raw kinetic energy. A Choice-cited exponential decay
    synthesizes the sustained-ring WAV via radiance.bell_acoustics;
    propagate_in_air (real speed_of_sound) reports arrival time and
    inverse-square loudness at ``observer_distance_m``.

    Every bell/stone/observer parameter left at its default (as EVERY
    current text request does -- Materia's translator has no slot
    extraction yet for "how big"/"how heavy"/"how far") is recorded as a
    Choice, not silently assumed (Choice doctrine, 2026-07-16).

    SIMPLIFIED_MODEL, stated plainly: one SUSTAINED vibrational mode only
    (real bells ring with several inharmonic partials; the impact "click"
    is reported as a frequency/duration number but NOT synthesized into the
    WAV as a second audio layer), the bell's collision envelope is its
    bounding SPHERE (not the true conical SDF), the impact-to-loudness
    curve is a Choice-flagged scale on the REAL dissipated energy (no
    first-principles energy-to-SPL/radiation-efficiency model on file), and
    atmospheric propagation is inverse-square-only (no frequency-dependent
    absorption).
    """
    import math
    import os
    from ..field.interface.acoustics import ring_frequency
    from ..field.interface.impact import (impact_energy_partition,
                                          impact_sound_frequency,
                                          hertz_contact_duration)
    from ..materia.engine import _material_density, _DensityMaterial
    from ..kernel.shapes import Cone, Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import WeldJoint
    from ..dynamics.stepper import step
    from ..dynamics.mechanisms.plug import Plug
    from ..dynamics.mechanisms.choice import Choice
    from .bell_acoustics import synthesize_bell_tone, propagate_in_air

    scenario_choice = Choice(
        "bell material/size, stone mass/material, and observer distance",
        {"bell_material": bell_material, "bell_diameter_m": bell_diameter_m,
         "stone_mass_kg": stone_mass_kg, "stone_material": stone_material,
         "observer_distance_m": observer_distance_m},
        checked="materia/translator.py strike_bell routing (no slot "
        "extraction yet for tool/stone size or observer distance from "
        "free text)",
        reason="the request named neither a bell size/material, a stone "
              "mass, nor an observer distance, so these are this "
              "function's own defaults, not measured/cited values")
    bell_r = bell_diameter_m / 2.0
    height_choice = Choice(
        "bell height-to-diameter proportion", 0.85,
        checked="no cited casting on file for a specific bell",
        reason="a typical bell silhouette is somewhat taller than its rim "
              "radius; picked for a legible shape, not measured")
    bell_h = height_choice.value * bell_diameter_m

    density_bell, bell_name = _material_density(bell_material, T)
    density_stone, stone_name = _material_density(stone_material, T)
    stone_r = (3.0 * stone_mass_kg / (4.0 * math.pi * density_stone)) ** (1.0 / 3.0)

    crown_y = 2.0 * bell_h
    bell = PhysicsParcel(Cone(bell_r, bell_h), _DensityMaterial(density_bell),
                         position=Vec3(0.0, crown_y - 0.75 * bell_h, 0.0),
                         label=bell_material)
    # near the rim, not dead-center, and just above bell.radius so the drop
    # is short -- a real, legible impact, not a long empty fall
    drop_choice = Choice(
        "stone start offset from the bell rim and drop height",
        {"rim_frac": 0.7, "drop_height_over_stone_r": 4.0},
        checked="no cited throw trajectory for this demo",
        reason="a short drop near the rim gives a real nonzero impact "
              "velocity without a long empty-fall simulation window")
    x0 = drop_choice.value["rim_frac"] * bell_r
    drop_h = drop_choice.value["drop_height_over_stone_r"] * stone_r
    stone_y0 = bell.position.y + bell.radius + drop_h + stone_r
    stone = PhysicsParcel(Sphere(stone_r), _DensityMaterial(density_stone),
                          position=Vec3(x0, stone_y0, 0.0), label=stone_material)

    world_pin = WeldJoint(bell, None, bell.position)
    support = Plug.support(f"{bell_name.lower()} bell hung from a fixed "
                           "point, without the chain/rope modeled")
    scene = PhysicsScene([bell, stone], ground=False, constraints=[world_pin])

    def _poses():
        out = []
        for p in (bell, stone):
            pos = p.position
            out.append({"pos": [pos.x, pos.y, pos.z],
                        "quat": [round(v, 6) for v in p.orientation]})
        return out

    contact_r = bell.radius + stone.radius
    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    t_max = 3.0
    struck_at, v_impact = None, None
    while scene.time < t_max:
        d_before = (stone.position - bell.position).length()
        pre_v = stone.velocity.length()
        step(scene, dt=dt, sub_steps=1)
        d_after = (stone.position - bell.position).length()
        if struck_at is None and d_before > contact_r and d_after <= contact_r:
            struck_at = scene.time
            v_impact = pre_v
        if scene.time >= next_rec:
            frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
            next_rec += frame_dt
        if struck_at is not None and scene.time >= struck_at + 0.4:
            break
    frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
    t_end = scene.time

    f_ring = ring_frequency(bell_material, bell_diameter_m)

    # Real Hertzian contact physics (field.interface.impact), NOT invented
    # from raw kinetic energy -- the SAME COR/energy-partition model
    # record_fall's own bounce restitution already uses. E_dissipated_J is
    # the fraction (1-e^2) of impact KE that becomes plastic deformation,
    # heat AND sound (impact.py's own energy-conservation derivation) -- it
    # is the physically correct input to a loudness estimate, unlike the
    # full impact KE (part of which stays as the stone's rebound motion,
    # never becomes sound at all). impact_sound_frequency/
    # hertz_contact_duration give the short high-frequency CLICK transient
    # (Hertz 1882: f ~ 1/(2*t_contact)) -- distinct from, and typically far
    # higher than, ring_frequency's SUSTAINED shell resonance; reported
    # here, not synthesized into the WAV (SIMPLIFIED_MODEL, see docstring).
    if v_impact:
        partition = impact_energy_partition(bell_material, velocity=v_impact,
                                            mass_kg=stone.mass, radius_m=stone_r)
        e_dissipated_j = partition["E_dissipated_J"]
        f_click = impact_sound_frequency(bell_material, velocity=v_impact,
                                         radius_m=stone_r, mass_kg=stone.mass)
        t_contact_s = hertz_contact_duration(bell_material, velocity=v_impact,
                                             radius_m=stone_r, mass_kg=stone.mass)
    else:
        e_dissipated_j, f_click, t_contact_s = 0.0, 0.0, 0.0

    tau_choice = Choice(
        "bell ring decay time constant", (decay_tau_s if decay_tau_s is not None
                                          else 2.5),
        checked="no material Q-factor table on file for bell decay",
        reason="a several-second ring-down is typical for a struck metal "
              "bell; not measured from field.interface data (SIMPLIFIED_MODEL "
              "-- real decay is material- and mode-dependent)")
    amp_choice = Choice(
        "bell strike amplitude vs. reference DISSIPATED impact energy "
        "(impact_energy_partition's E_dissipated_J, not raw kinetic energy)",
        {"reference_dissipated_energy_j": 0.1, "min_amplitude": 0.05},
        checked="field.interface.impact.impact_energy_partition (real "
        "Johnson-Thornton energy conservation) for the ENERGY input; no "
        "cited energy-to-SPL/radiation-efficiency curve on file for the "
        "SCALE",
        reason="amplitude scales monotonically with the real dissipated "
              "energy but the absolute loudness scale itself is not "
              "measured (SIMPLIFIED_MODEL)")
    amplitude = min(1.0, max(amp_choice.value["min_amplitude"],
                             (e_dissipated_j
                              / amp_choice.value["reference_dissipated_energy_j"]) ** 0.5))
    tau = float(tau_choice.value)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "web", "data"))
    os.makedirs(data_dir, exist_ok=True)
    audio_path = os.path.join(
        data_dir, f"bell_strike_{bell_material}_{int(bell_diameter_m * 1000)}mm.wav")
    # ring_frequency scales as 1/diameter -- a small/stiff bell can genuinely
    # ring above 22050 Hz's default 11025 Hz Nyquist limit (verified live: a
    # 150 mm iron bell rings at ~12.6 kHz). Undersampling wouldn't just lose
    # detail, it would ALIAS to the wrong audible pitch -- a real correctness
    # bug, not a quality tradeoff. Sample rate tracks the tone with margin.
    sample_rate = max(22050, int(math.ceil(3.0 * f_ring / 1000.0)) * 1000)
    synthesize_bell_tone(audio_path, f_ring, amplitude, tau, sample_rate=sample_rate)
    propagation = propagate_in_air(f_ring, observer_distance_m, source_level=amplitude, T=T)

    r = bell_r
    materials = {bell_material: _bake_material(bell_material, density_bell),
                stone_material: _bake_material(stone_material, density_stone)}
    leaves = [
        {"op": "add", "material": bell_material, "body": 0,
         "shape": {"type": "Cone", "center": [0, 0, 0], "radius": bell_r,
                  "height": bell_h}},
        {"op": "add", "material": stone_material, "body": 1,
         "shape": {"type": "Sphere", "center": [0, 0, 0], "radius": stone_r}},
    ]
    scene_spec = {
        "name": f"{bell_name.lower()} bell struck by a {stone_name.lower()} stone",
        "bodies": [{"pivot": [0, 0, 0], "label": "bell (hung)"},
                  {"pivot": [0, 0, 0], "label": "stone"}],
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": bell.mass + stone.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-4 * r, 4 * r], [-1.0, crown_y + r], [-4 * r, 4 * r]],
        "camera": {"target": [0.0, bell.position.y, 0.0],
                   "orbit_radius": max(6 * r, 0.3), "fov_deg": 45.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.6, "el0": 0.3},
        "identified": True,
        "plugs": [support.to_dict()],
        "choices": [scenario_choice.to_dict(), height_choice.to_dict(),
                   drop_choice.to_dict(), tau_choice.to_dict(),
                   amp_choice.to_dict()],
        "source": (f"bell ring frequency from field.interface.acoustics."
                  "ring_frequency (f=v_L/(pi*d), the same formula the "
                  "acoustics() Materia verb cites for bells); impact "
                  "energy partition and click frequency from field."
                  "interface.impact (Johnson-Thornton/Hertz contact "
                  "mechanics, the SAME model record_fall's bounce "
                  "restitution already uses) -- not invented from raw "
                  "kinetic energy; stone impact resolved by the engine's "
                  "own pairwise contact solver (dynamics/collision.py), "
                  "not scripted. "
                  + support.cite() + ". " + scenario_choice.cite() + ". "
                  + height_choice.cite() + ". " + drop_choice.cite() + ". "
                  + tau_choice.cite() + ". " + amp_choice.cite() + ". "
                  "SIMPLIFIED_MODEL: single sustained vibrational mode "
                  "synthesized (the impact click is reported as a number, "
                  "not a second audio layer), sphere-approximated bell "
                  "collision envelope, inverse-square-only atmospheric "
                  "propagation."),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": ["bell (hung)", "stone"],
            "validation": {
                "struck": struck_at is not None,
                "struck_at_s": struck_at,
                "v_impact_m_s": v_impact,
                "impact_energy_j": (0.5 * stone.mass * v_impact ** 2)
                                   if v_impact else 0.0,
                "dissipated_energy_j": e_dissipated_j,
                "ring_frequency_hz": f_ring,
                "impact_click_frequency_hz": f_click,
                "contact_duration_s": t_contact_s,
            },
        },
        "kind": "trajectory",
        "audio_path": audio_path,
        "acoustics": {
            "ring_frequency_hz": f_ring,
            "impact_click_frequency_hz": f_click,
            "contact_duration_s": t_contact_s,
            "dissipated_energy_j": e_dissipated_j,
            "decay_tau_s": tau,
            "amplitude": amplitude,
            **propagation,
        },
    }


def record_hand_tool_actuation(tool: str = "pliers",
                               swing_amplitude_deg: float = 15.0,
                               speed_rad_s: float = 4.0,
                               motor_max_torque: float = 0.02,
                               n_cycles: float = 3.0, T: float = 288.15,
                               dt: float = 1.0 / 960.0, frame_dt: float = 0.01,
                               target_watch_s: float = 8.0) -> dict:
    """Open and close a two-jaw hand tool (pliers/tongs/scissors/...) over
    time -- the hinge arc's payoff demo. The pivot is NOT declared: it is
    DISCOVERED by ``deckard.hand_tools.build_hand_tool_field`` +
    ``infer_dodoxel_joints`` from real dodoxel contact geometry, then bridged
    to a live ``RevoluteJoint`` by ``dodoxel_parts_to_scene`` (hinge arc
    Phase D). This function's own job is orchestration only: pin one handle
    to the world (as if held by a hand), set the discovered joint's swing
    limits, and drive it with ``OscillatingRevoluteActuator``.
    """
    import math
    from ..deckard.hand_tools import build_hand_tool_field
    from ..deckard.dodoxelize import dodoxel_parts_to_scene
    from ..materia.engine import _material_density, _DensityMaterial
    from ..dynamics.joints import WeldJoint
    from ..dynamics.stepper import step
    from ..dynamics.mechanisms.actuator import OscillatingRevoluteActuator
    from ..dynamics.mechanisms.plug import Plug
    from ..dynamics.mechanisms.choice import Choice
    from .scene_export import _shape_to_dict

    field, joint_info, meta = build_hand_tool_field(tool)
    material = meta["material"]

    def density_of(name):
        return _material_density(name, T)[0]

    def material_obj_of(name):
        density, _ = _material_density(name, T)
        return _DensityMaterial(density)

    bridge = dodoxel_parts_to_scene(field, material_obj_of, density_of)
    scene = bridge["scene"]
    lever_a, lever_b = bridge["parcels"]
    joint = bridge["joints"][0]        # confirmed revolute by the builder

    amplitude = math.radians(swing_amplitude_deg)
    joint.limits = (-amplitude, amplitude)
    joint.motor_max_torque = motor_max_torque

    world_pin = WeldJoint(lever_a, None, lever_a.position)
    scene.constraints.append(world_pin)
    support = Plug.support(f"{meta['tool']} handle held in place as if by a "
                           "hand, without the hand")
    plug = Plug(joint, "artificial opening/closing motion from external "
               "actuation", motor_speed_rad_s=speed_rad_s,
               motor_max_torque_nm=motor_max_torque,
               equivalent="a hand squeezing and releasing the handles")
    cycle_choice = Choice(
        f"{meta['tool']} jaw swing amplitude and actuation speed/torque",
        {"swing_amplitude_deg": swing_amplitude_deg, "speed_rad_s": speed_rad_s,
         "motor_max_torque_nm": motor_max_torque},
        checked="dynamics/mechanisms/actuator.py (no cited product cycle rate "
        "exists to check against)",
        reason="picked for a legible several-cycle render at this tool's "
              "tiny (gram-scale) inertia, not measured from a real product")

    act = OscillatingRevoluteActuator(joint, speed_rad_s=speed_rad_s)

    def _poses():
        out = []
        for p in (lever_a, lever_b):
            pos = p.position
            out.append({"pos": [pos.x, pos.y, pos.z],
                        "quat": [round(v, 6) for v in p.orientation]})
        return out

    frames = [{"t_sim": 0.0, "bodies": _poses()}]
    next_rec = frame_dt
    e0 = scene.total_kinetic_energy()
    half_period = (2.0 * amplitude) / speed_rad_s
    t_max = n_cycles * half_period * 2.2               # slack for settling
    target_reversals = int(round(2 * n_cycles))
    seen_min, seen_max = 0.0, 0.0
    while scene.time < t_max and act.reversals < target_reversals:
        step(scene, dt=dt, sub_steps=1)
        act.step(scene.time)
        th = joint.angle()
        seen_min, seen_max = min(seen_min, th), max(seen_max, th)
        if scene.time >= next_rec:
            frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
            next_rec += frame_dt
    frames.append({"t_sim": round(scene.time, 4), "bodies": _poses()})
    t_end = scene.time
    de = scene.total_kinetic_energy() - e0

    reach = meta["jaw_len_m"] + meta["handle_len_m"]
    half = max(reach * 0.65, 0.01)
    density_a, mat_name = _material_density(material, T)
    materials = {material: _bake_material(material, density_a)}
    leaves = [
        {"op": "add", "material": material, "body": 0,
         "shape": _shape_to_dict(lever_a.shape)},
        {"op": "add", "material": material, "body": 1,
         "shape": _shape_to_dict(lever_b.shape)},
    ]
    scene_spec = {
        "name": f"{meta['tool']} — opening and closing (dodoxel-discovered "
                "pivot, RevoluteJoint SOLVED not scripted)",
        "bodies": [{"pivot": [0, 0, 0], "label": "handle (held)"},
                  {"pivot": [0, 0, 0], "label": "jaw (actuated)"}],
        "csg_leaves": leaves,
        "materials": materials,
        "physics": {"mass_kg": lever_a.mass + lever_b.mass, "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-half, half], [-half, half], [-half, half]],
        "camera": {"target": [0.0, 0.0, 0.0], "orbit_radius": max(3 * half, 0.02),
                   "fov_deg": 45.0, "up": [0.0, 1.0, 0.0], "az0": 0.6, "el0": 0.5},
        "identified": True,
        "plugs": [plug.to_dict(), support.to_dict()],
        "choices": meta["choices"] + [cycle_choice.to_dict()],
        "source": (f"{meta['tool']} pivot DISCOVERED by dodoxel joint "
                  "inference (deckard/dodoxel_articulate.py) from real "
                  "contact geometry, driven by OscillatingRevoluteActuator "
                  "(dynamics/mechanisms/actuator.py) via a SOLVED "
                  "RevoluteJoint (dynamics/joints.py), not scripted. "
                  + plug.cite() + ". " + support.cite() + ". "
                  + cycle_choice.cite()),
    }
    suggested_rate = max(1e-6, t_end / target_watch_s)

    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": suggested_rate,
            "body_labels": ["handle (held)", "jaw (actuated)"],
            "validation": {
                "motor_work_j": joint.motor_work_j,
                "energy_gain_j": de,
                "energy_ledger_ok": de <= joint.motor_work_j + 1e-6,
                "reversals": act.reversals,
                "cycles_completed": act.cycles_completed,
                "angle_range_rad": [seen_min, seen_max],
                "limits_rad": list(joint.limits),
                "joint_axis_family": joint_info.get("axis_family"),
                "joint_snap_error_deg": joint_info.get("snap_error_deg"),
            },
        },
        "kind": "trajectory",
    }
