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
