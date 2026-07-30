"""record_push_off — push one object of a composed stack and watch what follows.

The follow-up half of the doughnut/plate/table arc: ``items`` is the same
bottom-to-top list the SCENE stack route composed ("1700s table", "plate",
"Doughnut"), rebuilt here per-object via ``deckard.scene.build_stack`` so each
body keeps its own real mass/bbox at its composed pose. The pushed body gets
an initial horizontal velocity (+x); everything above it rides on friction —
and whether it rides, slips, or is left behind is DECIDED by the mu values,
not scripted. With the default numbers the plate slides out from under the
doughnut exactly like the tablecloth trick: the icing-on-glaze friction
cannot supply the acceleration the escaping plate demands, the relative slip
exceeds the plate's own radius, and the doughnut drops to the table while the
plate carries on over the edge.

Simulated in the stack's own z-up frame (bodies translate, camera up is +z) —
no axis juggling between the composed statics and the dynamics.

SIMPLIFIED_MODEL, stated plainly: planar motion along the push axis (+x)
only; no tipping, rolling, or rotation (a body whose CoM passes its support
edge transitions to free fall, it does not pivot); support areas are bbox
rectangles (a round plate uses its bounding square); a falling body lands on
the highest surface under its CoM, with no mid-air body-on-body recontact; no
deformation. Friction and restitution are Choices, because NO measured
friction dataset exists on disk (TPSX carries shear moduli, not friction
coefficients) and the derived Bowden-Tabor coefficient is the wrong regime
(clean metals in vacuum, ~1.0 for every pair — see
materia.engine.make_material_friction and tests/test_degeneracy_probe.py).
"""
import math

G = 9.80665


def record_push_off(items, pushed_index: int = 1, push_speed_m_s=None, *,
                    mu_overrides: dict | None = None,
                    allow_llm: bool = False, dt: float = 1.0 / 960.0,
                    frame_dt: float = 1.0 / 120.0, t_max: float = 6.0) -> dict:
    from ..deckard.scene import build_stack, _slug
    from ..dynamics.mechanisms.choice import Choice
    from .scene_export import _bake_material, _shape_to_dict, _bounding_sphere

    built = build_stack(items, allow_llm=allow_llm)
    n = len(built)
    if not (1 <= pushed_index < n):
        raise ValueError(f"pushed_index {pushed_index} out of range for "
                         f"{n}-item stack (the base cannot be pushed)")

    names = [name for name, _c in built]
    cons = [c for _n, c in built]
    bbox = [c.bbox for c in cons]
    mass = [c.mass_kg for c in cons]
    com = [list(c.com_m) for c in cons]
    bot_off = [com[i][2] - bbox[i][2][0] for i in range(n)]   # CoM above own bottom
    floor_z = bbox[0][2][0]                                   # the base stands ON the floor

    half_w = bbox[0][0][1]
    push_choice = Choice(
        f"push speed applied to the {names[pushed_index]}",
        round(push_speed_m_s if push_speed_m_s is not None else 2.2, 3),
        checked=(f"table half-width {half_w:.2f} m and plate-table mu below: "
                 f"the shove must beat sqrt(2*mu*g*d) ~ "
                 f"{math.sqrt(2 * 0.35 * G * max(half_w, 0.1)):.1f} m/s to "
                 f"reach the edge at all"),
        reason="the request says 'push', not how hard; picked hard enough "
               "that the outcome is decided by friction, not by a feeble tap",
        adjustable=True)
    v0 = float(push_choice.value)

    mu_choice = Choice(
        "kinetic friction coefficients (pairwise)",
        {"plate_on_table": 0.35, "topper_on_plate": 0.50,
         "topper_on_table": 0.45, "anything_on_floor": 0.40},
        checked="no measured friction data on disk (TPSX has none; "
                "materials_engineering.json has no friction field); the "
                "derived Bowden-Tabor value is a vacuum-regime constant ~1.0 "
                "(degeneracy probe) and would make nothing ever slide",
        reason="typical engineering ranges: glazed ceramic on finished wood "
               "~0.3-0.4; tacky icing on glaze taken sticky ~0.5. Adjustable, "
               "and the OUTCOME (rides vs left behind) flips with them — that "
               "sensitivity is the physics, not a bug",
        adjustable=True)
    MU = dict(mu_choice.value)
    # the Choice says adjustable=True — this is the adjustment surface. An
    # override is still the SAME disclosed quantity, just a caller's value.
    if mu_overrides:
        unknown = set(mu_overrides) - set(MU)
        if unknown:
            raise ValueError(f"unknown mu key(s): {sorted(unknown)}; "
                             f"valid: {sorted(MU)}")
        MU.update({k: float(v) for k, v in mu_overrides.items()})
        mu_choice.value.update(MU)                 # the disclosure shows what RAN
    e_choice = Choice(
        "restitution at landings",
        {"plate": 0.30, "topper": 0.05},
        checked="impact.coefficient_of_restitution needs a MATERIALS-table "
                "key; 'stoneware'/'leavened dough' are Deckard vocabulary, "
                "not in that table — honest Choice instead of a bad alias",
        reason="stiff ceramic bounces a little; a doughnut lands like a "
               "doughnut", adjustable=True)

    def _mu(body, support):
        if support == "floor":
            return MU["anything_on_floor"]
        if body == pushed_index and support == 0:
            return MU["plate_on_table"]
        if body > pushed_index and support == pushed_index:
            return MU["topper_on_plate"]
        return MU["topper_on_table"]

    def _e(body):
        return e_choice.value["plate" if body == pushed_index else "topper"]

    # per-body state; support is "floor", a body index, or None while falling.
    dx = [0.0] * n
    dz = [0.0] * n
    vx = [0.0] * n
    vz = [0.0] * n
    support = [None] + [i - 1 for i in range(1, n)]
    vx[pushed_index] = v0

    def _rect(i):
        (x0, x1), _yy, _zz = bbox[i]
        return x0 + dx[i], x1 + dx[i]

    def _on(i, s):
        if s == "floor":
            return True
        x = com[i][0] + dx[i]
        x0, x1 = _rect(s)
        return x0 <= x <= x1

    def _contact_dz(i, s):
        top = floor_z if s == "floor" else bbox[s][2][1] + dz[s]
        return top + bot_off[i] - com[i][2]

    def _land_surface(i, z_bottom_prev):
        """Highest shelf under body i, qualified against the PRE-step bottom.
        Qualifying against the post-step position tunnels: a substep that
        carries the bottom 1 mm past a surface disqualifies that surface as
        "below" in the same breath it should have landed on it (caught live:
        the doughnut fell straight through the table top to the floor)."""
        x = com[i][0] + dx[i]
        best, best_z = "floor", floor_z
        for j in range(n):
            if j == i:
                continue
            if j != 0 and support[j] is None:
                continue                                   # a falling body is no shelf
            x0, x1 = _rect(j)
            top = bbox[j][2][1] + dz[j]
            if x0 <= x <= x1 and best_z < top <= z_bottom_prev + 1e-9:
                best, best_z = j, top
        return best

    def _energy():
        ke = sum(0.5 * mass[i] * (vx[i] ** 2 + vz[i] ** 2) for i in range(1, n))
        px = sum(mass[i] * vx[i] for i in range(1, n))
        pz = sum(mass[i] * vz[i] for i in range(1, n))
        ly = sum(mass[i] * ((com[i][2] + dz[i]) * vx[i]
                            - (com[i][0] + dx[i]) * vz[i]) for i in range(1, n))
        return {"energy_j": ke, "momentum": [px, 0.0, pz],
                "angular_momentum": [0.0, ly, 0.0]}

    def _frame(t):
        return {"t_sim": round(t, 4),
                "bodies": [{"pos": [round(dx[i], 5), 0.0, round(dz[i], 5)],
                            "quat": [0.0, 0.0, 0.0, 1.0]} for i in range(n)],
                **_energy()}

    frames = []
    t, next_rec = 0.0, 0.0
    EPS_V = 0.02
    while t < t_max:
        if t >= next_rec:
            frames.append(_frame(t))
            next_rec += frame_dt
        for i in range(n - 1, 0, -1):                      # riders first
            s = support[i]
            if s is None:                                  # ballistic
                z_bottom_prev = com[i][2] + dz[i] - bot_off[i]
                vz[i] -= G * dt
                dx[i] += vx[i] * dt
                dz[i] += vz[i] * dt
                land = _land_surface(i, z_bottom_prev)
                cdz = _contact_dz(i, land)
                if dz[i] <= cdz:
                    dz[i] = cdz
                    e = _e(i)
                    vimp = abs(vz[i])
                    vz[i] = e * vimp
                    mu = _mu(i, land)
                    if vx[i]:
                        dvx = min(abs(vx[i]), mu * (1 + e) * vimp)
                        vx[i] -= math.copysign(dvx, vx[i])
                    if vz[i] < 0.25:                       # too slow to bounce visibly
                        vz[i] = 0.0
                        support[i] = land
                continue
            # supported: one Coulomb branch covers both cases — a rider on a
            # moving support (doughnut on the escaping plate) and the pushed
            # body on a static surface (plate on the table). The normal load
            # is the body's own weight PLUS everything resting on it, so the
            # plate's table-friction feels the doughnut pressing down.
            sv = vx[s] if isinstance(s, int) and s != 0 else 0.0
            vrel = vx[i] - sv
            load = mass[i] + sum(mass[j] for j in range(1, n) if support[j] == i)
            mu = _mu(i, s)
            if abs(vrel) > EPS_V:
                # friction force mu*N opposes the RELATIVE slide; N = load*g
                a_i = mu * G * (load / mass[i])
                dv = min(abs(vrel), a_i * dt)              # never overshoot to reverse slip
                vx[i] += -math.copysign(dv, vrel)
                if isinstance(s, int) and s != 0:
                    # equal-and-opposite reaction drags the moving support
                    vx[s] += math.copysign(mu * G * load / mass[s] * dt, vrel)
            else:
                vx[i] = sv                                 # locked to the support
                # static-adequacy is NOT checked (SIMPLIFIED_MODEL: mu_s ~ mu_k
                # and a support decelerating harder than mu*g would re-slip;
                # with the default numbers that regime is never entered)
            dx[i] += vx[i] * dt
            if not _on(i, s):
                support[i] = None                          # over the edge
        t += dt
        if all(support[i] is not None and abs(vx[i]) < EPS_V and vz[i] == 0.0
               for i in range(1, n)):
            break
    frames.append(_frame(t))

    # ── scene spec: every construct's own leaves, tagged with its body ──
    csg_leaves, materials = [], {}
    for i, c in enumerate(cons):
        prefix = f"{_slug(names[i])}{i}__"
        label_material = {L.name: L.material for L in getattr(c, "layers", [])}
        for leaf, op in c.composed._leaves:
            plabel = prefix + leaf.material
            leaf_dict = {"op": op, "material": plabel, "body": i,
                         "shape": _shape_to_dict(leaf.shape)}
            bs = _bounding_sphere(leaf.shape)
            if bs is not None:
                leaf_dict["bounds"] = {"center": list(bs[0]), "radius": bs[1]}
            csg_leaves.append(leaf_dict)
            if plabel not in materials:
                substance = label_material.get(leaf.material, leaf.material)
                info = _bake_material(substance,
                                      c.density_by_label.get(leaf.material))
                info.pop("choice", None)
                materials[plabel] = info
    # a rendered floor so the fall lands somewhere visible (grey staging slab,
    # same honest convention as the theaters' floor)
    fx = max(1.5, 3.0 * half_w)
    csg_leaves.append({"op": "add", "material": "floor",
                       "shape": {"type": "Box",
                                 "center": [fx * 0.3, 0.0, floor_z - 0.011],
                                 "x": 2 * fx, "y": 2.0, "z": 0.02}})
    floor_info = _bake_material("floor")
    floor_info.pop("choice", None)
    materials["floor"] = floor_info

    top_z = bbox[-1][2][1]
    scene_spec = {
        "name": f"{names[pushed_index]} pushed off the {names[pushed_index - 1]}",
        "bodies": [{"pivot": [0, 0, 0], "label": names[i]} for i in range(n)],
        "csg_leaves": csg_leaves,
        "materials": materials,
        "physics": {"mass_kg": sum(mass), "com_m": [0, 0, 0],
                    "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-fx, fx], [-1.0, 1.0], [floor_z - 0.05, top_z + 0.3]],
        "camera": {"target": [0.3 * fx, 0.0, 0.5 * top_z],
                   "orbit_radius": max(2.2, 2.8 * half_w), "fov_deg": 45.0,
                   "up": [0.0, 0.0, 1.0], "az0": 0.5, "el0": 0.25},
        "identified": True,
        "plugs": [{"kind": "drive",
                   "description": f"the push — an applied impulse on the "
                                  f"{names[pushed_index]} (no hand modeled)",
                   "variables": {"push_speed_m_s": v0}, "adjustable": True}],
        "choices": [push_choice.to_dict(), mu_choice.to_dict(),
                    e_choice.to_dict()],
        "source": ("Each object re-researched and re-compiled at its composed "
                   "stack pose (deckard.scene.build_stack — same citations as "
                   "the static scene). Push, friction and restitution are "
                   "disclosed Choices; whether the topper rides or is left "
                   "behind follows from the stacked-block friction "
                   "inequality, not a script. "
                   + push_choice.cite() + ". " + mu_choice.cite() + ". "
                   + e_choice.cite() + ". SIMPLIFIED_MODEL: planar +x motion, "
                   "no tipping/rolling/rotation, bbox-rectangle support "
                   "areas, no mid-air recontact, no deformation."),
    }
    outcomes = {names[i]: ("floor" if support[i] == "floor"
                           else f"on {names[support[i]]}" if support[i] is not None
                           else "airborne")
                for i in range(1, n)}
    return {
        "scene": scene_spec,
        "trajectory": {
            "frames": frames,
            "t_end_s": t,
            "natural_timescale_s": t,
            "suggested_rate": max(1e-6, t / 5.0),
            "body_labels": names,
            "validation": {
                "pushed": names[pushed_index],
                "push_speed_m_s": v0,
                "final_support": outcomes,
                "pushed_left_base": bool(support[pushed_index] != pushed_index - 1),
            },
        },
        "kind": "trajectory",
    }
