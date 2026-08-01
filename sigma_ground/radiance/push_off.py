"""record_push_off — push one object of a composed stack and watch what follows.

The follow-up half of the doughnut/plate/table arc: ``items`` is the same
bottom-to-top list the SCENE stack route composed ("1700s table", "plate",
"Doughnut"), rebuilt here per-object via ``deckard.scene.build_stack`` so each
body keeps its own real mass/bbox at its composed pose. The pushed body gets
an initial horizontal velocity (+x); everything above it rides on friction —
and whether it rides, slips, or is left behind is DECIDED by the mu values,
not scripted. With the default hard push the plate slides out from under the
doughnut exactly like the tablecloth trick: the dough-on-glaze friction (the
doughnut is iced on TOP only — its bottom is fried dough) cannot supply the
acceleration the escaping plate demands, the relative slip exceeds the
plate's own radius, and the doughnut drops to the table while the plate
carries on over the edge. Vary the PUSH SPEED — the one artificial force in
the scene, and per the design rule the preferred knob precisely because it
is already supernatural — and the same friction yields different endings:
below ~1.7 m/s everything stays on the table (the plate stops short of the
edge); above it, the tablecloth trick. Probed empirically: there is NO
mid-speed window where the doughnut rides the plate off — and the anatomy
correction is exactly why. Dough-on-glaze (~0.38) barely exceeds
plate-on-table (~0.35), so the doughnut can never lock onto the plate in
time AND still reach the edge; a full-coat sticky-icing doughnut (the old
wrong model, mu ~0.5+) WOULD have had a ride-along window. The Captain's
"it's fried dough underneath" is the reason the doughnut always gets left
behind.

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

# The recorder's disclosed Choice VALUES, shared with the live loop so both
# run the same matter (the Choice objects with their full checked/reason
# text are still built in record_push_off — these are the numbers).
DEFAULT_MU = {"plate_on_table": 0.35, "topper_on_plate": 0.38,
              "topper_on_table": 0.45, "anything_on_floor": 0.40}
DEFAULT_E = {"plate": 0.30, "topper": 0.05}


def make_lookups(pushed_index, mu=None, e=None):
    """(mu_lookup, e_lookup) for StackSim from the pairwise Choice dicts."""
    MU = dict(DEFAULT_MU if mu is None else mu)
    E = dict(DEFAULT_E if e is None else e)

    def _mu(body, support):
        if support == "floor":
            return MU["anything_on_floor"]
        if body == pushed_index and support == 0:
            return MU["plate_on_table"]
        if body > pushed_index and support == pushed_index:
            return MU["topper_on_plate"]
        return MU["topper_on_table"]

    def _e(body):
        return E["plate" if body == pushed_index else "topper"]
    return _mu, _e


class StackSim:
    """The stacked-block integrator, step-callable — physics in ONE place.

    record_push_off() bakes frames through this exact object; the live
    god-hand loop (radiance.web.serve /live websocket) steps the same
    object in real time. There is no second integrator to diverge.

    Live artificial-force entry points (the closed taxonomy,
    dynamics.mechanisms.applied — Golden Rule 7's vocabulary):
      impulse — instantaneous Δv on one body
      force   — sustained horizontal acceleration over a window
      tilt    — re-aim gravity relative to the scene (the tray in the
                operator's hands); ramps to angle_deg over duration
      hold    — pin one body against the world (an artificial bracket)
    SOURCE (heat) and DRIVE (joints) have no state in this planar stack
    model — the live loop refuses them honestly rather than faking.

    SIMPLIFIED_MODEL scope is record_push_off's docstring, unchanged:
    planar +x motion, no tipping/rotation, bbox support rectangles.
    """

    def __init__(self, built, *, mu_lookup, e_lookup):
        self.names = [name for name, _c in built]
        cons = [c for _n, c in built]
        n = self.n = len(built)
        self.bbox = [c.bbox for c in cons]
        self.mass = [c.mass_kg for c in cons]
        self.com = [list(c.com_m) for c in cons]
        self.bot_off = [self.com[i][2] - self.bbox[i][2][0] for i in range(n)]
        self.floor_z = self.bbox[0][2][0]
        self._mu = mu_lookup
        self._e = e_lookup
        self.dx = [0.0] * n
        self.dz = [0.0] * n
        self.vx = [0.0] * n
        self.vz = [0.0] * n
        self.support = [None] + [i - 1 for i in range(1, n)]
        self.t = 0.0
        self.EPS_V = 0.02
        # god-hand state
        self.gx, self.gz = 0.0, G           # gravity vector (tilt re-aims it)
        self._forces = []                   # [(body, ax_m_s2, t_end), ...]
        self._holds = set()                 # pinned body indices
        self._tilt = None                   # (angle_rad_target, t0, t1)

    # ── the closed taxonomy's live entry points ────────────────────────────
    def apply_impulse(self, body, dv_x):
        self.vx[body] += dv_x
        self.support[body] = self.support[body]     # (state unchanged)

    def apply_force(self, body, fx_n, duration_s):
        self._forces.append((body, fx_n / self.mass[body],
                             self.t + duration_s))

    def apply_tilt(self, angle_deg, duration_s):
        self._tilt = (math.radians(angle_deg), self.t,
                      self.t + max(duration_s, 1e-6))

    def hold(self, body, on=True):
        (self._holds.add if on else self._holds.discard)(body)

    # ── geometry helpers (verbatim physics from the recorder) ──────────────
    def _rect(self, i):
        (x0, x1), _yy, _zz = self.bbox[i]
        return x0 + self.dx[i], x1 + self.dx[i]

    def _on(self, i, s):
        if s == "floor":
            return True
        x = self.com[i][0] + self.dx[i]
        x0, x1 = self._rect(s)
        return x0 <= x <= x1

    def _contact_dz(self, i, s):
        top = self.floor_z if s == "floor" else self.bbox[s][2][1] + self.dz[s]
        return top + self.bot_off[i] - self.com[i][2]

    def _land_surface(self, i, z_bottom_prev):
        """Highest shelf under body i, qualified against the PRE-step bottom.
        Qualifying against the post-step position tunnels (caught live: the
        doughnut fell straight through the table top to the floor)."""
        x = self.com[i][0] + self.dx[i]
        best, best_z = "floor", self.floor_z
        for j in range(self.n):
            if j == i:
                continue
            if j != 0 and self.support[j] is None:
                continue                               # a falling body is no shelf
            x0, x1 = self._rect(j)
            top = self.bbox[j][2][1] + self.dz[j]
            if x0 <= x <= x1 and best_z < top <= z_bottom_prev + 1e-9:
                best, best_z = j, top
        return best

    def step(self, dt):
        """One substep — the recorder's loop body, verbatim physics, plus
        the god-hand inputs (sustained forces, gravity tilt, holds)."""
        n, dx, dz, vx, vz = self.n, self.dx, self.dz, self.vx, self.vz
        support, mass, com = self.support, self.mass, self.com
        if self._tilt is not None:
            ang, t0, t1 = self._tilt
            frac = min(1.0, max(0.0, (self.t - t0) / (t1 - t0)))
            self.gx = G * math.sin(ang * frac)
            self.gz = G * math.cos(ang * frac)
        self._forces = [f for f in self._forces if f[2] > self.t]
        for i in range(n - 1, 0, -1):                  # riders first
            if i in self._holds:
                vx[i] = vz[i] = 0.0                    # the artificial bracket
                continue
            s = support[i]
            # a support in free fall exerts NO normal force — its rider falls
            # with it (caught by probing the ride-along regime).
            if isinstance(s, int) and s != 0 and support[s] is None:
                support[i] = s = None
            # god-hand inputs enter as accelerations BEFORE the contact
            # logic, so friction reacts to them like to any other motion
            a_ext = self.gx + sum(a for b, a, _te in self._forces if b == i)
            if s is None:                              # ballistic
                z_bottom_prev = com[i][2] + dz[i] - self.bot_off[i]
                vx[i] += a_ext * dt
                vz[i] -= self.gz * dt
                dx[i] += vx[i] * dt
                dz[i] += vz[i] * dt
                land = self._land_surface(i, z_bottom_prev)
                cdz = self._contact_dz(i, land)
                if dz[i] <= cdz:
                    dz[i] = cdz
                    e = self._e(i)
                    vimp = abs(vz[i])
                    vz[i] = e * vimp
                    mu = self._mu(i, land)
                    if vx[i]:
                        dvx = min(abs(vx[i]), mu * (1 + e) * vimp)
                        vx[i] -= math.copysign(dvx, vx[i])
                    if vz[i] < 0.25:                   # too slow to bounce visibly
                        vz[i] = 0.0
                        support[i] = land
                continue
            # supported: one Coulomb branch covers rider-on-moving-support and
            # pushed-body-on-static-surface. Normal load = own weight + riders;
            # under tilt the normal uses the gz component.
            vx[i] += a_ext * dt
            sv = vx[s] if isinstance(s, int) and s != 0 else 0.0
            vrel = vx[i] - sv
            load = mass[i] + sum(mass[j] for j in range(1, n)
                                 if support[j] == i)
            mu = self._mu(i, s)
            if abs(vrel) > self.EPS_V:
                a_i = mu * self.gz * (load / mass[i])
                dv = min(abs(vrel), a_i * dt)          # never overshoot to reverse slip
                vx[i] += -math.copysign(dv, vrel)
                if isinstance(s, int) and s != 0:
                    vx[s] += math.copysign(mu * self.gz * load / mass[s] * dt,
                                           vrel)
            else:
                # locked to the support UNLESS a live input pushes past
                # static grip this substep (mu_s ~ mu_k, SIMPLIFIED_MODEL)
                if abs(a_ext) <= mu * self.gz * (load / mass[i]) + 1e-12:
                    vx[i] = sv
            dx[i] += vx[i] * dt
            if not self._on(i, s):
                support[i] = None                      # over the edge
        self.t += dt

    def settled(self):
        return all(self.support[i] is not None
                   and abs(self.vx[i]) < self.EPS_V and self.vz[i] == 0.0
                   for i in range(1, self.n)) and not self._forces \
            and self._tilt is None

    def energy(self):
        n, mass, vx, vz = self.n, self.mass, self.vx, self.vz
        com, dx, dz = self.com, self.dx, self.dz
        ke = sum(0.5 * mass[i] * (vx[i] ** 2 + vz[i] ** 2) for i in range(1, n))
        px = sum(mass[i] * vx[i] for i in range(1, n))
        pz = sum(mass[i] * vz[i] for i in range(1, n))
        ly = sum(mass[i] * ((com[i][2] + dz[i]) * vx[i]
                            - (com[i][0] + dx[i]) * vz[i]) for i in range(1, n))
        return {"energy_j": ke, "momentum": [px, 0.0, pz],
                "angular_momentum": [0.0, ly, 0.0]}

    def frame(self):
        return {"t_sim": round(self.t, 4),
                "bodies": [{"pos": [round(self.dx[i], 5), 0.0,
                                    round(self.dz[i], 5)],
                            "quat": [0.0, 0.0, 0.0, 1.0]}
                           for i in range(self.n)],
                **self.energy()}


def record_push_off(items, pushed_index: int = 1, push_speed_m_s=None, *,
                    mu_overrides: dict | None = None, prebuilt=None,
                    allow_llm: bool = False, dt: float = 1.0 / 960.0,
                    frame_dt: float = 1.0 / 120.0, t_max: float = 6.0) -> dict:
    from ..deckard.scene import build_stack, _slug
    from ..dynamics.mechanisms.choice import Choice
    from .scene_export import _bake_material, _shape_to_dict, _bounding_sphere

    # ``prebuilt``: the [(name, Construct), ...] from a prior build_stack call.
    # A push-speed VARIANT SWEEP (Golden Rule 7: the artificial force is the
    # one legitimate during-sim knob) re-simulates the same matter at several
    # speeds — research once, push many times.
    built = prebuilt if prebuilt is not None else build_stack(items, allow_llm=allow_llm)
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
        dict(DEFAULT_MU),
        checked="no measured friction data on disk (TPSX has none; "
                "materials_engineering.json has no friction field); the "
                "derived Bowden-Tabor value is a vacuum-regime constant ~1.0 "
                "(degeneracy probe) and would make nothing ever slide",
        reason="typical engineering ranges: glazed ceramic on finished wood "
               "~0.3-0.4. topper_on_plate is FRIED DOUGH on glaze — the "
               "doughnut is iced on top only (Captain's anatomy correction, "
               "2026-07-30; the archetype's coat is clipped to the upper "
               "half), so the bottom contact is dough, slightly greasy, taken "
               "~0.38 — NOT the old sticky-icing 0.5, which assumed a "
               "full-coat doughnut sliding on its icing. Values remain "
               "adjustable, but per the design rule the PREFERRED knob is "
               "the artificial force (push speed): matter is fact, the plug "
               "is already supernatural",
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
        dict(DEFAULT_E),
        checked="impact.coefficient_of_restitution needs a MATERIALS-table "
                "key; 'stoneware'/'leavened dough' are Deckard vocabulary, "
                "not in that table — honest Choice instead of a bad alias",
        reason="stiff ceramic bounces a little; a doughnut lands like a "
               "doughnut", adjustable=True)

    _mu, _e = make_lookups(pushed_index, MU, e_choice.value)

    # the integrator lives in StackSim — the SAME object the live god-hand
    # loop steps; the recorder just bakes its frames.
    sim = StackSim(built, mu_lookup=_mu, e_lookup=_e)
    sim.vx[pushed_index] = v0                # the impulse, as initial condition

    frames = []
    next_rec = 0.0
    while sim.t < t_max:
        if sim.t >= next_rec:
            frames.append(sim.frame())
            next_rec += frame_dt
        sim.step(dt)
        if sim.settled():
            break
    frames.append(sim.frame())
    t, support = sim.t, sim.support

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
        # the push, disclosed through the closed artificial-force taxonomy
        # (dynamics.mechanisms.applied): an IMPULSE at the CoM, J = m*v0.
        # Spec and disclosure are the same object — Golden Rule 7's vocabulary.
        "plugs": [_push_af(names[pushed_index], mass[pushed_index], v0).to_plug()],
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
        # sim provenance: enough for the viewer/server to re-run this same
        # matter with a DIFFERENT artificial force (the /apply endpoint) —
        # research is re-done from the names, so nothing here is a cache of
        # physics, only of what the scene IS.
        "sim": {"builder": "push_off", "items": list(items),
                "pushed": names[pushed_index], "pushed_index": pushed_index},
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


def _push_af(target: str, target_mass_kg: float, v0: float):
    """The push as a member of the closed artificial-force set."""
    from ..dynamics.mechanisms.applied import AppliedForce
    return AppliedForce(
        "impulse", target,
        label=f"the push — an applied impulse on the {target} (no hand modeled)",
        vector=[round(target_mass_kg * v0, 4), 0.0, 0.0],
        variables={"push_speed_m_s": v0})
