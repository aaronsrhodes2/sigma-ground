"""Generate the viewer's demo data — the exact JSON the browser loads.

Two bundles into web/data/:
  cup.json  — the coffee cup (static): orbit a CSG construct on the GPU.
  drop.json — a copper sphere dropped onto a concrete floor (trajectory): play
              it with the time-rate knob. The floor is a STATIC leaf; the sphere
              is a DYNAMIC leaf moved by the per-frame pose.
The cup also ships Python ground-truth SDF samples for the in-page self-check.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground import deckard
from sigma_ground.radiance import (construct_to_scene, record_fall,
                                  record_object_fall, record_motor_spin,
                                  record_gear_train_spin, record_escapement_clock,
                                  record_gear_mesh_spin, record_clock,
                                  record_windmill_spinup, record_windmill_theater)
from sigma_ground.radiance.scene_export import (sdf_samples, _bake_material,
                                               _default_lighting)

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)


def _write(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    print(f"  {name}: {os.path.getsize(path)//1024} KB")


# Each bundle is independent — a regression in one (e.g. a Deckard cup change)
# must NOT block the others. Wrap each, report per-bundle, keep going.
import traceback
_FAILED = []

def _bundle(name, fn):
    try:
        fn()
    except Exception as e:
        _FAILED.append((name, e))
        print(f"  !! {name} SKIPPED — {type(e).__name__}: {e}")
        traceback.print_exc()


# ── 1) coffee cup — static, orbit it ────────────────────────────────────
def _build_cup():
    cup = deckard.identify("coffee cup")
    cup_spec = construct_to_scene(cup)                 # all leaves static (no `body`)
    cup_spec["kind"] = "static"
    cup_spec["sdf_samples"] = sdf_samples(cup, 4)     # ground truth for the self-check
    _write("cup.json", cup_spec)
_bundle("cup.json", _build_cup)

# ── 2) dropped sphere + floor — trajectory, play it ─────────────────────
def _build_drop():
    out = record_fall("copper", radius_m=0.05, start_altitude_m=1.5,
                      dt_max=0.005, frame_dt=0.02, target_watch_s=8.0)
    scene = out["scene"]                              # sphere already tagged body:0
    scene["csg_leaves"][0]["temperature_k"] = 1700.0  # a HOT copper ball (sim layer) — in path-trace
    scene["name"] = "hot copper sphere (1700 K) on a cold floor"  # mode it lights the cold concrete
    scene["pt_env"] = 0.12                             # dim environment → the ball is the dominant light
    scene["csg_leaves"].append({                      # a static concrete floor at STP (no body)
        "op": "add", "material": "concrete",
        "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
                  "x": 3.0, "y": 0.1, "z": 3.0}})
    scene["materials"]["concrete"] = _bake_material("concrete", 2400.0)
    scene["camera"] = {"target": [0.0, 0.55, 0.0], "orbit_radius": 2.6,
                       "fov_deg": 42.0, "up": [0.0, 1.0, 0.0]}
    scene["bbox"] = [[-1.5, 1.5], [-0.1, 1.6], [-1.5, 1.5]]
    _lit = _default_lighting([0.0, 1.0, 0.0])
    scene["lights"] = _lit["lights"]
    scene["ambient"] = _lit["ambient"]
    scene["kind"] = "trajectory"
    _write("drop.json", {"scene": scene, "trajectory": out["trajectory"],
                         "kind": "trajectory"})
    print(f"  frames: {len(out['trajectory']['frames'])}  "
          f"fall: {out['trajectory']['t_end_s']:.2f}s  "
          f"rate: {out['trajectory']['suggested_rate']:.3f} sim-s/wall-s")
_bundle("drop.json", _build_drop)

# ── 3) emergent color — metals (Drude/Fresnel) + semiconductors (band gap) ──
def _build_materials():
    from sigma_ground.field.interface.surface import MATERIALS
    from sigma_ground.radiance.scene_export import _bake_band_gap
    print("\nEmergent color — metals = Drude/Fresnel, semiconductors = band-gap absorption; nobody chose them:")
    METAL_ROWS = [["copper", "gold", "silver", "aluminum"],
                  ["iron", "nickel", "titanium", "lead"],
                  ["tungsten", "platinum", "depleted_uranium", "steel_mild"]]
    # 4th row: the gap, not the metal model, sets the hue — yellow, lime, blue-grey, white.
    BANDGAP_ROW = ["cadmium_sulfide", "gallium_phosphide", "silicon", "titanium_dioxide"]
    ROWS = METAL_ROWS + [BANDGAP_ROW]
    r, sx, sy = 0.06, 0.18, 0.18
    leaves, mats = [], {}
    for row, rowmats in enumerate(ROWS):
        y = (1.5 - row) * sy                          # 4 rows, centered on origin
        band = rowmats is BANDGAP_ROW
        for col, mk in enumerate(rowmats):
            leaves.append({"op": "add", "material": mk, "dynamic": False,
                           "shape": {"type": "Sphere",
                                     "center": [(col - 1.5) * sx, y, 0.0], "radius": r}})
            if mk not in mats:
                mats[mk] = _bake_band_gap(mk) if band else \
                    _bake_material(mk, MATERIALS[mk]["density_kg_m3"])
                c = mats[mk]["color_rgb"]
                tag = f"Eg={mats[mk]['band_gap_ev']}eV" if band else "metal"
                print(f"  {mk:18s} #{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                      f"  emergent={mats[mk]['emergent']:d} {tag}")
    _lm = _default_lighting([0.0, 1.0, 0.0])
    _write("materials.json", {
        "name": "emergent color — metals + band-gap semiconductors", "csg_leaves": leaves,
        "materials": mats,
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.34, 0.34], [-0.34, 0.34], [-r, r]],
        "camera": {"target": [0, 0, 0], "orbit_radius": 1.5, "fov_deg": 40.0,
                   "up": [0, 1, 0], "az0": 0.0, "el0": 0.16},   # face-on: it's a flat 4×4 grid

        "lights": _lm["lights"], "ambient": _lm["ambient"], "identified": True,
        "source": "metals: Drude/Fresnel reflectance · bottom row: band-gap absorption — every color emergent, none chosen",
        "kind": "static"})
_bundle("materials.json", _build_materials)

# ── 3b) emergent ceramic glazes — color from the chromophore ION ─────────
def _build_glazes():
    # The color of a ceramic glaze EMERGES from its transition-metal ion via
    # crystal-field d-d absorption. Headline pair: Cr3+ is RED in an oxide host
    # (ruby) but GREEN in a silicate host (emerald) — same element, the crystal
    # field sets the color. Nobody picks it.
    ROW = ["ceramic", "cobalt_glaze", "chrome_glaze", "emerald_glaze",
           "copper_glaze", "titanium_glaze"]
    r, sx = 0.07, 0.185
    leaves, mats = [], {}
    print("\nEmergent ceramic glazes (crystal-field d-d; the ION sets the color):")
    for i, mk in enumerate(ROW):
        leaves.append({"op": "add", "material": mk,
                       "shape": {"type": "Sphere",
                                 "center": [(i - 2.5) * sx, 0.0, 0.0], "radius": r}})
        mats[mk] = _bake_material(mk)
        c = mats[mk]["color_rgb"]
        print(f"  {mk:16s} #{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
              f"  {mats[mk]['mechanism']}")
    _lt = _default_lighting([0.0, 1.0, 0.0])
    _write("glazes.json", {
        "name": "emergent ceramic glazes - color from the chromophore ion",
        "csg_leaves": leaves, "materials": mats,
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.6, 0.6], [-0.12, 0.12], [-r, r]],
        "camera": {"target": [0, 0, 0], "orbit_radius": 1.7, "fov_deg": 40.0,
                   "up": [0, 1, 0], "az0": 0.0, "el0": 0.14},
        "lights": _lt["lights"], "ambient": _lt["ambient"], "identified": True,
        "source": "glaze color = crystal-field d-d of the metal ion; Cr3+ is RED in oxide (ruby) but GREEN in silicate (emerald) - same element",
        "kind": "static"})
_bundle("glazes.json", _build_glazes)

# ── 4) kinematic chair tip — proves rigid ROTATION + MULTI-BODY playback ──
# HONEST LABEL: this is a *kinematic preview*. The chair's tip angle and the
# ball's bounce are SCRIPTED, not solved — it exists to prove the renderer can
# play back per-body rotation+translation and >1 independent body. The real
# tipping point and the clatter arrive when Materia's rigid-contact stage lands;
# the renderer is then ready to display whatever pose stream it produces.
def _build_tip():
    import math
    H = 0.5
    q_about_x = lambda a: [math.sin(a * H), 0.0, 0.0, math.cos(a * H)]

    # a crude chair (one rigid body) from CSG boxes, authored upright, base at y=0
    seat_y, seat_t = 0.45, 0.06
    parts = [([0.0, seat_y, 0.0], [0.46, seat_t, 0.46]),            # seat
             ([0.0, seat_y + 0.25, -0.20], [0.46, 0.50, 0.05])]     # backrest
    for lx in (-0.19, 0.19):                                        # 4 legs
        for lz in (-0.19, 0.19):
            parts.append(([lx, seat_y * 0.5, lz], [0.05, seat_y, 0.05]))

    CHAIR, BALL, FLOOR = "steel_mild", "copper", "concrete"
    leaves = [{"op": "add", "material": CHAIR, "body": 0,
               "shape": {"type": "Box", "center": c, "x": d[0], "y": d[1], "z": d[2]}}
              for c, d in parts]
    ball_r, ball_c = 0.13, [0.64, 0.13, 0.26]
    leaves.append({"op": "add", "material": BALL, "body": 1,
                   "shape": {"type": "Sphere", "center": ball_c, "radius": ball_r}})
    leaves.append({"op": "add", "material": FLOOR,                  # static (no body)
                   "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
                             "x": 4.0, "y": 0.1, "z": 4.0}})

    EDGE = [0.0, 0.0, -0.19]                    # chair tips about its back-leg floor edge
    bodies = [{"pivot": EDGE, "label": "chair (tips)"},
              {"pivot": ball_c, "label": "ball (drops)"}]

    dt, t_total = 0.02, 2.2
    g, rest, e = 9.81, ball_c[1], 0.45
    by, bvy = 1.25, 0.0
    th_max = math.radians(88.0)                 # rest just before the backrest clips the floor
    frames = []
    for i in range(int(t_total / dt) + 1):
        t = i * dt
        if t < 0.25:        theta = 0.0
        elif t < 1.15:      theta = th_max * ((t - 0.25) / 0.90) ** 2   # accelerating fall
        else:               theta = th_max                             # settled (no clatter yet)
        frames.append({"t_sim": round(t, 4), "bodies": [
            {"pos": EDGE, "quat": [round(x, 6) for x in q_about_x(-theta)]},   # −θ = tips backward
            {"pos": [ball_c[0], round(by, 5), ball_c[2]], "quat": [0.0, 0.0, 0.0, 1.0]},
        ]})
        bvy -= g * dt; by += bvy * dt           # scripted free-fall + restitution
        if by < rest: by, bvy = rest, -bvy * e

    _lt = _default_lighting([0.0, 1.0, 0.0])
    scene = {
        "name": "chair tip — kinematic preview (rigid rotation + multi-body)",
        "bodies": bodies, "csg_leaves": leaves,
        "materials": {CHAIR: _bake_material(CHAIR, 7850.0),
                      BALL: _bake_material(BALL, 8960.0),
                      FLOOR: _bake_material(FLOOR, 2400.0)},
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.6, 0.9], [-0.1, 1.3], [-1.25, 0.6]],
        "camera": {"target": [0.1, 0.32, -0.15], "orbit_radius": 3.4, "fov_deg": 42.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.9, "el0": 0.28},
        "lights": _lt["lights"], "ambient": _lt["ambient"], "identified": True,
        "source": "KINEMATIC preview — scripted tip + bounce proving per-body rotation & multi-body playback (NOT solved dynamics)",
    }
    _write("tip.json", {"scene": scene, "kind": "trajectory",
                        "trajectory": {"frames": frames, "t_end_s": round((len(frames) - 1) * dt, 4),
                                       "natural_timescale_s": t_total,
                                       "suggested_rate": max(1e-6, t_total / 6.0),
                                       "body_labels": ["chair", "ball"]}})
    print(f"  tip: {len(frames)} frames · chair 0->88° about back edge (body 0) · ball bounces (body 1)")
_bundle("tip.json", _build_tip)

# ── 5) clatter — emergent BOUNCE: each material rebounds by its own restitution ──
def _build_clatter():
    from sigma_ground.radiance.trajectory import bounce_heights
    # Dropped together, each sphere bounces with its OWN velocity-dependent
    # coefficient of restitution (Hertz/Johnson, derived from cohesive energy).
    # rubber boings back near full height (COR~1); titanium gives a modest hop;
    # copper thuds dead (COR~0.1). Nobody scripts the heights.
    ROW = [("rubber", -0.5), ("titanium", 0.0), ("copper", 0.5)]
    r, drop, dt, T = 0.07, 0.45, 0.02, 3.0
    leaves, mats, bodies, series = [], {}, [], []
    print("\nEmergent clatter (restitution + ring pitch from cohesive energy):")
    for bi, (mk, x) in enumerate(ROW):
        leaves.append({"op": "add", "material": mk, "body": bi,
                       "shape": {"type": "Sphere", "center": [x, r, 0.0], "radius": r}})
        mats[mk] = _bake_material(mk, None)
        bodies.append({"pivot": [x, r, 0.0], "label": mk})
        series.append(bounce_heights(mk, radius_m=r, drop_height_m=drop, dt=dt, t_total=T))
        c = mats[mk]
        print(f"  {mk:10s} restitution_ref={c.get('restitution_ref')}  ring={c.get('ring_frequency_hz')} Hz")
    leaves.append({"op": "add", "material": "concrete",        # static floor
                   "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
                             "x": 3.0, "y": 0.1, "z": 1.2}})
    mats["concrete"] = _bake_material("concrete", 2400.0)
    n = len(series[0])
    frames = [{"t_sim": round(i * dt, 4), "bodies": [
        {"pos": [ROW[bi][1], series[bi][i], 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}
        for bi in range(len(ROW))]} for i in range(n)]
    _lt = _default_lighting([0.0, 1.0, 0.0])
    scene = {
        "name": "clatter - emergent bounce (rubber boings, copper thuds)",
        "bodies": bodies, "csg_leaves": leaves, "materials": mats,
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.8, 0.8], [-0.1, 1.0], [-0.6, 0.6]],
        "camera": {"target": [0.0, 0.33, 0.0], "orbit_radius": 2.7, "fov_deg": 42.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.0, "el0": 0.08},
        "lights": _lt["lights"], "ambient": _lt["ambient"], "identified": True,
        "source": "each sphere rebounds by its OWN derived restitution (Hertz/Johnson from cohesive energy); rubber COR~1 boings, copper COR~0.1 thuds - nobody scripts it",
    }
    _write("clatter.json", {"scene": scene, "kind": "trajectory",
                            "trajectory": {"frames": frames, "t_end_s": round((n - 1) * dt, 4),
                                           "natural_timescale_s": T,
                                           "suggested_rate": max(1e-6, T / 8.0),
                                           "body_labels": [m for m, _ in ROW]}})
    print(f"  clatter: {n} frames, {len(ROW)} materials bouncing by emergent restitution")
_bundle("clatter.json", _build_clatter)

# ── 6) water — emergent Fresnel REFLECTION + wind-driven RIPPLES (no polygons) ──
def _build_water():
    from sigma_ground.radiance.water_waves import wind_wave_components
    # A pond stirred by a 5 m/s fan, a copper ball half-sunk so its reflection
    # shows in the surface. The ripples are REAL gravity-capillary waves (the GPU
    # evaluates the height field + analytic normal); the reflection is a Fresnel-
    # weighted ray off water (n=1.333) — clear looking down, mirror at grazing.
    comps = wind_wave_components(wind_speed=5.0, wind_dir_rad=0.5, n=7, amplitude=0.010)
    level, hx, hz, depth = 0.0, 1.0, 0.7, 0.35
    leaves = [
        {"op": "add", "material": "water",
         "shape": {"type": "Water", "center": [0.0, level - depth / 2, 0.0],
                   "x": hx, "z": hz, "depth": depth, "level": level}},
        {"op": "add", "material": "copper",
         "shape": {"type": "Sphere", "center": [0.35, 0.10, -0.15], "radius": 0.14}},
    ]
    mats = {"water": _bake_material("water", None), "copper": _bake_material("copper", 8960.0)}
    # A deep pond's BODY is dark: the blue-green that survives absorption, at a low
    # albedo (most light transmits down into the dark and never returns). That is
    # the surface body — not the near-clear thin-film of a drinking glass. The
    # brightness on the water comes from the Fresnel REFLECTION, not the body.
    try:
        from sigma_ground.field.interface.optics import dielectric_color_rgb
        hue = dielectric_color_rgb("water", "water_blue", 2.5)        # blue-green survivor
        mats["water"]["color_rgb"] = [round(0.13 * v, 4) for v in hue]  # low deep-water albedo
        mats["water"]["render_note"] = "deep-water body: absorption hue at low albedo"
    except Exception:
        mats["water"]["color_rgb"] = [0.02, 0.06, 0.08]
    _lt = _default_lighting([0.0, 1.0, 0.0])
    print(f"  water: {len(comps)} wave components, reflect_r0={mats['water'].get('reflect_r0')}, "
          f"n={mats['water'].get('refractive_index')}")
    _write("water.json", {
        "name": "water - emergent Fresnel reflection + wind ripples (no polygons)",
        "csg_leaves": leaves, "materials": mats,
        "water": {"components": comps, "wind_speed_m_s": 5.0, "wind_dir_rad": 0.5},
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-1.1, 1.1], [-0.4, 0.4], [-0.8, 0.8]],
        "camera": {"target": [0.0, 0.02, 0.0], "orbit_radius": 2.4, "fov_deg": 44.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.6, "el0": 0.10},
        "lights": _lt["lights"], "ambient": _lt["ambient"], "identified": True,
        "source": "ripples = gravity-capillary dispersion (w^2 = gk + (gamma/rho)k^3) driven by a 5 m/s fan; reflection = Fresnel ray off water n=1.333, mirror at grazing",
    })
_bundle("water.json", _build_water)

# ── 7) FEATHER DROP — Deckard's shape · Materia's drag · Radiance renders ──
def _build_feather():
    # Deckard OWNS the shape: identify() compiles the real feather (cone rachis +
    # ellipsoid vane, keratin) — we author NO geometry here (the old hand-built
    # Box feather + Box floor is retired). record_object_fall drops that construct
    # through Materia's atmospheric drag and emits the viewer bundle — the exact
    # path the front-door dispatcher uses for "drop a feather". No floor: black is
    # empty space (the real floor shows through on the passthrough glasses).
    feather = deckard.identify("feather", allow_llm=False)
    out = record_object_fall(feather, start_altitude_m=8 * 0.3048)   # 8 ft → 2.4384 m
    _write("feather.json", out)
    tr = out["trajectory"]
    shapes = [l["shape"]["type"] for l in out["scene"]["csg_leaves"]]
    print("\nFEATHER DROP — Deckard shape · Materia drag · Radiance renders:")
    print(f"  shapes {shapes} · {len(tr['frames'])} frames · "
          f"fall {tr['natural_timescale_s']:.2f}s (no hand-authored geometry)")
_bundle("feather.json", _build_feather)


def _build_deckard_feather():
    """Render Deckard's ACTUAL identified feather — not a Box stand-in.

    Deckard.identify('feather') researches + compiles a primitive-kit flight
    feather: a tapered keratin shaft (Cone) + a flattened webbed vane (Ellipsoid),
    co-axial along +z. construct_to_scene() serializes the compiled SDF straight
    through; Radiance renders the cone+ellipsoid directly (newly taught to the
    viewer). Colour is the measured keratin reflectance; mass/CoM/inertia are
    Deckard's integrated values. The shape is deliberately approximate (dims
    flagged confidence 0.40) — honest, not faked.
    """
    feather = deckard.identify("feather", allow_llm=False)
    spec = construct_to_scene(feather)
    spec["kind"] = "static"
    # The auto-camera frames by bounding-diagonal, which is too tight for a long
    # thin feather (it overflows the fov). Pull back and take a 3/4 view so the
    # broad vane face AND the full 125 mm length read; z is up (Deckard's axis).
    bb = spec["bbox"]; cz = 0.5 * (bb[2][0] + bb[2][1])
    spec["camera"] = {"target": [0.0, 0.0, cz], "orbit_radius": 0.34, "fov_deg": 40.0,
                      "up": [0.0, 0.0, 1.0], "az0": 0.62, "el0": 0.18}
    spec["source"] = ("Deckard-identified feather: Cone rachis (0.8mm x 120mm) + "
                      "Ellipsoid vane (0.6 x 24 x 100mm), keratin -- Radiance renders the "
                      "compiled SDF directly. Primitive-kit approximation, dims conf 0.40.")
    print("\nDECKARD FEATHER -> Radiance:")
    print(f"  leaves   : {[(l['shape']['type'], l['material']) for l in spec['csg_leaves']]}")
    print(f"  mass     : {spec['physics']['mass_kg']*1000:.3f} g  (keratin, density [estimated])")
    print(f"  colour   : keratin {spec['materials'][spec['csg_leaves'][0]['material']]['color_rgb']}")
    _write("deckard_feather.json", spec)
_bundle("deckard_feather.json", _build_deckard_feather)

# ── 8) WINDWARD FIELD DEMO — the flagship's per-cell channel, as a demo file ──
def _build_windward():
    """The drag-heated iron ball with its per-cell windward temperature field:
    each interval's drag dissipation lands on the leading face (Newtonian cosθ)
    and Fourier-conducts inward (diffuse_fvm), adiabatic to match the cited
    f=1 bulk ΔT — leading face crosses the Draper point first, the whole ball
    glows by impact. The exact bundle the front door's "yes" produces for
    "does an iron sphere heat up falling from 30 km?"."""
    from sigma_ground.radiance.thermal_record import record_fall_thermal
    out = record_fall_thermal("iron", 0.05, 30_000.0, windward_field=True)
    _write("windward_iron_ball.json", out)
    f = out["scene"]["csg_leaves"][0]["fields"]["temperature_k"]
    print(f"  windward field: {len(f['keyframes'])} keyframes @ 24^3 u8, "
          f"T in [{f['t_min']:.0f}, {f['t_max']:.0f}] K "
          f"(deposited {out['trajectory']['validation']['windward_deposited_J']:.0f} J)")
_bundle("windward_iron_ball.json", _build_windward)

# ── 9) MOTOR SPIN — the clock demo's Phase 0: SOLVED rotation, not scripted ──
def _build_motor_spin():
    """A bare disc, spun by dynamics/joints.py's torque-capped RevoluteJoint
    motor — the first trajectory bundle whose `quat` per frame comes from the
    constraint solver itself rather than a scripted angle (contrast the
    "chair tip" bundle above, explicitly labeled kinematic). No gear teeth,
    no blueprint data yet: this proves only the render/dynamics bridge that
    the clock demo's gear train will run through."""
    out = record_motor_spin(motor_speed_rad_s=-4.0, motor_max_torque=1.5,
                            t_max=6.0, frame_dt=0.02, target_watch_s=6.0)
    _write("motor_spin.json", out)
    val = out["trajectory"]["validation"]
    print(f"  motor spin: {len(out['trajectory']['frames'])} frames · "
          f"motor_work={val['motor_work_j']:.4f} J · "
          f"energy_ledger_ok={val['energy_ledger_ok']}")
_bundle("motor_spin.json", _build_motor_spin)

# ── 10) GEAR TRAIN — the clock demo's Phase 1: coupled multi-body rotation ──
def _build_gear_train():
    """Three wheels: wheel 0 motor-driven, wheels 1-2 chained through
    dynamics/joints.py's GearCouplingJoint at fixed ratios — a kinematic
    rate constraint, not simulated tooth contact. Still placeholder wheels
    (no real tooth geometry/ratios — those arrive with blueprint
    extraction); this proves multi-body SOLVED rotation renders correctly."""
    out = record_gear_train_spin(ratios=(1.8, -1.5), motor_speed_rad_s=-4.0,
                                 motor_max_torque=1.5, t_max=6.0,
                                 frame_dt=0.02, target_watch_s=6.0)
    _write("gear_train.json", out)
    val = out["trajectory"]["validation"]
    print(f"  gear train: {len(out['trajectory']['frames'])} frames · "
          f"{len(out['scene']['bodies'])} wheels · "
          f"motor_work={val['motor_work_j']:.4f} J · "
          f"energy_ledger_ok={val['energy_ledger_ok']}")
_bundle("gear_train.json", _build_gear_train)

# ── 11) ESCAPEMENT CLOCK — the clock demo's Phase 3: spring + escapement ──
def _build_escapement_clock():
    """The Phase 3 capstone: dynamics/mechanisms/spring.py's MainspringState
    driving an escape wheel through dynamics/mechanisms/escapement.py's
    Escapement, gated by a real pendulum's own half-period — both pieces
    independently gated against closed forms before being wired together
    here. Still placeholder shapes and no real gear train yet; proves the
    highest-risk phase's physics alone, watchably."""
    out = record_escapement_clock(t_max=14.0, target_watch_s=14.0)
    _write("escapement_clock.json", out)
    val = out["trajectory"]["validation"]
    print(f"  escapement clock: {len(out['trajectory']['frames'])} frames · "
          f"{val['ticks']} ticks · "
          f"spring_wound_out={val['spring_wound_out']} · "
          f"energy_ledger_ok={val['energy_ledger_ok']}")
_bundle("escapement_clock.json", _build_escapement_clock)

# ── 12) MESHING GEARS — the clock demo's Phase 4: real involute teeth ──
def _build_gear_mesh():
    """Two real InvoluteGear shapes (kernel/gear.py, adversarially verified)
    meshing at the standard center distance, counter-rotating at the exact
    tooth ratio via GearCouplingJoint. Module is [estimated] — the Kelly
    catalog source cites tooth counts but no module (flagged gap)."""
    out = record_gear_mesh_spin(t_max=12.0, target_watch_s=12.0)
    _write("gear_mesh.json", out)
    val = out["trajectory"]["validation"]
    print(f"  gear mesh: {len(out['trajectory']['frames'])} frames · "
          f"ratio={val['ratio_commanded']:.3f} · "
          f"center_distance={val['center_distance_m'] * 1000:.1f}mm · "
          f"energy_ledger_ok={val['energy_ledger_ok']}")
_bundle("gear_mesh.json", _build_gear_mesh)

# ── 13) THE CLOCK — Phase 5: the full assembly, keeping real time ──
def _build_clock():
    """Kelly (1944) cited going train + spring + pendulum-gated escapement +
    real involute teeth + hands. The minute hand turns 2pi per 3600
    simulated seconds, DERIVED from the cited ratios (observed 0.07%
    accuracy in the gates), not tuned."""
    out = record_clock(t_max=60.0, target_watch_s=20.0)
    _write("clock.json", out)
    val = out["trajectory"]["validation"]
    print(f"  clock: {len(out['trajectory']['frames'])} frames · "
          f"{val['ticks']} ticks · pendulum L={val['pendulum_length_m']:.4f} m · "
          f"energy_ledger_ok={val['energy_ledger_ok']}")
_bundle("clock.json", _build_clock)

# ── 14) WINDMILL — the first fully NATURAL drive (nothing plugged) ──
def _build_windmill(wind_speed, slug):
    """Wind on pitched blades spins the rotor to its emergent terminal tip
    speed — flat-plate model (flagged), RigidBearing mount (KNOWN_GAPS.md).
    Two wind speeds recorded: 'adjustable wind' as a parameter sweep of
    frozen runs, per the renderer-plays-frozen-output doctrine."""
    out = record_windmill_spinup(wind_speed_m_s=wind_speed, t_max=40.0,
                                 target_watch_s=15.0)
    _write(slug, out)
    val = out["trajectory"]["validation"]
    print(f"  windmill {wind_speed:g} m/s: {len(out['trajectory']['frames'])} "
          f"frames · omega(end)={val['final_omega_rad_s']:.2f} of "
          f"omega*={val['terminal_omega_expected_rad_s']:.2f} rad/s · "
          f"plugs={out['scene']['plugs']}")
_bundle("windmill_10ms.json", lambda: _build_windmill(10.0, "windmill_10ms.json"))
_bundle("windmill_5ms.json", lambda: _build_windmill(5.0, "windmill_5ms.json"))

# ── 15) WINDMILL THEATER — Arc A capstone: gearset + slider-crank + pump ──
def _build_windmill_theater(wind_speed, slug):
    """The vision statement's second worked example, finished: wind spins a
    rotor -> BearingGearCoupling (load-blind, KNOWN_GAPS.md) -> real 2-stage
    InvoluteGear spur train -> a slider-crank (gated against the closed
    form s(theta)=r*cos(theta)+sqrt(l^2-r^2*sin(theta)^2)) -> a
    ReciprocatingPumpState (SIMPLIFIED_MODEL) filling a cosmetic reservoir.
    Two wind speeds recorded as a frozen parameter sweep, same doctrine as
    the plain windmill demo."""
    out = record_windmill_theater(wind_speed_m_s=wind_speed, t_max=40.0,
                                  target_watch_s=15.0)
    _write(slug, out)
    val = out["trajectory"]["validation"]
    print(f"  windmill theater {wind_speed:g} m/s: "
          f"{len(out['trajectory']['frames'])} frames · "
          f"omega(end)={val['final_omega_rad_s']:.2f} of "
          f"omega*={val['terminal_omega_expected_rad_s']:.2f} rad/s · "
          f"pump_strokes={val['pump_strokes']} · "
          f"pump_volume_m3={val['pump_volume_m3']:.6f} of "
          f"tank_capacity_m3={val['tank_capacity_m3']:.6f} · "
          f"plugs={len(out['scene']['plugs'])} · "
          f"choices={len(out['scene']['choices'])}")
_bundle("windmill_theater_10ms.json",
       lambda: _build_windmill_theater(10.0, "windmill_theater_10ms.json"))
_bundle("windmill_theater_5ms.json",
       lambda: _build_windmill_theater(5.0, "windmill_theater_5ms.json"))

if _FAILED:
    print(f"\n!! {len(_FAILED)} bundle(s) skipped: " + ", ".join(n for n, _ in _FAILED)
          + " -- others written OK.")
print("\nDemo data written. Serve with:  python -m sigma_ground.radiance.web.serve")
