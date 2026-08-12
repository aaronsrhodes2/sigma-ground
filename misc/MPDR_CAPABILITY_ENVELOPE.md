# MPDR Capability Envelope — what the engine can host, today and at each roadmap phase

**MPDR** = Multi-Physics Dodoxel Renderer (named at the game-engine pivot,
`misc/SESSION_LOG.md:2492`). Audience: the volume-and-game-design thread.
Question answered: how many independent objects, zoomed how close, at what
frame rate, with what physics population — TODAY, and at R3 / R4 / R5 / E7.

**Measurement environment (all [MEASURED] rows):** ANGLE (NVIDIA GeForce RTX
4070 Ti SUPER (0x00002705) Direct3D11 vs_5_0 ps_5_0, D3D11), vendor
Google Inc. (NVIDIA), devicePixelRatio=1 throughout. Measured 2026-08-04 on
this machine via a dedicated CDP-driven Chrome 150 instance (procedure in §7).

**Citation pin (added by the 2026-08-05 number audit):** all [CODE-VERIFIED]
file:line citations resolve against commit `80a277c` (repo HEAD when this
document was written). The working tree already carries uncommitted R0-lane
edits to viewer.js that shift later line numbers by roughly +50 (e.g. the
§2.1 field guard sits at :1282 there, not :1229) — verify quotes against the
pinned commit, not the drifting working copy. All other cited files were
unmodified at audit time.

## Tag legend (epistemic law — every number carries exactly one)

| Tag | Meaning |
|---|---|
| **[MEASURED]** | Obtained 2026-08-04 on this machine; procedure and environment stated. |
| **[CODE-VERIFIED]** | A ceiling/constant read from a named file:line, code quoted. |
| **[PROJECTED]** | Roadmap estimate with arithmetic shown inline and assumptions named as disclosed Choices. |

Anything that cannot carry a tag is omitted or listed in §7 under
"unmeasured — and how we will measure it".

---

## 0. One-page summary for game designers

| Object class | TODAY | R3 (splats) | R4 (unified brick grid) | R5 (streaming) |
|---|---|---|---|---|
| Voxel-textured bodies per scene | **15** [CODE-VERIFIED §2.1] | 15 (unchanged) | **~12,000** at today's planet resolution under a 4 GB atlas budget [PROJECTED §3.2] | same count; zoom floor becomes per-view, not per-bake [PROJECTED mechanism §3.3] |
| Analytic CSG leaves per scene | **44 completes in ~59 s, 64-88 hang (>90s), 128+ HARD FAILS to compile** ("Expression too complex") [MEASURED §2.2]; growth exponent b≈1.74, isolated to the SDF/compose axis, not material lookups | unchanged (Phase 1's material-LUT fix buys ~2.4x constant factor, confirmed NOT to fix the exponent [MEASURED §2.2]) | ceiling **dies** — shader becomes leaf-count-independent [PROJECTED §3.2; R4.7 Phase 2 targets this axis specifically] | n/a |
| Tiny/distant bodies (sub-3-px) | each costs a full leaf (see above) | **tens of thousands** as GL_POINTS; real bound = per-frame CPU upload loop, not GPU [PROJECTED §3.1] | absorbed into grid | streamed |
| Animated field leaves (e.g. water) | **≤ 7** at 0 voxel bodies; F ≤ floor((15−V)/2) [CODE-VERIFIED §2.1] | unchanged | re-architected with the grid (no committed design number) | — |
| Rigid bodies at 960 Hz real time | **≤ 50** (0.789 ms/step at n=50 vs 1.042 ms budget) [MEASURED+CODE-VERIFIED §6] | — | — | E7: **~220–290** real-time; **~500** at 3.2–5.3 ms/step [PROJECTED §3.4] |
| Planar stack bodies (StackSim) | n=3 measured at **0.0024 ms/step**; O(n) chain, no coded cap [MEASURED+CODE-VERIFIED §6] | — | — | — |
| N-body gravity population | 101 bodies at 52.2 ms per 1 s sim-step (far faster than real time at native cadence) [CODE-VERIFIED §6] | — | — | E7 same 15–25x lever [PROJECTED §3.4] |
| Zoom floor, Earth voxel sphere | **293 km/cell** [CODE-VERIFIED §4] | unchanged | **~150 km/cell** (96³ tier) or **~75 km/cell** (192³ tier), trading object count [PROJECTED §4] | floor set by per-view VRAM budget while flying [PROJECTED mechanism §4] |
| Camera range (real scale) | 1 AU → surface works; float32 rebase landed (R1-lite) [CODE-VERIFIED §2.7] | — | — | — |
| Frame rate | 60 fps vsync-capped in 4 of 5 measured scenes; 48-leaf CSG scene 2–5 fps [MEASURED §1] | — | — | — |

Headline honesty notes: 60 fps readings are **vsync-capped** — GPU headroom
above 60 is unmeasured (§5). Path tracing is **progressive**: a still camera
converges at fps×3 samples/px/s; a moving camera never converges (§5).

---

## 1. Today's measured envelope

All rows: window pinned 1280×800 outer, dpr=1, `document.visibilityState ==
'visible'` verified on every accepted reading (validity requirement — a hidden
tab throttles requestAnimationFrame and fakes the number). Canvas backing
HEIGHT varies per scene (206–609 px) because the Mentat HUD/chat chrome
differs per scene layout — fragment-bound results do not transfer across
resolutions without rescaling by pixel count (disclosed Choice). Sim scenes
measured as-loaded (paused at t=0); fps under physics playback unmeasured.

| Scene | fps [MEASURED] | Canvas backing | Convergence [MEASURED] | Notes |
|---|---|---|---|---|
| fall_sphere_iron (ceiling reference) | 60, 60, 60 | 1264×609 | 179.9 samples/px/s = 60 fps × 3 spp/frame exactly ((2466−1746) spp over 4.002 s) | tiny scene; true GPU cost hidden by vsync |
| water (wave field) | 60, 60, 60 | 1264×590 | bounded shutter by design: holds ~46–48 photons/px in a 24-frame rolling exposure (NMAX=24, viewer.js:1542); spp/s = fps × 2 = 120 into the window; never converges further | animated branch |
| clock (heaviest analytic SDF) | 60 (reads 2–3; read 1 caught pre-first-frame state — compile blocked ~8 s past the 15 s settle) | 1264×206 | 154 spp/s = 86% of the 180 ideal — counter said 60 but real frame delivery dropped ~14% over that window | small canvas: 206 px tall — do not quote this as full-window 60 fps |
| scene_doughnut_on_plate_on_1700s_table (48 CSG leaves) | **2–5** (reads 5/2/4; late read 5) | 1264×445 | 12.0 spp/s (= 4 fps × 3; (90−66) spp over 2.001 s) | first frame ~10.5 s after navigation — buildPathShader compile blocks the main thread (live evidence of the §2.2 mechanism) |
| solar_system_theater, default camera (30 bodies, 9 voxel) | 60 (reads 2–3; read 1 caught the first drawn frame) | 1264×507 | 180.4 spp/s = 60 × 3 ((1461−240) spp over 6.770 s) | cold navigation-to-first-frame ~35+ s (single run, undecomposed: 9 voxel texture fetches + shader compile; server sends no-store so every load re-fetches) |
| solar_system_theater, Earth close-up (`locateBody('Earth')` → true, HUD 'follow Earth') | 60, 60, 60 | 1264×507 | never converges deeply: 48 spp → 33 spp → 'live · 2 spp/frame · orbiting' — follow mode tracks the orbiting Earth, accumulation alternates converging bursts (tens of spp) with 2-spp moving exposures | "converged close-up of a moving body" does not exist in the current renderer; the close-up is a perpetually-noisy-but-fluid 60 fps live view |

**Visual settling, doughnut scene [MEASURED, subjective criterion disclosed]:**
criterion = "grain no longer the dominant visual impression at normal viewing
distance", judged from PNG screenshots by a single observer. Result: NOT
settled at any captured time. t=10.6 s post-navigation: pure noise. t=14.2 s
(30 spp): geometry fully readable but uniform heavy grain. t=17.1 s: noisier
again — a boot-time camera ease was still moving the view and resetting
accumulation ('orbiting (clears when you stop)'). t=29.8 s: at the close
framing, reset to 24 spp, still heavily grainy. At the measured 12 spp/s with
a still camera, even a few hundred spp takes tens of seconds — this scene
does not reach a clean image in interactive time today.

**Boot-time camera ease [MEASURED — affects every time-to-settle claim]:**
scene loads begin with a camera ease toward the scene framing (scene JSON
holds only a static orbit; the motion is the R2 controller adopting it).
While easing, the renderer sits in the moving branch (viewer.js:1547-1548;
:1586 is that branch's HUD string) — 2-spp
one-frame exposures, accumulation restarts on stillness. Settling clocks
start when the ease ends, not at page load.

---

## 2. The ceilings and their mechanisms [CODE-VERIFIED]

### 2.1 Voxel-body texture units: 15 voxel leaves; fields cost 2 units each

`MAX_TEXTURE_IMAGE_UNITS = 16` [MEASURED, WebGL2 context this session].
Unit 0 is reserved: viewer.js:1504 — "voxel 3-D textures on units >=1
(TEXTURE0 is the PT accumulation buffer)"; voxels bind at TEXTURE0+1+u
(viewer.js:1505-1507), so **V ≤ 15**. Field leaves bind TWO units each
(bracketing keyframes, viewer.js:1519-1522) with a graceful-drop guard,
viewer.js:1229-1231:

> `if(1+V+2*(list.length+1) > 16){ console.error(`field on leaf ${i} dropped — texture-unit budget exceeded (1 accum + ${V} voxels + ${2*(list.length+1)} field units > 16); bulk T still shows`)`

Max fields at V voxel bodies: **F ≤ floor((15−V)/2)** — e.g. 7 fields at V=0.
**Asymmetry:** `prepareVoxels` (viewer.js:1190-1198) has NO budget guard — a
16th voxel leaf would bind past unit 15, unguarded; behavior is unexercised
and undefined. The solar builder avoids it by policy
(build_solar_system_theater.py:13-16: "binds one 3-D texture per Voxel leaf,
and the browser's texture-unit budget caps that at roughly 15").

### 2.2 CSG-leaf shader growth: [MEASURED 2026-08-06] real curve, root cause isolated

**[MEASURED 2026-08-06]** — the gap this section used to flag as "in between
unmeasured" is closed. `tools/measure_compile_time.py` (R4.7) drove
`tools/gen_leaf_stress_scene.py`-generated scenes (analytic Sphere/Cylinder/
Cone/Box only, op:add, one shared material) through `compile_probe.html`'s
isolated compile+link timing, one fresh dedicated-Chrome instance per data
point, on this same machine/driver stack. Full campaign in
`misc/COMPILE_TIME_REPORT.json`.

| leaves | total compile+link (median of 3) | outcome |
|---|---|---|
| 4 | 842 ms | ok |
| 8 | 2,626 ms | ok |
| 11 | 4,067 ms | ok |
| 16 | 6,728 ms | ok |
| 22 | 12,059 ms | ok |
| 32 | 29,618 ms | ok |
| 44 | 58,951 ms | ok |
| 64 | — | **TIMEOUT** (>90 s) |
| 88 | — | **TIMEOUT** (>90 s), all 3 reps |
| 128 | — | **hard FAIL: "ERROR: Expression too complex."** |
| 176 | — | **hard FAIL: "ERROR: Expression too complex."** |

A power-law fit (`ms ≈ a·n^b`) on the leaves-that-completed gives **b≈1.74**
(r²=0.989) — confirmed worse-than-linear, not just suspected. New finding
this measurement surfaces: the failure mode changes character between 88 and
128 leaves — 64–88 is a slow **hang** (ANGLE/D3D11 eventually would finish,
just past any reasonable timeout, matching the pre-existing 101-leaf "tab
wedged" incident), but 128+ is a hard, **immediate** compiler rejection
("Expression too complex") — a real ceiling the compiler itself enforces,
not merely a patience problem.

**Root cause, isolated (not assumed):** the same campaign built two stubbed
shader variants from the real generated text — `stub_matchains` (the 8
per-leaf material-lookup branches collapsed to a constant, SDF/compose/
material-select left at full leaf count) and `stub_sdf` (the inverse: SDF/
compose/material-select collapsed, material chains left full) — isolating
which half of the per-leaf codegen actually drives the growth:

| variant | exponent b | r² |
|---|---|---|
| full (unmodified) | 1.741 | 0.989 |
| stub_matchains (material chains removed) | 1.743 | 0.998 |
| stub_sdf (SDF/compose/matsel removed) | 1.193 | 0.989 |

**The material-lookup chains are NOT the driver of the superlinear growth**
— removing them (`stub_matchains`) leaves the exponent essentially
unchanged (1.743 vs 1.741). Removing the SDF-evaluation/CSG-composition
code instead (`stub_sdf`) drops the exponent to near-linear (1.193). The
SDF/compose/material-select axis is unambiguously what's driving both the
worse-than-linear scaling and the hard 128-leaf ceiling. `stub_matchains`
DOES buy a real, useful constant-factor win regardless — e.g. at n=44,
24,820 ms vs full's 58,951 ms, roughly 2.4x — and even survives to n=88
(2,736 ms) where `full` and `stub_matchains` both time out; only the
SDF-heavy axis remains genuinely un-fixed by a material-side change alone.

This is the empirical justification for R4.7's phased plan
(`C:\Users\aaron\.claude\plans\expressive-whistling-squid.md`, "R4.7"
section): Phase 1 (material-LUT generalization to legacy leaves) is real,
validated, worthwhile relief — but this data confirms in advance that it
will not, by itself, meaningfully raise the ~88-128 leaf ceiling. The
deeper fix (Phase 2, a texture-based "Analytic Primitive Table" making the
SDF/compose axis itself leaf-count-independent, mirroring how SVO already
achieves this for voxel geometry) is what actually targets the dominant
driver this measurement isolated — gated on Phase 1 landing first and its
own ANGLE probe spike, not started until then.

Mechanism (unchanged from the original diagnosis): `buildPathShader`
(viewer.js, `buildPathShader` function) emits per-leaf GLSL — a `float s{i}`
SDF declaration per leaf, a min/max compose chain, an else-if material-
select chain, and (pre-R4.7) 8 per-leaf material-lookup branches (one, 
`reflfn`, was confirmed dead code in the path tracer and deleted this
session — never reached 8 live chains here, only 7). Mitigation in code:
`display_max_particles=32` evenly-spaced render-only subsample
(phase_e_export.py:33, 59-65) — a workaround for the render, not a fix for
compile time. This ceiling **dies at R4/R4.7-Phase-2** (leaf-count-
independent codegen, §3.2).

### 2.3 Rigid stepper: all-pairs O(n²), documented intent < 50

stepper.py:43-44 verbatim: "All O(n²) pairs are tested per step. For small
scenes (< 50 objects) this is acceptable. For large scenes, replace with a
spatial hash (see fluid/sph.py)." The E0 measurement agrees: n=50 at
0.789 ms/step is the last row inside the 960 Hz budget (§6).

### 2.4 N-body: pure-Python O(n²) per step

field/interface/nbody.py:399-413: `for i in range(n): ... for j in range(n):
... acc[i] += gm_j * r_ij / (r_sq * r)` — plus optional Einstein-Infeld-
Hoffmann 1PN per pair (:415+) and a second n×n loop at :385-396. Measured
scaling matches: 52.23/13.56 = 3.85x for 1.98x n (§6).

### 2.5 Continuum SPH: brute-force O(n²) neighbor search

continuum.py:376-394: density, neighbor list, and self-gravity all in one
n×n pass; no spatial hash (stepper.py:44 points at fluid/sph.py for one).

### 2.6 StackSim: one body per stack item, O(n) linear support chain, no coded cap

push_off.py:97 `n = self.n = len(built)`; :109 `self.support = [None] +
[i - 1 for i in range(1, n)]` — each body supported by the one below, so
stepping is O(n). Scope (:91): planar +x motion, no tipping/rotation, bbox
support rectangles. No ceiling constant exists; the bench pins n=3 and
harness.py:65-66 states "its real use case is a small stack, many steps, not
a large-N sweep".

### 2.7 float32 / real-scale camera: R1-lite rebase landed

Commit `eae812a` "R1-lite: camera-anchored coordinate rebase + view-scale
epsilons + locate" (2026-08-03). viewer.js:1457-1471: every GPU position is
expressed relative to the eye, "computed here in JS float64 BEFORE the
Float32Array cast"; uEye uploads as exactly [0,0,0] (:1491); "without it,
float32 can't represent a body's own few-km surface detail once its
coordinate exceeds roughly 1e11 m". View-scale epsilons at :1478-1488.
Result: bodies at real ~1.5e11 m positions render cleanly. The specific
"~1.7 km precision cliff" figure exists only in misc/SESSION_LOG.md:2497-2500
(session narrative); **no automated regression test covers the rebase** —
verification was a live pixel-match vs a near-origin control per the commit
message.

### 2.8 WorldHost cadence

host.py:71-73 verbatim: `SIM_DT = 1.0 / 960.0`, `SEND_DT = 1.0 / 30.0  # sim-
time between broadcast snapshots`, `MAX_CATCHUP = 4000  # bounded catch-up:
max steps per iteration`. Loop at host.py:437-442 steps until sim.t reaches
the wall-clock target or MAX_CATCHUP, emitting a snapshot when
sim.t ≥ next_send — the wire carries 30 snapshots per SIM-second regardless
of physics rate. PACE_MIN/PACE_MAX = 0.01/10.0 (:74).

### 2.9 E1 ECS: no entity ceiling by design; Region membership is the real bound

ecs.py:1-9 — generational (index, generation) handles; stale handles raise
instead of aliasing. Storage is plain unbounded Python lists (ecs.py:184-186:
`self._slots: list = []` etc.) — no capacity constant anywhere. The actual
engine-population bound today is the Region layer: exactly one adapter
exists, region.py:100 `class StackSimRegion` (grep for `class \w+Region`
returns only this), wrapping the planar StackSim verbatim (:101-113) — so
the whole engine-hosted world today is one small planar stack, with
SOURCE/DRIVE force kinds refused honestly (:136-137).

---

## 3. Roadmap envelope [PROJECTED — arithmetic shown, Choices named]

### 3.1 R3 splats: sub-3-px bodies as GL_POINTS

Cost model: O(1 vertex + 1 point-sprite fragment) per body. GPU side:
100,000 points × 60 fps = **6M vertices/s** — orders of magnitude below the
vertex throughput of a 4070 Ti-class GPU (billions/s class), so the GPU is
not the bound. The real bound is the **per-frame JavaScript upload**: each
body needs a float32 rebased position (R1-lite: float64 subtract per body in
JS, §2.7), so 100,000 × 3 floats × 4 B × 60 Hz = **72 MB/s** buffer traffic
— feasible, but the per-body JS loop (rebase + pack) is CPU work.
**Choices:** (a) 100k population is an assumed Kuiper-belt-scale target, not
a requirement; (b) the CPU loop cost is the thing to measure when R3 lands —
no number is claimed for it here. Verdict: thousands to tens of thousands of
splat bodies at 60 fps is plausible on this arithmetic, pending the CPU-loop
measurement.

**[MEASURED 2026-08-05] R3 CPU loop (classify + occlude + pack), as shipped
in splats.js, in-browser on the measurement environment (§ header):**

| Population | median ms/frame | p95 ms/frame | Procedure |
|---|---|---|---|
| 30 bodies (live solar theater) | 0.10 (mean 0.070) | 0.20 | `window._r3.stats.packMs` sampled over 600 rAF frames at 60 fps (max 0.30; browser timer quantum 0.1 ms) |
| 10,000 (synthetic) | 0.5 | 0.7 | `window._r3.bench(10000, 100)` — index-derived golden-spiral positions, no RNG, medians over 100 reps |
| 100,000 (synthetic) | 4.8 | 6.1 | `window._r3.bench(100000, 100)` (max 8.2) |

At 100k the pack loop consumes ~29% of a 16.7 ms frame — the §3.1 verdict
("thousands to tens of thousands at 60 fps") is CONFIRMED, and 100k itself
fits with headroom in the loop alone (draw/upload not included in these
numbers). Measured traffic arithmetic as shipped (split buffers): DYNAMIC
12 B/body/frame float32 rebased positions — at 100k × 60 Hz exactly the
projected **72 MB/s** — plus 1 B/body/frame visibility flags (+6 MB/s at
100k; eye-occlusion and per-emitter eclipse bits, a disclosed addition to
the original 12 B projection) and a STATIC 32 B/body buffer uploaded once
per scene (radius, albedo.rgb, static temp, emissivity.rgb).

### 3.2 R4 unified brick grid: the leaf ceilings die; VRAM becomes the cap

Shader size becomes leaf-count-independent — the §2.2 compile-hang mechanism
is removed entirely, and voxel bodies stop consuming one texture unit each
(§2.1) because they live in one brick atlas.

VRAM arithmetic (Choices: 16³ bricks; SDF band stored R16F = 2 B/voxel;
material id R8UI = 1 B/voxel; **dense per-object allocation** — narrow-band
sparsity would raise all object counts and is not assumed):

- 16³ = 4,096 voxels/brick × 3 B = **12,288 B ≈ 12.3 KB/brick**
- Atlas cap: MAX_3D_TEXTURE_SIZE = 2048 [MEASURED] → 2048³/16³ = 128³ =
  **2,097,152 brick slots** × 12,288 B = **25.8 GB** — exceeds the 16 GB
  VRAM on this GPU, so **VRAM is the real cap**, not the texture dimension.
- Honest budgeted envelope, **4 GB atlas budget** (Choice — leaves 12 GB for
  accumulation buffers, framebuffers, and everything else):
  4×10⁹ B / 12,288 B = **~325,520 bricks**.

Objects per 4 GB budget at resolution tiers (dense allocation,
bricks/object = (cells/16)³):

| Per-object resolution | Bricks/object | Objects in 4 GB | Earth-equivalent pitch* |
|---|---|---|---|
| 32³ | 2³ = 8 | ~40,690 | ~431 km/cell |
| 48³ (today's planet tier) | 3³ = 27 | **~12,056** | ~293 km/cell |
| 96³ | 6³ = 216 | ~1,507 | ~150 km/cell |
| 192³ | 12³ = 1,728 | ~188 | ~75.5 km/cell |

\* Pitch uses the shipped builder convention (§4): pitch = 2r × 1.15/(N+2),
r = 6.371×10⁶ m — a disclosed Choice inherited from
build_solar_system_theater.py; the naive diameter/N is ~10% finer.

**R4 ARCHITECTURE DIRECTIVE (Captain, 2026-08-04/05) — supersedes the
flat-atlas-only reading of the table above:** the unified voxel structure
is to be **hierarchical-sparse (SVO)** — a sparse voxel octree or a
GigaVoxels-style octree-of-bricks hybrid — driven by the stated zoom
requirement: seamless traversal "between viewpoint of every celestial
body and solar system and all the way up to a single satellite made of
metal and solar panels." The flat brick atlas survives only as a possible
leaf level inside the hierarchy. Consequences for this table: the
dense-allocation numbers above become **worst-case floors** — SVO
sparsity stores only occupied space (a 10 m satellite at 1 cm voxels is a
sparse surface shell, not a dense 1000³ block), so real object counts per
VRAM budget rise, by a factor the R4 design must measure, not assume.
Scale math for the directive: 1.5×10¹¹ m span → 1 cm leaf ≈ **44 octree
levels** — trivial for sparse storage; the R4 design panel must settle
(a) one world tree vs per-body trees anchored in body frames + R3 splats
bridging interplanetary space (the orchestrator's stated lean: per-body),
(b) traversal cost vs depth, (c) float precision (R1-lite rebase +
float64 CPU-side node addressing). Dodoxel note: FCC physics sites embed
in a half-spacing cubic grid (x+y+z even), so one cubic SVO can index
both render bricks and sparse FCC physics data — subject to the standing
**dodoxel kill-switch** (Captain, 2026-08-05): if the dodecahedral
lattice becomes untenable or a friction point at R4/E5, it is cut in
favor of cubic all the way through.

### 3.3 R5 streaming: zoom floor becomes per-view budget-bound

Mechanism (no numbers claimed): with bricks streamed against a fixed VRAM
budget, resolution near the camera is refilled per view instead of baked per
body — the floor stops being a property of the scene bake and becomes a
property of SUSTAINED VRAM under the budget while flying (evict far/coarse,
fetch near/fine). The projection here is the mechanism only; sustained-
streaming throughput on this machine is unmeasured (§7).

### 3.4 E7 numpy physics (Von Neumann lane's phase)

Base point [CODE-VERIFIED §6]: rigid n=50 = 0.7888 ms/step. O(n²)
extrapolation to n=500 scalar: 0.7888 × (500/50)² = **78.9 ms/step**.
Precedent for the vectorization lever: sdf_batch, **"15-25x measured"**
(misc/SESSION_LOG.md:2329-2331, Phase D1 — vectorized SDF across the CSG
tree + all 7 analytic primitives). Applying it:

- n=500: 78.9/25 = **3.2 ms/step** … 78.9/15 = **5.3 ms/step**
- Real-time n at 960 Hz (budget 1.0417 ms/step): solve
  0.7888×(n/50)²/S ≤ 1.0417 → n ≤ 50×√(1.0417·S/0.7888) →
  **n ≈ 222 (S=15) to 287 (S=25)**

**Choices named:** (a) clean O(n²) scaling holds from 50 to 500 (measured
only to n=50); (b) the sdf_batch 15-25x speedup transfers to the rigid
stepper's arithmetic profile. Both are assumptions until E7's own bench rows
exist.

---

## 4. Zoom envelope

**Range (camera):** real-scale 1 AU → surface already works — R1-lite rebase
[CODE-VERIFIED §2.7], exercised live in the R2 browser session (solar theater
`locateBody('Earth')` [MEASURED §1]).

**Floor (resolution) today — 293 km/cell for Earth, NOT the naive 265:**
build_solar_system_theater.py:45 `VOXEL_CELLS_PER_DIAMETER = 48`;
`_voxel_sphere` (:60-62): `margin = 1.15`, `n = cells + 2` (= 50),
`vs = (2.0 * radius_m * margin) / n`. Earth radius_m = 6.3710×10⁶
(body_physical_params.json:9). Actual pitch = 2 × 6.371×10⁶ × 1.15 / 50 =
14,653,300 / 50 = **293,066 m ≈ 293 km/cell** [CODE-VERIFIED]. The naive
12,742 km / 48 = 265 km/cell is not what ships — margin and the +2 padding
cells make the real pitch 10.4% coarser (1.15 × 48/50 = 1.104 exactly), and
the sphere actually spans
12,742,000 / 293,066 ≈ **43.5 cells**, not 48. Scope: 9 voxel bodies
(Sun + 8 planets), 21 moons as analytic Sphere leaves, 30 bodies total
(build_solar_system_theater.py:43-44, 20, 124-125). Features smaller than
the pitch simply do not exist in the bake — that is the zoom floor.

**Floor at R4 tiers [PROJECTED — same arithmetic as §3.2, same Choices]:**
pitch = 14,653,300/(N+2) m for Earth:

| Tier | Earth pitch | Cost (dense bricks/object) | Objects in a 4 GB atlas |
|---|---|---|---|
| 48³ (today) | 293 km/cell | 27 | ~12,056 |
| 96³ | 150 km/cell | 216 | ~1,507 |
| 192³ | 75.5 km/cell | 1,728 | ~188 |

The trade is explicit: within a fixed VRAM budget, halving the pitch costs
8x the bricks per object. **At R5** the floor decouples from the bake (§3.3):
the near-camera tier is streamed, so close zoom stops costing global VRAM —
per-view sustained budget becomes the bound, unmeasured today.

---

## 5. Frame-rate & convergence reality

**Vsync:** four of five measured scenes read exactly 60 fps — the display
refresh cap. The viewer has no uncapped mode, so **GPU headroom above 60 is
unmeasured** for the light scenes; the only scene revealing its true cost is
the 48-leaf doughnut (2–5 fps, far below cap) [MEASURED]. The fps element
updates every 500 ms (viewer.js:2326).

**Path tracing is progressive.** The image is an accumulation, not a frame:

- **Still camera, static scene:** the converging branch uses **spp=3 per
  frame** (viewer.js:1554 `reset=(accN===0); prevScale=1.0; spp=3;`), so
  **spp/s = fps × 3** — confirmed by measurement (60 fps scenes accumulated
  exactly 180 samples/px/s, §1). Note: `liveSPP=2` (viewer.js:1312) applies
  only to MOVING exposures (:1548) and ANIMATED water exposures (:1551-1552)
  — an earlier internal "spp/s = fps × 2" note is wrong for static
  convergence [CODE-VERIFIED].
- **Moving camera:** 1-frame 2-spp exposures; accumulation resets when the
  view key changes (:1528). A tracked/orbiting close-up therefore never
  converges — the Earth follow view is permanently in the tens-of-spp range
  [MEASURED §1].
- **Animated scenes (water):** bounded 24-frame shutter (NMAX=24, :1542) —
  holds ~48 photons/px steady state by design; it is never supposed to
  converge further [MEASURED §1].

**What "playable" means today:** interactive-noisy at 60 fps is the product
for anything moving; settled images belong to still cameras on static scenes
and arrive at fps×3 spp/s. The 48-leaf scene's 12 spp/s means no clean image
in interactive time (§1). No "converged at N spp" claim is made anywhere in
this document — the one settling attempt (doughnut) never settled within
30 s [MEASURED, criterion disclosed in §1].

---

## 6. Physics population envelope (E0)

Committed baseline rows [CODE-VERIFIED,
engine/bench/baselines/harness_baseline.json:1-75; method: median of
best-of-5 wall-clock runs, harness.py:31,43-50; no environment metadata in
the JSON — attribution to this machine rests on SESSION_LOG.md (E0, commit
6216f03, 2026-08-03)]:

| Backend | dt | Rows (n → ms/step) |
|---|---|---|
| stacksim | 1/960 s | n=3, 500 steps → 29.309 (**but see below — build-dominated**) |
| rigid_stepper (sub_steps=1) | 1/960 s | 5 → 0.0216 · 10 → 0.0515 · 25 → 0.2250 · 50 → 0.7888 |
| nbody | 1.0 s | 11 → 0.703 · 26 → 3.610 · 51 → 13.558 · 101 → 52.227 |
| continuum SPH | 1.0 s | 11 → 1.538 · 26 → 3.609 · 51 → 7.327 · 101 → 15.452 |

**The stacksim 29.3 ms/step row is ~99.99% scene build, not stepping
[MEASURED].** The timed closure builds the scene inside the run
(harness.py:69-72: `def run(): s = _build_stacksim(); for _ in range(steps):
s.step(1.0/960.0)`; `_build_stacksim` = Deckard build_stack(['1700s
table','plate','Doughnut'], allow_llm=False) + lookups + ctor,
harness.py:54-61). Measured split this session (median of 5,
time.perf_counter): build alone = 14,826.7 ms; 500 steps on a prebuilt sim =
1.22 ms total = **0.0024 ms/step**; (build+steps)/500 = 29.66 ms/step,
reproducing the committed 29.31 within 1.2%. The other three benches also
build inside run() but negligibly [MEASURED]: rigid n=50 builder 0.128 ms
(~0.16% of the row), nbody n=101 builder 0.474 ms (~0.02%), continuum n=101
builder 0.643 ms (~0.2%).

**Real-time fit at 960 Hz** (budget = 1000/960 = **1.0417 ms/step**)
[CODE-VERIFIED arithmetic on the rows above]:

| Backend | Fits the 960 Hz budget TODAY |
|---|---|
| rigid_stepper | through **n=50** (0.789 ms/step, 1.32x headroom); its own docstring caps intent below 50 anyway (§2.3) |
| StackSim | n=3 pure stepping at 0.0024 ms/step — **~430x headroom** |
| nbody | n=11 only (0.703 ms) — **hypothetical re-cadencing**: the bench steps dt=1.0 s (its native cadence, harness.py:127); n=26+ (3.61+) does not fit |
| continuum SPH | no measured n fits (1.538 ms at n=11) — same re-cadencing caveat (harness.py:153) |

At their **native 1 s steps**, every nbody/continuum row is far faster than
real time (52 ms wall per sim-second at n=101). Caveat: rigid rows reflect
slow scenes with one solver pass per step — stepper.py's internal CFL logic
can subdivide dt for fast-moving scenes. Host cadence context: physics at
960 Hz, wire at 30 snapshots/sim-second, MAX_CATCHUP 4000 (§2.8). Engine-
hosted population today = one planar stack via StackSimRegion (§2.9).

---

## 7. Benchmark coverage: measured, added here, and still open

**Measured before this document (E0, committed):** the four physics backends
in §6 — physics only; no render benchmark existed in the repo.

**Added by this document:** the first render-FPS numbers (§1) and the
reproducible procedure:

1. Serve the viewer on an **ephemeral** port: `python -m
   sigma_ground.radiance.web.serve <port>`; md5-verify the served viewer.js
   against `sigma_ground/radiance/web/viewer.js` in the repo checkout
   (guards the known stale-copy cwd hazard).
2. Launch a **separate Chrome instance under CDP**: `--remote-debugging-port`,
   fresh `--user-data-dir` in scratch, `--remote-allow-origins=*`,
   `--window-size=1280,800` at 0,0. Drive via Python websocket-client:
   `Page.bringToFront` + `Runtime.evaluate`.
3. Read `$("fps").textContent` (updates every 500 ms, viewer.js:2326) after
   ≥15 s settling; take ≥3 readings 2 s apart; record canvas backing W×H,
   devicePixelRatio, and the GPU fingerprint (WEBGL_debug_renderer_info).
4. **Validity gate:** every accepted reading must carry
   `document.visibilityState == 'visible'` (and hasFocus). A hidden tab
   throttles rAF and the number is INVALID. This is why the CDP instance is
   the instrument of record: the Claude Browser pane never displays in
   non-interactive sessions (visibilityState stays 'hidden' — those readings
   were discarded), and the user's own Chrome via the extension was also
   hidden (collapsed MCP tab group, no tab-activation tool).
5. Convergence rate: read samples/px from the debug HUD twice, Δspp/Δt.
6. Stop the server, `Browser.close` the CDP Chrome, verify the port freed.

**Unmeasured — and how we will measure it:**

| Gap | Instrument |
|---|---|
| Uncapped GPU headroom for the 60 fps scenes (vsync hides it) | the **render-perf probe** — the follow-up instrument to be hosted on the R0 probe infrastructure — with an uncapped/offscreen mode |
| ~~CSG compile-hang threshold between 48 and 101 leaves~~ **RESOLVED [MEASURED 2026-08-06]** — see §2.2's real curve (b≈1.74 fit), the 64-88 hang / 128+ hard-fail split, and the SDF-axis-not-material-axis root-cause isolation | `tools/measure_compile_time.py` + `tools/gen_leaf_stress_scene.py` + `compile_probe.html` (R4.7, shipped instruments — `misc/COMPILE_TIME_REPORT.json` holds the raw campaign) |
| fps with physics playback running (all §1 rows are paused at t=0) | same probe, play/pause axis |
| fps vs canvas size (backing height varied 206–609 px across scenes) | same probe, pinned full-window canvas |
| ~~R3 per-frame CPU upload loop cost (the splat bound, §3.1)~~ **RESOLVED [MEASURED 2026-08-05]** — see the §3.1 measured table (30-body live 0.10 ms median; 10k 0.5 ms; 100k 4.8 ms) | `window._r3.bench(n)` + live `window._r3.stats.packMs` (shipped instruments) |
| R4 brick atlas actuals (upload, sampling cost) | R4's own bench rows |
| R5 sustained streaming throughput under a VRAM budget | R5 flight-path bench |
| E7 numpy rows replacing the §3.4 projections | E0 harness re-run on the numpy backends |
| Automated regression for the R1-lite rebase (§2.7 — none exists) | pixel-match test vs near-origin control, candidate for the R0 pixel-regression net |
| Solar theater ~35 s cold load decomposed (fetch vs compile; server sends no-store) | probe with cache axis + timestamped load events |

**Standing caveats on this document's numbers:** timing splits in §6 ran
2026-08-04 while a concurrent E4 workflow may have consumed CPU (the split
reproduced the committed baseline within 1.2%, so interference was minimal,
but they are single-session medians, not committed baselines). FPS readings
came from a separate Chrome instance, not the user's profile — same
GPU/ANGLE stack (fingerprint recorded); extensions/profile differences are a
disclosed Choice. The doughnut settling judgment is single-observer from
PNGs with the criterion stated inline; one accumulation reset (90→24 spp at
t≈27-28 s) coincided with the boot camera ease but was observed once and not
root-caused.
