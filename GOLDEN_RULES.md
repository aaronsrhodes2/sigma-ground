# Golden Rules of sigma-ground Physics

These rules govern how all physics — and rendering — code is written in this project.
They encode lessons learned building the library and ensure every module
stays consistent, correct, and genuinely connected to the σ-field framework.

**Note:** Some rules that used to live here now live in [`PLATINUM_RULES.md`](../PLATINUM_RULES.md)
because they turned out to be universal, not physics-specific — see the audit
log at the bottom of this file for what moved and why.

---

## Rule 1 — Derive, Don't Assume

Every non-measured value shows its derivation from measured constants
inline. Comment the formula.

```python
# Derived: α = e² / (4πε₀ℏc)
ALPHA = E_CHARGE**2 / (4 * math.pi * EPS_0 * HBAR * C)  # ≈ 1/137.036
```

If a derivation requires more than a line, put it in the docstring.

---

## Rule 2 — Explicit Domain Bounds

**Formal definition:** **Design by Contract** — every model declares and *enforces*
its preconditions (its domain of validity) and documents its breakdown condition.
Use the SAFE / EDGE / WALL / BEYOND classification from `sigma_ground/field/bounds.py`
when integrating with other field modules.

```python
def lorentz_factor(v):
    """Valid for 0 ≤ v < c. Diverges at v = c."""
    if v >= C:
        raise ValueError(f"v={v} ≥ c={C}: Lorentz factor undefined")  # the precondition, enforced
```

---

## Rule 3 — Nature Already Has the Best Answer

Prefer exact analytic solutions over approximations. Use numerical methods
only when no closed form exists. Cite the source (textbook, PDG, NIST).

```python
# Hawking temperature — exact analytic result
# Hawking 1974, Comm. Math. Phys. 43, 199
def hawking_temperature(M):
    return HBAR * C**3 / (8 * math.pi * G * M * K_B)
```

---

## Rule 4 — No Averages, No Assumptions, No Gaps

Track all terms. Don't silently drop small contributions. When approximating,
state what you are dropping and why it is safe to drop it.

```python
# OK: explicit approximation
def binding_energy_approx(A, Z):
    """Bethe-Weizsäcker formula. Drops pairing term for simplicity."""
    ...

# Not OK: silent drop
def binding_energy(A, Z):
    volume = ...  # pairing term silently missing — don't do this
```

**Enforcement (added 2026-07-01):** this rule kept getting restated because it
kept getting quietly violated — fallback/legacy chains accumulated in the
pipeline despite the stated principle. Stating an ideal isn't enough; gaps
need to be visible. From now on:

- Any fallback, legacy path, or dropped term gets a `# PHYSICS_GAP:` comment
  at the point it's introduced, explaining what's missing and why it was
  acceptable at the time.
- Every `# PHYSICS_GAP:` tag gets a matching one-line entry in
  `KNOWN_GAPS.md` at the project root (create it if it doesn't exist yet).
- `KNOWN_GAPS.md` is meant to be grepped and periodically reviewed — a gap
  that's still there a year later is a decision to revisit, not a secret.

---

## Rule 5 — Test Against a Volume of Matter

Always test physics against a volume of matter, unless specifically testing
an isolated entity. Conductivity, resistance, superconductivity, phase
transitions, and bonding are collective phenomena — they emerge from many
atoms interacting in a volume. A single-atom calculation that ignores
the solid-state environment will miss the physics that matters.

```python
# Wrong — isolated atom, misses d-band formation
lambda_ep = pseudopotential_scattering(Z)  # free-electron model

# Correct — accounts for d-band in the solid
lambda_ep = d_band_coupling(Z)  # tight-binding: bandwidth, coordination, hopping
```

If a property is intrinsically collective (band structure, phonons,
screening, phase transitions), the model must include the volume.
If a property is intrinsically atomic (ionization energy, electron config),
test the atom.

---

## Rule 6 — Fix the Physics, Never Patch the Picture

Radiance renders the way nature does — by simulating light transport (path
tracing). Appearance and dynamics are **consequences** of the physics, never
things painted on afterward. When the image looks wrong, the fix lives in the
physics, or in how we *integrate* it — **never in a post-hoc graphics trick** (a
fake renderer, a denoiser, a fudge applied after the fact). A trick that hides
one artifact is a new lie that resurfaces as another.

```
# Wrong — patch the picture after the fact
Water flickers under path tracing  →  fall back to the headlight/ambient "fast"
shader (does reflection, NOT refraction).  Flicker gone — and the water is now
opaque. One lie stacked on the last.

# Correct — fix it in the light transport
Water flickers because a path tracer can't average a MOVING surface.  The frame
is a real camera EXPOSURE: integrate photons over a bounded shutter while the
ripples advance.  Clean, transparent, AND animated — from the one renderer.
```

There is exactly one renderer: the path tracer. The only quality knob is
**photons** — samples integrated over an exposure — because that is the only knob
nature has. No second "fast" path in the real pipeline; no denoiser guessing the
image instead of computing it. If it's noisy, gather more photons. If it's slow,
make the light transport cheaper. Never fake the light.

**Corollary — measure reality before you "fix" it.** A "0.2 fps performance wall"
once proved to be the browser *pausing* an occluded window
(`document.visibilityState === "hidden"` → `requestAnimationFrame` frozen), not
the renderer. It nearly bought real compromises — frozen ripples, quarter
resolution — to solve a problem that did not exist. Confirm the bottleneck is
real (window focused, frames actually advancing) before trading physics to chase it.

---

## The Golden Goal — Physics to Screen, Nothing in Between

The Golden Rules are constraints; this is the aspiration they serve. Radiance's
rendering model, as closely as we can possibly muster, is **physics to screen with
nothing in between.**

Every photon that reaches the eye should trace back to a real quantity the
simulation actually produced — a temperature, a refractive index, an emission
spectrum, a surface the dynamics actually formed — with no artist's fudge, no
stylization pass, no "looks about right" constant slipped between the physics and
the pixel. The renderer is a *window onto the simulation*, not a painter
interpreting it.

We will never close the gap completely — a screen is not the universe, and finite
photons are finite photons. But the distance between the physics and the picture is
the error term, and the whole craft is driving it toward zero. **When in doubt,
remove a layer rather than add one.**

---

*These rules exist because the σ-field framework lives or dies on the
precision of its foundations. Every magic number is a crack in the theory.
Every undocumented approximation is a place where the framework secretly fails.
Every missing database entry is a thing the simulation cannot see.*

*"Nature already has the best answer — we just need to find it."*

---

## Vision — The Matter Information Cascade

This library is built on a discovery: **you can derive all material properties
from a small number of measured atomic inputs through physics equations.**

For metals: ~15 measured numbers per element → hundreds of derived properties.
For organics: ~42 measured atomic properties (7 atoms × 6 each) plus 7
homonuclear bond energies → all covalent bonds → all molecular geometries →
all intermolecular forces → all bulk properties.

The equations *are* the compression. You never need to store what you can derive.

This is the same principle that lets the inventory system load the entire
observable universe and query any particle in it without a massive thinking
machine. A lookup table of every molecule's properties would be infinite.
The derivation chain that produces them from atoms fits in a few kilobytes.

**The goal:** virtual matter that behaves exactly like real matter in every way,
constructed from first principles plus a minimal set of measured seeds. No
lookup tables of bulk properties. No fitting parameters disguised as physics.
No magic numbers that haven't been traced to a measurement or a derivation.

The cascade flows downward:
```
measured atomic properties (χ, r_cov, IE₁, mass, D(A-A))
  → bond energies (Pauling)
    → bond lengths (Schomaker-Stevenson)
      → molecular geometry (VSEPR)
        → intermolecular forces (H-bond, London, dipole-dipole)
          → bulk properties (density, viscosity, surface tension, ...)
```

Each step is a physics equation, not a lookup. Each equation cites its source.
Each measured input is labeled MEASURED. Each derivation is labeled
FIRST_PRINCIPLES. The provenance is the proof.

*"We proved we could load the entire universe and query any particle in it.
The matter information cascade is the same insight applied to chemistry:
the compression is the physics itself."*

---

## Audit Log

**2026-07-01** — Golden rules audit, done alongside a Platinum Rules audit:
- Old Rule 1 (One Source of Truth for Constants) removed — duplicated Platinum §3, which already covers `sigma_ground/field/constants.py` as the constants source of truth.
- Old Rule 4 (Wire to σ) and Rule 5 (Physics Never Imports Rendering) removed by the Captain's choice.
- Old Rule 8 (Tests Prove the Physics) removed — duplicated Platinum §6 (Test Against Reality), which already lists sigma-ground's NIST/PDG validation standard as its example.
- Old Rule 9 (If One, Then All) **promoted to Platinum §10** — the exact same failure mode (touch one instance, forget to update the rest) turned up independently in music-collection's genre tags and a game's cleanup-method call sites, not just physics databases. See PLATINUM_RULES.md.
- Old Rule 7 (No Averages, No Assumptions, No Gaps) enhanced with a concrete enforcement mechanism (`# PHYSICS_GAP:` tags + `KNOWN_GAPS.md`) after operatic-archive evidence showed the stated principle getting quietly violated by fallback-chain accumulation.
- Remaining rules renumbered 1-6.
- Rule 2 (Explicit Domain Bounds) anchored to its formal term, **Design by Contract** — the precondition/domain-of-validity concept it was restating in custom prose. Same audit day, formal-term compression pass.
