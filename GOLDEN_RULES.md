# Golden Rules of sigma-ground Physics

These rules govern how all physics — and rendering — code is written in this project.
They encode lessons learned building the library and ensure every module
stays consistent, correct, and genuinely connected to the σ-field framework.

---

## Rule 1 — One Source of Truth for Constants

All measured values live in `sigma_ground/field/constants.py`.
Every module imports from there. **No magic numbers anywhere else.**

```python
# Correct
from sigma_ground.field.constants import C, HBAR, G

# Wrong — never do this
C = 3e8
```

Cosmological constants (G, C, ℏ), nuclear data (proton mass, quark masses),
electrostatics (e, ε₀), and SSBM parameters (ξ, η, σ_conv) all live there.

---

## Rule 2 — Derive, Don't Assume

Every non-measured value shows its derivation from measured constants
inline. Comment the formula.

```python
# Derived: α = e² / (4πε₀ℏc)
ALPHA = E_CHARGE**2 / (4 * math.pi * EPS_0 * HBAR * C)  # ≈ 1/137.036
```

If a derivation requires more than a line, put it in the docstring.

---

## Rule 3 — Explicit Domain Bounds

Every model defines where it is valid. Document the breakdown condition.
Use SAFE / EDGE / WALL / BEYOND classification from `sigma_ground/field/bounds.py`
when integrating with other field modules.

```python
def lorentz_factor(v):
    """Valid for 0 ≤ v < c. Diverges at v = c."""
    if v >= C:
        raise ValueError(f"v={v} ≥ c={C}: Lorentz factor undefined")
```

---

## Rule 4 — Wire to σ

Every physics module must include at least one function showing how the
σ-field modifies its standard result. This is what makes sigma-ground
different from a generic physics utilities package.

The pattern:
```python
def sigma_X(sigma, standard_input):
    """How X changes under σ-field compression."""
    return standard_result * scale_factor(sigma)
```

The σ-field value in everyday matter is negligible (Earth surface: ~7×10⁻¹⁰).
The σ-functions become meaningful approaching black hole accretion disks and
at the Big Bang (σ → σ_conv ≈ 1.849).

---

## Rule 5 — Physics Never Imports Rendering

The ARCHITECTURE.md rule. Inviolable.

```
sigma_ground.field     ✓ may import sigma_ground.dynamics
sigma_ground.field     ✗ may NOT import matter_shaper or any renderer
sigma_ground.dynamics  ✓ may import sigma_ground.field
```

---

## Rule 6 — Nature Already Has the Best Answer

Prefer exact analytic solutions over approximations. Use numerical methods
only when no closed form exists. Cite the source (textbook, PDG, NIST).

```python
# Hawking temperature — exact analytic result
# Hawking 1974, Comm. Math. Phys. 43, 199
def hawking_temperature(M):
    return HBAR * C**3 / (8 * math.pi * G * M * K_B)
```

---

## Rule 7 — No Averages, No Assumptions, No Gaps

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

---

## Rule 8 — Tests Prove the Physics

Every new function gets at least one test checking known values against
standard references (NIST, PDG, textbooks). Physics tests are not unit
tests — they are verification against reality.

```python
def test_electron_rest_energy():
    # NIST CODATA: electron rest energy = 0.51099895 MeV
    E_mev = rest_energy(M_ELECTRON_KG) / (E_CHARGE * 1e6)  # J → MeV
    assert abs(E_mev - 0.511) < 0.001  # within 0.2%
```

---

## Rule 9 — If One, Then All

When adding a property for one material, element, or entity, add it for
**every applicable entry** in that database. Incomplete databases are silent
gaps — they compile, they run, and they give wrong answers when simulated.

```python
# Wrong — cherry-picked data
SUPERCONDUCTORS = {
    'aluminum': {'T_c_K': 1.175, 'kappa': 0.01, ...},
    'niobium':  {'T_c_K': 9.25,  'kappa': 1.05, ...},
    # 45 other superconductors silently missing — simulation sees only 2
}

# Correct — every known superconductor, every field populated
SUPERCONDUCTORS = {
    'aluminum': {'T_c_K': 1.175, 'kappa': 0.01, 'kappa_source': 'measured', ...},
    'niobium':  {'T_c_K': 9.25,  'kappa': 1.05, 'kappa_source': 'measured', ...},
    'titanium': {'T_c_K': 0.40,  'kappa': 0.09, 'kappa_source': 'derived',  ...},
    # ... all 53 elements + compounds, no gaps
}
```

If a value cannot be measured, derive it and flag the provenance. If it
truly cannot be determined, flag it explicitly — never silently omit the entry.

---

## Rule 10 — Test Against a Volume of Matter

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

## Rule 11 — Fix the Physics, Never Patch the Picture

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
