# Railgun scenario: the offline half (verb, self-checks, recorder)

*2026-10-11. First slice of `SKIPPYVERSE_PHYSICS_REUSE_MAP.md` section 8: steps a, b, c, d, f.
Step e (the in-game player) is not built. Mentat the container was down, so everything ran in-process
against the same `sigma_ground` code Mentat serves.*

**Every number in this document was produced by the library code (verb `rail_shot`,
`materia/railgun.py`, `radiance/railgun_record.py`) on 2026-10-11. None was hand-computed.**
The only numbers typed by a person are the example *inputs* (marked "illustrative input").

## What exists

| Piece | File | Role |
|---|---|---|
| Physics | `sigma_ground/materia/railgun.py` | `simulate_rail_shot` (the ODE), `check_rail_shot` (self-checks), `rail_shot_impact` (muzzle state into `kinetic.interact`), `NeedsData`, `NOT_MODELLED` |
| Verb | `sigma_ground/materia/scenarios.py` `rail_shot`, registered in `SCENARIOS` and `manifest.py` `VERB_MANIFEST` | `run_simulation('rail_shot', {...})`, chainable |
| Recorder | `sigma_ground/radiance/railgun_record.py` `record_rail_shot` (exported from `sigma_ground.radiance`) | frame bundle for a player |
| Tests | `tests/test_railgun.py` (27 tests) | agreement, ledger, refusals, mutations, sanity, determinism, chain |

Chain wiring: `supersonic_projectile` now declares `launch_mach` `bind_from="muzzle_mach"` and `mass_kg`
`bind_from="projectile_mass_kg"` (additive, in `manifest.py`), so `_auto_bind` threads a muzzle exit into the flight
verb. `kinetic.interact` is not a Materia verb, so impact is a helper, `rail_shot_impact(outputs, layer, thickness_m, radius_m)`.

## Slots of `rail_shot`

| Slot | Unit | Source | Note |
|---|---|---|---|
| `cap_F`, `volts` | F, V | design input | bank energy via `electronics.energy_stored` |
| `rail_len_m` | m | design input | |
| `rail_radius_m`, `rail_spacing_m` | m | design input | rails are modelled as round wires; gap = spacing - 2 r |
| `L_per_m` | H/m | optional **owner-entered override** | default is `mobius.parallel_pair_inductance_per_m(r, d)`; the step list labels an override "NOT from the library" |
| `armature_material`, `rail_material` | key | library | must have resistivity(T), specific heat and melting point (14 metals today); otherwise `NeedsData` lists the supported ones |
| `armature_mass_kg` | kg | design input | the armature is a bar filling the rail gap, cross-section A = m / (density x gap) (assumption) |
| `circuit_R_ohm`, `circuit_L0_H` | ohm, H | **owner-entered, required** | bank ESR/ESL, busbar, switch, contacts; no library source, so no default: missing means `NeedsData` |
| `T0_K` | K | default 300 K | the library's resistivity reference temperature |
| `tolerance` | | default 1e-3 | energy-ledger tolerance (fraction of bank energy) |

Named outputs (chainable): `muzzle_speed_m_s`, `muzzle_mach`, `projectile_mass_kg` (only on a validated muzzle
exit), `speed_at_failure_m_s` (melt case instead), `peak_current_A`, `peak_force_N`, `delta_T_K`, `peak_T_K`,
`muzzle_energy_J`, `efficiency`, `shot_time_s`, `armature_melted`, `exited_rails`, `bank_reversed`.
On a failed self-check, or a shot that melts before the muzzle, the muzzle outputs are withheld, so a chained step
fails with a `KeyError` naming the missing output instead of flying on a bad speed.

## Physics that is IN

State (V_c, I, x, v, dT, three Joule ledgers), stepped by `sg_solve_ivp(method='rk4')` one fixed step at a time
(dt = the bank's own LC or L/R time over 200). The event (end of rails, or melt) is located by bisecting the last step.

- Capacitor: `C dV_c/dt = -I`.
- Circuit with moving load: `L(x) dI/dt = V_c - R_tot I - L' v I`, `L = L0 + L' x`; `R_tot = R_circuit + R_rails(x) + R_arm(T)`.
- Force: `F = 1/2 L' I^2` (rails behind the armature only; the 1/2 is the semi-infinite-rail result).
- Armature Joule heating: `m c_p(T) dT/dt = I^2 R_arm(T)`, `R_arm(T) = rho(T) gap / A`, with library Bloch-Gruneisen
  `electronics.resistivity` and Debye `thermal.specific_heat_j_kg_K` (sampled on a 96-point grid up to the library
  melting point and interpolated linearly, so the resistivity-temperature feedback is live).
- Rail resistance grows with armature position: `2 rho(T0) x / (pi r^2)` (DC).
- Stop at the muzzle (`end_of_rails`), at the onset of melting (`melt`, library `PHASE_DATA` T_melt), or on a spent
  circuit: when the remaining circuit energy is below 1e-10 of the bank energy the armature coasts out at constant
  speed exactly (no friction modelled). A `t_max` stall is refused.
- Bank voltage reversal (no crowbar or diode) is reported, not prevented.

## What is NOT in (the physics is bigger than this)

- **Skin effect and current diffusion.** The current is uniform across the armature and rails. Real millisecond pulses
  crowd into a skin and heat a thin layer far above the bulk number, so the reported armature heating is a **lower bound**.
  This is why a hobby armature in these runs only warms a few kelvin.
- Sliding contact resistance, contact arcing, transition to a plasma armature (contact loss exists only inside `circuit_R_ohm`).
- Rail erosion, gouging, rail heating (rails stay at T0), rail recoil and deflection.
- Rail inductance gradient simplifications: L' is the round-wire pair formula (or an override). Flat rails, proximity
  effect, frequency dependence, end effects and the breech splice are not modelled. The library's `ln(d/r)` form also
  differs from the field-integral `ln((d-r)/r)`; for d = 5 r that is about 15 percent in L'.
- Magnetic saturation (non-magnetic conductors only), barrel friction, air column ahead of the armature, gravity.
- Latent heat, liquid or vaporising armature (the shot ends at the onset of melting).
- Switch dynamics, crowbar, pulse-forming network (a single capacitor discharge only).
- Library quirk worth knowing: library c_p for aluminium at 300 K is 807 J/(kg K), the Debye-model value, not a handbook table value.

## Self-checks (each is a key in `validation['checks']`; any failure sets `passed: False`)

1. `energy_ledger`: bank energy = remaining capacitor energy + 1/2 L I^2 + kinetic + Joule (circuit, rails, armature), worst residual over the whole shot, tolerance 1e-3 of the bank energy.
2. `armature_heat_balance`: armature Joule integral equals `m * integral c_p dT` over the recorded temperature history.
3. `speed_below_energy_ceiling`: `v <= sqrt(2 E / m)`.
4. `peak_force_vs_library_lorentz`: `field.electrodynamics.magnetic_force(...)` (its magnitude; the tier-4 `electromagnetic_force_analysis` computes the same value) with `q v = I * gap` and `B = L' I / (2 gap)` must equal the model's peak force. **Honest limit:** B is built from the same L', so this checks the force law, the factor 1/2 and the units against the library's cross product, not the value of L' itself.
5. `x_monotone`, `speed_monotone`, `no_negative_temperature` (T > 0 and non-decreasing), `no_loss_gain`.
6. `shot_terminated`: the armature must reach the muzzle or melt; a shot that never clears the rails is refused.

The recorder adds: the bundle must agree with the verb's scalar answer within 2 percent (speed, peak current, temperature rise, shot time), the frames must
obey their own integrals (x = integral v dt, bank charge = integral I dt, within 1 percent), and the last frame must sit at the muzzle. A failed shot is never recorded.

Refusals exercised by the tests: a force law missing its 1/2 (fails `energy_ledger` and `peak_force_vs_library_lorentz`, muzzle outputs withheld, recorder
raises); a deleted back-EMF term (fails `energy_ledger`); a recorder whose physics differs from the expected scalars (raises); a time budget that ends on the rails (fails `shot_terminated`);
a chain after a melt (raises `KeyError: muzzle_mach`); missing bank data (raises `NeedsData`, and `run_simulation` returns a declined result).

## Examples (all outputs from the library)

Illustrative inputs: the bank resistance and inductance are made-up round numbers, **not measurements**.

**Small hobby railgun.** 20 mF at 450 V (2025 J), 0.5 m of copper rails (radius 5 mm, spacing 25 mm), 1 g aluminium armature, bank R 5 mohm, L0 3 uH.

| Quantity | Value |
|---|---|
| L' (library, round-wire pair) | 0.6438 uH/m |
| Peak current | 27.59 kA |
| Peak force | 245.1 N (about 25,000 g on the armature) |
| Muzzle speed | 128.9 m/s (Mach 0.38), after 4.52 ms |
| Kinetic energy / efficiency | 8.31 J / 0.41 percent |
| Where the 2025 J went | circuit 2003.0 J, rails 5.97 J, armature 6.55 J, left in capacitor 0.037 J, magnetic 1.11 J |
| Armature temperature | 300 K to 308.1 K (bulk, lower bound) |
| Energy ledger worst residual | 5.2e-12 of the bank energy |
| Bank voltage | reverses to -232.7 V (no crowbar) |

Same bank with a 0.1 g armature: it reaches the aluminium melting point (933.47 K) after 7.3 cm at 557.8 m/s and the verb reports `armature_melted`, no muzzle outputs.
The recorded bundle ends on an `armature_melt` event with a non-zero glow colour.

**Large railgun.** 1 F at 10 kV (50 MJ), 10 m of copper rails (radius 5 cm, spacing 25 cm), 3 kg aluminium armature, bank R 2 mohm, L0 2 uH.

| Quantity | Value |
|---|---|
| Peak current | 2.961 MA |
| Peak force | 2.822 MN (about 96,000 g) |
| Muzzle speed | 2233 m/s (Mach 6.56), after 6.52 ms |
| Kinetic energy / efficiency | 7.48 MJ / 15.0 percent |
| Where the 50 MJ went | circuit 41.63 MJ, rails 0.127 MJ, armature 0.0113 MJ, capacitor 0.030 MJ, magnetic 0.718 MJ |
| Armature temperature | 300 K to 304.6 K (bulk, lower bound) |
| Energy ledger worst residual | 2.0e-12 |

Chained: `supersonic_projectile` (3 kg, 0.1 m diameter, Mach 6.56) goes subsonic (final Mach 1.96 at the verb's stop) after 2223 m;
`rail_shot_impact` into 5 cm of mild steel penetrates (Poncelet, x_max 0.123 m) and exits at 782 m/s.

**Plausibility, stated honestly.** I compared the outputs to what I recall of railgun behaviour in general, not to retrieved
sources. They are order-of-magnitude checks: small capacitor-driven hobby railguns are generally reported at well under
a few percent efficiency and tens to a few hundred m/s (here 0.4 percent, 129 m/s); naval-class railguns are generally reported with
peak currents of order a few MA, muzzle speeds of order 2 to 2.5 km/s and tens of g-kilo accelerations (here 3 MA, 2.2 km/s); a single
capacitor discharge is expected to be less efficient than a pulse-forming network (here 15 percent). Nothing here is calibrated to a specific gun.
The weakest part is armature heating (bulk Joule only).

## Data that is missing and must be owner-entered (with a citation)

- `circuit_R_ohm`, `circuit_L0_H`: capacitor-bank ESR/ESL plus busbar, switch and contact resistance; per-bank datasheet or measurement.
- Flat-rail inductance gradient `L_per_m` (measured or field-solved) if the round-wire pair is not representative.
- Materials beyond the 14 metals that have resistivity(T), c_p and melting data; and any contact-resistance or skin-depth rule for the armature.
- Any pulse-forming network, switch and crowbar parameters (not modelled at all).

## The bundle and the future player (step e)

`record_rail_shot(**rail_shot_slots, n_frames=240)` returns `{scene, trajectory, kind: "trajectory", thermal: True, railgun: True}`.
The frame contract extends `trajectory.py` additively (all new keys optional):

```
frame = {"t_sim": s,
         "bodies": [{"pos": [x, y, z], "quat": [0,0,0,1], "temperature_k": T,
                     "rgb": [r,g,b], "emit": 0..1}],
         "events": [{"kind": "discharge_start" | "peak_current" | "bank_reversal" | "visible_glow" | "muzzle_exit" | "armature_melt", ...}]?,
         "readouts": {"I_A", "V_cap_V", "v_m_s", "F_N", "T_K"}}
trajectory.readout_spec = {name: {label, unit, source}}   # units and sources once, not per frame
trajectory.events = [... with t_sim ...]
```

`rgb` is the library's `thermal_emission_rgb` (black below the Draper point, `blackbody` stand-in for metals without measured n,k).
`emit` is the relative Stefan-Boltzmann ratio (T / T_melt)^4, a brightness cue, not calibrated radiance. Frames are the solver's own states
picked evenly by step index. `suggested_rate` = shot time / 15 s (hobby 3.0e-4, large 4.3e-4: play a millisecond shot at about 1/3000 speed).
Positions run along +x with the armature centre at the rail axis height; rails are two static `Box` leaves, the armature a `Box` body (render-only geometry derived from its mass).
Bundle size: 240 frames is about 90 KB JSON (about 23 KB compressed); 60 frames about 27 KB.
Both examples are deterministic: the same parameters give a bit-identical bundle.

What step e must do (all in the off-limits GMod side, untouched here):

1. **Extend `export_gmod`.** I ran `export_scenario` on a hobby bundle: it accepts the bundle but (a) drops the static rail leaves (only body leaves are exported, so rails must be spawned as props by the player), (b) ignores `rgb`/`emit`/`events`/`readouts` (it keeps `temperature_k`, "exported but not displayed until Phase 3"), (c) warns that a 1 g armature is outside GMod's PhysObj mass range of 0.1 to 50,000 kg (clamped; fine for kinematic replay, wrong for released motion).
2. Apply `rgb`/`emit` to the armature (replacing the Phase 3 temperature stub), fire events (muzzle flash on `muzzle_exit`, melt on `armature_melt`), and draw `readouts` on a bench board using `readout_spec` units and sources.
3. Play at `suggested_rate` through `mentat_play / rate`; keep the 64 KB client limit in mind (use `n_frames` or `decimate`).
4. For the live path, the verb needs a push, a new `SIGMA_GROUND_REF` in the SkippyTel image and a rebuild (reuse map section 9); until then the bundle is a recorded file.
