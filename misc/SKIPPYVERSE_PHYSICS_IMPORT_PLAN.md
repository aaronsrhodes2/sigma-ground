# Importing all physics into the SkippyVerse

*Status: plan, 2026-10-10. Nothing here is built. It rests on a read of `stage/gmod/README.md`, `misc/GMOD_COVERAGE_PLAN.md`,
`stage/gmod/experiments/mentat_cache.json`, `ARCHITECTURE.md` and the `sigma_ground/engine`, `dynamics` and `materia` packages.*

## 1. The question: are we realizing the data, or ignoring it?

**What was executed.** Every one of Mentat's 289 entries (221 MCP tools and 68 simulation verbs) is classified in a machine-checked
ledger (`stage/gmod/coverage/`). That is an accounting, not an execution. The engine experiments S01 to S06 called Mentat for 36
distinct tools (659 cached answers in `experiments/mentat_cache.json`) and graded Havok against them. The other 250 or so entries have
a class and a planned scenario, and no running code.

**What reaches the simulation, by mode.**

| Mode | What it is | How much |
|---|---|---|
| Realized by an engine | Havok moves bodies using Mentat's numbers (mass, friction, restitution, inertia); an independent grader checks it against Mentat | S01 to S06: falling, projectiles, ramps and friction, collisions, rolling, pendulum and spring. 21 entries (7 %) are independent checks. |
| Displayed, not simulated | Frozen sigma-ground trajectories replayed in Garry's Mod or the web viewer. The doctrine is explicit: "a field is frozen sim output; renderers display it, never integrate it." | 44 of 86 bundles export. The bundles carry `temperature_k` per body; the Garry's Mod stage ignores it (no thermal colouring, "Phase 3"). The contract has no deformation, so nothing can dent or break. |
| Holodeck props | Pose and, since this week, real mass | Material, friction, temperature, composition: ignored. A prop has no state besides where it is. |
| Classified only | Everything else | Heat, gas, optics, circuits, fields, chemistry: planned as Lua ports (67 entries), none built. |
| Out of regime | Atomic, astronomical, relativistic, theory | 100 entries (35 %). |

**So: mostly ignored, by design.** Mentat knows far more than the world uses, because the world's only live physics is a rigid-body
solver that has no concept of temperature, material, charge or composition.

**But the repo already has the missing piece, and it is not connected to the SkippyVerse.** `sigma_ground/engine` is an authoritative
world stepper (`WorldHost`, an entity table, a `Region` seam for physics backends, a deterministic command log with exact replay, and a
closed set of world verbs: impulse, force, drive, hold, source, tilt). Behind it sit real solvers: continuum and SPH fluids, N-body with
Barnes-Hut, heat fields, joints, collisions. It already refuses honestly what a backend has no state for ("SOURCE, thermal" and "DRIVE,
joints" are refused, never faked). The Garry's Mod stage and the web viewer only ever saw its frozen output.

## 2. The plan in one sentence

Make the SkippyVerse's physics a **set of regions stepped by one authoritative world host**, driven by **state on each object's card**,
with **every Mentat law assigned an integration mode**, so that nothing is silently ignored: each law is realized, shown, gated by scale,
or declared missing data with a reason.

## 3. Principles

1. **One authority.** The world host steps the world. Clients (the web page, Garry's Mod) send intents and display state. No second
   integrator, which is sigma-ground's own non-negotiable.
2. **Do not write what engines already do.** Keep a proper rigid-body solver (Havok in Garry's Mod, Rapier in the browser). The host owns
   what those lack: heat, phase change, ignition, fluids, fields, chemistry, deformation. It reads rigid contacts and feeds results back
   (impulse becomes heat, a contact list becomes a conduction path, a melted body changes its mass and shape).
3. **State lives on the object.** The object card grows from size and mass to composition per part (from the lazy-repair design), and then
   temperature, specific heat, conductivity, ignition point, charge, and so on. Laws read and write these fields.
4. **Mentat is the authority on numbers and the oracle in tests.** Runtime solvers cannot make a web call per tick, so the kernels are
   the pure functions inside sigma_ground, run by the host. A port to another language (as the Lua ports were planned) is allowed only
   with parity tests against Mentat. Parity is not truth, so independent oracles are added where they exist (JPL Horizons and REBOUND for
   orbits, CRC handbook data for materials).
5. **Say what is real.** Each object and region carries its fidelity ("simulated", "displayed", "approximate", "guess"), and Skippy says so.

## 4. The integration modes (this is the fixable / not fixable map)

| Mode | Meaning | Examples | Fixable? |
|---|---|---|---|
| M1 Property | A lookup feeds a solver | density, friction, restitution, specific heat, conductivity, melting point | Yes. Needs real materials on the card, and Mentat data for the missing ones. |
| M2 Process | A region steps state over time | rigid motion, conduction and radiation, phase change, fluid flow, gas laws, circuits, reaction kinetics | Yes, one region per domain. Rigid is done by the engine; the others use the existing dynamics and thermal code. |
| M3 Coupling | A rule links two domains | ignition, friction heating, current heating, buoyancy, thermal expansion, burning mass loss | Yes, once the regions exist. This is where "hot tungsten on hay" lives. |
| M4 Instrument | A readout computed from state | meters, spectrometers, scopes | Yes, cheap. |
| M5 Scale-gated | Valid only in a regime a room never reaches | orbits, black holes, relativistic speeds, atoms | Yes as separate **scale scenes** (a planetary scene with its own units and an N-body region), entered on command, not inside a 6 m room. |
| X Not fixable here | No honest model exists in our means | pure theory entries (9), Garry's Mod's own rigid shapes deforming, true fluids inside Source (no metric solver), anything needing molecular dynamics at room scale | Display only, or leave out and say so. Deformation and fluids are fixable only outside Source (own solver or a voxel layer). |

## 5. Phases (each ends with tests, not a demo)

**P0, days: the law registry and a sweep.** Turn the ledger into a registry where every one of the 289 entries has a mode, the state
fields it reads and writes, its valid regime, and a status (realized, shown, gated, missing data). Add a headless sweep that calls every
entry once through the world host with real inputs, so we stop guessing how much is wired. Output: a table of real numbers for the 289.
This is cheap and it answers your question for all of them.

**P1: state on the card.** Composition per part, temperature, thermal properties derived from materials. The lazy-repair work gives
materials; Mentat's gaps are filled by its owner with sourced values (fabric, foam, ceramic, cardboard, steel, plant matter, ignition
temperatures). Nothing is invented by me; each value cites a handbook.

**P2: the world host in the loop.** Run the host for a SkippyVerse room. Rigid stays with the rigid solver; the host reads contacts.
Garry's Mod and the web page become clients of the same state stream, each with a "this is simulated / this is displayed" mark.

**P3: thermal, phase and ignition.** Conduction at contacts, radiation, simple convection, melting and boiling, an ignition rule with
dwell time, glow by blackbody colour, burning that removes mass. First cross-domain acceptance scenario of about ten.

**P4: fluids and gases.** The SPH and continuum code behind a region, buoyancy back into the rigid solver, the gas laws.

**P5: fields and instruments.** Electromagnetism, optics, circuits as regions and readouts.

**P6: deformation and fracture.** Stress laws plus a geometry that can change: this is where the micro-voxel layer earns its place,
on demand and sparse. The hardest phase; treat it as research.

**P7: scale scenes.** Planetary, stellar and atomic contexts with their own regions and independent oracles.

**Acceptance for every phase:** (1) every law the phase claims has a parity test against Mentat; (2) conservation checks (energy, momentum,
charge) pass over long runs; (3) at least one cross-domain scenario passes; (4) the registry status changes from "classified" to "realized"
and a test fails if it does not; (5) the object and Skippy report the fidelity honestly.

## 6. Risks

- **Scope.** 289 entries is a lot; P0 exists to stop us pretending. Phases are ordered by how much of the registry they turn from
  classified into realized.
- **Coupling stability.** Heat, phase and mass changing together is where simulations explode; the conservation checks are the guard.
- **Speed.** Python at interactive rates needs measuring; regions can run at their own rates (thermal at 1 to 10 Hz is fine).
- **Validation.** Mentat is both the implementation and the oracle. Independent references and physical sanity checks are needed, not just
  parity.
- **Data.** Many laws need material numbers Mentat does not have. Those are owner-entered and sourced, and they gate phases.
- **Determinism and replay.** The host's command log gives exact replay; every region has to keep that property.

## 7. Decisions that are Aaron's

1. Keep a standard rigid solver (Havok in Garry's Mod, Rapier in the browser) and put the host's regions around it? Recommended: yes.
2. Where the host runs for the portal: the home PC, or a small always-on service in the cloud. Recommended: start on the PC.
3. Who enters the missing Mentat materials, and from which handbooks.
4. Go-ahead for P0 (registry plus sweep), which changes no behaviour.
