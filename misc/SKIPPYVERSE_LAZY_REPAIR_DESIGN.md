# Lazy repair, materials and derived mass (and the micro-voxel question)

*Status: design, 2026-10-10, from a research workflow (four effective tracks, a synthesis, a red team). Nothing is built. Companion to SKIPPYVERSE_OBJECTS_DESIGN.md. The pipeline track answered the wrong question, so the state machine and card schema are the thinnest part.*

## Short answer for Aaron

Your question first: no. Our Universe Sandbox check (misc/GMOD_COVERAGE_PLAN.md, section 5) found no scripting interface. Saves were zip files of editable JSON and graphs export to CSV, so it can show a scene we generate, but it cannot host or drive GMod-scale celestial physics, and it recomputes some values on load, so it is not a test oracle either. Solar System Simulator has no mods or export. Two researchers searched again today and found nothing newer. One caveat: the "editable JSON" claim comes from 2020 to 2022 posts, and Update 36 (2026-03-30) changed the save structure, which nobody could inspect because the game is not installed. The credible route for celestial scale is our own N-body code replayed on a scaled stage, checked against REBOUND and JPL Horizons. GMod cannot do that yet: the addon refuses any stage scale other than 1.

On your lazy-repair and micro-voxel idea, here is the rendering-versus-simulation split. GMod draws each prop as a thin skin of triangles, and the graphics card paints those very fast. It simulates physics on a different, crude stand-in: the median prop in your library collides as 4 convex lumps built from 60 triangles. Drawing and bumping are two cheap layers, and that is why GMod is fast. Your Radiance viewer is different: a path tracer that gets pretty by accumulating frames. A voxel (a tiny cube that knows what it is made of) is excellent for physics questions and poor for drawing. A 1 m object at 1 mm is 1e9 cells, at 0.1 mm 1e12, at 10 micrometres 1e15.

So micro-voxels belong under the object as an optional, lazy, sparse physics tier, used when something breaks, is cut, heats, or must be certified. They should be stored as the fraction of material per cell, not as yes/no cells. They are never a render path and never a default step. A closed single-material part needs no voxels for mass: exact mass, centre of mass and inertia come from the triangles in a few milliseconds.

The measurements back your lazy approach. Real meshes are mostly not sealed: about half of random Objaverse models, 8% of ShapeNetSem models, and none of the 50 PartNet models tested. Deckard's current voxelizer is unsafe on them. A tiny hole in a sphere loses 87% of its volume, a 3 mm cup reads 68% high, and one PartNet chair halves in mass each time the grid gets finer (50 kg, 26 kg, 13 kg). A replacement that fires rays along each axis was within 1% on 57 of 66 deliberately damaged cases, and the researchers' prototype handled a 1M-triangle mesh in about a quarter of a second.

Real materials help, but less than the wall thickness does. A steel filing cabinet is steel, but only a 1 mm skin of it, so volume times density is 10 to 58 times too heavy. Getting the material right changes mass error only from about x2.2 to x2.0, because the unknown is how much of the box is solid. A shell rule (surface area, times wall thickness, times density) lands near x1.5 for cabinets, appliances, tubs and mugs. Frame-and-panel furniture is best served by a learned bulk density per class (about x1.2 to x1.3). Honest accuracy is about x1.4, which is also the spread of your own sourced ranges, so you cannot claim better. The practical plan: keep sourced masses as the default authority, have the repaired mesh say solid versus hollow and supply centre of mass, inertia and collider, and let the derived mass override only when the mesh models its own wall or no sourced family exists.

Material choice is about 84% right from Valve's own surface labels, 63% from Deckard's current name guess, and 61 to 67% from the only local vision model. All lean toward wood and miss metal, which is the costly mistake because metal is about 11 times denser. Combining independent evidence gets about 86% on unseen families. Mentat has no density for steel, cast iron, ceramic, fabric, foam, cardboard, paper, leather, plywood or MDF. It returns a silent null for them instead of an error.

# Lazy repair, materials and derived mass: one design

Synthesis of four research tracks (mesh repair, part materials, mass derivation, micro-voxel). The fifth track (pipeline) answered only the Universe Sandbox question and did no pipeline design. Nothing here has been built. Every repo was read-only. Labels: **[measured]** means run by a researcher on repo or local data. **[published]** means read from a paper or doc, not reproduced. **[inferred]** means reasoning. **[guess]** means a judgement call.

## 0. The Universe Sandbox question

No valid scriptable appliance. Sources:
- Our earlier check (misc/GMOD_COVERAGE_PLAN.md section 5). Universe Sandbox has no scripting API. Saves were zip plus JSON, graphs export to CSV, and some orbital and climate values are recomputed on load. Solar System Simulator (Steam 2355330) has no Workshop, mods or export. Neither is a host or an oracle.
- Re-checked today by three tracks [web, summaries]. No scripting or API on the 2026 roadmap or in Steam announcements through 2026-08-20. A GOG build record says "Script interpreter: False". There is an unofficial 2022 script-library mod and an unaudited DLL import/export mod.
- **One correction to the earlier write-up.** "Saves are editable JSON" rests on 2020 to 2022 posts. Update 36 (2026-03-30) introduced a new save structure that nobody could inspect, and the game is not installed here. Suggested wording: "were (2020 to 2022 sources); Update 36 changed the format, unverified."
- The real route for celestial scale is in-house. sg_nbody (Forest-Ruth plus 1PN) is replayed as a scaled stage (GMOD_STAGE_PLAN Phase 4, not built; server.lua rejects stage_scale other than 1), with REBOUND and JPL Horizons as independent oracles. REBOUND is not installed.
- Scale arithmetic [inferred from bundle values]: on a 20 m stage, true-scale Earth is 0.00072 HU and the Earth-Moon distance is 0.041 HU, so radii must be exaggerated and labelled.
- solar_system_theater.json is a single static frame of 30 bodies.
- The web client could consume the same trajectory bundle without GMod.

## 1. Rendering versus simulation, and where micro-voxels go

| Layer | What it is | Cost driver | Micro-voxels? |
|---|---|---|---|
| Render (GMod) | Triangles rasterised by the GPU. A prop is a few thousand triangles. | Triangles per frame | No |
| Render (Radiance) | WebGL2 path tracer on SDF bricks, about 2 samples per pixel per frame, accumulating. svo.js stores geometry only; one material per body. | Pixels x samples | No |
| Collision (GMod/Source) | Convex hull pieces. Median prop is 60 triangles and 4 convex pieces [measured, measured.json]. | Piece count | No |
| Collision (Rapier, web) | Convex or trimesh colliders plus mass, centre of mass and principal inertia | Colliders | No |
| Mass properties | Exact integrals over a closed surface, or surface quadrature for shells | Triangles, milliseconds | Not needed |
| Heterogeneous or changing matter | Per-cell material, damage, temperature, cuts | Surface area / pitch^2 | **Yes, lazily** |

Micro-voxels are a lazy physics backend and never a render path. Cost [measured and arithmetic]:
- Dense 1 m cube: 1e9 cells at 1 mm, 1e12 at 0.1 mm, 1e15 at 10 um (1 GB, 1 TB, 1 PB at one byte per cell).
- Sparse surface band on a real chair: 37,640 surface cells at 7.8 mm, growing x4 per pitch halving (N^2). That extrapolates to about 2.3 M cells at 1 mm and 0.23 G at 0.1 mm.
- Whole-object 10 um stays infeasible even sparse. Mass only needs a homogenised density anyway.
- Radiance's sparse bricks beat dense storage only above about 150 cells across (0.83x at 192, 0.46x at 384) [measured].

## 2. Reconciling the tracks

1. **Geometry exactness is not mass exactness.** The repair track gets volume to under 1% on 57 of 66 damaged cases. The mass track shows volume times density is x10 to x58 off for hollow things. Both are right. Repair yields a trustworthy geometry and a solid-or-hollow verdict. Mass then needs the stock form and wall thickness.
2. **Two voxel fixes, one story.** The repair track uses ray-parity column intervals. The micro-voxel track recommends centre-sample with a winding number and a distance band for sheets. Decision: column intervals are the default fill and the sparse micro-voxel representation. Winding number is a targeted arbiter on disputed voxels. A distance band applies only to open sheets, after a thickness is assigned.
3. **Open sheets have no volume.** PartNet parts are open, often coincident double layers (a seat sheet is 39 up faces plus 39 down faces; mesh.area is 2x the physical area). Volume is null. Mass comes from area x assigned thickness, with duplicates removed first.
4. **Sourced thickness versus default.** The materials track found a sourced thickness gives x1.64 rms on 33 shell models. The mass track found sourced (x1.44) barely beats the stock default (x1.50). Reconcile: the error floor is structure (drawers, liners, double walls the hull lacks), not the thickness number. Do not overinvest in thickness precision.
5. **Circularity warnings.**
   - The adopted masses in fidelity.json are clamp(reference x s^k, low, high), and 44 of 120 sit on a bound.
   - The mug 95% result fits scale to the card bbox and shares the wall thickness with the sourced range.
   - The materials fusion reliabilities were set on the same 121 models.
   - The sourced ranges themselves carry sqrt(high/low) of about x1.39 median.
   - Treat these as consistency tests. A derived mass cannot honestly claim better than about x1.4.
6. **GPU:** it is an RTX 4070 Ti SUPER with 16 GB, not 12 GB, and it was shared and 100% busy in every track. Treat GPU timings as upper bounds and never depend on a free GPU.
7. **libigl:** the old comment in voxelize.py ("no Python 3.13 wheel") is stale. libigl 2.6.3 has a cp312-abi3 Windows wheel.

## 3. The pipeline

Trigger: first physical need for an object (spawn needs only a ghost box and a family-prior mass shown immediately). Everything below runs async, per part, cached, and never blocks the spawn. Cache key: sha256 of source bytes + repair version + material table version. Valve and ShapeNet meshes run on the PC only and never go to the portal.

**Stage 0. Frame** (ms).
- Normalise to metres, +Z up, front +X, origin on the floor under the footprint centre.
- Scale always comes from the card size, never from the file (only 58% of Objaverse files have a longest edge between 0.1 and 10 units; ShapeNetSem file units span 55,000x).
- Up axis comes from the format convention (glTF +Y; ShapeNetSem +Z in 87.1% of 396 labelled models; Valve Z-up, 0.01905 m per HU), then is validated against the card height. Do not use physical-stability sign heuristics (tables are right only 4% of the time).
- A 10% size error is a 33% mass error for solids and 21% for shells.

**Stage 1. Diagnose (L0)** (ms). Edge incidence after a tolerance weld (1e-5 of bbox diagonal; UV-seam splitting must be undone first), components, flat ratio, signed volume against hull volume, and the odd-crossing column fraction.

**Stage 2. Repair (L1)** (10 ms to 3 s).
- Drop degenerate and duplicate faces. An opposite-orientation duplicate pair is a double-sided sheet; keep one.
- Fix winding only if inconsistent. Do NOT use trimesh fix_normals(multibody=True) blindly, because it turns cavities into solids.
- Make orientation nesting-aware: a reversed inner component is a cavity, a same-way inner component is redundant, and a lone inverted component is flipped.
- Drop floaters (<0.2% of main volume) and ground slabs or planes.
- Fan-close small holes only when perimeter is small relative to the part's thinnest extent.
- Handle overlapping closed parts by pairwise inclusion-exclusion.

**Stage 3. Volume and cross-check (L2).**
- Exact polyhedral volume, centre of mass and inertia for closed parts.
- 3-axis column-interval fill as an independent estimate.
- Winding number only on disputed voxels or overlap boxes (libigl fast_winding_number if approved).
- The cross-check is skipped when feature size 2V/A is under 3 cells.

**Tiering.** A = derive. B = approximate. C = volume refused (null), shell rule or prior used. Thresholds in section 5 are provisional.

**Stage 4. Parts.** Split by the cheapest boundary that exists: glTF primitives, materials and nodes; Valve collision solids and VMT; PartNet semantic leaves. Cap at about 6 parts by mass share and merge parts with identical material posteriors. A learned segmenter (PartField, P3-SAM) is a later option and needs Linux. Connected components are poor evidence (raw PartNet leaves fall apart into 737 to 22,780 components versus 6 to 25 semantic leaves).

**Stage 5. Materials, per part.** Fuse evidence **by group**, not by source:
- Group A, asset metadata: Valve surfaceprop, VMT, glTF names and PBR, source tags.
- Group B, pixels: a vision-language model forced-choice with logprobs and the object name given.
- Group C, name: a curated kind card, a Deckard name prior and a qwen3:8b prior, counted as ONE group because the errors are correlated.

Output per part: a posterior over a canonical vocabulary (class, then about 40 material ids, then stock form), a confidence label, and alternatives. Ask Aaron only when the groups disagree AND the top two candidates differ in mass by more than about 25% of the object's mass. Store the answer on the card and propagate it to the kind.

**Stage 6. Mass router, per part, with Monte Carlo p5/p50/p95.**
- Measured wall: the mesh models its inner surface. Volume x density is exact, and thickness comes from ray or max-sphere (trimesh).
- Closed sheet-shell families (appliances, cabinets, drums, tubs, vessels): area x thickness x density, with area from the de-duplicated surface.
- Frame-and-panel and anything unknown: the learned bulk-density prior by topology.
- Cast-solid kinds (anvil, ball): volume x density, with a solid-or-hollow guard.
- Upholstery: foam bulk (20 to 50 kg/m3) over cushion volume.
- Density from Mentat only; proxies are flagged.

**Reference override rule.** If the card has a family with a sourced range, intersect it with the derived 90% interval. If they are disjoint by more than x1.5, keep the sourced scaled value, flag "derivation disagrees xN" and lower confidence. Derivation wins only when (a) the mesh has a measured wall, (b) no family exists, or (c) the object is outside the sourced size range. A mesh-derived mass never silently overrides a sourced total. A derived number is never labelled stronger than its weakest input.

**Stage 7. Mass properties.**
- Closed parts: exact (Mirtich, trimesh), about 3 ms for 5k triangles.
- Shelled parts: surface triangle quadrature (inertia 0.3 to 1.2% error, versus 43 to 45% for a bulk-density lump).
- Assemblies: parallel-axis theorem, not the Python grid.
- Emit principal moments plus frame. Rapier takes mass, centre of mass, principal inertia and a local-frame quaternion. GMod takes SetMass and SetInertia (diagonal in the local frame).
- Engine inertia for solid boxes reads x1.21 to x1.75 the uniform value for an unexplained reason, so set inertia explicitly and measure it back.

**Stage 8 (optional). Micro-voxel tier.** Built only on demand (break, cut, heat, certification).
- Representation: sparse column intervals, or surface-band bricks, with per-cell **material fraction** (mass-conserving), not binary occupancy.
- Pitch at most t/4 of the thinnest assigned wall (band volume error is 0.07% once t/p is 9 or more), capped at about 5e7 occupied cells.
- Evidence: fog voxels conserve mass to 1e-12 and give inertia within 0.3 to 0.5% at 10 mm for 0.6 to 6 mm walls. Binary axis-aligned sampling of a 1.5 mm wall gives sd 28% at p=2t and 104% at p=8t, and Deckard's _integrate returned 0.000 mass at n>=32.
- Cached beside the card; evicted when cold.

## 4. Object card additions (card stays about 0.5 KB; detail goes in sidecar `repair/<sha256>.json`, 1 to 3 KB)

In the card:
- `geometry.repair`: `{v, hash, level, tier A|B|C, solid_kind}`. The solid_kind values are closed_solid, closed_with_cavity, closed_thin_or_hollow_form, thin_solid, open_shell, sheet, multi_part_assembly, not_an_object. Rename the repair track's closed thin slab from "sheet" to thin_solid.
- `volume_m3`: `{value, low, high, basis measured|estimated, method}`. Null when tier is C.
- `solidity`, `thickness_m` (basis estimated; method 2V/A or ray), `com_m`, `inertia_unit_density` (6 numbers).
- `frame`: `{scale_to_m, scale_basis, scale_from, up, up_basis, origin}`.
- `mass.derived`: `{kg, low, high, basis estimated, path volume_x_density|shell_rule|topology_prior|sourced_anchor, vs_family_range}`.
- `parts[]`: `{id, name, role, closed, volume_m3, area_m2, com, bbox_occupancy, material, material_basis, stock_form, thickness_mm, alternatives}`.

In the sidecar: before/after counts (unwelded vertices, degenerate, duplicate, opposite-duplicate pairs, boundary and non-manifold edges, components); actions with parameters; estimator values and spread; per-part rows; timings; microvoxel file reference; ask-log.

Basis vocabulary stays measured / sourced / estimated / guessed. Add "catalogued" for table-backed material choices. Voxel masses carry the pitch and a two-pitch stability figure.

## 5. Tier rules (provisional; calibrated on 66 synthetic cases and 108 mugs only)

- **A (derive):** closed after repair; closed-hole total at most 2% of area, each at most 1.5x the part's thinnest extent; overlap at most 2% of volume; exact volume within 5% of the winding check when resolvable.
- **B (approximate):** holes at most 10% of area and 6x thickness; overlap at most 10%; estimators within 15%.
- **C (refuse volume):** open area over 10%, estimator spread over 10%, flat or sheet-like. Volume is null; use shell rule or prior.
- Mug sample: A 65, B 24, C 19 of 108.
- A whole missing part (a table leg) is undetectable from the mesh.

## 6. Experiments: what has been run

All run by researchers, read-only, in the scratchpad (wf4/*). See `experiment_plan`.

**Repair**
- 66-case defect suite: pipeline under 1% in 57, 5% or more in 5 (4 flagged); a trimesh-only baseline was 5% or more off in 21, silently.
- 298 Objaverse GLBs: 49% closed, 74% with 2 or more components.
- 400 ShapeNetSem: 7.8% closed.
- 50 PartNet: 0 had a closed part; Deckard confidence is a flat 0.50.
- Deckard voxelize versus an independent parity fill on PartNet: median ratio 0.70, only 2 of 50 within 25%, up to 90 s, one model over 4 GB.
- Scanline: 0.02 to 0.25 s at 10k to 1M triangles.

**Mass**
- Hybrid router x1.47 median on 120 models, 42% inside the sourced range.
- Frame-and-panel prior x1.22 to x1.28.
- Shell rule x1.31 to x1.5.
- Solid volume x density x10.55 and bbox x27.5.
- Deckard-built kinds x1.69 median (chair x4.2, side table x5.2).

**Materials**
- Surfaceprop 84.3%; Deckard name prior 62.8%; qwen3:8b 68.6 to 71.1%; llava 61.2% image-only, 66.9% with the name.
- LOFO fusion 86.0%; all three groups agreeing 98% (n=49).

**Micro-voxel**
- Chair mass halves per pitch halving; fog voxels exact in mass.
- Three voxelize() defects (section 9).

## 7. Known weak points, said plainly

- No per-object ground truth exists, and no per-part material ground truth. All material numbers are object-level.
- No generated-mesh data (TripoSR, Hunyuan3D, TRELLIS.2) was measured. TRELLIS.2 needs 24 GB or more VRAM and Linux.
- Tier thresholds are tuned on tiny synthetic cases.
- Only 4 images were viewed by the materials researcher; labels are one annotator's reading.
- Valve wiki pages returned 403, so several Valve facts rest on search summaries.
- Only llava 7B was tested as a vision model. A stronger one (qwen3-vl:8b, 6.1 GB) is untested.
- Weak mass classes (solid/plant, cardboard, ceramic, lamps) have 1 to 2 references.
- The fill-holes loop closer leaves some cup holes open, and non-convex holes are wrong with fans.

## 8. Mentat gaps (physics rule: densities must come from Mentat)

`mentat_call.sh density` returns **ok:true with value null** for missing materials (no error). Callers must check for null. Missing: steel, stainless_steel, cast iron, brass, bronze, ceramic, porcelain, brick, granite, fabric, cotton, leather, foam, polyurethane, paper, cardboard, plywood, particleboard, MDF, PMMA, polystyrene, polyethylene, nylon, titanium, cork. Present: 35 materials, including wood_oak 770, wood_pine 530, glass 2500, plastic 1100, rubber 1100, iron 7874, aluminum 2700, concrete 2400, copper 8960.

Review items:
- wood_pine 530 ("Eastern white pine nominal") versus PNNL/Marks white pine 430 and Deckard's FPL-cited 420.
- rubber 1100 versus PNNL natural rubber 920.
- Deckard's material_profile() silently returns an estimated 1000 kg/m3 at confidence 0.2 for cast iron, plywood, MDF, polystyrene, memory foam, stoneware, vitreous china and about a dozen other names. A labelling step will hit these constantly.
- The materials researcher extracted about 40 citable values (page references) from PNNL-15870 Rev.2 for the Mentat owner to enter. Examples: carbon steel 7872, stainless 304 8030, gray cast iron 7150, brass 8070, PET 1380, PE 930, PP 900, PVC 1406, plate glass 2400. Not entered or invented by anyone.
- Porous and woven materials (foam, fabric, cardboard, boards) must be stored as BULK or AREAL values with a flag (fibre density x bbox is the 51x error).
- Mentat has no thin-shell mass tool and no box/cylinder inertia tool. rotational_dynamics offers only rod, parallel axis and a sphere factor.
- Interim proxies, flagged on the card: steel and cast iron to iron (7874 versus about 7850, 0.3%), ceramic to glass (2500 versus about 2400), with the proxy named in `material_basis`.

## 9. Deckard defects to fix before anything else relies on it

1. voxelize() surface-touch plus fill: O(pitch) overestimate (+37% box and +44% sphere at 40 mm); collapses on holes; PartNet crust shrinks with pitch.
2. floor((world-lo)/pitch) on lattice-aligned meshes loses 20 to 27% of cells (-24% at 5 mm, -14% at 10 mm).
3. `except Exception: continue` silently drops a part that trimesh cannot voxelise (reproduced with max_iter exceeded at 1 mm).
4. construct._integrate: pure Python grid; 6 to 10% inertia error on a table; erratic mass versus n; zero mass for a 1.5 mm wall.
5. Inertia of shelled parts is a bulk lump (43 to 45% error).
6. watertight_frac is 0.000 for PartNet, so confidence is a flat 0.50.
7. The axis-orientation fix is deliberately not applied (HOLODECK_PHYSICS_AUDIT.md) and stays out of scope here.
8. Memory: 4.4 GB (once 12.8 GB) on a 9k-face, 34-part PartNet model. The cause was not isolated.

## 10. Licences and platform

- libigl 2.6.3: MPL-2.0 (CGAL copyleft parts not checked in the wheel).
- manifold3d: Apache-2.0.
- Open3D: MIT.
- trimesh: MIT.
- pymeshlab: GPL-3. pymeshfix: AGPL-3. Both are a real question if the portal serves other people. Avoid them unless Aaron decides otherwise.
- ManifoldPlus: non-commercial.
- VoMP: code Apache-2.0, weights NVIDIA Open Model License, Linux installer, solid-fill bias on hollow objects. Optional independent density opinion only.
- Objaverse per-asset licences are uploader-asserted. Valve, ShapeNet and Sketchfab assets must not go to the portal (SKIPPYVERSE_OBJECTS_DESIGN.md).
- Portal side: manifold-3d WASM (manifoldness check, union volume) and a Worker port of the scanline. Not measured.



## Experiments

- **R1. Defect-injection suite (RAN)**: Staged repair plus tiering beats trimesh-only and the current Deckard path. No single estimator survives every defect, so cross-check is required. How: 4 shapes of known volume x 17 defects, estimators compared (raw signed integral, Deckard path, 3-axis parity, winding number, trimesh-only repair, repair_v0). Pass if: Repair under 1% on most cases and every 5%-or-worse case flagged. Result: v0 under 1% in 57 of 66, 5% or more in 5 (4 flagged C, 1 silent: a whole missing leg). Baseline 21 of 66 at 5% or more, unflagged, worst +1350%. Estimator medians: raw 4.65%, Deckard 58%, parity 3.9%, winding 0.91%. (done)
- **R2. Real-mesh defect surveys (RAN)**: Lazy repair is mandatory. 'Watertight' cannot be assumed for found or dataset meshes. How: 298 Objaverse GLBs, 126 mugs, 400 ShapeNetSem OBJs, 50 PartNet models. Pass if: Measures how often repair is needed. Result: Objaverse 49.3% closed; ShapeNetSem 7.8%; PartNet 0 of 50 with any closed part; Deckard confidence is 0.50 for all PartNet models. (done)
- **R3. Deckard voxelize versus independent parity on PartNet (RAN)**: Replace the voxel fill. The reference itself is not validated on open parts. How: 50 comparable models, Deckard unchanged at 1/100 pitch against a per-part 3-axis parity fill. Pass if: Ratio near 1. Result: median 0.70, p5 0.10, p95 15.3; 2 of 50 within 25%; up to 90 s; one model over 4 GB; scanline reference 0.8 s and 0.18 GB on the same model. (done)
- **M1. Mass methods on 48 independent references and 120 audited models (RAN)**: The router design. The noise floor of the sourced data is about x1.39. Adopted masses share bbox scaling with the methods, so agreement is partly circular. How: Solid, shell, Deckard occupancy and learned bulk density, leave-one-family-out, against sourced masses and ranges. Pass if: Median error factor and share inside the sourced range. Result: topology prior x1.46 (frame x1.22); shell x1.50 on shells; occupancy x1.99; solid x25.6; hybrid router x1.47 with 42% in range on the 120. (done)
- **M2. Deckard-built objects versus sourced (RAN)**: Occupancy fixed the x10 to x100 errors but is still heavy; frame-and-panel kinds should use the topology prior. How: 29 kinds from physics_audit.json compared with the family reference scaled to size. Pass if: Within x2 for most. Result: median x1.69, 69% within x2, 28% inside range; chair x4.2, side table x5.2. (done)
- **T1. Material labelling sources on the 121 models (RAN)**: Group-level fusion with agreement gating. Raw VLM confidence is overconfident. This is object-level only; no per-part truth exists. How: Surfaceprop, Deckard name prior, qwen3:8b, llava (JSON and logprob forced choice), self-consistency, leave-one-family-out fusion. Pass if: Primary-material accuracy and per-class recall. Result: surfaceprop 84.3% (metal recall 93%); Deckard 62.8% (metal 19%); qwen3 68.6 to 71.1%; llava 61.2% / 66.9%; LOFO fusion 86.0%; all three groups agreeing 98% (n=49); pixels+text agreeing 82% (n=73), disagreeing 34% (n=41). (done)
- **T2. Shell model with sourced thickness (RAN)**: Stock form and thickness decide mass, not the material label. Rho values in this experiment were NOT from Mentat. How: m = rho x t x S on 33 shell models of 15 families; back out implied thickness. Pass if: Within x2 for most. Result: rms x1.64, 76% within x2; implied thicknesses physically sensible (sheet steel 0.55 to 3.4 mm, china 6.4 to 8.5 mm). (done)
- **V1. Fog versus binary voxels on thin walls, and Deckard _integrate (RAN)**: Micro-voxel tier must store partial volume. The current integrator is unsafe for walls. How: Hollow-box sampling at 0.5 to 12 mm; real construct._integrate on solid, hollow and table cases. Pass if: Mass conserved and inertia under 1%. Result: fog mass exact, inertia 0.3 to 0.5% at 10 mm; binary sd 28% at p=2t; _integrate returned 0.000 for a 1.5 mm wall. (done)
- **V2. Column-interval accuracy and cost (RAN)**: Column intervals serve as both the fill and the sparse microvoxel representation. Thin features under about 3 cells degrade. How: Scanline intervals versus trimesh exact on sphere, torus, table and cup at 32 to 256 cells; 100k-triangle torus down to 0.49 mm. Pass if: Volume under 1% at 64 cells. Result: sphere +0.146% at 32 and 0.000% at 128; cup dips -2.8% at 64 (wall 1.9 cells); table +10.4% at 32; 0.19 s and 22 MB at 0.49 mm. (done)
- **E1. Gate on the 121 sourced models with the full pipeline (NOT RUN)**: The end-to-end claim, scored against the repo's ground truth. Must report the circularity caveat. How: Run the ported pipeline on a mesh for each model (Valve models only on the PC; if meshes are unavailable, use PartNet or Objaverse exemplars of the same families). Score the p50 and p5-p95 interval against fidelity.json sourced ranges, leave-one-family-out for any fitted prior. Pass if: At least 45% of derived p50 inside the sourced range (current hybrid baseline is 42%, Deckard 28%), median error at most x1.5, p5-p95 coverage at least 85%, no tier A or B result more than x3 outside the range. (2 days after increment 3)
- **E2. Labelled real-mesh quality set (NOT RUN)**: Whether the 1.5x / 6x hole thresholds hold outside synthetic cases, and how generated meshes behave. How: Hand-label about 100 real meshes (closed solid / hollow / open / sheet / scene) with a measured or sourced reference volume. Repeat on 30 meshes each from TripoSR, Hunyuan3D-2 and TRELLIS.2 when available. Pass if: Tier A precision at least 95% (derived volume within 10% of reference); silent failures (tier A or B with error above 15%) under 2%; tier thresholds re-tuned. (3 to 4 days)
- **E3. Stronger VLM rerun (NOT RUN)**: How much of the ask-rate (about 1 in 2 for found assets) a better model removes. How: ollama pull qwen3-vl:8b (6.1 GB; over this run's download limit), rerun vlm_run.py and lp_run.py unchanged, then score_all.py and lp_score.py. Pass if: Beat llava+kind 66.9% primary and metal recall 40%; top-1 at least 75%; lower overconfidence in the reliability bins. (half a day)
- **E4. Per-part materials ground truth (NOT RUN)**: Part-level quality, currently unmeasured. How: Highlighted-part renders for PartNet exemplars and about 20 hand-labelled Objaverse assets; compare with Aaron's labels or PhysXNet if released. Pass if: Mass-weighted per-part accuracy at least 80%. (2 days)
- **E5. Voxel certification test (NOT RUN)**: Voxels add no error beyond the thickness input, justifying the tier as an optional certificate. How: For the 121 models, build the thickness-band or fog voxel field lazily at t/4 (cap 5e7 cells) from card thickness and material, integrate mass, repeat at two pitches. Pass if: Mass stable within 5% across pitches, and agreeing with the exact or shell-rule mass within 5%. (1 to 2 days)
- **E6. libigl, manifold3d, Open3D timing (NOT RUN, needs install approval)**: Whether the heavier libraries are worth their licences. How: Venv install; time fast_winding_number, boolean-union volume, fill_holes on the 100k and 1M-triangle test meshes. Pass if: fast_winding_number under 2 s for 100k triangles at 64^3; union volume within 1% of inclusion-exclusion. (1 to 2 days)
- **E7. Universe Sandbox save round-trip and REBOUND versus sg_nbody (NOT RUN)**: Whether Update 36 saves are still editable; whether the in-house integrator is a valid ground truth. How: Only if Aaron already owns the game: edit a 2-body save and compare period with Mentat orbital_period. Separately, REBOUND (IAS15) versus sg_nbody versus JPL Horizons over 1 year. Pass if: Save still readable JSON and edits survive; per-planet position errors below Horizons uncertainty. (half a day each)

## Increments

- **I1. Stop the bleeding in Deckard's voxel fill**: Replace trimesh voxelized + binary_fill_holes in sigma_ground/deckard/voxelize.py with a numba 3-axis column-interval fill (same VoxelField contract, part labels kept). Round instead of floor on grid indices. Raise or flag a part that fails instead of silently skipping it. Report post-repair watertight state in watertight_frac. Test: Sphere, torus, cup and table volumes within 1% at 64 cells (cup within 3% when the wall is under 3 cells). Identical volume for an aligned versus sub-voxel-shifted box. A deliberately failing part yields a flagged low-confidence result, not a silent zero. The 50-model PartNet comparison runs with p90 under 2 s and peak memory under 1 GB. Existing deckard tests pass, including test_occupancy.py. (1 to 2 days)
- **I2. mesh_repair module (L0 and L1)**: sigma_ground/deckard/mesh_repair.py: diagnose, weld, dedupe, double-sided handling, components, nesting-aware orientation, floater and floor removal, guarded hole closing, overlap inclusion-exclusion, tiering. Output repair sidecar JSON, keyed by sha256 + version. Test: The 66-case defect suite as pytest: at least 57 cases under 1%, no tier A or B case at 5% or more except the documented missing-leg case, and a trimesh-only baseline recorded alongside. 108 mugs: no crash, tiers within 3 points of the published A 65 / B 24 / C 19. Planes and the leaf are tier C with null volume. A 100k-face mesh does not call fix_winding unless inconsistent. (3 to 4 days)
- **I3. Mass properties and the router**: Exact closed-mesh properties; shell quadrature inertia; parallel-axis assembly; topology bulk-density prior fitted from fidelity_research.json; part-class router; Monte Carlo p5/p50/p95; sourced-reference override rule; principal moments plus frame. Test: Synthetic hollow box exact to 1e-9 relative; shelled-part inertia within 2% of the analytic hollow body; hybrid router reproduces the published x1.47 median and 42% in-range on the 120-model table within noise; the override rule flags 'derivation disagrees' on a forced x3 mismatch; Monte Carlo covers at least 85% on leave-one-family-out. (3 days)
- **I4. Mentat gaps closed and the material vocabulary**: Mentat owner adds the missing densities with citations. A canonical vocabulary (class, about 40 material ids, stock form with typical thickness and cited density, bulk/areal flags for porous materials). Deckard's 1000 kg/m3 fallback replaced by a loud 'unknown material' result. Null density handled explicitly. Test: mentat_call.sh density returns a sourced value for every id in the vocabulary, or the id is listed as 'proxy: X' with the proxy named on the card. No name in the vocabulary reaches the 1000 fallback. A test fails on any ok:true / value:null density. (1 to 2 days (Mentat owner))
- **I5. Free-label readers and the kind-card table**: Readers for Valve surfaceprop/VMT/solid (from measured.json first), glTF material names, PBR hints and tags; an alias table (dirt/ceiling_tile/carpet to upholstered seat, computer to plastic/metal/glass); kind cards seeded from fidelity_research.json. Material fusion by evidence group with agreement gating. Test: On the 121 models, surfaceprop-based primary at least 84%; fused LOFO primary at least 85% with metal recall above Deckard's 19%; confidence bins calibrated within 10 points; ask-rate with metadata at most 20%. (2 to 3 days)
- **I6. Card additions and cache**: Card fields and sidecar from design section 4, lazy async trigger, never blocking spawn, cache invalidation by repair/material-table version, ghost-box plus family-prior mass until ready. Test: Spawning an uncached object returns immediately with a prior mass; the derived card replaces it when ready; a second spawn is a cache hit under 50 ms; a changed material-table version invalidates the entry; no derived field is labelled stronger than its weakest input. (2 days)
- **I7. Gate run on the sourced set (E1) and tier calibration (E2)**: Scored report against fidelity.json; tier thresholds re-tuned on a labelled set of about 100 real meshes. Test: E1 and E2 pass criteria from the experiment plan. (4 to 5 days)
- **I8. Better vision model and per-part prompts**: Pull qwen3-vl:8b (or gemma4:12b), rerun the harness, add highlighted-part prompts with logprobs, timeouts and retries so a stalled shared GPU never blocks the game. Test: E3 and E4 pass criteria; a stalled Ollama call is abandoned after the timeout and the fallback path is used. (2 to 3 days)
- **I9. Sparse fog micro-voxel tier (optional)**: On-demand sparse column-interval or band store with per-cell material fraction, pitch at most t/4, 5e7 cell cap, stored beside the card; used for break/cut/heat and for mass certification. Test: E5: mass stable within 5% across two pitches and within 5% of the exact or shell-rule mass; inertia within 1% at 10 mm for the cabinet; build under 10 s; stored size 3 to 10 MB; never loaded by the render path. (4 to 5 days)
- **I10. Portal-side cheap stages (optional)**: Worker port of the scanline (about 150 lines) and a manifold-3d manifoldness check for found open-licensed assets. Valve and ShapeNet never go to the portal. Test: Worker result within 0.5% of the Python version on the test set; a 100k-triangle mesh is processed in under 3 s in the browser; a Valve-derived asset is refused at the portal boundary. (3 days)

## Decisions for Aaron

- **Are you willing to replace Deckard's voxelize fill now (increment 1)? It is the weakest link, and the repair work depends on it.** Recommendation: Yes, do it first. Three defects are reproduced: an O(pitch) overestimate, a lattice-alignment bug that loses 20 to 27% of cells, and a silent per-part skip. It changes the numbers in the physics audit, so regenerate the audit afterwards. The axis-orientation fix can stay deferred.
- **May I install libigl, manifold3d and Open3D in a scratch venv to time them (increment E6)? Which licences are acceptable for the portal?** Recommendation: Allow libigl (MPL-2.0), manifold3d (Apache-2.0) and Open3D (MIT) for evaluation. Do not use pymeshlab (GPL-3) or pymeshfix (AGPL-3) in anything the portal serves. ManifoldPlus is non-commercial.
- **Who adds the missing densities to Mentat, and which proxies are acceptable meanwhile?** Recommendation: Mentat's owner enters steel, cast iron, stainless, brass, ceramic/porcelain, fabric, foam, cardboard, paper, leather, plywood, MDF and the common plastics, using the PNNL-15870 Rev.2 values with page references. Until then proxy steel and cast iron to iron and ceramic to glass, with the proxy named on the card. Also decide whether wood_pine 530 and rubber 1100 are intended (PNNL gives 430 and 920).
- **Should a sourced family mass always beat a mesh-derived mass?** Recommendation: Yes by default. A derived mass wins only with a measured wall, no family, or an out-of-range size. When they disagree by more than x1.5, keep the sourced value and flag it. Nothing claims better than about x1.4.
- **Should I pull a stronger local vision model (qwen3-vl:8b is 6.1 GB; gemma4 and qwen3.5 are other candidates)?** Recommendation: Yes; it is the cheapest way to lower the ask-rate for found assets. The harness reruns unchanged. The GPU is shared at 100% load with other jobs, so calls need timeouts and an async fallback.
- **Where do micro-voxels sit in the plan?** Recommendation: As the optional last tier (increment 9), stored as material fractions, sparse, lazy and cached, built on demand for break, cut, heat or certification. Not a render path and not a default step. Postpone until increments 1 to 7 pass.
- **How much do you want to be asked?** Recommendation: Ask only when evidence groups disagree and the mass at stake is over about 25% of the object. Expect about 1 question in 5 for Valve models and about 1 in 2 for found assets with the current small vision model. One answer per kind (for example 'filing cabinets here are steel') removes most repeats.
- **Do you want the Universe Sandbox write-up corrected, and is celestial-scale GMod still wanted?** Recommendation: Correct the 'editable JSON' sentence (Update 36 changed the format). Because the portal is now the main client, drive the solar system from the web scene with the same sg_nbody trajectory bundle. Treat GMod scaled stages (Phase 4) as lower priority. Validate sg_nbody against REBOUND and JPL Horizons once REBOUND is allowed.
- **Is the pipeline researcher's missed assignment a problem?** Recommendation: No. That track's text did not match the task and it answered only the Universe Sandbox question. The design above uses the other three tracks and the earlier check, so nothing is missing for the pipeline itself.

## Needed from Mentat
- Densities with sources for: steel (carbon, e.g. 7872 PNNL 1008), stainless_steel (304 8030, 316 8000), cast_iron (gray 7150), brass (8070), bronze, titanium.
- Densities for: ceramic and porcelain/vitreous china, brick (common silica 1800), granite, terracotta/stoneware.
- Bulk or areal values flagged as porous or woven: fabric/cotton, leather, foam (PU flexible ~21, memory ~50), cardboard/corrugated, paper (newsprint 650, printer 690), cork.
- Panel materials: plywood, particleboard, MDF, hardboard (secondary sources only so far: MDF 500 to 1000, birch plywood ~680, conifer plywood 460 to 520).
- Plastics: PMMA/Lucite 1190, polystyrene 1060, polyethylene 930, polypropylene 900, PVC 1406, polycarbonate 1200, nylon 6/6 1140, PET 1380, ABS (Deckard cites 1050), PTFE 2250.
- Fix: density currently returns ok:true with value null for unknown materials; it should return an explicit not-found error.
- Review: wood_pine 530 versus 430 (PNNL/Marks) and 420 (Deckard, FPL); rubber 1100 versus natural rubber 920.
- Tools that do not exist: thin-shell mass (area x thickness x density), box/cylinder/shell inertia, parallel-axis assembly of many bodies (rotational_dynamics has only rod, parallel axis and a sphere factor).
- Provenance tags for every new value (PNNL-15870 Rev.2 page references already extracted by the materials researcher; reuse terms unverified).


---

## Red team
Direct answer to the Captain's question: NO. The Universe Sandbox check found no valid scriptable appliance. It has no scripting API, its 2022 script-library mod is unofficial, and the save-format claim is stale after Update 36. Sourced from earlier research summaries, not re-verified by me today. The right celestial route is in-house sg_nbody with REBOUND and Horizons as oracles. Design verdict: the architecture (lazy, cached, per-part, router plus sourced override, fog voxels as optional tier) is sound, but several headline numbers are weaker than presented. (1) The 42% in-range and x1.47 median are scored against masses that are themselves clamp(reference x s^k) with 44 of 120 on a bound, so they test self-consistency, not accuracy. (2) The tier thresholds come from 66 synthetic cases plus 108 mugs. (3) Material accuracy is object-level only, and 86% LOFO fusion is tuned on the same 121 models. (4) No generated-mesh data exists. (5) Nothing is built, so the design's claims about laziness and security are untested. I re-checked only simple arithmetic this session: the 1e9/1e12/1e15 cell counts, the chair surface extrapolation to 2.3 M cells at 1 mm and 0.23 G at 0.1 mm, and the 10% size error giving +33% (solid) and +21% (shell) mass. These are correct. I did not re-run any experiment, so every other number rests on the researchers. I did note an inconsistency: the design gives true-scale Earth as 0.00072 HU on a 20 m stage, but my arithmetic (Earth radius over 1 AU, times 20 m, over 0.01905 m/HU) gives about 0.045 HU for the Earth-to-Sun-distance ratio, i.e. a different quantity, so the quoted figure depends on an unstated stage mapping and should be recomputed or its basis stated.

### [high] Validation is partly circular, so 'pass' criteria can be met without being right
Adopted masses in fidelity.json are clamp(reference x s^k, low, high), 44 of 120 sit on a bound, and the methods use the same bbox scaling. E1's pass bar (45% in range vs 42% baseline) is a 3-point gain on a noisy, partly self-referential target; sourced ranges themselves carry about x1.39 noise. A pipeline can pass E1 and still be wrong on real objects.

**Fix:** Add a truly independent holdout: 15-20 objects Aaron or we physically weigh or that have manufacturer spec-sheet mass and a matching mesh. Report accuracy on those separately. Replace the +3-point threshold with a bootstrap CI and require the lower bound to beat the baseline. Never claim better than x1.4.

### [high] Reference-override rule makes the derivation decorative and hides failures
The rule keeps the sourced scaled value whenever derived and sourced disagree by more than x1.5, so the headline mass for most family objects is the family prior, not a derivation. The pipeline's value is then only visible on objects without a family, which is exactly where the design has the least validation (weak classes have 1 to 2 references, no generated-mesh data).

**Fix:** Report two numbers on every card (prior-scaled, derived) and score each separately. Track how often override fires; if it fires over about 30%, the derivation is not earning its cost. Make 'derivation disagrees' a first-class output, not a silent fallback.

### [high] Untrusted-mesh processing has no sandbox, resource caps or format hardening
Found GLB/OBJ and generated meshes are parsed by trimesh/numba in the main process. Risks: decompression/billion-triangle bombs, huge texture or buffer-view claims, NaN/inf vertices, degenerate sizes that make 2V/A or voxel counts explode (cap of 5e7 cells is stated but pre-parse limits are not), glTF external URI references (SSRF/local file read), Draco/meshopt decoders, pickle-based caches. Deckard already hit 4.4 to 12.8 GB on a 9k-face model, so memory is an unbounded-resource issue even for honest input.

**Fix:** Parse in a subprocess with hard RLIMIT/job-object memory (say 1 GB) and a wall-clock timeout; pre-check file size, triangle count, accessor counts before loading; reject external URIs; reject non-finite values; never unpickle; run Valve/ShapeNet jobs on PC only as designed, and key the cache by sha256 only after size checks. Treat every numba kernel as needing bounds checks (unchecked indexing can corrupt memory).

### [high] 'Lazy' is not enforced: first physical need is undefined and spawn can still block
The design says never block spawn, but trigger (what counts as first physical need: pickup, collision, weight query?), queueing, back-pressure, and a stalled shared GPU/Ollama are unspecified. Repair at 3 s plus VLM calls plus Monte Carlo plus optional 10 s voxel builds can stack up when a voice command adds many objects ('three couches' triggers three jobs of one kind). The GPU was 100% busy in every track.

**Fix:** Dedupe by cache key before enqueue (one job per sha256), single worker queue with priority, a per-job deadline (e.g. 5 s for L0-L2, 30 s for materials) after which the prior stands and the job is marked stale, and an explicit test: spawn 20 uncached objects while Ollama is hung and assert spawn latency is unchanged.

### [high] Material labelling by LLM/VLM is overconfident and its error cost is not propagated
Group-level fusion gets 86% object-level, but metal recall from the name prior is 19%, llava is 61 to 67%, and no per-part truth exists. A wrong class changes density by 5x to 15x (foam 30 vs steel 7850). The ask rule (ask only when top two differ by more than 25% of object mass) depends on calibrated posteriors, but raw VLM confidence is stated as overconfident, and fusion reliabilities were fit on the same 121 models. 98% when all three agree is on n=49 and agreement is correlated with easy objects.

**Fix:** Carry material uncertainty into the Monte Carlo as a categorical draw from the posterior, not a point choice, so the p5-p95 widens honestly. Calibrate on a held-out split (not LOFO on the fitting set) and report ECE. Cap confidence for any part whose material came only from group B/C. Make the stronger-VLM rerun (E3) a gate before material labels are used for anything but display.

### [medium] Wall-thickness error can reach 10x and the design underplays it
Thickness sets shell mass linearly. Sourced vs default barely differs (x1.44 vs x1.50), which the design reads as 'error floor is structure'; the alternative reading is that neither thickness source is informative at this resolution, so any reported interval narrower than about x1.5 is false precision. Ray/max-sphere thickness on found meshes with double walls, coincident layers or thickened UV seams can give 10x errors; 2V/A fails for ribs and tubes.

**Fix:** Put a prior on thickness spanning at least x3 per stock form in the Monte Carlo, require two independent thickness estimates to agree within x2 before basis 'estimated' (else 'guessed'), and flag any part where thickness differs from the stock-form typical by more than x3.

### [medium] Repair can silently change the object, and the safeguards are heuristic
Dropping floaters under 0.2% of main volume, ground slabs, duplicate faces, fan-closing holes and nesting-aware orientation all alter geometry. A thin legitimate part (a handle, a chair leg far from a big seat) can be removed; a whole missing leg is undetectable; inverted-component flips can turn a real cavity into solid or the reverse. Silent failures were 1 of 66 on synthetic cases only.

**Fix:** Record every dropped or modified component with its volume share in the sidecar and cap total repair-induced volume change (e.g. 5%); if exceeded, tier C. Keep the original-mesh hash and show a before/after bbox/volume diff on the card. Add a symmetry or family-prior check (expected part count) to catch missing parts.

### [medium] Tier thresholds are overfit to synthetic data
The 1.5x/6x hole thresholds and 2%/10% limits were set on 66 synthetic cases and 108 mugs, a single family. Mug tier split (65/24/19) is a distribution fact about mugs, not a calibration. Objaverse is only 49% closed, ShapeNetSem 7.8%, PartNet 0%: most real input lands in B/C, so the main path is the less-validated one.

**Fix:** Treat thresholds as provisional until E2; until then default ambiguous cases to B with widened intervals. Report the share of real inputs reaching each tier up front so Aaron knows what fraction gets a real derivation.

### [medium] Micro-voxel cost claims rest on extrapolation and the optional tier has a hidden memory risk
The chair figure (37,640 cells at 7.8 mm) is one measurement extrapolated by N^2 to 2.3 M at 1 mm and 0.23 G at 0.1 mm; thin or fractal surfaces (foliage, fabric, meshes with internal sheets) scale worse. A 5e7 cell cap with material fractions is 50 to 400 MB depending on the encoding, not the stated 3 to 10 MB. Two-pitch stability needs two builds, doubling cost.

**Fix:** Measure cell counts on at least 10 varied meshes (including a leaf and a table) before promising 3 to 10 MB. Define the on-disk encoding (run-length intervals plus fraction byte) and report bytes per cell. Enforce the cap at the plan stage by estimating cells from area/pitch^2 before building.

### [medium] Densities and Mentat integration have unflagged hazards
Mentat returns ok:true with value null for missing materials, so callers that cast to float get NaN or 0 mass; wood_pine 530 and rubber 1100 disagree with the cited 420 to 430 and 920; interim proxies (steel to iron, ceramic to glass) are small but the 1000 kg/m3 fallback in Deckard is not. The T2 shell experiment used densities NOT from Mentat, so its x1.64 does not demonstrate the rule-compliant pipeline.

**Fix:** Fail loudly on null; add a test that every vocabulary id resolves or is a named proxy; rerun T2 and the router with Mentat-sourced densities and report the delta; get the pine/rubber discrepancy resolved before relying on them.

### [medium] Rigid-body inertia output is not verified against the engine
The design notes engine inertia for solid boxes reads x1.21 to x1.75 the uniform value for an unexplained reason, yet plans to emit principal moments and 'measure it back'. Unknown engine behaviour means the derived tensor can be applied wrong while the sidecar looks perfect. Inertia error on assemblies depends on correct parallel-axis frames, which depend on the unvalidated Stage 0 up-axis.

**Fix:** Add a GMod/Rapier round-trip test for 3 shapes with analytic inertia before any card claims inertia; add a unit test that applies a known tensor and reads back angular response. Resolve the x1.21 to x1.75 discrepancy first.

### [low] Frame/scale step trusts the card size, which can itself be wrong
Scale 'always comes from the card size'; but the card size for generated or found objects may itself be a guess, and a 10% error is 33% in solid mass. Up-axis validated against card height can pass for symmetric objects (cube-like) where both axes fit.

**Fix:** Propagate card-size uncertainty into the Monte Carlo (basis-dependent sigma) and flag objects whose bbox is near-cubic as 'up-axis unverifiable'.

### [low] Scale arithmetic for the celestial stage and the Universe Sandbox write-up need correction
The 0.00072 HU Earth radius figure is not reproducible from the stated 20 m stage without an unstated mapping, and the 'saves are editable JSON' claim rests on 2020 to 2022 posts while Update 36 changed the format.

**Fix:** State the mapping (metres per AU) in the doc and recompute from Mentat. Reword the save-format claim as the design itself suggests; no further Universe Sandbox work is warranted.

