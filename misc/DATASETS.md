# Dataset Inventory — sigma-ground data lane

Every external dataset on disk, its license posture, and what we distilled from
it. Doctrine: **raw data stays local + gitignored; only distilled, cited
aggregates are committed** (`sigma_ground/inventory/data/*`). Updated
2026-07-08 (final pass — all downloads/distills complete except PartNet-Mobility).

## Shape oracles

| Dataset | Location | Size | Status | License / ToU |
|---|---|---|---|---|
| PartNet `data_v0` | `D:/datasets/shapenet/PartNet/` | 23 GB (extracted), 129 GB archives | **32,537 models, 24 categories**, censused; per-part semantic hierarchy (`result.json`) present per model. Exemplar pools redistilled **K=4 → K=8** (184 files, 23/24 categories — keyboard has no viable candidates in the geometry data) | ShapeNet ToU — non-commercial research; distilled aggregates only in git |
| ShapeNetCore v1+v2 | `D:/datasets/shapenet/ShapeNetCore*` | 23 GB processed + 55 GB archives | **120 models extracted, 5 target synsets** PartNet lacks entirely: car(30/3533), airplane(30/4045), guitar(20/797), watercraft(20/1939), motorcycle(20/337) — real engineering assemblies, not furniture. Every model ships a pre-computed `model_normalized.solid.binvox` — a trusted solid voxelization, alternative to our fill-heuristic path | ShapeNet ToU |
| ShapeNetSem | `D:/datasets/shapenet/ShapeNetSem-archive/` + `D:/datasets/shapenetsem/` | 12 GB | Two layers: (1) aggregate stats already distilled (`shapenetsem_sizes.json`, `materials.csv` distills) — per-model weight/solidVolume/isContainerLike/staticFrictionForce columns are genuinely EMPTY (0/12,288) in this copy, confirmed against raw rows. (2) **A full shape source we'd never opened**: 12,288 models, real OBJ+MTL meshes + an independent `models-binvox-solid` grid per model, `up`/`front` real orientation (30% filled, 49 distinct up-vectors — NOT all Z-up) + `unit` real-world scale (77% filled). **30 gap-category models extracted**: Hammer(22), ScrewDriver(1), Motorcycle(7) — categories confirmed absent from PartNet. Its `fullId` hash is the SAME id scheme as ShapeNetCore/PartNet `model_id` (verified on our own chair 44164) | ShapeNet ToU |
| Objaverse LVIS (targeted) | `D:/datasets/objaverse/` | 4.4 GB | **864/864 models, 13 categories** (hammer 59, mallet 79, frying_pan 31, saucepan 26, teakettle 69, wineglass 101, pitcher 73, screwdriver 40, wrench 53, lightbulb 83, **motor 54**, cup 70, mug 126). NOT in LVIS: axe, skillet, anvil, feather, gear | **Per-object** Sketchfab licenses — see `inventory/data/objaverse_ledger.csv` (821 CC-BY, 15 CC-BY-NC, 9 CC-BY-SA, 14 CC-BY-NC-SA, 5 CC0). **No real-world scale annotation** — size grounding must come from typical-size/Wikidata at integration |
| KiCad packages3D (sparse) | `D:/datasets/electronics/kicad-packages3D/` | 76 MB | 8 THT component families (Resistor, Capacitor, Diode, LED, TO/SOT packages, Battery, Relay, Fuse) — real STEP/WRL models for future working-circuit internals | CC-BY-SA 4.0 (with KiCad libraries exception) |
| PartNet-Mobility (SAPIEN) | — | est. 10–20 GB | **NOT on disk** — awaits the Captain's registration at sapien.ucsd.edu (2,346 articulated models, 46 categories, URDF joints — the actuation enabler) | SAPIEN ToU, non-commercial; account required |

## Physics / materials data

| Dataset | Location | Size | Distilled aggregate (committed) | License |
|---|---|---|---|---|
| refractiveindex.info database | `D:/datasets/optics/` | 70 MB (4,198 files) | `inventory/data/optics_nk.json` — measured n,k at 650/550/450 nm for 16 materials, paper cited per entry (Johnson & Christy, Rakić, Werner, Inagaki, Aspnes & Studna, Hale & Querry) | open database; papers cited per entry |
| NIST atomic weights + isotopic compositions | `D:/datasets/chemistry/nist_atomic_weights.txt` | 1.4 MB | `inventory/data/atomic_weights.json` — 118 elements, 3,352 isotopes (verify: Hg = 200.592) | public domain (US gov) |
| Burcat/ATcT thermochemical database | `D:/datasets/chemistry/burcat.thr` | 2.3 MB | `inventory/data/combustion_thermo.json` — NASA-7 polynomials, 11 combustion species incl. n-/iso-octane (Hf298 recomputed from polynomials matches textbook: CO₂ −393.5, H₂O(g) −241.8 kJ/mol) | free for scientific use (cited) |
| CODATA constants | `D:/datasets/chemistry/codata_allascii.txt` | 60 KB | `inventory/data/codata_constants.json` — 12 core constants (verify: c = 299,792,458 m/s) | public domain (US gov) |
| Wikidata electrical properties | (API, polite cached) | — | `inventory/data/electrical_properties.json` — P5679 resistivity + P2068 thermal conductivity, temperature-qualified nearest 20 °C (verify: Cu ρ = 1.7×10⁻⁸ Ω·m, exact textbook match). **Honest coverage gap**: only 7/17 attempted elements have either property curated on Wikidata at all; only copper has resistivity (verified against raw claims, not a script bug) | CC0 |
| NASA TPSX materials database | `D:/datasets/tpsx/` (cached HTML) | — | `inventory/data/tpsx_materials.json` — **700 named, cited materials**, 2,685 property rows (value/units/uncertainty/source/reference, page id cited). Caught + fixed a real title-extraction bug before shipping (everything landed "(unnamed)" while values were already correct). Verified: gold density 19,300 kg/m³ (exact), k=318 W/(m·K) (real ≈317) | US gov work, freely retrievable |
| JPL DE440s ephemeris kernel | `~/.materia/ephemeris/de440s.bsp` | 31.2 MB | (consumed directly by `field/interface/adapters/_jplephem_bridge.py`; verified: EMB at J2000 = 1.469×10⁸ km from SSB) | public domain (NAIF/JPL) |

## Voxel bake cache (local, gitignored, regenerable)

`sigma_ground/deckard/voxel_cache.py` caches finished `VoxelField`s under
`local-cache/voxels/<anno>_<pitch>mm_<VERSION>.npz`. Warmed for all 23 real
PartNet categories' canonical exemplar (53s total) — `identify(fidelity=
"voxel")` on any of bag/bed/bottle/bowl/cabinet/chair/clock/dishwasher/door/
earphone/faucet/hat/knife/lamp/laptop/microwave/monitor/mug/refrigerator/
scissors/table/"trash can"/vase is now instant instead of a multi-second
voxelize. Nothing to commit here — the cache is gitignored (`*.npz`) and
free to regenerate.

## Distill tools (committed, rerunnable)

- `tools/fetch_shapenet_archives.py` — resumable HF-vault fetch (the original 242 GB)
- `tools/distill_partnet.py` — census / geometry / verify / exemplar pools (top-K per category)
- `tools/distill_shapenetsem.py`, `tools/distill_shapenet_taxonomy.py`, `tools/distill_quickdraw.py`
- `tools/fetch_objaverse_lvis.py` — targeted LVIS category pulls + per-object license ledger
- `tools/distill_optics_nk.py` — rii database → RGB n,k cited JSON
- `tools/distill_chemistry.py` — atomic weights + Burcat NASA-7 + CODATA
- `tools/distill_electronics.py` — Wikidata P5679/P2068 (polite, 429-backoff)
- `tools/fetch_tpsx.py` — TPSX id sweep (1 req/s, HTML cached) + property distill
- `tools/extract_shapenetcore_synsets.py` — selective per-synset extraction + census
- `tools/extract_shapenetsem_gap_categories.py` — Hammer/ScrewDriver/Motorcycle from ShapeNetSem

## Cross-dataset findings (unexploited, flagged for the main lane)

- **PartNet ↔ ShapeNetCore/Sem join key exists and is unused.** Every PartNet
  model's `meta.json` carries `model_id` — the literal ShapeNetCore synset
  hash (verified: chair anno_id 44164's model_id `ee2ea12a2a2f8eb71335bcae
  6f5543ce` is `ShapeNetCore.v2/03001627/<same hash>/`). This unlocks, per
  PartNet model: ShapeNetCore's independently-computed `solid.binvox` as a
  free cross-check against our own fill-heuristic voxelization, AND a
  **higher-resolution 256³ solid binvox** (`model_normalized.256.solid.
  binvox`) neither dataset's aggregate stats hinted existed.
- **ShapeNetSem's `up`/`front` orientation is real ground truth, mostly
  unused.** 30% of 12,288 rows carry it; 88% of those are Z-up (matching our
  convention) but 49 distinct up-vectors exist — proof our single global
  Y-up→Z-up rotation rule (`mesh.load_parts(to_z_up=True)`) is measurably
  wrong for some real fraction of models when ground truth is checkable.
- **Taxonomy hierarchy discarded.** `shape_aliases.json` keeps only flat
  WordNet lemma groups from `taxonomy.json`, not its parent-child synset
  tree — no "is-a" category fallback (an unmatched subtype can't fall back
  to its parent category).
- **ins_seg_h5.zip / sem_seg_h5.zip (21.2 + 8.6 GB) — deliberately NOT
  pulled.** These are the official PartNet ML-benchmark point-cloud
  segmentation files, built for training PointNet-style neural segmenters.
  We already have something strictly better for our purposes: the
  ground-truth mesh-level `result.json` hierarchy + real triangle geometry,
  not a sampled point-cloud approximation of it. A conscious skip.

## Honest gaps

- **No articulation data on disk** (PartNet-Mobility pending the Captain's
  account registration at sapien.ucsd.edu — non-commercial ToU, I cannot
  create the account. The one dataset with real URDF joint axes/limits).
- **No internal combustion engine mesh in any dataset** — the actuatable engine
  is an assembly-from-engineering-data frontier; Burcat gives its combustion
  thermochemistry, Objaverse `motor` + ShapeNetSem `Motorcycle` give adjacent shapes.
- **Confirmed absent across ALL THREE ShapeNet-family sources** (PartNet's 24
  categories, ShapeNetCore's synsets, ShapeNetSem's 355 category tokens) —
  not a PartNet-specific gap: wrench, axe, anvil, saw, drill, pliers, gear,
  engine. Objaverse remains the only source for these.
- ShapeNetSem's own per-model weight/solidVolume/isContainerLike columns
  are empty → voxel-mass cross-checks need another source (TPSX or Wikidata,
  both already distilled this session, cover some overlapping materials).
- Objaverse and ShapeNetCore meshes carry **no real-world scale** (both
  verified this session — a sampled Objaverse hammer and a ShapeNetCore car
  are both in an arbitrary model-space unit, not metres). Size grounding for
  either must come from the existing typical-size/Wikidata layer.
- Wikidata electrical resistivity coverage is thin (only copper of 17
  attempted elements) — a denser table needs a non-CC0 source.
- **Deep-research physics audit** (this session, `wf_74792794-e99`) hit an
  infrastructure rate-limit storm during adversarial verification — the
  geometry/rendering half (SDF storage, sparse structures, voxelization) got
  5 confirmed findings; the physics half (MPM/FLIP/LBM, joint representation,
  fracture, adaptive resolution) surfaced real sources (NVIDIA/DeepMind/
  Disney's Newton, Genesis, NVIDIA Warp) but zero adversarially-confirmed
  claims — needs a re-run before being treated as load-bearing.
