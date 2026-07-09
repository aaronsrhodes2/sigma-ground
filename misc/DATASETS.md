# Dataset Inventory — sigma-ground data lane

Every external dataset on disk, its license posture, and what we distilled from
it. Doctrine: **raw data stays local + gitignored; only distilled, cited
aggregates are committed** (`sigma_ground/inventory/data/*`). Updated 2026-07-08.

## Shape oracles

| Dataset | Location | Size | Status | License / ToU |
|---|---|---|---|---|
| PartNet `data_v0` | `D:/Aaron/datasets/shapenet/PartNet/` | 23 GB (extracted), 129 GB archives | **32,537 models, 24 categories**, censused; per-part semantic hierarchy (`result.json`) present per model. Exemplar pools **being redistilled at K=8** (was K=4; in progress) | ShapeNet ToU — non-commercial research; distilled aggregates only in git |
| ShapeNetCore v1+v2 | `D:/Aaron/datasets/shapenet/ShapeNetCore*` | 23 GB processed + 55 GB archives | 55 synset zips **unextracted** (P1 pending: selective extraction for car/guitar/airplane/…) | ShapeNet ToU |
| ShapeNetSem | `D:/Aaron/datasets/shapenet/ShapeNetSem-archive/` | 12 GB | dims + category material ratios distilled (`shapenetsem_sizes.json`, `materials.csv` distills). **Per-model weight columns are EMPTY in this copy** — no mass cross-check possible | ShapeNet ToU |
| Objaverse LVIS (targeted) | `D:/Aaron/datasets/objaverse/` | 4.4 GB | **864/864 models, 13 categories** (hammer 59, mallet 79, frying_pan 31, saucepan 26, teakettle 69, wineglass 101, pitcher 73, screwdriver 40, wrench 53, lightbulb 83, **motor 54**, cup 70, mug 126). NOT in LVIS: axe, skillet, anvil, feather, gear | **Per-object** Sketchfab licenses — see `inventory/data/objaverse_ledger.csv` (821 CC-BY, 15 CC-BY-NC, 9 CC-BY-SA, 14 CC-BY-NC-SA, 5 CC0). **No real-world scale annotation** — size grounding must come from typical-size/Wikidata at integration |
| KiCad packages3D (sparse) | `D:/Aaron/datasets/electronics/kicad-packages3D/` | 76 MB | 8 THT component families (Resistor, Capacitor, Diode, LED, TO/SOT packages, Battery, Relay, Fuse) — real STEP/WRL models for future working-circuit internals | CC-BY-SA 4.0 (with KiCad libraries exception) |
| PartNet-Mobility (SAPIEN) | — | est. 10–20 GB | **NOT on disk** — awaits the Captain's registration at sapien.ucsd.edu (2,346 articulated models, 46 categories, URDF joints — the actuation enabler) | SAPIEN ToU, non-commercial; account required |

## Physics / materials data

| Dataset | Location | Size | Distilled aggregate (committed) | License |
|---|---|---|---|---|
| refractiveindex.info database | `D:/Aaron/datasets/optics/` | 70 MB (4,198 files) | `inventory/data/optics_nk.json` — measured n,k at 650/550/450 nm for 16 materials, paper cited per entry (Johnson & Christy, Rakić, Werner, Inagaki, Aspnes & Studna, Hale & Querry) | open database; papers cited per entry |
| NIST atomic weights + isotopic compositions | `D:/Aaron/datasets/chemistry/nist_atomic_weights.txt` | 1.4 MB | `inventory/data/atomic_weights.json` — 118 elements, 3,352 isotopes (verify: Hg = 200.592) | public domain (US gov) |
| Burcat/ATcT thermochemical database | `D:/Aaron/datasets/chemistry/burcat.thr` | 2.3 MB | `inventory/data/combustion_thermo.json` — NASA-7 polynomials, 11 combustion species incl. n-/iso-octane (Hf298 recomputed from polynomials matches textbook: CO₂ −393.5, H₂O(g) −241.8 kJ/mol) | free for scientific use (cited) |
| CODATA constants | `D:/Aaron/datasets/chemistry/codata_allascii.txt` | 60 KB | `inventory/data/codata_constants.json` — 12 core constants (verify: c = 299,792,458 m/s) | public domain (US gov) |
| Wikidata electrical properties | (API, polite cached) | — | `inventory/data/electrical_properties.json` — P5679 resistivity + P2068 thermal conductivity, temperature-qualified nearest 20 °C (verify: Cu ρ = 1.7×10⁻⁸ Ω·m, exact textbook match). **Honest coverage gap**: only 7/17 attempted elements have either property curated on Wikidata at all; only copper has resistivity (verified against raw claims, not a script bug) | CC0 |
| NASA TPSX materials database | `D:/Aaron/datasets/tpsx/` (cached HTML) | — | `tools/fetch_tpsx.py` sweep **in progress** — polite id crawl (1 req/s, HTML cached so reruns are free); distills to `inventory/data/tpsx_materials.json` (property sheets: value/units/uncertainty/source/reference, page id cited) | US gov work, freely retrievable |
| JPL DE440s ephemeris kernel | `~/.materia/ephemeris/de440s.bsp` | 31.2 MB | (consumed directly by `field/interface/adapters/_jplephem_bridge.py`; verified: EMB at J2000 = 1.469×10⁸ km from SSB) | public domain (NAIF/JPL) |

## Distill tools (committed, rerunnable)

- `tools/fetch_shapenet_archives.py` — resumable HF-vault fetch (the original 242 GB)
- `tools/distill_partnet.py` — census / geometry / verify / exemplar pools (top-K per category)
- `tools/distill_shapenetsem.py`, `tools/distill_shapenet_taxonomy.py`, `tools/distill_quickdraw.py`
- `tools/fetch_objaverse_lvis.py` — targeted LVIS category pulls + per-object license ledger
- `tools/distill_optics_nk.py` — rii database → RGB n,k cited JSON
- `tools/distill_chemistry.py` — atomic weights + Burcat NASA-7 + CODATA
- `tools/distill_electronics.py` — Wikidata P5679/P2068 (polite, 429-backoff)
- `tools/fetch_tpsx.py` — TPSX id sweep (1 req/s, HTML cached) + property distill

## Honest gaps

- **No articulation data on disk** (PartNet-Mobility pending the Captain's account).
- **No internal combustion engine mesh in any dataset** — the actuatable engine
  is an assembly-from-engineering-data frontier; Burcat gives its combustion
  thermochemistry, Objaverse `motor` gives adjacent shapes.
- ShapeNetSem per-model weights empty → voxel-mass cross-checks need another source.
- Objaverse meshes carry **no real-world scale**.
