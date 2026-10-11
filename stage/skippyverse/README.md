# stage/skippyverse: the object data layer

Cards, an index and a resolver for the SkippyVerse "any object on command" inventory. Design: `misc/SKIPPYVERSE_OBJECTS_DESIGN.md`
(sections 3-6, increment 2). Python 3.13, standard library only, SQLite FTS5. Fully offline: nothing here opens a network connection
(a test replaces `socket` with a function that raises and runs every adapter and the resolver under it).

```bash
python stage/skippyverse/ingest_local.py            # catalog + Deckard + ShapeNetSem names/sizes -> build/cards.json, prints counts
python stage/skippyverse/index.py build             # -> build/index/index.sqlite, then version.json (written last)
python stage/skippyverse/index.py search "couch"
python stage/skippyverse/metrics.py --failures      # resolver on data/queries_seed.json
python stage/skippyverse/sources/cli.py --source fuel --raw listing.json   # parse a listing FILE you supply
python -m pytest sigma_ground/radiance/tests/test_skippyverse_inventory.py
```

## What exists

| File | What it does |
|---|---|
| `cards.py` | Card v1 schema, `validate_card`, the **licence gate** (allowed: CC0, CC-BY-3.0, CC-BY-4.0, PD-USGOV; Valve / ShapeNet / PartNet / Sketchfab / Quaternius sources never servable even if they claim CC0; NC/SA/ND not allowed), basis levels (`measured > sourced > estimated > guessed`, a derived number never stronger than its input), `honesty_clause` (Skippy's sentence from class/style/basis), `to_planner_footprint` / `to_planner_kind` (metres to Hammer units, catalog.json variant shape, ready for a later `addKind`), `strip_for_web`, `prior_mass`. |
| `ingest_local.py` | Existing data to cards: 126 curated Valve-model cards (46 kinds, measured sizes, fidelity masses, `redistributable=false`), 2 second-hand seed facts (car, jumbo jet, flagged unverified), 12 Deckard part-layout cards, ~360 placeholder cards (ShapeNetSem names + median sizes, 9 scalar-size objects). |
| `sources/` | Adapters `polyhaven`, `kenney`, `gazebo_fuel` (CC0 + CC-BY only), `nasa3d`, `smithsonian`: `parse_listing(raw)` and `parse_listing_with_report(raw)`; licence-gate by default (`include_disallowed=True` keeps flagged cards). `cli.py` parses a local file. `data/source_fixtures.json` holds hand-written fixtures. |
| `index.py` | SQLite FTS5 index: names, aliases, approx names, tags, category closure, WordNet synonym groups (`sigma_ground/inventory/data/shape_aliases.json`, 162 groups), a size-sanity check, `version.json` last, refusing to load a missing or mismatched build. `--web` stores `strip_for_web` copies. |
| `resolve.py` | The ladder, rungs 0, 1, 1b and 5, with stubs for 2/3/4. Exact/alias, then approx name, WordNet synonym, category question, typo, and a coverage + margin text match. No cosine cutoff. |
| `metrics.py`, `data/queries_seed.json` | 169 seed queries (133 positive, 36 negative). |

### Resolver behaviour (all tested)
Never a silent substitution: every non-exact result says what it did ("I read 'cuoch' as 'couch'", "The closest I have is a wood
stove", "I took 'settee' to mean couch"). Ambiguity returns a short list, not a pick. A bare category ("furniture") returns options.
Unknown things return `not_found` with no card ("dragon"). Too big for the room is refused with the size and "Try it outdoors"; a card
whose size fails the sanity check (ShapeNetSem "jet" is 0.27 m) is `untrusted_size`; a card without a size is `needs_size`. For
`audience="web"` a non-redistributable card is still reported (size, mass) but `servable=False`, with the game path removed and "it
would be a box".

## Measured (this machine, seed set, 2026-10-10)
133 positives: top-1 95.5%, top-3 95.5%. By tag: natural 71/71, plural 20/20, synonym 20/20, typo 16/16 (these are mostly easy:
exact names, plurals and one-letter typos of names the index holds), **hard paraphrases ("something to sit on") 0/6**: not solved,
and not claimed. 36 negatives: 1 confident wrong object ("chandelier" resolves to the WordNet synonym "pendant"; arguably right),
3 honest "closest I have" ("eiffel tower" to tower, "pogo stick" to stick, "hot tub" to bathtub), 2 ambiguity lists, 30 refused.
Caveat: the seed set was written by the builder in one sitting, and two resolver rules (glued names such as `wine glass` =
`wineglass`, a coverage floor on ambiguity) were added after seeing two of its failures. **Aaron should write the real 200-query set from his own
phrasing**; these numbers are a smoke test, not the design's acceptance test.

## Card counts (local ingest)
500 cards: curated 126, deckard 12, placeholder 362. **0 are redistributable**: everything local is Valve or ShapeNet derived, or a
second-hand fact, and the gate says so. Servable cards appear only when a source adapter (from real raw metadata) yields CC0 / CC-BY items.
Size basis: measured 126, sourced 4, estimated 361, guessed 9. Mass basis: estimated 121 (fidelity layer), sourced 1, guessed 378
(`41.9 x V^0.72`, the design's prior, labelled guessed; unaudited Valve engine masses also guessed).

## Stubbed or unverified
- **Rungs 2, 3, 4** return a `plan` (`executed: false`) describing a fetch, a generation, or a Deckard parts build. Nothing is fetched or generated.
- **Live fetching**: `--allow-network` exists and exits with status 3; no adapter has network code.
- **Every source fixture format is from memory and unverified** (Poly Haven's `dimensions` units and up-axis, Fuel's `license_name` strings and
  whether "Creative Commons - Attribution" is v4.0, NASA and Kenney listings, the Smithsonian response shape, all page URL patterns). Each adapter's docstring says so.
- The two seed facts (car, 747) are copied from the design document, which cites Wikidata; not re-checked. Licence `unverified`, so not servable.
- No embeddings, no vector file, no mesh handling (LOD, hashing, normalisation), no bucket publish, no portal routes, no GMod/planner wiring.
- Size-sanity bands (`taxonomy.py`) are judgement, not sourced or physics. The mass prior is the design's, not a computation of mine; the Mentat stack was down.
- Known imperfections: ShapeNetSem names contain junk (`voider`, `butt`, `down`); `lamb` is read as `lamp`; the keyword category guess
  is crude; a `kind` for ShapeNet groups is the shortest name (`grand` for `grand piano`).

## Needs Aaron
1. **Approve downloads / network** per source before any real listing is fetched; none of these were run.
2. **API keys and accounts**: Smithsonian api.data.gov key; optional Poly Pizza key (design decision 6). None are stored here.
3. **Confirm licence readings**: Gazebo Fuel "Attribution" = CC-BY-4.0? ABO (CC-BY 4.0 per red team) and old Quaternius packs are unhandled.
4. **ShapeNet-derived data in friends' browsers** (design decision 5): currently blocked by the gate (`redistributable=false`).
5. **Repo-original facts**: `object_sizes.json` and the seed facts carry licence "repo-original, unreviewed"/"unverified" and are treated as not servable until you say otherwise.
6. **Write the 200-query labelled set** (and the 30 "must not resolve" nouns) from your own phrasing.


## Rung 2: acquire, measure, resolve (increment 3)

```bash
python stage/skippyverse/live_acquire_check.py        # by hand, NOT pytest: 1 NASA + 1 Fuel + 2 Poly Haven models, a few MB
python -m pytest sigma_ground/radiance/tests/test_skippyverse_acquire.py      # offline: injected fetch, synthetic 2x1x3 meshes
```

| File | What it does |
|---|---|
| `acquire.py` | One of two modules allowed network imports. `acquire(card, fetch, cache_dir, force)` downloads ONE asset: NASA `.glb` from `raw.githubusercontent.com/nasa/NASA-3D-Resources/master/<quoted path>`, Fuel `https://fuel.gazebosim.org/1.0/<owner>/models/<name>.zip` (images are not kept), Poly Haven `api.polyhaven.com/files/<id>` then the 1k glTF + `.bin` only. Licence gate first; 40 MB per file unless `force=True`; cache in `cache/assets/<card id>/` (gitignored), marker written last. `acquire_and_measure` returns `{card, pose, asset, measurement, bytes}`. |
| `measure.py` | Pure-stdlib bounding boxes: GLB/glTF (node matrix/TRS, vertices read when not axis-aligned), OBJ, STL (binary+ascii), Collada (`<unit>`, `up_axis`, node transforms), SDF (`<scale>`, poses, box/sphere/cylinder; nested models and includes are refused). Everything is reported +Z up; unit-less formats say `unit_known=False`. |
| `data/nasa_real_sizes.json` | Published real sizes for NASA models; empty on purpose (see its `about`). |

Size basis after a download: `measured` when the units are known (glTF metres, Collada `<unit>`, SDF models, Poly Haven); NASA stays `guessed` (flag `scale_arbitrary`) unless a verified table entry makes it `sourced`. A download never replaces a stronger listing size; a measured size more than 5% off a `sourced` listing size is flagged `measured_size_disagrees_with_listing`. Mass is the labelled-`guessed` prior; front stays unknown. The `pose` is a row-major 4x4 (source scene coords -> metres, +Z up, origin on the floor under the footprint centre); no mesh is converted or re-exported.

`resolve(..., acquirer=None)`: by default nothing is fetched and a found card with no size gets `plan[0] = {action: 'fetch', card_id, source, executed: False}`. Pass `acquirer=acquire.acquire_and_measure` (partially applied with a cache dir if wanted) and rung 2 runs for a found, servable card with no size or no local copy; the sentence says "I don't have a ... on the shelf yet. Fetching a model from X..." then the honesty clause with the measured size and its basis. A failed fetch is `fetch_failed`, never hidden.

Live proof 2026-10-10: Poly Haven barrel_03 and dining_chair_02 measured from the downloaded glTF match the published dimensions to four decimals (axis handling confirmed on an asymmetric chair).
