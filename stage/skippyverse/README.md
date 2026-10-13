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
- No embeddings, no vector file, no mesh handling (LOD, hashing, normalisation), no bucket publish, no portal routes. (GMod wiring: see the last section.)
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
| `packcache.py` | The packed-payload cache: key `sha256(card, format, triangle target, meshpack.PACK_VERSION)`, atomic checksummed files in `cache/packed/`, source-file fingerprint, 500 MB LRU cap, one build per key under concurrency. The bridge's `/mesh` uses it; bump `PACK_VERSION` when the packer changes. Details: `stage/gmod/holodeck/README.md` "Object caches". |
| `measure.py` | Pure-stdlib bounding boxes: GLB/glTF (node matrix/TRS, vertices read when not axis-aligned), OBJ, STL (binary+ascii), Collada (`<unit>`, `up_axis`, node transforms), SDF (`<scale>`, poses, box/sphere/cylinder; nested models and includes are refused). Everything is reported +Z up; unit-less formats say `unit_known=False`. |
| `data/nasa_real_sizes.json` | Published real sizes for NASA models; empty on purpose (see its `about`). |

Size basis after a download: `measured` when the units are known (glTF metres, Collada `<unit>`, SDF models, Poly Haven); NASA stays `guessed` (flag `scale_arbitrary`) unless a verified table entry makes it `sourced`. A download never replaces a stronger listing size; a measured size more than 5% off a `sourced` listing size is flagged `measured_size_disagrees_with_listing`. Mass is the labelled-`guessed` prior; front stays unknown. The `pose` is a row-major 4x4 (source scene coords -> metres, +Z up, origin on the floor under the footprint centre); `acquire` itself converts nothing; `meshpack.py` (last section) does.

`resolve(..., acquirer=None)`: by default nothing is fetched and a found card with no size gets `plan[0] = {action: 'fetch', card_id, source, executed: False}`. Pass `acquirer=acquire.acquire_and_measure` (partially applied with a cache dir if wanted) and rung 2 runs for a found, servable card with no size or no local copy; the sentence says "I don't have a ... on the shelf yet. Fetching a model from X..." then the honesty clause with the measured size and its basis. A failed fetch is `fetch_failed`, never hidden.

Live proof 2026-10-10: Poly Haven barrel_03 and dining_chair_02 measured from the downloaded glTF match the published dimensions to four decimals (axis handling confirmed on an asymmetric chair).


## GMod wiring (increment 4): the holodeck summons any servable card as a placeholder box

```bash
python stage/skippyverse/ingest_real.py        # offline: cache/ listings -> build/cards.json + build/index (4602 cards, 4102 servable)
python stage/gmod/holodeck/live.py "add a fire hydrant" "add a birthday cake" --start-bridge --pos 64,0     # dedicated server, graded against the cards
python -m pytest sigma_ground/radiance/tests/test_gmod_found.py
```

`stage/gmod/holodeck/skippy_bridge.py` serves `POST /resolve {"noun", "audience": "game"}` (token, deadlines, 12 per minute and 4 per caller, one
download slot): it runs `resolve.resolve(noun, index, room=outdoor, audience="web")` (so only servable cards rank first and the answer is the
web-safe copy), turns a placeable answer into a planner kind with `cards.to_planner_kind` (Hammer units; made safe: no approximate names, no seat, no
surface, `entity: holodeck_found`, the card's id/name/source/licence/basis in `found`), and the game does the rest (`stage/gmod/holodeck/README.md`,
"Found objects"). The index is `build/index` when it holds found cards, else it is built from `cache/` into a temporary directory (about 5 s).
`index.load_index(..., shared=True)` allows the one read-only handle to serve the bridge's request threads.

What a game answer means: `place` is true only for status exact / alias / synonym, a servable `found` card, a size that passed the sanity check
(`index.size_ok`) and the outdoor room check; a card with no size triggers rung 2 only when the match is one the game would place; the sentence
always ends "For now it shows as a box of that size." when it places. Measured live on the dedicated server (2026-10-11): Poly Haven fire hydrant,
boombox, cannon and croissant, and a Gazebo Fuel birthday cake fetched in the run (2.3 MB, measured 0.4207 x 0.4148 x 0.2256 m): the engine's own
box matched the card to 0.000 % on all three axes, the engine mass matched the card's guessed prior (`41.9 x V^0.72`, label "guessed"), all asleep, none
shoved, sunk, toppled or overlapping.

Findings worth knowing. (1) Most nouns people say for these cards are not unknown to the holodeck's parser: "barrel", "chair", "table" and a few
dozen more are library words, and "fire" is an approximate name for a wood stove, so "fire hydrant" parses as a wood stove with "hydrant" left over
(handled: the whole phrase is asked). (2) Size-less Fuel/NASA cards are 3,581 of the 4,102 servable; every one of them costs a download before it can be
placed. (3) A model that is tiny or huge for its category is refused by the game (1 to 2000 HU on every side, so a 6 mm ball bearing and a 70 m
airliner are sentences, not boxes). (4) `build/index` from `index.py build` holds only the 500 local cards; `ingest_real.py` is the command that
builds the real one.

## Mesh payload (SVM1): the real shape of a found object, small enough to send to a game client

`meshpack.py` turns one cached glTF/GLB into a payload a client can draw. It is the visual half of swapping a placeholder box for the real thing;
collision stays the box (`stage/gmod/holodeck/README.md`, "The real shape of a found object").

```bash
python stage/skippyverse/meshpack.py <model.gltf|glb> --target-tris 1500 --codec lzma --out payload.bin    # codec zlib (default) or lzma
```

Wire format of the payload after decompression (little endian): `'SVM1' | u16 n_verts | u16 n_tris | f32 min[3] | f32 max[3] | u16 verts[n][3] |
u16 idx[t][3] | u8 rgb[t][3]` (32-byte header; vertices quantised across min..max; one flat colour per triangle: the material's `baseColorFactor`,
linear to display, grey 0.8 when there is none). Frame: metres, +Z up, x = the card's width, y = its depth, origin on the floor under the footprint
centre (glTF y-up turned to z-up the way `measure.py` does it; the authored front, glTF +Z, comes out as -y). Simplification is vertex clustering on a
grid sized to the target (crude: it keeps size and silhouette, drops detail). Real models (Poly Haven, 1k glTF): dining chair 22,013 -> 1,308
triangles, 7,982 bytes; boombox 10,168 -> 1,320, 6,501 bytes; barrel 1,473 -> 1,473, 5,630 bytes; packing takes 0.1 to 1.5 s in pure Python.

**Codecs.** `zlib` (the default) and `lzma`. Garry's Mod Lua has no zlib; `util.Decompress` is LZMA, and the framing it reads was checked in the
real game (2026-10-11, `stage/gmod/addon/lua/mentat/tools/lzma_probe.lua`): it is the legacy `.lzma` ("alone") stream, `props[5] | u64 uncompressed
size | stream`, which is also exactly what `util.Compress` emits. **The size field must hold the real length**: Python's
`lzma.compress(format=FORMAT_ALONE)` writes -1 ("unknown") and `util.Decompress` then returns nothing, as it does for `.xz`, a bare stream and a
Source-engine `LZMA` header; patching in the real size makes it work (`meshpack.lzma_alone`). liblzma, conversely, rejects that known-size form when
the stream also carries an end marker, so `meshpack.unlzma_alone` writes -1 back before decoding (tests only).

**Not done (SVM1; the SVM3 section below does textures and normals):** textures (every Poly Haven model is textured, so it comes out grey: the colour is only the material's base colour), normals from the
file (computed per triangle on the client), a collision hull, levels of detail, models over 4,000 triangles or 65,535 vertices, NASA and Gazebo Fuel
models (NASA units are a guess and would need the measured scale applied; Fuel is Collada), animation, and any caching of the payload beyond the
bridge's memory.

### SVM3: colours from textures, material cues, normals (look pass 1, 2026-10-12)

`python stage/skippyverse/meshpack.py <model> --fmt svm3 --target-tris 3000 --codec lzma` (the bridge's default; `SKIPPY_MESH_LEGACY=1` keeps SVM1).
`surface.py` holds the pieces (all pure Python; Pillow only decodes textures):

| piece | what it does |
|---|---|
| `Texture`, `tri_colour` | decode, box-reduce to 256 px, bilinear sample with repeat wrap; colour = base-colour factor x texture, in linear light, at the UV centroid of a triangle (four points for a large one) |
| `pbr_cue` | metal / roughness / emissive / alpha -> `(spec, metal, emit, alpha)` bytes (spec = (1 - rough)^1.5; unknown metal counts as 0) |
| `build_palette` | triangles -> <= 24 colour regions: split by material class (spec level, metal level, glows, see-through), each class cut by median cut, entries shared out by area |
| `smooth_normals` | vertex normals with creases split at 40 degrees (a box keeps hard edges, a sphere stays smooth) |

Sources of colour: glTF `baseColorTexture` (external file next to the model, `data:` URI or embedded bufferView; `EXT_texture_webp` sources too) with
`TEXCOORD_n`, the metal-rough map (G, B), `emissiveFactor`, `alphaMode`; Draco primitives use the decoder's UVs when it has them; Collada: material ->
effect -> diffuse colour or texture (sampler -> surface -> image, images found by name or by stem next to the `.dae`), `specular`/`shininess` for the gloss,
and a Gazebo model that says plain white but ships `materials/textures/*Diffuse*` gets that image; SDF primitives and OBJ keep flat colours; an SDF
`<submesh><name>` now selects that node/geometry of the Collada (before, every visual drew the whole file).

Wire format after decompression: `'SVM3' | u16 nv | u16 nt | f32 min[3] | f32 max[3] | u8 npal | npal x (u8 r g b spec metal emit alpha) | u16 verts[nv][3] |
u16 idx[nt][3] | u8 palette index[nt] | i8 normals[nv][3]` (triangles sorted by palette entry; `meshpack.unpack3` is the inverse, `unpack` still returns
(verts, faces, colours) for either magic). Simplification is still vertex clustering, but each cluster's vertex now sits at the quadric-error minimiser,
and a surviving face takes the dominant colour of the triangles around it. On the real cache (3000-triangle target, LZMA): kettle 25 KB,
boombox 25 KB, cannon 29 KB, croissant 18 KB, hydrant 26 KB, mailbox 1 KB, birthday cake 20 KB, Gemini 24 KB, Apollo Lunar Module 26 KB.

**Downloads (`acquire.py`).** The look pass needs images: Poly Haven's 1k base-colour and metal-rough images (the ones the glTF names; normal maps never; each
under 8 MB), the images in a Gazebo Fuel zip (under 8 MB each), nothing extra for NASA (embedded). `ensure_textures` tops up a model downloaded before this
pass, once. A `.bin` whose size differs from its glTF's declared buffer length is replaced from the glTF's own folder (Poly Haven's listing names the 8k `.bin`
for `cannon_01`'s 1k glTF); a mismatch that survives is an error, not a corrupt mesh.

**What is lost** (honest): texture detail below one triangle (labels, grain, logos), reflections, anything the clustering removes (thin parts break up);
cast shadows. What is kept: the main colours, the material split, the smooth/hard edge structure. Tests: `sigma_ground/radiance/tests/test_skippyverse_surface.py`.

### Draco-compressed glb (2026-10-11)

Most NASA glb files store geometry as `KHR_draco_mesh_compression`. `measure.decode_draco_primitive` decodes each Draco primitive with the OPTIONAL
official PyPI package **DracoPy** (`pip install DracoPy`, 2.2.0, Apache-2.0, wheel for CPython 3.13 win_amd64; wraps Google's Draco library). `measure.py` takes
bounding boxes from the decoded positions (node transforms applied); `meshpack.py` packs the decoded triangles and the material base colour like any glTF.
Without DracoPy the modules still import and a Draco model stays a box with the reason "Draco-compressed and the Draco decoder is not installed";
a corrupt Draco buffer is refused with a Draco reason (the box size falls back to the accessor min/max). `gltf_readable` now trial-decodes every Draco buffer,
so the bridge offers a mesh only for a file that really decodes. EXT_meshopt_compression is still unreadable. Live 2026-10-11 (NASA raw files): Apollo Lunar
Module 716,840 B, 97,588 -> 1,445 tris, 8,030 B payload (zlib 10,415), 6.4 x 6.4 x 5.0 native; Mars 2020 Perseverance Rover 4,987,176 B, 199,521 -> 1,476 tris,
8,447 B, 2.58 x 2.95 x 1.79 m packed; Cassini Assembly 2,388,308 B, 453,987 -> 1,469 tris, 8,203 B. Packing is pure Python: 7 s to 33 s for these.

### Resolver: part of a name, and a servable card instead of a blocked one (2026-10-11)

`how = "partial"`: a phrase of two words or more that is a subset of exactly one kind's name and ends in that kind's head noun ("perseverance rover" of
"mars 2020 perseverance rover") is taken as that kind and said so ("I took ... to mean ..."); status `synonym`, so the bridge may fetch and place it.
`how = "shared_instead"`: when the exact/alias match is a card the web/game audience may not be served, the resolver falls through to a servable card that names the
same noun (exact/alias first, then head-noun subset), strictly (never a loose text match), and says "I can't share my kettle, but I have a CC0 one: ...";
several servable kinds give an ambiguity question. Seed-set metrics unchanged: top-1 95.5%, false accepts 1/36.
