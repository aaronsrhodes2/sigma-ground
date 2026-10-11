# SkippyVerse: objects on command, in a browser page inside the portal

*Status: design, 2026-10-10. Produced by a research workflow (six tracks, a synthesis, a red-team pass); nothing here is built yet. The red-team findings that change the plan are in the last section. The lazy-repair and derived-mass design (a separate workflow) will be added as a companion.*

# SkippyVerse objects on command, in a browser page inside the portal

Design synthesis, 2026-10-10. Written for Aaron. It reconciles six research reports and rests on my own spot checks (section 1). "Measured" means somebody ran it on this PC. "Inferred" means arithmetic or a reading of documentation. "Guess" means neither.

## 1. The short version

1. **The client is an ordinary portal page, `/world/`.** It uses the same Google sign-in as the other Browser Games and has a card in that list. It draws with three.js. The holodeck brain (`parse.lua`, `plan.lua`, `layout.lua`) runs unchanged inside the browser, in a Web Worker, on Lua compiled to WebAssembly (wasmoon). Saying "add three couches" therefore needs no server and no physics engine. One spike in real Chrome 154 painted three couches in a U, open toward the player, 68 ms after Enter (24-38 ms for later requests). Planning took 2.4 ms.
2. **Every object is a small "card" (about 0.5 KB).** It holds size, front, flat top, mass, collider, where the mesh is, and where it came from. The planner needs only the card, so a translucent box can appear before any mesh exists. Meshes then stream in as separate hash-named files: a 7-10 KB proxy, then 50-80 KB, then 0.4-0.8 MB, then the full version.
3. **Lookup is a ladder:** known to the page, then cached, then found in a free source, then generated, then built from Deckard parts, then an honest refusal. At every rung Skippy says what he did. The words come from fields on the card, never from a model's opinion.
4. **Assets come only from sources we may redistribute:** CC0, CC-BY, and NASA. They are indexed, normalised and optimised on the home PC, then published to a private cloud bucket. The portal serves them. The Half-Life 2 and CS:S models stay in Garry's Mod. Sketchfab, ShapeNet, anything non-commercial, and (for now) the new Quaternius packs stay out.
5. **Build the first piece in about a week:** a car and a jumbo jet (plus about ten other things) found, scaled to their real dimensions, and shown in a browser. They carry credits and an honest "approximate" label. The PC can be off.
6. **Garry's Mod keeps working.** Its Lua files are not edited. The web bundle is generated from them, and the existing 850 offline tests stay the guard.

### What I checked myself in this pass

- **The portal runs on Cloud Run, not Docker on the home PC.**
  - `SkippyTel/.github/workflows/deploy.yml` lines 313-320 deploy `skippy-portal` with `--timeout 900 --max-instances 1 --memory 1Gi`. `portal/app.py:58` sets the host to `makerportal.live`.
  - My search of SkippyTel's `.py`, `.md`, `.yml`, `.html` and `.json` files found no `makernet.live`.
  - The cloud portal cannot call your PC. The PC polls it. This shapes the voice and generation design, and is decision 4 in section 14.
- **The planner copes with objects it has never seen, including very big ones** (my experiment, measured; the repo was not touched; the script is `scratchpad/wf3/synth/bigobj.py`).
  - I added a car (4.5 x 1.8 x 1.45 m) and a jumbo jet (70.66 x 64.44 x 19.4 m, Wikidata dimensions) to the in-memory catalog on the real planner (LuaJIT through lupa).
  - Rebuilding the parser and planner with the two new kinds took 3 ms. "Add a car" planned in 0.5 ms and "add three cars" in 0.4 ms. In the open, "add a jumbo jet" planned in 16 ms and put the jet 49 m ahead of the player.
  - In a 10.7 x 9.1 m room, or a 114 m square room with the player at the centre, the jet got "I could not find clear floor for that" after 132-167 ms. That is a correct refusal.
  - "A jumbo jet and three cars" in the 114 m room burned the whole 10 s test budget and returned "more than I can fit here" with nothing placed.
  - So: big things need an outdoor world, and the browser planner needs a time budget (0.5-1 s) in the worker.
- **ABO licence (the Amazon furniture scans):** I fetched the dataset's own page. It says CC BY 4.0, with no non-commercial clause. Another report said CC BY-NC. Unresolved, so ABO is excluded for now. We do not need it.
- **Quaternius:** I fetched the licence page, dated 2026-08-28. It forbids redistributing the assets as standalone assets or packs. It is silent on whether older packs are still CC0. Skipped until clarified.
- **Hosted TRELLIS.2 prices and trellis.cpp:**
  - fal.ai shows $0.25 / $0.30 / $0.35 per object at 512 / 1024 / 1536.
  - trellis.cpp v0.8.1 (25 Sep) does offer Windows CUDA zips of 682 MB and 612 MB. Its notes say a complete image-to-quad run on Windows is untested. A licence is not stated on the releases page.

## 2. Where the reports disagreed, and what I decided

| Question | What the reports said | Decision |
|---|---|---|
| Where does the portal run? | The brief says Docker on the home PC. Three reports and my check say Cloud Run at makerportal.live. | Design for Cloud Run. Anything needing the PC goes through a pass-guarded gate or the job queue, and the page must work with the PC off. |
| GPU | The brief says 12 GB 4070 Ti. Three reports ran nvidia-smi and got a 16 GB 4070 Ti SUPER, with 11.5 GB already in use at the time. | Plan for 16 GB, with an exclusive-GPU lock for generation. Generation needs the LLM and Whisper to let go of VRAM. |
| Does the brain give the same answer everywhere? | One report: 8 sentences identical on 6 Lua VMs. Another: the parser is identical on 962 of 962 sentences, but 122 of 992 planner requests differ between wasm and native Lua. Both results pass the independent grader. | Trust the larger test. The cause is last-bit sin/cos differences breaking symmetric ties. **One planner decides each placement and everyone else receives poses.** Two engines never both plan the same room. |
| Who is the authority in shared rooms? | The server re-plans with lupa (web-client), or the speaker's client plans and broadcasts (brain-port). | The speaker's client plans. The server only validates (inside the room, caps, cooldowns) and relays. No Lua on the server. Re-planning there would contradict the ghost the speaker just saw about 12% of the time. |
| Render the scene with the Radiance SDF viewer? | It compiles a shader per scene: 4.1 s at 11 CSG leaves, 59 s at 44, a timeout at 64 (the repo's own sweep). Another report proposed it as the zero-mesh path for Deckard. | No. Deckard parts are drawn as ordinary primitive meshes. Radiance stays as an "inspect: how Skippy built this" view for one object. |
| One free Funnel port | Both the voice gate and an asset gate want port 10000. 443 and 8443 are the music and video gates. | One new gate, one port, separate pass labels per route family. The asset hot path is deferred. |
| Public bucket or signed URLs? | Public-read with hashes, or signed URLs (signing from Cloud Run unverified). | In the first two increments, a private bucket behind a portal route for files up to 8 MB. Same origin, no CORS, no signing, no free CDN on your bill. Revisit when traffic or file sizes justify it. |
| ShapeNetSem sizes for instant ghosts | Two reports use them. Another notes the terms are non-commercial research only. | Not shipped to friends' browsers until you decide (decision 5). Sizes come from curated and sourced tables first. |
| Collision shapes | One Rapier convex hull is wrong for chairs and tables (hull is 3-14x the real volume). Rapier's own decomposition over-splits (15 hulls for 4 boxes). | A collider spec on the card. Default is an oriented box. Compound pieces come from Deckard parts or an offline decomposition on the PC. Anything over 8 m or 2,000 kg is static. |
| Mesh codec | Draco is smaller; meshopt decodes 25-39x faster (2020 desktop benchmark). | meshopt by default. Draco only for a large, static, full-detail version fetched in the background. |
| Viture glasses | Native SDKs only; no WebXR or browser path found. | A `?view=glasses` page: black background, large type, idle at 0 fps. It is a second screen, not a world-locked scene. |

## 3. Architecture

```
 BROWSER  https://makerportal.live/world/   (same origin, Google login, card in Browser Games)
 +-----------------------------------------------------------------------------------------+
 | UI: text box, hold-to-talk, beam/pointer, About + credits, honesty badges                 |
 | Brain worker: wasmoon (Lua 5.4 wasm) + parse/plan/layout, unmodified     1-5 ms/plan     |
 |   world interface (floor_z, clear, wall_probe, ceiling_z) answered from data / Rapier    |
 | Renderer: three.js, black base, render only when something moves                         |
 |   ghost box -> proxy -> LODs;  Deckard parts as primitive meshes;  same kind = instanced |
 | Physics (loaded after first paint): Rapier wasm, bodies created asleep                   |
 | Cache: Cache Storage keyed by sha256 (meshes), IndexedDB (cards), LRU                    |
 +--------+---------------------------------+-----------------------------+-----------------+
          | /world/api/*  (JSON)            | /world/a/<sha256> (bytes)   | voice, hot path
          v                                 v                             | (HMAC pass)
   PORTAL (Cloud Run, 1 instance)     private GCS bucket                  v
   resolve / search / cards /         hashed, immutable files,     skippy-gate on the PC
   asset / generate / jobs            published assets only        (Tailscale Funnel :10000)
   reads the index replica ---------> index.sqlite + version.json  Whisper, ElevenLabs,
   (version.json written last)        (same pattern as doomloop.py)   Ollama, generation worker
          ^
          | publish (PC is the producer)            LATER (increment 8)
   HOME PC: ingest -> normalise -> optimise ->      skippy-world, Cloud Run WebSocket:
   hash -> card -> index build -> upload;           rooms, registry, caps, undo, ticket from
   generation worker; job queue via the existing    the portal. No physics, no Lua.
   home reporter
```

**Where things run**
- **Brain:** in the browser, always. In shared rooms the speaker's client plans.
- **Search and cards:** the portal reads a replica the PC publishes to the bucket. It works with the PC off.
- **Bytes:** never through anything slow. The portal route streams small files in the first increments. Hot files from the PC come later, through the gate.
- **PC-only work:** index building, mirroring new assets, generation, voice, and the embedding fallback.
- **Garry's Mod:** a second client that loads the same four Lua files and keeps its Valve models locally.

**Page budgets**
- Before any asset: about 0.46 MB gzipped (computed from measured sizes: three.js 194 KB, wasmoon 150 KB, brain Lua 96 KB, footprint catalog 13 KB, plus extras).
- Rapier's 1.17 MB gzipped loads after first paint.
- Targets, not yet measured on phones:

| Device | Draw calls | Triangles | Physics per step |
|---|---|---|---|
| Laptop | at most 300 | 1.5 M | at most 2 ms |
| Phone | at most 150 | 500 k | at most 4 ms |
| XR headset | at most 100 | 300 k | n/a |

- Every client renders on demand, so an idle scene costs nothing.

**Portal changes**
- Ship the page files in the image at a pinned sigma-ground ref, as Radiance already is.
- Add a CSP for `/world` only, report-only first, with `wasm-unsafe-eval`.
- Add an optional `/world/embed` variant for use in an iframe.

## 4. The object card (v1)

One JSON record per thing. Units are metres and kilograms; +Z up; the front faces +X; the origin is on the floor under the footprint centre. The catalog generator converts to the planner's Hammer units (1 HU = 1.905 cm; three.js position = (x, z, -y) x 0.01905).

```json
{
  "card_v": 1,
  "id": "boeing-747.pp-49CLof4tP2V",
  "name": "Boeing 747",
  "kind": "jumbo jet",
  "aliases": ["jumbo jet", "747", "jetliner", "airliner"],
  "category": "vehicle/aircraft",
  "size_m": { "w": 64.44, "d": 70.66, "h": 19.40,
              "basis": "sourced", "source": "Wikidata Q906937" },
  "front": { "yaw_deg": 0, "source": "curated-by-eye" },
  "symmetric": false,
  "top": null,
  "mass": { "kg": 183523, "low": 160000, "high": 220000,
            "basis": "sourced", "source": "747-400 operating empty weight (Wikipedia table)" },
  "mobility": "static",
  "collision": { "mode": "box" },
  "visual": { "class": "found", "style": "low-poly", "blocky": true,
              "scale_applied": 0.0, "lods": [ { "l": 3, "tris": 1000, "bytes": 8000, "hash": "sha256:..." } ] },
  "provenance": { "source": "Poly Pizza", "url": "https://poly.pizza/m/49CLof4tP2V",
                  "author": "Miha Lunar", "license": "CC-BY-3.0", "fetched": "2026-10-xx" },
  "say": "approximate shape; size from published dimensions"
}
```

(`scale_applied` and the LOD numbers above are placeholders to be filled in at ingest.)

**Rules**
- **Basis levels.** Every number is labelled `measured` (true of the file we hold), `sourced` (a cited record), `estimated` (derived by a stated rule), or `guessed` (class default or unchecked LLM figure). A derived number is never stronger than its weakest input. A model's own "confidence: high" is ignored; it said "high" for a 45 g mug and a 2.5 m piano.
- **Provenance class.** `curated`, `found`, `generated`, `deckard`, or `placeholder`. Skippy's honesty clause is generated from `class`, `style`, `blocky` and the `basis` of size and mass.
- **Planner fields.** The planner reads only footprint, front, symmetric, top, seat, and wall preference. The footprint is about 230 B per object. `model` is passed through, so an asset id fits there.
- **Unknown front.** For found meshes the front is the weak point. The 127 curated fronts were checked by eye against photographs. `front.source` is `curated-by-eye`, `heuristic`, or `unknown`. `unknown` is treated as symmetric, and Skippy says so for directional things such as sofas.
- **Mass.**
  - Never from mesh volume. A solid jet mesh at aluminium density comes out 42.9x too heavy.
  - Order: a sourced total for a named thing, then the existing fidelity family table scaled by size, then a prior of 41.9 x V^0.72 labelled `guessed`.
  - Physical computations go through Mentat (Platinum 21).
- **Valve models.** They carry `redistributable: false` and are never sent to the web client.

## 5. The lookup ladder and what Skippy says

| Rung | When | Time | What Skippy says |
|---|---|---|---|
| 0. Known to the page | Starter kinds (46) and the shelf cards are in memory | Planner 1-5 ms (measured); first paint about 70 ms (measured) | The planner's own sentence, e.g. "Three couches, in a U, open toward you." If the card is not `curated`, add one clause: "The car is a low-poly model, CC0." |
| 1. Cached | In the index replica or already mirrored in the bucket or browser cache | One round trip: about 75 ms warm, about 200 ms on a fresh connection (measured to the live portal), plus about 5 ms search; mesh LODs add tens of ms at 10-50 Mbit/s (arithmetic) | "Putting a hydrant there." If the mesh fails: "I couldn't load the model, so I'm leaving the box." |
| 1b. Mid-band match | A close but not exact hit | same | "I don't have a floor lamp by that name. The closest I have is a standing light. Using that." |
| 2. Found | Listed in a free source (Poly Haven, Fuel, Kenney, NASA, Objaverse CC0/CC-BY) but not yet mirrored; the PC fetches, normalises and publishes it | Unmeasured; my guess is 10-60 s the first time, then rung 1 | "I don't have that on the shelf yet. Fetching a model from Poly Haven. The box is where it will go." PC off: "I can only use what's already on the shelf right now." |
| 3. Generated | No source has it. TRELLIS.2 on the PC, from a concept picture | Rough about a minute, detailed a few minutes (inferred from other GPUs; unmeasured here) | "I don't have a helicopter, so I'll make one. The box is where it will go. A rough version takes about a minute. Its size is my estimate: about 13 metres." Then: "Rough version is in. It's an AI guess from one picture, so thin parts like rotors may be wrong." Then: "The detailed version is ready; swapping it in." On a failed quality check: "That attempt came out lopsided. Trying again." |
| 4. Deckard parts | Generation failed or is unavailable, and the class has a template (vehicles first) | Seconds once cached | "I built a simple one from parts: body, cabin, four wheels. It's blocky, but the size and weight are right." |
| 5. Honest refusal | Nothing works, a quota is used up, the prompt is unsafe, or it won't fit | Instant | "I don't have a dragon, and I can't make new things right now. The closest I have is a gargoyle." / "That's bigger than this room (70 m). Try it outdoors." / "You've used your six new objects this hour." |

**Rules**
- **Ghost, then swap in place.** The ghost is a translucent labelled box sized from the card. When a mesh arrives it replaces the ghost at the same pose and is scaled to the card's footprint. **Nothing is re-planned.**
- **Never silently shrink.** If it doesn't fit, say so.
- **Do not use a cosine cutoff to decide "we have it".** On the test, in-domain top cosine ran 0.54-0.70 and out-of-domain 0.59-0.65. They overlap, so use alias and category evidence plus a margin.
- **Search order.** Exact or alias match first (FTS5 plus WordNet synonym groups from `shape_aliases.json`). Embeddings (nomic-embed-text through the PC) only when that is weak.
  - In the 34-query sanity check, top-3 accuracy was 23/34 for FTS5 alone, 30/34 for embeddings alone, and 29/34 for a blend. These are small numbers on my own labels.
  - Build a 200-query set from your own phrasing before trusting any threshold.
- **Dynamic kinds.** The parser reports unknown nouns. The client asks the resolver, calls `addKind`, re-parses (about 3 ms measured), then plans.

## 6. The asset service

**Ingest (home PC, Python and Node, offline).**
- **Sources, with scale handling:**

| Source | What | Notes |
|---|---|---|
| Poly Haven | 521 CC0 models | Real dimensions in mm for every model. Take the 1k variant. |
| Kenney | 50 packs, 4,402 GLBs, CC0 per each pack's licence file | Stylised, so scale must be normalised. |
| Gazebo Fuel | About 3,340 models, CC0 plus CC-BY 4.0; filter out the 47 NC/SA ones | Real scale, but Collada/OBJ that needs converting. |
| NASA 3D resources | 257 GLBs; Saturn V, Shuttle, Atlas | Credit NASA, imply no endorsement, leave out insignia. |
| Objaverse 1.0 | LVIS 46,207 objects (long tail, later) | Only per-object licence cc0 or by. Uploader-asserted labels, so spot-check. |

- **Normalise.** Origin at the footprint centre on the floor, front +X, +Y up, metres. Scale is arbitrary in most sources. A Kenney sedan is 1.5 x 1.45 x 2.55 units, and a NASA Saturn V is 2 x 13 x 2. The sourced real size pins the scale. Front direction is set by eye from a contact sheet in the first increments.
- **Optimise.**
  - gltfpack or glTF-Transform (not installed here yet) with quantisation, vertex-cache reordering and meshopt, LODs by simplification, textures to KTX2.
  - Vertex order matters: unordered meshopt streams are about 2x bigger.
  - Textures dominate furniture. A real sofa was 3.15 MB for 4,196 triangles, 3.0 MB of it PNG. A 2K map is 22.4 MB in GPU memory as RGBA8 and 5.6 MB as BC7/ASTC, so KTX2 is mandatory.
- **Scans.** A 100 MB scan is baked to a LOD chain; the original is never served. Measured on a 1.3 M-triangle test mesh, Draco LOD sizes were 3.9 MB / 435 KB / 50 KB / 7 KB. The synthetic mesh flatters this, since real artist assets cost 2-3x more per triangle.
- **Licence gate.** A card without a whitelisted licence, author and URL fails the build. A dated copy of the licence text is stored with each pack.
- **Hash and publish.** Files are named by SHA-256.

**Index.** `assets.sqlite` holds cards, FTS5 (names, aliases, tags) and a category closure table. A flat 768-d f16 vector file sits beside it. At 100k assets that is about 225-275 MB. A query measured 4 ms (FTS) plus 10 ms (brute-force cosine) plus 25-130 ms to embed. No vector database is needed below about a million vectors. `index.version.json` is written last, as `doomloop.py` does.

**Store and delivery.**
- A private GCS bucket holds `blobs/<sha256>`. The portal route `/world/a/<hash>` streams objects up to 8 MB with `Cache-Control: immutable`.
- Cloud Run caps non-streamed responses at 32 MiB, and the portal has 1 GiB with one instance. It must not proxy big files, so a signed-URL or R2 step comes when files outgrow this.
- Shared hash-named files dedupe across assets and friends. Generation requests are de-duplicated by prompt.

**API** (all behind the portal login and a new `world` grant):

| Route | Purpose |
|---|---|
| `GET /world/api/start` | Version, limits, starter cards hash |
| `POST /resolve` | Nouns to cards (up to 7 variants spanning the size range) |
| `GET /search`, `/cards`, `/asset/{id}` | Search, card batches, manifests |
| `POST /generate`, `GET /jobs/{id}` | Generation (increment 6) |

Rate limits reuse the portal's limiter: 120 resolves per minute per person, 6 new objects per hour per friend with one running job each, plus a global monthly spend cap.

**Client cache.**
- Cache Storage is keyed by hash, not by the pass-bearing URL.
- IndexedDB holds cards.
- Budget: 512 MB desktop, 160 MB phone. These are design values, not tested limits. Browsers can evict everything, so the cache is never the source of truth.
- Keep cards and proxies forever. Unload full-detail meshes and high texture mips by screen coverage and recency.

**When the PC is off.** Search, cards and everything already published still work. Only these stop: mirroring new assets, generation, voice, and the embedding fallback (search drops to names and aliases). Skippy says which.

## 7. The client

- **Rendering.** three.js (r186, pinned and vendored, as hls.js already is). Found assets arrive as glTF (meshopt, KTX2). Deckard parts draw as boxes, cylinders and spheres, with a lathe for vessels. Same-kind objects are instanced. The spike's naive per-part stools cost 63 draw calls for ten, so assets must be merged per material or instanced.
- **Physics (increment 3).**
  - Rapier 0.21 plain wasm, loaded after first paint.
  - Bodies are created asleep and wake on grab or shove. They go back to sleep.
  - Measured in Chrome: 39 compound props (couches, chairs, crates, stools) took 0.15 ms per step, were all asleep after 2.85 s, and a crate pile moved 5 mm.
  - The first step costs 82 ms, so do a warm-up step during load.
  - The planner's world questions can be answered by Rapier ray queries (7 us each).
- **Input.**
  - Desktop: a beam drawn from the lower right to the ray hit (the "finger instead of a held gun"). Click-drag grabs with a spring.
  - Phone: tap to point, long-press to grab.
  - Later, WebXR hands: the index-fingertip joint is the ray and a pinch is the grab, using three.js's `OculusHandPointerModel`. Controllers use trigger and squeeze.
  - Vehicles and anything over 8 m or 2,000 kg are static: "too heavy to lift."
- **Worlds.** An indoor room for furniture and an outdoor apron (flat ground, about 400 m) for vehicles. The planner got "no clear floor" for the jet in any room I tried.
- **Voice (increment 4).**
  - Hold-to-talk, MediaRecorder with Opus, uploaded to the gate. The transcript then takes the same path as typed text.
  - Opus at about 3-4 KB/s fits the 128 KB SkippyTel limit for 30+ s. WAV would allow 4 s.
  - Skippy's spoken reply is the planner's sentence plus the honesty clause.
  - Typed text always works.
- **Phone AR and glasses (increment 9).**
  - Android Chrome AR can feed hit-test and plane detection to the planner's floor and wall questions.
  - `?view=glasses`: black clear colour (black is transparent on the additive display), no dark materials, saturated palette, 0 fps idle.
  - iPhone has no WebXR.
- **Hidden tabs.** Workers get throttled. One test had the brain take over 100 s to start in a hidden pane. Handle `visibilitychange`, and don't make logic depend on `requestAnimationFrame`.

## 8. Where the holodeck brain runs, and Garry's Mod

- **Same four files, two hosts.** `parse.lua`, `plan.lua`, `layout.lua` and `library.lua` are the single source. GMod loads them with `include()`. The web build copies them and generates a footprint-only catalog from `data/catalog.json` (asset ids instead of `.mdl` paths, 13 KB gzipped).
- **The browser needs a few lines of host prelude**, not source changes:
  - `plan.lua:3092` calls `math.atan2`, which doesn't exist in Lua 5.3+. Alias it to `math.atan` in the prelude.
  - Provide `include()`.
  - Strip control characters from input (a NUL truncates strings).
  - Set the locale to C in test harnesses.
- **wasmoon gotchas (verified).**
  - After about 45-60 awaited `doString` calls the Lua state corrupts. Use `doStringSync` and call through a function handle (3,000 calls were fine).
  - The 1.16.0 build is UMD, so a named ES import makes a worker fail silently.
  - Return `undefined`, not `null`, from JS callbacks.
  - If wasmoon stalls, the hedges are fengari (30-100x slower) or our own Lua-to-wasm build. A JavaScript port is a last resort. A 499-line pilot passed 6,600 differential cases, but that was the easiest 7% of the code.
- **World interface.** The client passes the world as data where possible and through callbacks otherwise: `floor_z`, `clear`, `wall_probe`, optional `ceiling_z`. Callbacks cost about 5-17 us per call.
- **Budget.** Set 0.5-1 s in the worker. The game's 0.05 s and 0.25 s are too tight for wasm. Pre-existing cliffs: "a dozen crates" takes about 450 ms and places nothing at the game budgets. "Ten desks and ten chairs" takes 527 ms and places nothing in a 10.7 x 9.1 m room. Show partial results when `timed_out`.
- **Guard for GMod.**
  - A CI job runs the 850 existing tests on LuaJIT 2.1.
  - A Node job runs the same files under wasmoon: the parser must match LuaJIT exactly on the 962-sentence corpus, and the planner must have 0 grader problems and match a stored wasmoon snapshot. It is not compared bit for bit with LuaJIT.
  - If a brain edit is ever needed, make it in the addon with a fallback that behaves identically, and run both jobs.

## 9. What Deckard becomes

Deckard becomes an **offline card compiler**. It is never on the interactive path (the language-model step takes 18-86 s). It answers: what does this thing weigh, where is its centre of mass, what are its convex pieces, and what is a blocky stand-in?

- **Repair first** (found by the physical-spec track, each reproduced; the repo is untouched):
  - ShapeNetSem dimensions are `[x, y-up, z]`, but `research.py` sorts them. Couches come out 2.13 m tall and beds 2.32 m. The in-memory fix kept 412 Deckard-related tests green and raised the audit kinds with height within 0.67-1.5x of real from 18 to 28 of 33.
  - `catalog/a_car.md` is a frozen language-model answer flagged as identified. It is the whole "car is 2.5 kg" story. Quarantine it and stop freezing model output into the repo.
  - Add a per-class size sanity gate (ShapeNetSem "jet" is 0.27 m).
  - Strip articles and follow redirects in the Wikipedia lookup.
  - Set `think:false` (qwen3:8b takes 85.7 s thinking, past the 30 s timeout).
  - Remove the generic-vessel fallback for named things (return "unidentified" plus a stub box).
  - Vectorise the integrator (3.4-3.8x faster, identical numbers).
  - The README says the axis change moves pinned scenes, so you review those diffs.
- **Mass.** For ordinary things use the sourced fidelity family table (121 of 127 models). Deckard's bottom-up number is a cross-check only. It runs a median 1.75x high (chair 50 kg against a sourced 4.8). A one-line power-law prior beats it (rms x2.2 against x3.1).
- **Vehicles and aircraft (stage 2).**
  - A facts card from a curated spec table, then Wikidata, then a schema-constrained local model with a cross-check.
  - Then an archetype grammar: a road vehicle (body, cabin, engine, 4 wheels) or an aircraft (fuselage, 2 wings, tail, fin, engines).
  - Then a **mass anchor**: compile, then scale densities to the sourced total. Without it a 747-class spec drifted 14% across grid resolutions.
  - Prototypes with today's primitives landed on 1,565 kg for a mid-size sedan (sourced band 1,100-2,200 kg; your 1,300 kg is a compact-car figure inside it) and 183,523 kg for the 747 (empty weight), in 0.6-2.2 s.
- **Deckard should not** produce meshes or textures, be the authority on a named product's size or mass, guess when it doesn't know, resolve ambiguous names, or supply friction and restitution. Mentat's friction tool returns mu = 1 for wood on wood.
- **Surfaces.** Use a small curated class table (about 15 classes) labelled `guessed` where no source row exists.

## 10. Licensing rules for the shared portal

This is my reading of published texts, not legal advice.

| Class | Rule |
|---|---|
| **Allowed** | CC0. CC-BY 3.0/4.0 (credit panel with author, licence and link). NASA (credit NASA, imply no endorsement, no insignia). Apache/MIT code and weights. |
| **Garry's Mod only** | Half-Life 2 and CS:S models. The Steam Subscriber Agreement (2.G) bars hosting or distributing the Content to people Valve hasn't licensed. Footprints and masses are facts, but the **meshes never go to the portal**. The web catalog drops `.mdl` paths. |
| **Not used** | Sketchfab (the Download API requires each end user to log in; it can be discontinued without notice; ownership just changed hands). ShapeNet / ShapeNetSem / PartNet as served assets. Anything non-commercial, share-alike or no-derivatives. Thingiverse, Printables, Fab. The Quaternius QAL (2026-08-28) packs. RMBG-2.0 (CC BY-NC) as a cut-out step. |
| **Hold** | ABO (licence conflict). Old Quaternius packs (clarify first). Objaverse beyond the cc0/by filter. Smithsonian (needs an api.data.gov key). |
| **Generated meshes** | TRELLIS.2 code and weights are MIT, with no territory limit. Hunyuan3D is not usable for friends (EU/UK/South Korea excluded, user cap). The DINOv3 conditioning-weights licence was not read. Copyright of AI outputs and of the training sets (Objaverse family) was not researched. |

## 11. Costs

Prices were read from the pages named below; the Google Cloud ones are not confirmed against Google's own table.

| Item | Cost |
|---|---|
| three.js, Rapier, wasmoon, meshoptimizer, SQLite | $0 (MIT/Apache) |
| Cloud Storage (GCS, assumed $0.022/GB-month; official table did not load) | 1,000 assets at about 1 MB = 1 GB = about $0.02/month. 100,000 at 2 MB = 200 GB = about $4.40/month |
| GCS egress, $0.12/GB (assumed) | 10 friends x 30 MB x 10 sessions = 3 GB = about $0.36/month |
| Cloudflare R2 (alternative) | $0.015/GB-month, free egress, needs a new account |
| World service on Cloud Run (increment 8) | Probably inside the free tier. Not verified. |
| WebRTC TURN | Cloudflare: free below 1,000 GB/month, then $0.05/GB. Opus at 32 kbps is about 14 MB per stream-hour. LiveKit's free tier covers 5,000 minutes if rooms grow. |
| Hosted generation | fal.ai TRELLIS.2: $0.25 / $0.30 / $0.35 per object (verified today). About $30 per 100 objects at 1024. Tripo $0.10-0.30 (reported; I did not verify). Rodin needs a $120/month plan. |
| Local generation | Free apart from power. About 7-11 GB of weights, plus a 682 MB program. |
| ElevenLabs | Your existing credit. Per-friend daily character cap, like the bridge's allowance. |

## 12. Build order

Effort is working days of one focused developer (me, with you reviewing). It is inferred except where a spike already exists. Each increment is useful alone and leaves GMod untouched. The order is single-agent and sequential.

1. **Browser client and the vehicle shelf (about 6 days).** The first increment, and the only one here that is mostly proven by a spike.
2. **Index, resolver and ingest (about 9 days).**
3. **Physics and the beam (about 5 days).**
4. **Voice through the gate (about 4 days).**
5. **Deckard repair and sourced masses (about 3 days).**
6. **Generation on the PC (a 0.5-day bench, then about 10 days).**
7. **Deckard stand-ins for vehicles and aircraft (about 10 days).**
8. **Shared rooms and proximity voice (about 8 days).**
9. **WebXR hands, phone AR and glasses mode (about 6 days plus devices).**

The full acceptance tests are in the structured increments list.

## 13. Verified versus guessed

**Verified (measured, or read from code or a fetched page):**
- The spike numbers, the brain being identical across runtimes for the parser, and Rapier timings (Chrome 154 on this PC).
- Cloud Run deployment and the makerportal.live host.
- The big-object planner results in section 1.
- Source counts and licences for Poly Haven, Kenney, Fuel, NASA and Objaverse. Latencies from this PC to those sources.
- The ABO and Quaternius page readings.
- fal.ai prices.
- Codec sizes on the synthetic and Khronos sample meshes.
- Planner cost for typical sentences (0.5-5 ms).

**Inferred or guessed:**
- All phone, headset, Safari and Firefox performance. No device was available.
- Every generation time on this GPU, and the quality gate's thresholds (tuned on synthetic jets only).
- The real in-browser decode times of meshopt, Draco and KTX2.
- Funnel throughput, and whether WebSockets work through Funnel.
- Home upload speed.
- Cloud Run and GCS prices.
- Whether Whisper accepts browser-recorded webm/opus (the code suggests yes).
- Automatic front detection for found meshes.
- The days estimates.

## 14. Decisions for Aaron

These are listed in the structured field below.



---

## Increments (in build order)

### 1. Browser client and the vehicle shelf
- Ships: A /world/ page in the portal (login required, a card in Browser Games, no voice, no physics, no multiplayer). It runs the unmodified holodeck Lua in a Web Worker on wasmoon, drawn with three.js on a black base. The 46 starter kinds appear as labelled placeholder boxes. A hand-picked shelf of about 12 redistributable assets appears as real meshes, loaded into the parser at start with addKind, with real sizes and credits. Candidates for you to pick from a contact sheet: car (Kenney Car Kit sedan, CC0; Gazebo Fuel Hatchback, CC0, which needs Collada to GLB); jumbo jet (Poly Pizza Boeing 747 by Miha Lunar, CC-BY 3.0, described as a bit wonky; the OpenGameArt CC0 Jumbo Jet as a fallback); rocket (NASA Saturn V); couch (Kenney loungeSofa, Poly Haven Sofa_01 1k); bookshelf; floor lamp; fire hydrant (Fuel, 0.87 MB); bicycle (Fuel, CC-BY 4.0); palm tree (Kenney). Also: an indoor room and an outdoor apron world, a text box, a credits panel, honesty labels, and a build script that copies the four Lua files and emits the catalog. The shelf lives in a private bucket and the portal streams it. Includes the portal route, the Dockerfile copy line, a SIGMA_GROUND_REF bump, a report-only CSP and tests.
- Acceptance test: (1) With the PC off, on desktop Chrome and on your S23: sign in, page loads in at most 0.5 MB gzipped before any asset; brain ready within 1 s on a laptop (target within 3 s on the S23, unmeasured). (2) Type 'Skippy, add a car': a translucent labelled box appears within 100 ms of Enter; the textured car replaces it within 2 s on a warm connection; its long side measures 4.5 m plus or minus 0.2 m; it faces the way the planner chose; the label names the source, licence and 'low-poly'. (3) Type 'add a jumbo jet' outdoors: it lands at least 40 m ahead (my test: 49 m, 16 ms), long side 70.66 m plus or minus 1 m, wingspan 64.44 m plus or minus 1 m, label says approximate shape and size from published dimensions, credits panel shows Miha Lunar CC-BY 3.0 with a link. Indoors, the same sentence places nothing, Skippy says it is bigger than the room and suggests outdoors, and there is no hang beyond the 0.8 s worker budget. (4) 'Add three couches' makes a U open toward the player. The parser output is byte-identical to LuaJIT on the 962-sentence corpus; the planner output has 0 problems from the independent grader on the 8 golden sentences and equals a stored wasmoon snapshot. (5) GMod regression: the 850 existing holodeck tests pass on LuaJIT 2.1 and git shows no diff under stage/gmod/addon. (6) Licence gate: the build fails if any card lacks a licence in {CC0, CC-BY-3.0, CC-BY-4.0, public-domain US government}, an author, or a URL; a scan finds no Valve, ShapeNet or non-commercial content in the bundle. (7) A headless-Chrome smoke test reports objects drawn and no console errors; no request is made after a repeat sentence once cached. (8) Card sizes agree within 5% with the normalised mesh bounding boxes.
- Effort: About 6 working days: page, worker and loader from the existing spike 2; shelf ingest script and hand curation 2; portal route, Dockerfile, report-only CSP and tests 1.5; contact-sheet review with you 0.5.
- Depends on: Nothing. The spike in scratchpad wf3/web-client/spike is the starting point.

### 2. Index, resolver and ingest pipeline
- Ships: A PC-side ingest tool that reads Poly Haven (521), Kenney (4,402 GLBs), Gazebo Fuel (CC0 plus CC-BY 4.0 only) and the NASA tree into cards. It normalises scale and orientation to sourced sizes, optimises with meshopt and KTX2 into LODs, hashes the files and writes a licence snapshot. An index (SQLite with FTS5, WordNet aliases and category closure) is published to the bucket with version.json written last. The portal gets /world/api/resolve, search, cards and asset routes. Files are mirrored the first time someone asks (rung 2). The client runs the loop: unknown noun, resolve, addKind, re-parse, plan, ghost, stream. It shows a shortlist of up to 7 variants, a 'closest I have is X' middle band, and degrades to names and aliases when the PC is off. Embeddings through the PC are the fallback for paraphrases.
- Acceptance test: (1) A 200-query labelled set written from your own phrasing: in-index nouns are correct in the top 3 at least 85% of the time (the small trial scored 68% for names alone and 88% for embeddings; re-measure). (2) 30 nouns that are not in the index (for example 'dragon'): at most 2 get a wrong object; the rest get 'not found' or an honest 'closest I have'. (3) Resolve p95 at most 300 ms warm with the PC off. (4) An indexed but not yet mirrored asset (for example the Fuel piano) is usable within 60 s with the PC on (a target, unmeasured), and within 300 ms the second time. (5) Every served furniture GLB is at most 3 MB and every hero asset at most 8 MB. (6) The credits panel and licence scan pass for every served asset, with no NC/SA/ND, Valve, ShapeNet or QAL pack in the store. (7) The same asset requested by two accounts is stored and mirrored once. (8) Directional kinds with an unknown front are treated as symmetric and Skippy says so.
- Effort: About 9 working days: ingest and normaliser 4; optimiser 2; index and API 2; client resolver loop 1. The search-quality labelling is on top of that and needs you.
- Depends on: Increment 1.

### 3. Physics and the beam
- Ships: Rapier loaded after first paint. Bodies are made from each card's collider spec (oriented box by default, compound from Deckard-style primitives where present, hulls decomposed offline on the PC later), created asleep, woken by grab or shove, and put back to sleep. Anything over 8 m or 2,000 kg is static. Desktop beam from the lower right with a spring drag; phone tap and long-press; undo; the planner's floor/clear/wall questions answered by ray queries.
- Acceptance test: (1) A 39-body scene (3 couches, 6 chairs, a 10-crate pile, 20 stools) steps at a median of at most 0.5 ms on desktop Chrome, all bodies asleep within 5 s, the pile moving at most 2 cm. (2) First paint time is unchanged from increment 1 (physics loads afterwards). (3) Dragging a crate lifts it and releasing drops it and lets it sleep; dragging the car or the jet is refused with 'too heavy to lift'. (4) Idle scene draws 0 frames. (5) Target to confirm on the S23: median frame at most 16 ms with at most 150 draw calls. (6) The 850 GMod tests still pass.
- Effort: About 5 working days: glue 1; collider specs and pipeline field 1; beam and drag on desktop and phone 2; tests 1.
- Depends on: Increment 1 (cards with mass and collider fields; increment 2 helps but is not required).

### 4. Voice through a pass-guarded gate
- Ships: A small 'skippy-gate' container on the PC, published only through Tailscale Funnel on port 10000, with short per-person passes signed by the portal under a new label (a music or video pass never opens it), CORS limited to the portal origin, body limit of 128 KB and 25 s. Route: POST /stt to local Whisper. Hold-to-talk in the page (X key or a button), MediaRecorder with Opus, the transcript taking the typed-text path. Skippy's reply is text plus the browser's speechSynthesis; his ElevenLabs voice for friends only if you decide so (decision 1). Per-person quotas and a clear 'voice is unavailable' message when the PC is off.
- Acceptance test: (1) Spoken 'Skippy, add three couches' gives ghosts within 4 s of releasing the key (inferred 2-4 s; measure it). (2) The gate rejects: no pass, a music pass, an expired pass, a body over 128 KB, audio over 25 s, and any path other than the listed routes. (3) With the PC off, the page shows voice as unavailable and typing still works. (4) The portal tests for pass signing mirror the music-gate and video-gate test vectors. (5) Whisper accepts the browser's recorded format (webm/opus on Chrome; check mp4/aac on Safari). (6) No request to SkippyTel's own port 3003 is ever exposed.
- Effort: About 4 working days: gate 2; page voice and fallbacks 1; tests 1. Plus your decisions on voice and spend.
- Depends on: Increment 1.

### 5. Deckard repair and sourced masses
- Ships: The Deckard fixes from section 9: the axis order in the three ShapeNetSem sites, quarantine of a_car.md and the other frozen model answers, a per-class size gate, article stripping and redirects in the Wikipedia lookup, think:false with a structured facts prompt, no generic-vessel fallback for named things, and a vectorised integrator. Plus a small sourced table of class masses (including a car and a 747-400 empty weight) that cards use. Pinned scene outputs are regenerated and the diffs shown to you.
- Acceptance test: (1) The 261 plus 151 Deckard-related tests pass. (2) A couch compiles wider than tall (about 2.13 x 1.06 x 0.89 m, not 2.13 m tall); the audit kinds with height within 0.67-1.5x of real rise from 18 to at least 28 of 33. (3) 'A car' returns 1,000-2,200 kg and 'a jumbo jet' 150,000-220,000 kg, or an 'unidentified' stub, never a 0.6 kg mug. (4) Nothing writes model output into the repo catalog. (5) Compile time of the 9-part car drops from about 4 s to at most 1.5 s with identical mass, centre of mass and inertia. (6) You have reviewed the pinned-scene diffs.
- Effort: About 3 working days.
- Depends on: None technically. Cards benefit once increment 2 exists.

### 6. Generation on the PC
- Ships: Start with a half-day bench: the trellis.cpp v0.8.1 Windows CUDA program plus the Q4 weights (about 7 GB; your go-ahead needed), 5 objects at 512 and 1024, wall seconds and peak VRAM logged. Then a resident generation worker behind a single-slot queue and a GPU lock that evicts Ollama models and unloads Stable Diffusion for the job: a concept picture made from a fixed template, background cut with a threshold keyer (not RMBG-2.0), TRELLIS.2 at 512 as the rough preview, the automatic quality gate (floating parts, inside-out faces, thin walls, proportion bands), then the 1024 version in the background. Scale comes from a sourced size, not the model. A cache by normalised prompt, per-person and daily caps, a prompt filter, and the exact Skippy lines in section 5. Optional hosted fal.ai path if you set a budget.
- Acceptance test: (1) Bench: measured seconds and peak VRAM for 5 objects at 512 and 1024, recorded in a table, replacing my estimates (about 30-70 s at 512 and 1:45-5:00 at 1024). (2) 'Add a helicopter' (not on any shelf): a labelled box appears at once, a rough mesh swaps in under 3 minutes, and a detailed one later, with Skippy's three lines. (3) The quality gate rejects a mesh with floating parts or missing wing in the fixture set and passes a good one; after 30-50 real generations its thresholds are recalibrated. (4) A second friend asking for the same thing gets the cached result with no GPU use. (5) During a generation Ollama and Whisper still answer, or Skippy says generation is waiting. (6) The generated mesh gets a card (size from table, mass from table, box collider) and a credit-free 'generated' label.
- Effort: About 0.5 days for the bench, then about 10 days for the worker, lock, gate, picture step, caps and card writer.
- Depends on: Increments 2 and 5. Your approval for the downloads.

### 7. Deckard stand-ins for vehicles and aircraft
- Ships: Deckard v2 as a card compiler for classes with no good free model: a facts card, an archetype grammar for road vehicles and aircraft, and the mass anchor. Output is a part list drawn as primitive meshes in the page with Rapier primitive colliders (2.5-11 KB of JSON), used as an instant stand-in while a model generates and as the fallback when generation fails or the PC is off.
- Acceptance test: (1) 'A car' gives about 1,565 kg for the mid-size default, inside the 1,100-2,200 kg sourced band, in a 7-part body drawn with correct proportions. (2) 'A jumbo jet' resolves to the 747-400 with 70.66 x 64.44 x 19.40 m and 183,523 kg, in a 9-part body, mobility static. (3) The same spec compiled at three grid resolutions agrees within 1% after anchoring. (4) Each card says blocky and approximate. (5) Compile plus fact lookup is cached so a repeat request is a file read.
- Effort: About 10 working days.
- Depends on: Increment 5.

### 8. Shared rooms and proximity voice
- Ships: A separate small WebSocket service ('skippy-world') on Cloud Run, max one instance, with state saved to GCS. The portal mints a 10-minute HMAC ticket. The server keeps what exists, who owns it and where it rests. It enforces the GMod holodeck's caps, cooldowns and undo, validates every move, and relays. The speaker's client plans; the holder of an object simulates it and streams its pose at 15-20 Hz, then sends a rest pose. Late joiners receive the registry. Audio-only WebRTC (mesh up to 6-8 people) with Web Audio panning using the GMod numbers: normal 21 m with the last 40% fading, whisper 7 m, shout 46 m, walls muffle. STUN plus TURN.
- Acceptance test: (1) Two browser tabs in one room: a sentence in one shows the same poses in the other within 300 ms. (2) A third tab joining late sees every object at its rest pose. (3) A fourth object over the per-player cap, a request during cooldown, or a pose outside the room is rejected with a reason. (4) Undo removes the speaker's last group in all tabs. (5) Killing and restarting the service: clients reconnect and resync, and the registry survives. (6) Two headless tabs test the voice panning: volume falls with distance and drops behind a wall. (7) No Lua runs on the server.
- Effort: About 8 working days.
- Depends on: Increments 1 and 3. Your decisions 3 and 7.

### 9. WebXR hands, phone AR and glasses mode
- Ships: WebXR with hand tracking where it exists (Quest, Android XR): index-fingertip ray, pinch grab, controller fallback. Android phone AR where hit-test and plane detection supply the planner's floor and wall questions for a real room. A '?view=glasses' page for the Viture glasses as a second screen: black background, large type, 0 fps idle, 30-45 fps while moving.
- Acceptance test: (1) On a Quest or Android XR device: pinch grabs a crate and releases it. (2) On an Android phone: 'add a couch' places it on a detected floor at its real size. (3) On the glasses via the phone: text and objects are readable, black is see-through, and the phone does not overheat over 10 minutes (the native app ran it hot). (4) iPhone keeps the non-XR view.
- Effort: About 6 working days, plus the devices and testing time.
- Depends on: Increments 1 and 3.


## Decisions that are Aaron's

- **May friends hear Skippy's cloned voice?**
  Recommendation: No. Give friends text plus the browser's built-in speech or a stock voice, and keep the clone on your own account.
  Why it is his: tts.py says the cloned voice is for personal use only and needs permission from both rights holders to release. Sharing it with friends is your call on rights and on how it sounds.

- **What spend limits apply: ElevenLabs per friend per day, and hosted generation (fal.ai about $0.25-0.35 per object)?**
  Recommendation: Local generation only to begin with. Give hosted generation a monthly cap of your choosing (for example $10, about 33 objects at 1024). Allow 6 new objects per hour per friend and a per-friend daily character cap for speech.
  Why it is his: It is your money and your credit, and it decides how freely friends can ask.

- **Which is the real deployment: Cloud Run at makerportal.live, or Docker on the home PC at makernet.live?**
  Recommendation: I designed for Cloud Run because the deploy workflow, the Dockerfile and the app code say so, and 'makernet.live' appears nowhere in SkippyTel. If it moved, tell me: the voice gate and the world service get simpler, because they could sit next to the portal.
  Why it is his: Only you know what is actually running and what domain you intend to use.

- **Where should the shared-room service run (increment 8)?**
  Recommendation: Cloud Run, one instance, state in GCS, so rooms survive the PC being off. Keep the PC for voice and generation only.
  Why it is his: It is a hosting and availability choice with a small ongoing bill and an always-on public service.

- **May ShapeNet-derived data (the size medians and exemplar facts in Deckard and the public repo) appear in friends' browsers?**
  Recommendation: Not for now. Use curated and sourced sizes in the web client. Separately check the 184 exemplar JSON files (668 KB) in the public sigma-ground repo that derive from PartNet and ShapeNet.
  Why it is his: ShapeNet's terms are non-commercial research, and whether sharing derived numbers with friends is acceptable is a legal and ethical call about your project.

- **Which free-asset sources and accounts do we use, and do we open any accounts?**
  Recommendation: Start with Poly Haven, Kenney, Gazebo Fuel and NASA (no accounts). Skip the new Quaternius packs and ABO until their licences are clarified. Consider a free Poly Pizza API key and a Smithsonian api.data.gov key only at increment 2. Open a Cloudflare account for TURN only at increment 8.
  Why it is his: Signing up for services and accepting their terms is something you must do yourself, and each source's licence risk is yours to accept.

- **Do the Half-Life 2 / CS:S models stay in Garry's Mod only, with the web client using redistributable assets and the same footprints?**
  Recommendation: Yes. The footprints and masses are facts you measured and are low risk to share; the meshes never leave your machine.
  Why it is his: It is your reading of the Valve agreement for your own install, and it decides what the browser can show.

- **OK to download trellis.cpp (682 MB) and the Q4 weights (about 6.55 GB) for the generation bench?**
  Recommendation: Yes, at the start of increment 6, after increment 2. Run the 5-object bench before committing to the full build.
  Why it is his: Large downloads, and they take disk and the shared 16 GB GPU.

- **Is friends' microphone audio allowed to reach your home PC through a public gate?**
  Recommendation: Yes, with short passes, size limits and per-person quotas, only on listed routes, and only for friends you have granted the 'skippy' permission. Typed input always works without it.
  Why it is his: It opens a path from the internet to your machine and carries your friends' voices.


## Top risks

- Found assets are approximate: arbitrary scale, wonky shapes (the Poly Pizza 747 is described as wonky), and unknown fronts so sofas can face the wrong way. Mitigation: sizes pinned to sourced dimensions, fronts checked by eye for the first shelf, honesty labels from card fields, a symmetric fallback for unknown fronts.
- Licence mistakes. Uploader-asserted CC labels (Objaverse) can be wrong; Quaternius changed terms on 2026-08-28; the ABO terms conflict between sources; ShapeNet-derived numbers sit inside Deckard and the public repo. Mitigation: a hard whitelist in the build, a per-asset card and dated licence snapshot, manual spot checks, nothing from Valve, Sketchfab or ShapeNet on the portal. This is not legal advice.
- Planner cliffs and non-identical results across Lua engines. In my test a jet plus three cars used the whole 10 s budget and placed nothing; 'a dozen crates' takes about 450 ms; 12% of requests differ between wasm and native Lua. Mitigation: a worker budget of 0.5-1 s, partial results, an outdoor world for big things, and one planner per room.
- wasmoon is fragile. It had a state-corruption bug after about 45-60 awaited calls (avoid by using synchronous calls only), its stable release is nearly three years old, and only Chrome has been tested. Mitigation: a golden-test job, a pinned version, and fallbacks (fengari, or our own Lua-to-wasm build).
- Nothing was measured on phones, headsets, Safari or Firefox. Draw-call blowup (63 calls for ten naive stools), GPU memory from textures, and heat on the phone are the likely failures. Mitigation: merged and instanced meshes, KTX2, render on demand, a device check in increments 1 and 3.
- PC-dependent features (voice, generation, mirroring new assets, embedding fallback) vanish when the PC is off or asleep, and nothing remote can wake it. Funnel's bandwidth limit is unpublished and WebSockets through it are undocumented. Mitigation: the page works without the PC, the gate carries short HTTPS calls only, and Skippy says what is unavailable.
- Cost and abuse on public-ish endpoints (ElevenLabs credit, GPU time, bucket egress). Mitigation: per-person passes and quotas, caps copied from the holodeck, a global monthly cap, a prompt filter, and hashed immutable files.
- Generation is unproven on this hardware. There is no measured time or VRAM on the 4070 Ti SUPER; 11.5 GB of the 16 GB was already in use; thin parts of aircraft may be lost or thickened; the quality-gate thresholds were tuned on synthetic jets; trellis.cpp is a one-maintainer community port whose Windows image-to-mesh run is untested.
- Deckard's old numbers can leak into cards (a car at 2.5 kg, couches 2 m tall) if increment 5 is skipped or its output is not anchored to sourced totals. Mitigation: sourced mass tables first, Deckard only as a cross-check until repaired.
- The brief's premises differ from the code (Cloud Run not Docker, makerportal.live not makernet.live, a 16 GB SUPER card, no WebXR path for Viture). If the portal has moved, parts of the gate and world-service design change.


---

## Red team: what changes the plan

The skeleton is sound: browser-first, a small card per object, a lookup ladder, and a licence gate. Most of the facts I re-checked hold. But the first increment rests on three things that are wrong or untested, and the security story is thin.

1. "Go outdoors" does not fix big objects. Mixed-kind sentences fall off a planner cliff even in open space. I measured this on the real planner.
2. The dynamic-kind loop listens for the wrong signal. The parser silently turns "fire hydrant" into a wood stove, and a hydrant is on your own shelf.
3. Increment 1 is about twice the stated size. It leaves out the engine-glue port (undo, move, remove) and four-source asset curation.

Security gaps:
- /world runs on the portal's own origin, with a CSP that only reports at first.
- Untrusted community 3D files get converted on the PC that holds the .env vault.
- Voice opens an inbound path to that same PC when simpler options exist.

Facts I confirmed:
- The portal is Cloud Run at makerportal.live: gunicorn with 1 worker and 8 threads, max 1 instance, 1 GiB. makernet.live does not resolve at all (NXDOMAIN), so decision 3 can be closed.
- The GPU is a 16 GB 4070 Ti SUPER.
- Poly Haven has 521 models, all with dimensions.
- Fuel has 3,387 models, 3,340 of them CC0 or CC-BY 4.0 (47 excluded), exactly as claimed.
- NASA has 257 GLBs. The Quaternius licence page, the fal.ai prices, the Wikidata dimensions and the Cloud Run limits all match.

Corrections to the design:
- ABO is CC BY 4.0 (changed from NC in 2023), so it is no longer a hold.
- The existing Browser Games are public, with no sign-in.
- The trellis.cpp repo is labelled MIT.
- Objaverse is a scrape of Sketchfab, which the design excludes.

Recommendation: proceed, but narrow increment 1 and fix items 1 and 2 before any asset work. Do the half-day generation bench before committing the 20+ days that depend on it. Experiment scripts and outputs are in C:\Users\aaron\AppData\Local\Temp\claude\D--Aaron-development-sigma-ground\6981a9aa-3c90-44ec-9872-031c4164ea1d\scratchpad\wf3\redteam\.

### [high] Outdoor apron does not fix the planner: mixed-kind sentences fall off a cliff and the refusal text is false
Measured on the real planner (LuaJIT via lupa) with kinds shaped like the design's own bigobj.py. Results are in apron_cliff_results.txt, apron_mixed_results.txt, apron_why_results.txt and pair_fuzz_results.txt. The kind records are synthetic, so the exact failing pairs depend on the category and wall tags the real ingest would emit.

- 'add a jumbo jet' alone on a 400 m apron: 2.6 ms.
- 'add a jumbo jet and a car': 4.3-7.5 s with a 10 s budget and 0 placed. The planner makes 61,828 world callbacks. It fails identically on a 400 m apron, a 1,000 m apron and an infinite open world.
- With the 0.8 s web budget it stops at 800 ms, places 0 and says 'That is more than I can fit here'. In an open field that is false.
- Ordinary outdoor pairs fail too. 'a car and a tree' takes 1.1 s and places 0; 'a bus and a tree' and 'a rock and a tree' fail the same way. 'a bus and a car', 'a car and a cone' and 'a jet and a hangar' work.
- Pair fuzz over 8 shelf-like kinds on the apron at 0.8 s: 6 of 28 pairs (21%) fail. Every failure involves the jet or the rocket.

Mechanism: a multi-kind sentence with no arrangement goes to room_attempt (plan.lua around line 2432). That is a furniture-room composer, and it then scans open_frames. Failing cases make about 30k floor_z and 5k clear calls. At the design's 5-17 us per wasm-to-JS call, that is 0.15-0.5 s of boundary crossing alone.

The design blamed the size of the room. That diagnosis is wrong.

**Fix:** Split a multi-kind sentence into sequential requests client-side whenever any kind is outside furniture, or the footprint ratio exceeds roughly 8x. I validated this: 'add a jumbo jet. add a car' places both in 2.7 ms, and 'add a jumbo jet, then add a car' in 36 ms. 'and then a car' still fails, so split on sentence breaks, not on that phrase. Also:
- Make a pairwise fuzz of every shelf pair a CI gate, in indoor and apron worlds, at the 0.8 s budget.
- Replace the timed-out message with something honest, such as 'I couldn't arrange those together, so I'm adding them one at a time'.
- Answer floor and clear questions from data held inside Lua, such as a heightfield or AABB list, instead of per-call JS callbacks.

### [high] The dynamic-kind loop is fed by the wrong parser signal, so common nouns never reach the resolver
The design says the parser reports unknown nouns. I ran 79 'add a <noun>' sentences through the real parser (hijack_results.txt).

- 53 came back as UNKNOWN, which would fire the resolver.
- 25 resolved to a catalog kind. Some are legitimate catalog nouns. Others are silent wrong objects that never reach the resolver:
  - 'fire hydrant' becomes wood stove (approx=true, 'hydrant' dropped)
  - 'fire truck' becomes wood stove ('truck' dropped)
  - 'drum kit' becomes barrel ('kit' dropped)
  - 'palm tree' and 'oak tree' become the generic plant
  - 'pool table' becomes table, and 'ottoman' becomes stool
- 'swing set' produces no item at all.
- That is roughly 7-11 of about 66 non-catalog nouns (11-17%).

Hydrant and palm tree are on the design's own increment-1 shelf. Other parser quirks:
- 'boeing 747' reports unknown 'boeing' and drops the 747.
- 'add a 1969 mustang' parses with count 1000. The planner caps it at 60, so Skippy would say 'I limited mustangs to 60'.

**Fix:** Treat any item with approx=true, a non-empty leftover, or a dropped head noun as unresolved, and send its whole phrase (the parser already returns the 'phrase' field) to the resolver before planning. Register shelf kinds with their multiword names before parsing so exact matches win. Strip bare numbers from product-name phrases before counting. Build the 200-noun corpus now and gate on the silent-hijack rate. Have Skippy say when a close match was used.

### [high] /world on the portal origin: same-origin script can reach admin endpoints, and the CSP only reports at first
The portal session cookie is HttpOnly and SameSite=Lax, so script cannot read it. Any script running on the portal origin can still call the same-origin endpoints as the signed-in user. Those include admin actions that start and stop the games VM, pair the home reporter and change settings.

/world will render untrusted strings: Fuel, Poly Pizza and Objaverse titles, authors, URLs and tags, plus friends' generated names and prompts. The design ships the CSP report-only at first, so nothing is enforced in increment 1. Credits links and alias or description fields are classic injection spots, and a javascript: URL is one example.

**Fix:** Serve the client from its own origin, such as world.makerportal.live or another domain. Mint a short ticket from the portal, the way increment 8 already does.

Enforce the CSP from day 1 (script-src 'self' 'wasm-unsafe-eval'; no inline script). Use textContent only. Allow https:// credit URLs only. Sanitise every card string at ingest. Add a test that feeds hostile card text through the credits panel.

### [high] Untrusted community 3D files are converted on the Captain's own PC, next to the .env vault, with a standing cloud credential, and a friend's sentence can trigger it
Rung 2 has the PC fetch, convert and publish an asset when someone asks for it.

- The sources are community-upload sites. Fuel has no upload review (a maintainer says so on Discourse). 1,230 of Fuel's 3,340 acceptable-licence models are community uploads (fuel_all.json).
- Collada, OBJ/MTL and .gltf files can reference external paths or URIs. Converters such as glTF-Transform and assimp resolve them relative to the input. A crafted model could pull a local file, for example the dev-root .env vault named in CLAUDE.md, into the published bundle. I did not test this; it is a known class (CWE-22). Pixel-count and decompression bombs and parser CVEs are the other risks.
- Publishing needs a GCS write credential on a Windows machine that also runs games, Discord and Docker.
- The design has no sandbox, no size caps and no approval step.

**Fix:** Run the converter in a locked-down container with no network and no mounts, read-only input, a non-root user, and memory and time limits.

Accept only .glb with embedded buffers after rewriting all URIs. Cap pixels, vertices and file size. Run gltf-validator.

Upload through one-time signed URLs that the portal mints per job, so the PC holds no standing credential. If you must keep a key, scope it to objectCreator on one prefix.

For rung 2, restrict on-demand mirroring to a pre-vetted allowlist of authors and packs. Put everything else in an approval queue.

### [medium] Voice (increment 4) is the only change that opens an inbound path to the PC, and simpler options exist
The existing music and video gates are read-only. This one accepts uploads of friends' audio. The Whisper service writes the bytes to a temp file and lets ffmpeg decode them. The model is medium on the GPU, with task=translate and a primer tuned to the Captain's commands (server.py).

Tailscale's Funnel documentation calls it beta, with bandwidth limits you cannot configure, and does not document WebSockets. Alternatives were not weighed:
- Browser SpeechRecognition, which is free and needs no ingress. Chrome's on-device mode is experimental (processLocally); the default sends audio to Google.
- Whisper running in the browser through transformers.js. Quoted download sizes disagree: about 40 MB for base q4 in one source, 150-290 MB in others.
- Relaying through the existing job queue, which keeps the PC outbound-only.

**Fix:** Ship v1 voice with feature-detected SpeechRecognition feeding the same text path. Then add in-browser Whisper or a queue relay if quality demands it. If you keep the gate:
- run ffmpeg unprivileged
- allow-list codec and container with ffprobe
- cap clips at about 10 s
- allow one in-flight request per pass
- make the 'voice unavailable' state first-class

### [medium] Asset bytes go through a single gunicorn worker with 8 threads, and the search index is sized for a library 12 times larger than the sources justify
Portal facts:
- The Dockerfile runs 'gunicorn --workers 1 --threads 8'.
- Cloud Run is set to max-instances 1 and 1 GiB.
- The writable filesystem is in memory and counts against that 1 GiB (Cloud Run container contract).

A phone at 10 Mbit/s holds a thread for about 6.4 s per 8 MB blob. Nine concurrent downloads would then block login, chat and /home/report. The home state flips to 'home is silent' after 60 s without a report.

A private bucket buys nothing, because the design's own licence rule means every served file is redistributable.

The real source total is about 8.5k assets: 521 + 4,402 + 3,340 + 257. That is roughly 1 MB gzipped of names and aliases plus about 13 MB of 768-d f16 vectors. The design's 225-275 MB figure is for 100k assets.

**Fix:** Serve blobs from a public-read, hash-named, immutable bucket, or V4 signed URLs. Add Cache-Control: immutable and CORS. Keep the portal to HTML and JSON.

Ship the names and aliases index to the browser (cached) and search it in the worker. That removes the 75-200 ms round trip, the single-instance bottleneck and the PC dependency. Keep sqlite and the vector file for the PC build step only.

### [medium] Search acceptance targets conflict: the PC-off mode cannot reach 85%
Embedding the query needs the PC (Ollama). With the PC off, search drops to names and aliases, which scored 23/34 = 68% (Wilson 95% interval about 51-81%). Acceptance (1) needs at least 85% and acceptance (3) tests the PC-off path.

Embeddings scored 30/34 = 88% (interval about 73-95%). The intervals overlap, and the 34 queries are the author's own labels.

In-domain and out-of-domain cosine ranges also overlap (0.54-0.70 vs 0.59-0.65), as the design itself says.

**Fix:** Embed queries off the PC, with a small ONNX model on Cloud Run or in the browser. Or set two targets (for example at least 85% with the PC on, and at least 70% plus an honest 'closest I have' with it off).

Build the 200-query set from your own phrasing before choosing thresholds. Use embeddings only to re-rank full-text candidates.

### [medium] Aliases from source tags can shadow curated nouns in the parser
The design registers found assets with addKind(names=aliases). Aliases come from source titles, tags and descriptions. A Fuel model tagged 'couch', or an Objaverse tag, would compete with the curated couch for everyone. The parser already shows how loose its approximate matching is (the 'fire' to wood stove case above).

**Fix:** Curated names always win. Source aliases only match when no curated kind matches. Limit them to the normalised title plus at most 3 hypernyms, and never use free-text descriptions. Namespace asset kinds (for example asset:fuel/xyz). Add a regression test that the 46 curated nouns still resolve to curated kinds after loading the full index.

### [medium] Increment 1 is about twice the stated 6 days, and the engine-glue port is missing
The 'brain' is only parse, plan, layout and library. The engine glue is holodeck.lua at 1,144 lines. The planner reads ctx.registry, ctx.last and ctx.mine, and returns moves and removals. The glue also handles groups, undo, turn, recolour, resize, caps and cooldowns.

The README flagship needs it: 'put a lamp next to the couch... Undo. Clear the room.' The acceptance tests are add-only.

Also in the increment:
- 12 assets from 4 sources in 3 formats, with scale and front set by eye
- an indoor and an outdoor world
- a wasmoon CI job
- a headless smoke test
- the portal route, grant and tests
- an S23 test

The 46 starter kinds appear as boxes. If you later swap in different meshes, their footprints will not match the HL2-measured card sizes. That leaves two sources of truth.

**Fix:** Split it:
- 1a: add-only page, brain, boxes, and 3 assets from one pre-converted source
- 1b: registry, undo and the glue
- 1c: the shelf and the contact-sheet review

Define what a starter kind looks like on the web. Either it stays a labelled box, or it gets a stand-in whose footprint is re-measured and written to the card. Re-estimate at 10-14 days.

### [medium] The jet acceptance test cannot be met honestly, and the 'sourced' mass cherry-picks
The 747-400 has a length-to-wingspan ratio of 70.66/64.44 = 1.097. The Poly Pizza mesh is described as 'wonky' (the page gives no triangle count), so its ratio will differ. Requiring both length at 70.66 +/-1 m and span at 64.44 +/-1 m (about 1.4-1.6%) forces a non-uniform stretch or a failing test.

The Wikipedia table gives two operating empty weights for the 747-400: 183,523 kg for the PW engines and 163,701 kg for the GE engines. That is a 12% spread. The card stores one as basis 'sourced'.

The Wikidata item (dimensions verified) is the 747-400, while the mesh is an unspecified 747.

**Fix:** Scale uniformly on the longest axis and report the other axes as measured, with a tolerance of about 5%. Store mass as a range with both sources and call the object a '747-style jetliner'. Do not stretch non-uniformly without saying so on the card.

### [medium] CC-BY compliance gaps: no adaptation notice, forks credited wrongly, API attribution missing
- CC BY 3.0 (the Poly Pizza 747) requires a credit identifying use in an adaptation. CC BY 4.0 requires an indication of modification. The card has no 'modified' field, yet every asset is rescaled, simplified and re-encoded.
- Fuel (fuel_all.json): of 3,340 acceptable models, 381 are exact name-and-size duplicates. 232 name-and-size groups have more than one owner, which means forks. 471 of 1,230 community uploads have an identical twin. 198 community uploads are self-labelled CC0. Crediting the forker can be wrong, and the original author is the one who must be credited.
- Poly Haven's API terms require 'Powered by Poly Haven' attribution and a unique User-Agent, even though the assets are CC0.
- NASA's repo lists outside contributors (Ball Aerospace, DigitalSpace). NASA says its insignia are not public domain and may not appear with AI-generated imagery.
- Objaverse is a scrape of Sketchfab that ignored NoAI tags. Its licence labels are uploader-asserted. The design excludes Sketchfab but lists Objaverse as a later source.
- ABO is CC BY 4.0 since 2023 (AWS Open Data registry), so the 'hold' is stale.

**Fix:** Add modified:true and a change note to the card, and print it in the credits panel. De-duplicate Fuel by content hash. Prefer OpenRobotics and GoogleResearch owners. Require a manual review for community-uploaded CC0. Add the Poly Haven and NASA credit lines.

Drop Objaverse, or limit it to CC0 with a manual check. Move ABO to the first-tier candidates for furniture: 7,900+ products, glTF 2.0, real scale. Keep a dated licence snapshot per asset and a takedown path.

### [medium] The free-source ladder is thinner than it reads, and some hits are semantic traps
Poly Haven has 521 models, but 343 are scenic or collection pieces and only 231 are everyday indoor objects. It has no car (only 'covered_car'), no aircraft, no bicycle and no piano.

Fuel has no Boeing at all. The airplane hits are four 'Advanced Plane' copies, a 0.17 MB 'jet 1' and a Cessna. The helicopter hit is a Google Scanned Objects toy of 8.7 MB. 1,033 of its 3,340 usable models are such consumer-product scans. 'sofa' includes a '3D_Dollhouse_Sofa' of 11.7 MB, a toy.

The hero demos depend on two single models of unknown quality: the Kenney sedan and the Poly Pizza 747. The 'long tail' depends on either Objaverse, which is licence-compromised, or generation, which is heavy.

**Fix:** Carry a class flag (toy, miniature, product) from the source and exclude it from furniture and vehicle nouns by default. Get a contact-sheet review of the actual jet and car meshes before promising the demo. Plan the coverage honestly: starter kinds, then ABO, Kenney and Fuel's curated owners, then generation. Log every unresolved noun to decide what to mirror next.

### [medium] The generation plan rests on a community port with no tested path on this machine
Upstream TRELLIS.2 says it needs at least 24 GB of GPU memory. It was verified on A100 and H100 under Linux (3 s at 512, 17 s at 1024 on H100). Upstream issues report VRAM growth and leaks (#63, #136) and a peak above 63 GB (#24).

trellis.cpp's own README measured an RTX 5060 Ti (16 GB) at resolution 1024: 3:16 for a light input and 7:23 for a heavy one, including load. The heavy case was slower than the Python reference. Its v0.8.1 notes say a complete Windows image-to-quad run is untested. A thin-winged airliner is a heavy input.

The design's 30-70 s at 512 is a reasonable ratio from the H100 numbers, but no desktop 512 timing is published.

GPU contention on the machine:
- OLLAMA_KEEP_ALIVE=-1 with up to 2 models
- Whisper medium resident, about 2 GB
- Stable Diffusion
- the display

Evicting Ollama makes the next voice command pay a reload.

The design has no queue-depth cap. 6 objects per hour for each of 10 friends is about 60 per hour against a capacity of roughly 15-25 per hour.

The pipeline hard-wires facebook/dinov3 (a gated Meta licence you must accept yourself) and briaai/RMBG-2.0 (CC BY-NC). The design's pre-matting bypass avoids RMBG-2.0, but trellis.cpp v0.6.0 mentions 'robust BiRefNet'. Check which weights it bundles.

**Fix:** Run the half-day bench first, before increments 2-5 are justified by it. Run each job as a subprocess so VRAM is freed on exit. Add a queue cap with an ETA, and an admin 'reject and regenerate' for cached results. Extend the prompt filter to real persons and trademarked characters. Read the DINOv3 licence yourself. Confirm the matting weights are not NC.

### [medium] Quotas, queues and the spend cap live in one Cloud Run instance's memory
ratelimit.py says it is in process and that the portal runs one instance. The home job queue has a 600 s JOB_TTL and also lives in process. Both reset on every deploy.

A 'monthly spend cap' and '6 per hour' quotas need durable counters.

Jobs reach the PC on the reporter cadence of about 15 s, or every 1.5 s only inside a 300 s fast window that follows chat activity. So rung 2 adds up to 15 s before any work starts.

MAX_REPORT is 1,000,000 bytes, so result bytes cannot ride the reporter.

**Fix:** Keep counters in a GCS object or Firestore, with a cap that fails closed. Open the fast-poll window when someone loads /world. Return results by signed-URL upload. Add GCP budget alerts plus a kill switch (budgets alert but do not cap).

### [medium] Client delivery is coupled to the shared SIGMA_GROUND_REF pin and to full portal deploys
deploy.yml uses a single SIGMA_GROUND_REF for both the skippy-mentat service and the portal image. Every client tweak would mean a sigma-ground commit, a pin bump, Mentat tests and deploy, a portal deploy and a smoke test. Mentat is a public API, and it would pick up whatever else is in sigma-ground master (the working tree currently holds uncommitted daily-job changes). The repo already has a better pattern in doomloop.py: a bucket folder, version.json written last, and no redeploy.

**Fix:** Publish the world client to a bucket channel read by the portal, or served from its own origin. Use the pin only for the generated Lua bundle if you need it at all.

### [low] wasmoon is the only runtime, and cross-engine behaviour is unmeasured
wasmoon 1.16.0 was published on 2023-12-08, nearly three years ago. The repo still gets commits (pushed 2026-09-07) but has had no release. The design itself found state corruption after about 45-60 awaited calls and a silent UMD import failure. 122 of 992 planner outputs (12%) differ between wasm and native Lua, so layout is sensitive to last-bit differences. Only V8 was tested; Safari and Firefox may differ.

**Fix:** Build and pin your own Lua 5.4 to wasm with emscripten, which is small. Run the parity corpus in Chromium, WebKit and Firefox. Keep 'one planner decides' as a hard rule, and never promise GMod-to-web parity.

### [low] The page budget fails on its own arithmetic
Measured from jsDelivr (three@0.186.1, gzip transfer). The two core files total about 196 KB, which matches the design's 194 KB. The design's 0.46 MB leaves out:
- GLTFLoader, 26 KB
- MeshoptDecoder, 8 KB
- KTX2Loader, 9 KB
- basis_transcoder.js, 15 KB

That is about 512 KB, over the 0.5 MB acceptance line. basis_transcoder.wasm is another 248 KB, bringing the total to about 760 KB. The Draco decoder wasm adds 89 KB.

**Fix:** Lazy-load the loaders and decoders at the first mesh. State the budget as at most 0.5 MB to interactive, and about 0.8 MB with decoders. Use meshopt only.

### [low] Factual slips in the design text
- 'Same Google sign-in as the other Browser Games' is wrong. The /play pages are public (app.py around line 1000: 'public like the experiments'). The real pattern for friend grants is portal_required('audio') or ('video').
- Decision 3 is answerable. makernet.live is NXDOMAIN, and makerportal.live returns 200 on Google addresses.
- ABO is CC BY 4.0, not unresolved.
- The trellis.cpp repo is labelled MIT, and the v0.6.0 notes mention an 'intended MIT license'.
- The '850 tests' figure could not be reproduced. I collected 1,263 tests in test_gmod_*.py, and 220 in the holodeck and glue files alone.

**Fix:** Correct the text. Decide explicitly whether /world has a public viewer mode with login only for voice and generation, or is login-only through a new portal grant. Name the exact command that yields the baseline count.

### [low] Increments 5 and 7 (about 13 days of Deckard) buy mostly a blocky stand-in
The design itself says Deckard must not be the authority on size or mass, and that its bottom-up mass runs a median 1.75x high. The remaining value is primitive meshes plus a mass anchor, which just reproduces the sourced total. Increments 6-9 are speculative but written as a committed order.

**Fix:** Replace increment 7 with about 12 hand-built primitive templates (car, truck, plane, boat, tree, house and so on) carrying sourced size and mass. That is about 2 days and needs no language model. Keep the Deckard repair on its own merits, and treat increments 6-9 as options to re-decide after increment 2.


### Missing
- A written threat model: a malicious or compromised friend account, per-account caps, a kill switch, and a hard spend cap. GCP budgets only alert.
- Takedown and removal: a CC-BY or rights-holder request process, hash revocation, an audit log of what each friend requested and received, and moderation plus 'regenerate' for cached generated assets.
- Demand logging. Record every unresolved noun, anonymised. It is the cheapest signal for what to mirror next.
- A decision on what the 46 starter kinds look like on the web (box or stand-in), and a per-source front and scale convention for auto-ingested assets. Unknown fronts will make cars and jets face sideways, and the card's 'treat as symmetric' answer is visible for those kinds.
- Outdoor rendering spec: terrain, sky, shadows, fog, far plane and a 400 m apron's draw budget on a phone.
- Behaviour spec for the PC-off cold path. The first request after scale-to-zero pays a Python, Authlib and google-cloud-storage start-up that was not measured.
- Voice transcript privacy and retention, and a friend-facing terms or consent note.
- Backup and rollback of the bucket and index (version.json written last helps, but bad publishes need a rollback).
- iOS Safari: storage eviction, WebGL2 memory limits and audio-context gestures.
- A defined first-run experience (example chips) and what Skippy says when the parser's own approximate match was used.

### Claims to recheck
- '850 existing tests': I collected 1,263 in test_gmod_*.py and 220 in test_gmod_holodeck.py plus test_gmod_glue.py. Name the exact command and set.
- The cause of the 12% wasm-versus-LuaJIT divergence (last-bit sin/cos) is a hypothesis. Planner parity in Safari and Firefox was never run.
- meshopt decoding 25-39x faster than Draco rests on a 2020 desktop benchmark. Real KTX2 and meshopt decode times on phones were not measured.
- Kenney '4,402 GLBs in 50 packs', the Objaverse LVIS count of 46,207, the Tripo and Rodin prices, and Hunyuan3D's territory exclusions were not independently verified.
- GCS prices ($0.022 per GB-month, $0.12 per GB egress), the Cloud Run free-tier claim for the world service, and the Cloud Run cold-start time with the PC off. The GCS pricing page did not load.
- trellis.cpp Q4 weights (about 6.55 GB) and their quality. Whether a 16 GB card fits TRELLIS.2 with Ollama, Whisper and Stable Diffusion resident. The README's only published timings are RTX 5060 Ti and Apple M5, none for 40-series cards.
- Which background-removal weights trellis.cpp bundles (BiRefNet versus RMBG-2.0), and the DINOv3 licence terms. The Hugging Face page is behind a login wall.
- Whisper accepting browser webm/opus and Safari mp4/aac. Probable because faster-whisper decodes through PyAV, but not run.
- Poly Pizza 747 quality, triangle count and file size. The page states no triangle count, and the Poly Pizza terms and Gazebo Fuel terms pages returned 404.
- Tailscale Funnel WebSocket support and throughput. The docs say bandwidth limits cannot be configured and do not mention WebSockets.
- Rapier 1.17 MB gzipped, the phone and XR frame budgets, the S23 time to brain ready (Node showed 1,029 ms, Chrome 602 ms wall), and home upload speed.
- My own apron, parser and pair-fuzz results use synthetic kind records patterned on the design's bigobj.py. Re-run them with the real ingest output.
