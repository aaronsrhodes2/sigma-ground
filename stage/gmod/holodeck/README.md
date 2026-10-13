# The holodeck (Garry's Mod)

Ask for things in plain words and real models appear, arranged the way a person would arrange them:

> **Skippy, add three couches.** → three couches in a U, open toward you, facing each other
> **Skippy, a table with six chairs.** → the smallest table that seats six, the chairs tucked in and looking at it
> **Skippy, a living room.** → couch facing the TV, coffee table between, armchairs flanking, a lamp and a plant in the corners
> **Skippy, put a lamp next to the couch. Turn the chair to face me. Remove the lamp. Undo. Clear the room.**

It never asks a clarifying question. Whatever is unclear resolves to the most sensible reading, and the reply says what it did
("there was no table, so I added one", "I don't have a dragon in my library").

## Using it

- **Chat or voice:** any sentence that starts with **Skippy** (or another name you gave him: see "Wake words (per player)") is a request. Everything else players say is left alone, so players with
  a live mic talk normally. In the console, `holodeck <words>` needs no wake word. Also `holodeck_status`, `holodeck_undo`, `holodeck_clear`,
  `holodeck_reload` (superadmin; re-reads which models the server has).
- **What it can build:** 46 kinds of everyday thing (seats, tables, beds, storage, lamps, appliances, bathroom, electronics, containers,
  plants) as 127 real Half-Life 2 and Counter-Strike: Source models, and seven whole rooms (living room, dining room, kitchen, bedroom,
  office, waiting room, bathroom). Colours ("a red couch") tint the model; size words pick a bigger or smaller model; counts are
  exact ("a dozen crates", "ten stools in a circle"). Arrangements: a row, a circle, a grid, a U or L, facing each other, stacked or
  scattered, against a wall or in a corner, around / beside / in front of / behind / on something that stands or that it adds.
- **What the models need:** the server must have the Half-Life 2 and Counter-Strike: Source content (the holodeck checks and leaves out
  what is missing); `holodeck_content hl2` restricts it to Half-Life 2 models, which every player has.
- **Safety on a shared server:** everyone's props belong to whoever made them (admins change anything) and are created as theirs
  (`SetCreator`, CPPI), the server's own spawn rules apply (`PlayerSpawnObject` and `PlayerSpawnProp`, asked with `hook.Run` so registered hooks and the gamemode's own function both count, and `sbox_maxprops` where the gamemode has no such function; an explicit `false` is a veto), a cooldown between requests (checked first, longer
  after an expensive one), caps per player and overall that undo also obeys, props of players who left make way for newcomers, a small
  planner time budget for ordinary players, Skippy's lines go only to the requester and those within earshot, and the bridge is
  rationed per player and per server. Details and the convars: [../CLOUD_SERVER.md](../CLOUD_SERVER.md).

## How it is built

```
words ─ parse.lua ─▶ request ─ plan.lua ─▶ poses ─ holodeck.lua (engine glue) ─▶ props in the world
          │                       │  uses layout.lua (geometry), catalog.lua (the library), library.lua (what this server has)
          └─ agent (optional): when the words build nothing, skippy_bridge.py asks a local model to turn the wish into a request,
             checked against the library before the game sees it
```

- `catalog_src.py` (curated: which model stands for which thing) + `measure.py` (the ENGINE's own bounds, mass, material, collision
  body, flat tops) + `data/front_overrides.json` (which side is the front, checked by eye on a photograph of every model:
  `contact_sheet.py`) → `build_catalog.py` → `addon/lua/mentat/holodeck/catalog.lua` (generated, do not edit).
- `parse.lua`, `layout.lua`, `plan.lua`, `library.lua` are pure Lua (no GMod globals) and are tested offline under LuaJIT.
  `holodeck.lua` is only engine glue (traces for floors and walls, spawning, chat, undo, ownership, the limits). It is tested offline
  too: `gmod_mock.lua` is a small stand-in for the Garry's Mod API (virtual clock, recorded net messages and HTTP calls, a flat world
  with walls and blockers you place) and `test_gmod_glue.py` runs the REAL `holodeck.lua`, `voice.lua` and the client files against it.

## Testing

```bash
pip install lupa pytest                                     # lupa = LuaJIT, the VM Garry's Mod uses (optional, test-only)
python -m pytest sigma_ground/radiance/tests/test_gmod_holodeck.py sigma_ground/radiance/tests/test_gmod_glue.py   # the planner/parser and the engine glue, a few seconds
python stage/gmod/holodeck/eval_parse.py --show 20          # the parser against 223 independently written and audited requests
python stage/gmod/holodeck/lab.py                           # (import) run any request in a synthetic world and grade it
```

The offline grader (`lab.py`) rebuilds every footprint from the engine-measured bounds, never from the planner's own numbers, and
checks floors, overlaps, walls, facing, tops. The live checks need the game files:

```bash
python stage/gmod/holodeck/live.py "add three couches" "a living room" --pos 64,0        # dedicated server, real physics, settled
python stage/gmod/holodeck/live.py "make it cozy" --start-bridge --persona --voice       # with the agent and Skippy's voice
python stage/gmod/holodeck/shots.py "a living room" --tag lr --top                       # photographs from several sides
python stage/gmod/holodeck/skippy_e2e.py                                                 # a spoken request through the real game
```

`live.py` grades what the physics engine left standing: nothing shoved, sunk, toppled, inside a wall, or overlapping, and everything
asleep after settling.

## Skippy: agent, voice, ears (all optional)

| Piece | Where it runs | What it does |
|---|---|---|
| `skippy_bridge.py` | next to the game server | `/plan` (a local model turns a vague wish into a validated request), `/reply` (Skippy's persona says what was done), `/speak` (an mp3). Fixed actions only, behind a token. |
| `skippy_ears.py` | each player's own PC | reads the microphone into a short memory (about a second and a half), and when the game says the player is transmitting, has the sentence recognised by the LOCAL Whisper; only a sentence that starts with "Skippy" is handed to the game (`--list` shows the microphones it can see) |
| `skippy_client.lua` | the game client | tells the ears when the player transmits (and keeps saying so while the key is down, so a crash is noticed within seconds); plays Skippy's voice in 3D from where he built things |

**Running the bridge.** It refuses to start without a token: put the same value in `SKIPPY_BRIDGE_TOKEN` (or `--token-file`) for the
bridge and in the server's `holodeck_bridge_token`; `--no-token` exists for a bridge on loopback only. Requests from a web page, without a
JSON body, or with a foreign Host are refused. Its work is bounded (a few connections, short queues for the model and the voice,
per-minute budgets per endpoint and per player, a daily budget for the paid voice); when it is full it answers 429 or 503 (and 501 where
the machine has no voice at all) and the game falls back to plain text. Everything the model says is cleaned and checked before it is
spoken or shown, and the game rations its own calls on top (see [../CLOUD_SERVER.md](../CLOUD_SERVER.md)). On Linux there is no
Windows speech: use `--tts auto` or `elevenlabs`.

Voices: `--tts auto` (the launcher's default: Skippy's own ElevenLabs voice through SkippyTel, Windows speech if that is not available;
it costs API credit per character, capped by the bridge's daily allowance), `elevenlabs` (his voice only, or nothing) or `sapi` (Windows speech,
free; the bridge's own default when you start it by hand). The game prints "Skippy's voice is now: elevenlabs" (or sapi) when the voice that
speaks changes. A dedicated server reaches the bridge on localhost with `-allowlocalhttp` (already in `gmod_ds_ctl.sh`);
a game that hosts its own world must use the machine's Tailscale address (the engine's flag works on servers only).

## Wake words (per player)

Every player has their own list of names Skippy answers to. The default is **Skippy** (with the spellings speech recognition makes of it).
You may add any name you like, rude ones included: it is your right, and it is private to you. Nobody else's sentences are ever matched
against your list, and everything Skippy says about it goes to you alone (a private line, never spoken aloud, never shown to those near
you). A name command typed in public chat is hidden from the other players' chat.

| Say (typed, aloud or in the console; start with any of your names, the console needs none) | Effect |
|---|---|
| `Skippy, answer to Rex` / `call yourself Rex` / `your name is also Rex` | add a name |
| `stop answering to Rex` / `forget the name Rex` | remove one (any, `Skippy` too, but never the last) |
| `only answer to Rex` | replace the whole list |
| `what do you answer to` / `what are your names` | he lists them, privately |
| `reset your names` | back to Skippy |
| console: `skippy_wake_add Rex`, `skippy_wake_remove`, `skippy_wake_only`, `skippy_wake_list`, `skippy_wake_reset` | the same |

Rules: 1 to 3 words, 2 to 24 characters, letters, digits, apostrophe, hyphen, space (case, quotes and accents do not matter); at most 8
names; a name must not be a word the parser needs (a command such as `add` or `come`, an object such as `couch`, a number, a pronoun),
because a sentence beginning with it would wake him by accident: he refuses that one politely and says why. Nothing is refused for being
rude. Only the **start** of a sentence counts, so a name later in a sentence does not wake him; `Rex's couch is ugly` talks about Rex and
only wakes him before an order. Spoken forms are tolerated by a conservative edit-distance rule scaled by the name's length (none under
5 letters, one slip from 5, two from 10; the first letter must match), and `ass hole` is the same as `asshole`.

How it is wired (three places, one list per player):

1. **The server** (`addon/lua/mentat/holodeck/wakenames.lua`, `wakewords.lua`): `Holodeck.Wake.strip(ply, text)` replaces the old
   `Parse.strip_wake` call at the chat hook and for what a player said aloud; `Parse.strip_wake` is still the matcher for the default
   name. The lists live in `data/mentat/wake_words.json` (by SteamID64; written through a temp file and read back; capped; any corruption
   means the default). `Holodeck.Handle` catches the name commands before the parser sees anything.
2. **The microphone bridge** (`skippy_ears.py`, `wakewords.py` is its twin of `wakewords.lua`): the server sends the list to its owner's
   client alone (net `holodeck_wake_names`, on join, on every change, and when the client asks), `skippy_client.lua` writes it to the
   client's `data/mentat/wake_names.json`, and the bridge re-reads that file when it changes (missing or corrupt means `Skippy`), so a
   sentence beginning with your name is passed on exactly like one beginning with Skippy. `+skippy` / the talk key is unchanged.
3. **The avatar** (`avatar.lua`): "`<name>, come here`" and "`<name>, go away`" use the speaker's names; when called by a custom name he
   answers privately ("Here I am, Rex.") instead of saying anything near others.

Tests: `sigma_ground/radiance/tests/test_gmod_wake.py` (Lua and Python share `wake_vectors.json`).

## Found objects: anything in the SkippyVerse inventory, as a labelled placeholder box

Say a thing the 46-kind library lacks ("Skippy, add a fire hydrant", "a boombox", "a birthday cake") and, when the bridge is set up, the holodeck asks
it about the noun (`POST /resolve`, the SkippyVerse resolver in `stage/skippyverse`). If the inventory has a **servable** card of that name with a
size we can trust, the card becomes a *virtual kind* in the live library (one variant: a box of the card's size, its mass, front unknown so
symmetric) and the request runs again through the normal pipeline, so layouts, overlap rules, "remove it", "undo", the per-player and server limits,
ownership and the server's own spawn rules all treat it like any other prop. What stands is an `holodeck_found` entity
(`addon/lua/entities/holodeck_found/`): a physics box made at runtime with `PhysicsInitBox`, **no model and no game content** (so it works on the cloud
server that lacks HL2/CS:S), the card's mass (respecting `holodeck_real_mass`; with 0 the engine's own), asleep once settled. Every client draws it
as a translucent `#00FF88` box with solid edges and a floating label `<name> - <source>, <licence>, size <basis>` within about 10 m. The label
strings, the box and the mass are NetworkVars on the entity.

- **Honesty, never a substitution.** Only an exact / alias / listed-synonym match for a licence-gated (CC0, CC-BY, PD-USGOV) found card with a
  trusted size is placed. "The closest I have is X", a typo reading, "not found" ("I don't have a dragon"), too big, size untrusted, size unknown
  ("I don't know how big that is") and **non-redistributable cards (Valve- and ShapeNet-derived, which includes the 126 curated models)** come back as
  Skippy's sentence and place nothing. Where the parser only *guessed* ("fire" is the library's approximate name for a wood stove, with "hydrant"
  left over), the bridge is asked about the whole phrase in case the inventory has the thing itself; if it has not, the parser's own stated guess
  stands unchanged and the bridge's "no" is not spoken.
- **A card with no size** (Gazebo Fuel and NASA listings carry none) may cost ONE download: the bridge fetches and measures that asset (rung 2 of the
  resolver) inside a budget of 40 MB in all, https only, the four download hosts only, a per-request timeout and a 22 s deadline, one download at a
  time (a second gets 503), and only for a card the game could then place (never for a "closest" or typo reading). `SKIPPY_FOUND_FETCH=0` turns
  downloads off: the answer is then "I know of X, but I don't know how big that is". Downloaded assets are kept in `stage/skippyverse/cache/assets/`
  (gitignored). The box is the measured size; the real shape (Poly Haven cards) follows it, see "The real shape of a found object" below.
- **The game side never blocks and never trusts.** The call is asynchronous (the bridge's per-minute and per-player allowances, the in-flight cap and
  `holodeck_bridge_inflight` apply; one lookup per player at a time; a definite "no" is remembered for 5 minutes), the answer is re-validated in Lua
  (finite sizes between 1 and 2000 HU, one variant of the right entity class, no word the library already knows) and the box is rebuilt from the
  card's w, d, h alone. At most 3 nouns are asked per sentence and 64 found kinds are kept per session (the oldest that nothing stands as makes way).
  Found kinds live in `Holodeck.found_kinds`, are re-applied by `holodeck_reload`, and a standing thing whose kind has left stays *legacy* (movable,
  removable). `holodeck_found 0` switches the whole thing off; `holodeck_status` counts the kinds.
- **The real shape.** `ENT:SetVisual(kind, data)` takes `"box"` or `"mesh"` (`{ id = n }`); see "The real shape of a found object" below.
- **Tests.** `sigma_ground/radiance/tests/test_gmod_found.py` (58): the real bridge function and HTTP handler against a synthetic inventory (no
  network: a guard fails any non-loopback connection), the real `holodeck.lua` + entity against `gmod_mock.lua` (now with scripted-entity support:
  `MOCK.load_entity`), fed the bridge's real answers; footprint and mass vs the card, label strings, undo, limits, rules, reload, legacy, the
  refusals, rate limits, the download budget, and the client drawing. `live.py` grades found objects against the CARD, not the planner:
  `python stage/gmod/holodeck/live.py "add a fire hydrant" "add a birthday cake" --start-bridge --pos 64,0`.
- **Verified in the real game (2026-10-11).** The box, its label and the translucent fill render in the real client (a modelless `base_anim` is
  networked and drawn fine); see `data/shots/` (`boombox_box.jpg` and friends). `barrel`, `chair`, `couch` and the other 46 kinds stay the curated
  Valve models (the parser knows them, so the bridge is never asked); "dining chair" is a chair.

### The real shape of a found object (mesh swap, prototype)

*(2026-10-11: the swap now also covers NASA glb models and Gazebo Fuel models, see "Other sources" at the end of this section.)*

A found object starts as the labelled box (the ghost). When its card is a Poly Haven model, the real triangles follow and are drawn **in place of the
box, at the same pose, with no re-plan**. No model file is installed on any client: the geometry travels over the network.

```
 /resolve says  found.mesh = <16 hex key>  ->  the entity spawns as a box  ->  H.Mesh.attach(): ONE POST /mesh {card, key} per model
 bridge: acquire (cache or one bounded download) -> meshpack 1500 tris -> LZMA -> base64 in JSON
 server (meshnet.lua): checks it -> sends 16 KB chunks, one per 0.1 s, to everyone -> ent:SetVisual("mesh", {id})  (NetworkVar MeshId)
 client (meshclient.lua): reassembles -> verifies -> util.Decompress -> SVM1 decode (svm1.lua) -> IMesh -> ENT:Draw
```

- **Bridge to server:** `POST /mesh {"card", "key", "who"}` returns `{"ok", "key", "format": "svm1-lzma", "bytes", "crc32", "tris", "verts", "bbox_m",
  "b64"}`. Base64 inside JSON because the engine's HTTP callback hands the body over as a string, and a JSON field reaches it whole whatever the
  bytes are. The key is a hash of the card id and the recipe; the bridge only answers for a card it offered (`mesh_offer`: servable, `found`, source
  Poly Haven); 12 a minute and 4 per caller; one download at a time; one pack at a time; `SKIPPY_FOUND_FETCH=0` still means no download (the box stays).
- **Server to clients:** net message `holodeck_mesh`, one per chunk: `UInt8 mesh number (1..64) | UInt8 chunk (1-based) | UInt8 chunk count | UInt20
  total bytes | UInt32 CRC-32 of the whole payload | UInt16 bytes here | Data`. The payload is the bridge's LZMA bytes untouched. Sent **once per model**
  (not per entity) to everyone connected when it is ready. A client that joins later, or missed a chunk, asks with `holodeck_mesh_want` (`UInt8`
  number, 0 = everything) on loading and whenever it draws an entity whose mesh it lacks (at most once per 8 s per mesh); the server answers that
  player alone and rations asks to 12 a minute per player. Entities carry the number in a NetworkVar (`MeshId`, with `VisualKind`), so a late client
  learns it with the entity.
- **Limits.** Payload <= 256 KB (the bridge, the server's check and the client each refuse more), 16 KB chunks (<= 16 per payload), <= 64 models per
  session, <= 4,000 triangles and 12,000 vertices on decode, an LZMA header that declares its inflated size (<= 1 MB: a "bomb" or an unknown size is
  refused before anything is inflated), 8 sends queued, a half-received mesh dropped after 30 s, a failed model asked for again after 60 s (three tries
  at most). Only the server can reach a client's `net.Receive`; the mesh receiver has no server-side twin, so a client cannot inject one.
  `holodeck_mesh 0` (server convar) = boxes only (`Holodeck.Mesh.resume()` asks for what waited); `holodeck_status` counts ready/waiting/failed shapes.
- **Drawing.** Built once per client from flat per-triangle colours with computed normals and a fixed light baked into the colours (the material is
  unlit `UnlitGeneric` with `$vertexcolor`: no scene lighting, no shadows). Same pose as the entity (`Matrix` Translate + Rotate; a `SetTranslation` +
  `SetAngles` pair misplaced it in the real game whenever the yaw was not 0). The payload's frame has x = the card's width and y = its depth, the
  entity's box has them the other way round, so the mesh is turned a quarter circle (the authored front lands on +x, where the planner puts a front)
  and put on the box's footprint centre and floor. Triangle order is flipped for the engine (Source's front faces are clockwise); the normals follow.
  The label stays. `holodeck_found_outline 1` (client convar) keeps a faint outline of the card's box around the shape; off by default.
- **Failure takes the placeholder down** (changed with the transporter effect, below). No answer, a bad answer, a checksum or length that does not
  match, a payload that does not decode, a bridge that stays busy or silent past `holodeck_found_timeout`: the beam collapses red, the placeholder is
  removed and Skippy says why. (A model whose three tries are used up, or `holodeck_mesh 0`, leaves plain boxes: nothing waits, so there is no beam.)
  A client that never gets the chunks, or whose `util.Decompress` returns nothing, keeps drawing the box on that client (the server cannot know).
- **Collision is still the card's box.** Only the drawing changes. The mesh is NOT a collision hull: things rest on the box, so a chair's open back is
  solid to the physics.
- **LZMA verdict (real game, this machine).** `util.Decompress` reads exactly the legacy `.lzma` stream with a known size in its header, which is also
  what `util.Compress` writes; Python's `lzma` module produces it after patching the size field (`meshpack.lzma_alone`). `.xz`, an unknown size (-1), a
  stream without the 13-byte header and a Source `LZMA` header are all refused (`util.Decompress` returns nil). A 6.5 KB boombox payload went
  bridge, server, client, passed the CRC-32 on both sides, inflated and drew. No fallback (raw bytes, RLE) was needed.
- **Live proof.** `python stage/gmod/holodeck/shots.py "add a boombox" --tag boombox --start-bridge --two-phase --outline --yaws 0,90,180,270 --dist 1.6 --top`
  (the real game for about 80 s; the bridge listens on the Tailscale address because a game that hosts its own world cannot use localhost; no language
  model is started) photographs the box (`holodeck_mesh 0`), then lets the shape be fetched and photographs the same views again. `--synthetic` serves
  a made-up six-colour block (`synthetic_found.py`: +x red, -x cyan, top green, +z blue, -z yellow) instead, which is how colour, axes and winding
  were read off the pictures. Pictures: `data/shots/{boombox,boombox_close,pillar}_{box,mesh}.jpg`.
- **What the pictures show (honestly).** Poly Haven boombox, 1,320 triangles, 6,501 bytes: the box is replaced in place by a recognisable boombox
  (handle, speaker grilles, antenna) filling the card's footprint, upright and on the floor, with the outline box around it when asked. It is
  **white-grey, not coloured**: Poly Haven materials are textured and carry no base colour, so every face is the default grey, shaded by the baked
  light. The six-colour block shows that colours come through (from each of four camera yaws the face seen was the one the mapping predicts, and the
  top is green), that the faces are not inside out (the near faces are the ones drawn), and that nothing z-fights (the box is not drawn under the
  mesh; only the optional outline is). Defects: the baked light is crude (some faces look dark), the 1,500-triangle clustering rounds off thin parts
  (the antenna is a thin line), and the fade-in (`render.SetBlend` with the entity's colour alpha) was not looked at separately.
- **Not done / not verified.** Textures; a collision hull; levels of detail; models over 4,000 triangles; NASA and Fuel models; client-side caching
  between sessions (a reconnect asks again; the server keeps the bytes for the session only); scene lighting and shadows. Offline tests only
  (`sigma_ground/radiance/tests/test_gmod_found_mesh.py`, 63): chunk reassembly out of order and twice, a late joiner, rationing, hostile and malformed
  answers, the Lua decoder against Python's unpacker on the real chair. **Not tried in the real game:** a dedicated server with a remote client (the
  live net path was the listen server's own loopback, so a late joiner's `holodeck_mesh_want` and a payload of several chunks have only run in the
  mock), several different models at once, a weak connection.

**Other sources (2026-10-11, live-tested with `found_live.py`).** The bridge offers a mesh only when it can really read the geometry, so a promise
never turns into a failure: **NASA** glb models once downloaded (the /resolve that sizes the card downloads them) and **not meshopt-compressed**, and Draco-compressed ones too when the optional decoder is installed (`pip install DracoPy`, Apache-2.0; `meshpack.gltf_readable`; without it, or with a corrupt Draco buffer, the model stays a labelled box). Live: the Apollo Lunar Module (Draco, 97,588 triangles) arrives as 1,445 triangles, 8,030 bytes, in-game, packed at the card's size (file units
taken as metres while the size basis is guessed; scaled to the published length when `nasa_real_sizes.json` has a verified entry); **Gazebo Fuel**
models whose SDF visuals are box/sphere/cylinder or Collada/OBJ meshes (`meshpack.sdf_drawable`; scale and poses applied, colours from the visual's
diffuse else grey, no textures; STL stays a box). Poly Haven is unchanged. Live driver: `python stage/gmod/holodeck/found_live.py <tag> "add a mailbox" --purge <card id> [--slow-send]`
(real bridge, real download, per-stage timings from the bridge, a time-lapse, six final views, the engine's own size/pose of what stands, a frame-cost
comparison; pictures `data/shots/<tag>_*.jpg`, `<tag>_sheet.jpg`, `<tag>_report.json`).

### The transporter effect: a materialise whose coherence is the pipeline's progress

While a found object's real shape is on its way, the placeholder box is replaced by a Star Trek style beam: a column of sparkles in the card's
footprint box. Its job is **feedback about progress** (players need to tell whether we gave up or not): the beam is *sparse, wide and flickery* at 0, and
*condenses* as the pipeline really advances (noise falls, the sparkles are pulled onto the box's edges and faces, the outline sharpens, a scan band
sweeps the column, the silhouette fills in). When the mesh arrives it is revealed floor-up behind a bright band and the beam fades (0.8 s). A beam that
has heard nothing for a while **destabilises** (jitter, dimming, drifting to amber); one that is given up on **collapses red** and the object is removed.
Procedural only: billboard squares of an additive `color/white` material, quads, `DrawWireframeBox`; no model, material file or sound from the game.

- **Files.** `mentat/fx/transporter_model.lua` (pure: stages, progress bands, the coherence function; tested offline), `mentat/fx/transporter.lua`
  (client drawing), `mentat/holodeck/foundprog.lua` (server: progress events, heartbeat, giving up, saying why), `mentat/fx/beam_net.lua` (Skippy's
  beams), the entity (`holodeck_found/shared.lua` NetworkVars and `FxSet`, `cl_init.lua` drawing), `meshnet.lua`/`meshclient.lua` (events and per-chunk
  receive fraction), the bridge's `POST /progress`.
- **Progress comes from real events only** (published as NetworkVars `FxStage`, `FxProgress`, `FxBeat` on the entity, so a late joiner sees the same).
  Stage bands on a 0..1 scale: resolving 0-0.08, fetching 0.08-0.50, measuring 0.50-0.58, packing 0.58-0.74, sending 0.74-0.92, decoding 0.92-1.
  *fetching* = bytes downloaded / bytes expected, from the bridge's `/progress` (polled once a second while pending; the bridge counts every 256 KB chunk
  of the real download, and learns the expected total from Poly Haven's file list: glTF + .bin, never textures; an unknown total gives `1 - exp(-bytes/1.5 MB)`);
  *packing* has no inner progress (one blocking call), so the bridge's "I am in packing" is a heartbeat, honoured for 25 s; *sending* = chunks sent / chunks
  (and, on each client, the bytes it has really received); *decoding*, then *done*, after the last chunk. Progress only rises; a stage never goes back.
  `resolving` and `measuring` exist in the table but are **not visible**: the /resolve (and the size measurement it may trigger) happens before the object
  exists, so the entity is born in *fetching* at 0.08.
- **Heartbeat.** Every event that adds progress (or a living packing worker) stamps `FxBeat`. Clients derive "silent for N seconds" from it and drift to
  amber from about a third of `holodeck_found_stall` (default 6 s, replicated). "Waiting in line" (the bridge downloads one thing at a time and answers
  503; the game asks again every 2 s) is silence too, so the beam shows the wait honestly. `holodeck_found_timeout` (default 75 s, counted from the request;
  the bridge's own /mesh deadline is 44 s) is the hard limit.
- **Failure** (`FxStage = failed`, from the server, immediately for a known error): over 1.5 s the column collapses from the top, flickers wildly, goes
  amber then red, the sparkles scatter and fade; then the placeholder is removed (1.7 s) and Skippy says why: "I could not bring in the cannon: it took too
  long and I lost the lock. I took the placeholder down." (also "the bridge could not fetch it", "the model file was damaged on the way", "the model is too
  big for me to bring in", "the download slot was busy", ...). One sentence per owner and model ("3 of the garden gnome").
- **Reveal technique.** `render.PushCustomClipPlane` with the entity's local up flipped (it keeps what is below the band) around `CM.draw`; a bright
  quad (the band) rides the clip height. It worked in the real game first time. A per-vertex alpha ramp was not needed.
- **Cost bounds.** At most 220 sparkles per beam (full detail inside 700 HU, a quarter of them by 2,500 HU, nothing beyond 3,500), the entity's render
  bounds do the off-screen culling, no allocation per sparkle. `holodeck_fx 0` (client) = the old translucent box and no beam; the avatar's body is then
  never held back.
- **Skippy.** `SkippyAvatarArrive` / `SkippyAvatarLeave` (avatar.lua, untouched) become a `mentat_fx_beam` message; the client draws a 1.0 s beam-in at the arrival
  point (following him while it plays) with his body hidden until it completes (`avatar_client.lua`: one line in `draw()`), and a 0.8 s beam-out at the old place.
- **Convars.** `holodeck_found_timeout`, `holodeck_found_stall` (server; the second is replicated), `holodeck_fx` (client). Live-test knobs (not convars):
  `Holodeck.Mesh.CHUNK` / `.PUMP` (a slower send, so progress inside *sending* can be photographed).
- **Bridge.** `POST /progress {"key"}` -> `{"ok", "stage": idle|queued|fetching|packing|done|failed, "bytes", "total", "age_s", "error"?}`: token-guarded, validated
  (16 hex), budgeted (300 a minute, 150 per caller); `/mesh` answers exactly as before (a game that never polls loses nothing).
- **Offline tests.** `sigma_ground/radiance/tests/test_gmod_transporter.py`: the coherence function condenses with progress, destabilises with a stall and
  collapses on failure (mutating it to a constant fails them); events -> NetworkVars, monotonic, heartbeat, timeout -> failure -> removal -> sentence, a late
  joiner, the bridge's counters and endpoint, the client drawing against fakes of the render library (sparkles inside the box rise with progress, amber on
  silence, red collapse, the reveal plane moves up, LOD, bound), the avatar beams. `test_gmod_found_mesh.py` was changed where it said "failure keeps the box".
- **Live proof (real game, this machine).** `python stage/gmod/holodeck/transporter_shots.py success|failure|avatar|perf` (the real bridge and real packing; only the
  *download* is replaced by a throttled delivery of cached or made-up bytes so each stage lasts long enough to photograph; the progress counters, `/progress`, chunks,
  heartbeat and effect are the production code). Pictures (gitignored): `data/shots/transporter_<scenario>_*.jpg` and `..._sheet.jpg`.
  *success* (boombox): 8 % sparse scattered sparkles; 25 % a faint box outline forming; 49 % tighter, the scan band visible; 71-80 % a dense box with the band
  sweeping; at arrival the grey boombox is revealed bottom-up inside the fading beam; done. "More coherent over time" is clearly visible.
  *failure*: a download that breaks (hydrant) flashes amber, collapses red, scatters and is gone in 1.7 s with a red "transport failed" label and
  Skippy's sentence in chat; a download that goes quiet (cannon) first goes amber with "no news for 6 s", then fails the same way at the timeout.
  *avatar*: the beam-out dissolves in 0.8 s; the beam-in condenses over 1 s without his body, which appears when it ends.
  *perf* (4 beams, about 160-250 sparkles in view): whole frame 6.85-6.87 ms (display-limited at 144 Hz) with beams, 6.87 with plain boxes, 6.90 with nothing: no
  measurable difference; the effect's own CPU time is 0.6-0.9 ms per frame for those four beams (about 0.2 ms each).
- **Not verified / known limits.** Late joiners in the real game (offline only: the NetworkVars are copied between mocks); a dedicated server with a remote
  client (the stall is read against `CurTime()`, which the client keeps close to the server's; a large offset would shift the amber by that much); a client
  whose decode fails keeps its box and never tells the server; the plain box still turns solid after 0.6 s as before (the card's box is the contract) rather
  than staying passable for the whole materialise; the stall colours and the band's strength were judged by eye at one distance; the sparkle size is a fixed
  fraction of the card's size (a very large object has large squares); the resolve step is invisible (above).

### Failure wisp: the reason, left hanging where the transport failed

When a found-object transport FAILS, Skippy's honest sentence is also left behind as a **wisp**: a small soft light at about eye height at the failed spot (after the
red collapse has played), carrying that same sentence. It is the silent, persistent version of what Skippy said and typed. The owner reads it by walking up and looking at
it; it notices, dissolves, and goes. (Aaron: "instead of a post-it note, a floating whisp that is dismissed after a few moments of being read/seen up close".)

- **What leaves one.** The resolver's definite no for a noun the library lacks ("I don't have a gargoyle, and I can't make new things right now."; the wisp is made
  where the thing would have stood: an arm's length in front of the asker, at eye height, kept out of walls), the game's own refusals of a bad card, a failed download,
  decode or checksum, and the hard timeout (at the failed object's place, the same `FP.sentence` Skippy says). Not a parser guess that stands, not a bridge that is down (no sentence).
  Colour by class, from the resolver's `status`: amber `#FFAA00` a real failure (not found, fetch/decode failure, timeout); cyan `#00CCFF` a neutral suggestion ("I read 'boombx' as
  'boombox'", ambiguous); red `#FF4444` a refusal (too big for here, "a model I may not share", the game rejecting a card's numbers). The text is the sentence, never made up on the
  client, cleaned and cut to 160 characters (at a word, ending in "..."; the whole sentence was also said in chat); each viewer's own chat filter is applied to it when drawn.
- **Reading.** The SERVER runs the state machine (it sees the owner): `appear` (1.2 s fade-in) -> `idle` <-> `reading` -> `dissolving` (1.0 s) -> gone. The owner is "reading" while within
  4 m, looking within about 28 degrees of it and with no wall between (a `MASK_SOLID_BRUSHONLY` trace). It brightens and the text types out (about 0.9 s) on a black plate that faces the reader
  (3D2D; scaled up a little with distance so 2-3 m stays readable). Reading time only counts while it is being looked at: dwell = max(2.5 s, characters / 18 per second) (63 characters: 3.5 s).
  A glance away shorter than 0.4 s does not fold the text up. When the dwell is full the text turns to sparks, the light flares and goes. Other players see only a dim light (no text) and cannot dismiss it.
- **Never read.** After `holodeck_wisp_life` (120 s) it dims over 2.5 s and goes by itself. A new wisp within 1.5 m of an old one of the same owner replaces it; at most `holodeck_wisp_max` (3) per player
  (the oldest goes first, gracefully) and 24 on the server. A wisp is **not a placed thing**: no registry entry, so undo / clear / the per-player and server limits never count it and "clear everything"
  leaves it alone; the owner leaving removes theirs.
- **Commands.** Say "clear the wisps" / "clean up the notes" / "dismiss the wisps" (only the wisps; "clear the couches" is still the holodeck's), or the console `mentat_wisps_clear`
  (the server console clears everyone's).
- **Convars.** `holodeck_wisps` (1), `holodeck_wisp_life` (120), `holodeck_wisp_max` (3); all server, archived.
- **Files.** `mentat/fx/wisp.lua` (pure: classes and colours, dwell, state machine, who can read, replacement/cap, text cap and wrap, the look of each state), `entities/holodeck_wisp/{shared,init,cl_init}.lua`
  (no model, no physics, NetworkVars carry text/class/state/owner, so a late joiner sees the same), `mentat/holodeck/wisps.lua` (server glue: drop, cap, tick every 0.1 s, clear, the command),
  hooks in `holodeck.lua` (`F.ask` classes, the resolve callback) and `foundprog.lua` (`give_up_entities`). Procedural only: a 64x64 soft-falloff texture painted once into a render target, additive
  sprites (halo, body, hot core, 7 embers rising slowly, slow bob and pulse), no model, material or sound from the game, so it works on the cloud server. Client cost: about 10 sprites idle, 20-30 with the text and sparks;
  nothing is drawn beyond 25 m or behind the viewer.
- **Offline tests.** `sigma_ground/radiance/tests/test_gmod_wisp.py` (80): the pure module (palette, dwell scaling, the state machine, a glance away, expiry, who can read, replacement and cap, text cap/UTF-8/wrap),
  the server (each failure path leaves a wisp with the SAME sentence, the right class, place and timing; owner-only reading; not in the registry; clear command and its non-matches; per-player and server caps, convars;
  owner leaving; NetworkVars) and the client against fakes of the render library (owner-only text, the chat filter, colour by class, far/behind draws nothing, bounded sprites, no game content in the source).
- **Live (real game, this machine; the real bridge and resolver, a made-up download for the fetch failures).** `python stage/gmod/holodeck/wisp_shots.py lapse|colours|unread|fetch|perf`; pictures (gitignored)
  `stage/gmod/holodeck/data/shots/wisp_<scenario>_*.jpg` and `_sheet.jpg`. *lapse*: the wisp fades in at eye height, drifts with a few embers; from 5.7 m it is a small glow; from 2.7 m the text types out on a
  dark plate, the light brightens, after 3.5 s of looking the text fades into sparks, the light flares and is gone. *colours*: amber (not found), cyan (a typo read as the closest), red (a model I may not
  share), clearly distinguishable in daylight. *unread*: dims and goes with `holodeck_wisp_life 7`. *fetch*: after the red collapse an amber wisp carries "the bridge could not fetch it" / "it took too long and I lost the lock".
  **Legibility verdict:** three lines of bold white on a dark plate read easily at 2-3 m at the game's native resolution; in a 960-pixel-wide contact sheet they are small but still readable. The additive light is
  weak against a bright daylight sky (the map's horizon sits at eye height, so the wisp is often seen against the sky) but visible; it will be stronger indoors or at dusk. "It noticed me" comes across: the
  light brightens while it is read, and the dissolve starts the moment the dwell is full. Frame cost: the wisps' own CPU time is about 0.16 ms per frame for three wisps (23-30 sprites); the whole-frame
  timings in the perf run (17-42 ms) were dominated by other load on the machine and are not usable for a comparison.
- **Not verified / known limits.** Several players (owner-only reading and dimness are tested offline only; the live runs were one player on a listen server); a dedicated server with a remote client (the owner
  is a NetworkVar entity, the state time is read against `CurTime()`); a wisp seen through glass or a prop (only the world's brushes block reading); the font ("Tahoma") on a Linux client; indoor lighting and a
  wisp in a dark room; a player who is looking at the wisp when it appears reads it at once (it appears 2 m in front of the asker, which is the point, but a very short sentence dissolves after 2.5 s);
  reading by voice/chat of the text is not a thing: only the view counts.

## Playing with your voice (one command)

```bash
bash stage/gmod/skippy_play.sh                 # the bridge, your microphone and the game, started in order and wired together
bash stage/gmod/skippy_play.sh --selftest-voice "add two couches" --stop-after     # prove the voice path with no human
```

`play.py` checks SkippyTel's Whisper, Ollama, ffmpeg, your microphone and your Tailscale address, starts the bridge behind a fresh token,
starts the game as a small listen server, connects the game to the bridge (`mentat/tools/skippy_play.lua`, run through the same
`data/mentat/tool.txt` mechanism as every other tool) and starts the microphone bridge. Then **hold X and say what you want** (no wake
word; `--key` or the `skippy_key` setting picks another key), or hold V and start with "Skippy" (or one of your own names for him). The key is read by the game itself,
because the server may not bind keys for a player. The self-test speaks a request with Windows speech to the microphone bridge's own
recogniser (the real Whisper), lets the game pick it up and checks that things were built and Skippy's answer played in the game;
only the physical microphone and the key press are left to a human. Sentences longer than about 4 seconds are compressed before they
are uploaded (SkippyTel refuses bodies over 128 KB).

## Proximity voice

Native to the server (`mentat/voice.lua`), installed as the gamemode's own `PlayerCanHearPlayersVoice` wrapped around the rule it
already had (so a gag or mute addon always wins): a normal voice carries about 21 m and fades smoothly over its last 40 %, crouching
whispers (about 7 m), holding `+mentat_shout` shouts (about 46 m; the server remembers a shout for 1.5 s at a time), walls muffle.
`mentat_voice_mode` `proximity | global | off` (the clients are told, so global and off do not fade), `mentat_voice_range`,
`mentat_voice_wall`. Bind a key: `bind v +voicerecord` is the game's own; `bind x +mentat_shout`.

## Where the data came from

**Masses.** The library's masses are real-world masses from a sourced reference table, scaled to each model's size and held inside the
family's sourced range (`fidelity.py`, `data/fidelity.json`, report in `misc/HOLODECK_FIDELITY.md`); the holodeck gives each spawned
prop that mass. Six models keep the engine's gameplay mass because no source describes them (catalog `mass_status` "unaudited"); the
engine's original mass stays in the catalog as `engine_mass`. A separate audit (`audit_physics.py`, report in
`misc/HOLODECK_PHYSICS_AUDIT.md`) sets the ENGINE's mass, size and collision body beside what Deckard compiles for the same name; it
reports, it does not correct. Deckard's own mass and size defects found by that audit were fixed in `sigma_ground/deckard` (aliases match
only as the head noun, so "a desk lamp" is a lamp and compounds keep their own head noun; each part is voxelised on its own grid, so thin
boards, slender members, glazing and foliage get a mass that does not move with the grid resolution; masses are held to the sourced ranges
in `fidelity.py`, and the mass-sanity bands in `tests/test_deckard_mass_sanity.py` come from those ranges, with three strict xfails for
the cases still outside them; a hollow-shell occupancy rule that depends on thickness and material); the one left on purpose is the axis-orientation change, which moves
pinned scenes and needs its own change.
