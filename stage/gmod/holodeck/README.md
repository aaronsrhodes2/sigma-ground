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
| `skippy_bridge.py` | next to the game server | `/plan` (a local model turns a vague wish into a validated request), `/reply` (Skippy's persona says what was done), `/speak` (an mp3), `/quip` (an unprompted line with its clip, under a daily budget: "Skippy alive" below), `/turn` (a chat line or question answered by the frontier model with a local fallback: "Solid turns" below). Fixed actions only, behind a token. |
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

**Other scripts.** A name may be written in Cyrillic, Greek, Armenian, Georgian, Hebrew, Arabic, Devanagari/Bengali/Tamil, Thai, Hangul, kana or Han
(the `LETTERS` table in `wakewords.lua`, mirrored in `wakewords.py`; any other script is refused with the reason, and so are emoji). The 2 to 24
limit counts characters, not bytes (the Lua side decodes UTF-8 by hand and refuses invalid UTF-8). Such a name is matched **exactly** after
case-folding (Greek final sigma included) and NFC (Cyrillic/Greek accents, Hangul jamo, kana dakuten); there is no fuzzy matching outside Latin,
and a homoglyph is never a "slip" of a Latin name. Known limits: the name must end at a space or punctuation (Chinese or Japanese text with no
space after the name is one unbroken word and does not wake him: `小明，加沙发` works, `小明加沙发` does not), the parser's collision list is English,
and a display name is shown as typed (only Latin names are capitalised).

**Privacy in the log.** The server console never shows a name or a name command: replies about the list are private `Tell`s that are not
printed, and the only line is `wake list changed for <steamid64>`.

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
- **Drawing (SVM1, the original; SVM3 supersedes it, see "The look of found objects" below).** Built once per client from flat per-triangle colours with computed normals and a fixed light baked into the colours (the material is
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
- **Not done / not verified.** (Textures: done since, as colours, see the look pass below.) A collision hull; levels of detail; models over 4,000 triangles; NASA and Fuel models; client-side caching
  between sessions (done since: see "Object caches" below); scene lighting and shadows. Offline tests only
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

### The look of found objects (look pass 1, 2026-10-12)

A found object used to arrive as a flat, unlit grey-white shape (SVM1: one base-colour factor per triangle, so every textured model lost all its
colour; a fixed light baked into the vertex colours). It now arrives as **SVM3** and is drawn with the same lights as Skippy's body.

```
 bridge: acquire (+ the 1k diffuse / metal-rough images the glTF uses) -> meshpack fmt=svm3, ~3000 tris -> colour sampled from the texture at each
         triangle's UV, material cue from metal/rough/emissive/alpha -> a palette of <= 24 colour regions -> smooth normals, creases split -> LZMA
 client: SVM1.decode3 -> one IMesh per palette entry -> drawn with a VertexLitGeneric material per entry (own colour, own gloss), lit by mentat/lit.lua
```

- **Colour without shipping a texture.** At pack time (Pillow, optional) the base-colour texture (a glTF `baseColorTexture` x `baseColorFactor`; a Collada
  diffuse map or colour per material; a Gazebo material script's loose diffuse image; NASA's embedded webp/png) is box-reduced to 256 px and sampled at the
  UV centroid of every original triangle (four points for a big one), in linear light. Clustering then gives each surviving face the DOMINANT colour among the
  triangles around it (binned, area-weighted; a plain mean turned a red cake purple). **What is lost:** every fine detail of the texture (labels, grain,
  logos, speaker cloth weave): each triangle is ONE flat colour, and the palette snaps those to at most 24 regions. Big colour areas, tonal gradients
  and the material split survive. No texture ever travels to a client; without Pillow a textured model keeps its base-colour factor (the old look).
- **Materials.** `metallicFactor`/`roughnessFactor` (and the metal-rough map, G = rough, B = metal) become a gloss and a metal level per palette entry:
  a kettle's polished metal glints (phong boost up to ~4x a plain plastic's), a plastic boombox has a soft highlight, a croissant is matte; a metal tints its
  highlight and has less diffuse. Emissive glows (flat, self-lit); `alphaMode` BLEND entries are translucent and MASK cut-outs lose their holes. Collada
  specular/shininess give the gloss. Palette entries are material classes first, colours second, so a metal red and a plain red stay two entries.
- **Draw.** `mentat/lit.lua` (extracted from `avatar/body_client.lua`, which now uses it) sets the engine's lights per object: the map's ambient and sun at the
  object, a warm key toward the viewer's upper left, a rim from behind, a fill, Phong gloss; one mesh per palette entry because a lit mesh ignores vertex
  colours (found for Skippy's body in cb4f52e; a vertex-colour VertexLitGeneric was not tried again here). Convars shared with the avatar: `mentat_avatar_amb / key / rim / spec`. `holodeck_found_lit 0` draws
  the old flat way; an old client (no lighting calls) bakes the shade from the normals into one colour per vertex. A soft **blob shadow** (Skippy's,
  sized to the object's footprint) sits under every object. The materialise fade uses translucent variants of the materials.
- **Format and compatibility.** `svm3-lzma` (bridge `MESH_FORMAT`); the server accepts `svm1-lzma` too and SVM1 payloads still decode and draw exactly as
  before. `SKIPPY_MESH_LEGACY=1` makes the bridge send the original SVM1 (1500 tris, flat colours): that is the "before" of the pictures below.
  SVM3 after decompression: `'SVM3' | u16 nv | u16 nt | f32 min[3] | f32 max[3] | u8 npal | npal x (u8 r g b spec metal emit alpha) | u16 verts[nv][3] |
  u16 idx[nt][3] | u8 palette index[nt] | i8 normals[nv][3]`. Limits on decode: 6000 triangles, 12000 vertices, 32 palette entries.
- **Size and cost** (this machine): typical payload 17 to 29 KB at 3000 triangles (kettle 25 KB, boombox 25 KB, cannon 29 KB, croissant 18 KB, hydrant 26 KB,
  mailbox 1 KB, cake 20 KB, Gemini 24 KB, Lunar Module 26 KB; before 6 to 8 KB at 1500; hard cap still 256 KB), so two chunks of 16 KB instead of one: about
  0.1 s more on the wire. Packing in pure Python: 0.3 to 7 s (the Draco Lunar Module 6.7 s, the hydrant 6.6 s), the cake 3.8 s (it was 14 s: see below).
  CPU cost of drawing one object in the real game (`holodeck_mesh_bench`: 200 draws inside the translucent pass, SysTime): **0.04 to 0.12 ms** for
  6 to 24 lit regions (kettle 0.10 ms, boombox 0.12, cannon 0.10, Gemini 0.08, Lunar Module 0.08, mailbox 0.05, croissant 0.04). The frame rate is capped by the
  display, so the GPU side could not be separated; at these numbers 100 objects would cost about 10 ms of CPU. The textures kept on disk for the nine models
  are about 10 MB in all (limit for this pass: 40 MB).
- **Fixed on the way.** (1) A Gazebo SDF that draws one `<submesh>` per `<visual>` (the birthday cake: 11 visuals of one 377,000-triangle Collada)
  drew the WHOLE file once per visual; `meshpack` now honours `<submesh><name>` (42,858 triangles, 14 s to 0.6 s). (2) Poly Haven's file list can name
  another resolution's `.bin` for a glTF (`cannon_01`: the 8k `.bin`, 1,198,792 B, for a 1k glTF that declares 1,143,768 B): indices ran past the vertex list
  and the cannon never arrived ("transport failed"). The glTF's declared length now decides, and the sibling file in the glTF's own folder is used.
  (3) Clustering puts each vertex where its triangles' planes meet (quadric error), not at their centroid, so a box stays a box.
- **Textures are fetched now, but only these.** Poly Haven: the images the glTF uses as base-colour or metal-rough maps at 1k (normal maps never), each
  under 8 MB; Gazebo Fuel: the zip's images (and a material script's textures) under 8 MB; a model downloaded before this pass tops up once
  (`acquire.ensure_textures`, remembered in `acquired.json`). New downloads for the whole pass: see the report; well under 40 MB.
- **Pictures.** `python stage/gmod/holodeck/found_look.py --phase before|after [--only kettle ...]` summons the standard set one after another through the
  real bridge (hands off: the window stays behind, no mouse) and photographs five fixed cameras plus a top view and times the draw; `--sheet` builds
  `data/shots/found_look_before_after.jpg` (before on top, after below, same cameras) and `found_look_before_after_close.jpg`; per-model views are
  `data/shots/look_{before,after}_<tag>_<view>.jpg` (gitignored with the other shots).
- **Verified in the real game (2026-10-12), hands off.** All nine models of the standard set arrive coloured and lit (kettle brown enamel with a wooden handle, boombox dark plastic, cannon black iron on a brown carriage, croissant golden, mailbox black with a red flag, cake red with white frosting, Gemini white with a dark nose, Lunar Module gold and silver), draw cost 0.05 to 0.14 ms each (before: 0.002 to 0.003 ms), contact shadow visible from above. The hydrant could not be summoned in the legacy (before) run, so its before tile is empty. While testing, every request froze the game: the water module wraps `Holodeck.Handle` on top of the avatar's wrapper and the avatar then re-wrapped THAT, so the chain called itself (fixed in `avatar.lua`'s `wrap`, test in `test_gmod_avatar.py`). The look driver also takes the water layer off for its run. **Avatar hat:** checked against the kit and in-game (`data/shots/hatcheck_*.jpg`): not rotated; the hat is the kit's own near-square bicorn, as wide as the reference photo's from the front (about 1.6 x the head) and long front-to-back in profile.
- **Still weak.** The geometry is vertex-clustered to ~3000 triangles: thin parts (antenna, handles, spouts) break up and a hydrant-like model with many
  small features turns to mush (a real decimator would fix it). One flat colour per triangle: no texture detail. No environment reflections, so polished metal
  shows only its highlight. The blob shadow is a blob, not the object's shadow, and no cast shadows. Glass is translucent but unsorted. The lighting
  follows the viewer, so highlights move as you walk around (as Skippy's do).

### Object caches (2026-10-13): a model is packed once and downloaded once per player

Two caches, so a model that anyone has asked for before is cheap for everyone after.

**1. The packed-payload cache (PC, `stage/skippyverse/packcache.py`, wired into `POST /mesh`).** A first request costs download + measure + pack (about 3 s
for a small Poly Haven model, up to 33 s for a big one, pure Python); the finished SVM payload and the measured facts (size, triangles, vertices, palette,
source, licence, size basis) are kept, so every later request, from any player in any session, is a file read.
- *Where:* `stage/skippyverse/cache/packed/<key>.svmc` (gitignored; with a scratch `MESH_CACHE_DIR`, a test or a tool, it sits beside that directory instead).
  One file: `SVMC1` magic, a JSON header, the payload bytes. The header holds the key, `PACK_VERSION`, the format, the triangle target, a fingerprint of
  the source files, the payload length and its SHA-256, and the facts.
- *Key:* `sha256(card id | format | triangle target | meshpack.PACK_VERSION)`, 32 hex. **Bump `meshpack.PACK_VERSION` whenever the packer, `surface.py` or the SVM layout
  changes what a model packs to**: it is a new key, so old entries are never read again and age out. (The binary header inside the payload stays `SVM1`/`SVM3`: the
  game client reads it, so the version lives in the key and the entry header, not in the wire format.)
- *Source check:* a read compares the entry's fingerprint (SHA-256 of the downloaded files the packer reads; for NASA cards also the verified-size table) with the files
  on disk now; a mismatch deletes the entry and packs again. When the download was pruned there is nothing to compare and the entry is trusted (an asset id names an
  immutable upload).
- *Safety:* written atomically (temp file in the same folder, fsync, `os.replace`; stale temp files are swept); every read re-hashes the payload, and a damaged,
  truncated, wrong-key or wrong-version file is deleted and counted as a miss. Two simultaneous first requests for one model take a per-key lock: one downloads and
  packs, the other reads its entry.
- *Cap and eviction:* 500 MB (`SKIPPY_PACK_CACHE_MB`, `0` switches the cache off), least recently used first (a read touches the file); the entry just written is never the one evicted.
- *The beam stays truthful:* a hit reports the stage `cached` and then `done` (the `/progress` answer says `"cached": true`, bytes 0); it never pretends to have downloaded.
- *Look, clear:* `python stage/skippyverse/packcache.py` lists entries and size; `--clear` deletes them all (or delete the folder).

**2. The client cache between sessions (game client, `addon/lua/mentat/holodeck/meshcache.lua`).** A payload a client receives is kept in its own data folder; on the
next join the server offers instead of sending, and a client that has the bytes says so.
- *Where:* `data/mentat/meshcache/<first 32 hex of the payload's SHA-256>.dat` (the LZMA bytes exactly as received) and `index.json` (`{v, tick, entries:[{sha, bytes, crc, u}]}`, `u` = a use counter,
  no clock, no SteamID anywhere). Only `.dat` and `.json` are ever written; the folder is made first; a write is read back and compared; the index keeps its previous copy as `index.bak.txt`
  (the engine rules above; `mentat/persist.lua`). Orphan `.dat` files that the index does not list are ignored and never read.
- *Handshake (new net messages, backwards compatible):* a client says `holodeck_mesh_hello` once on joining. From then on the server, for each model, sends `holodeck_mesh_offer`
  (mesh number, CRC-32, size, SHA-256) instead of the chunks; the client answers `holodeck_mesh_have` (number, CRC-32, have/want). **have**: nothing is sent. **want** (or no answer in 3 s): the chunks are
  streamed to that client alone. A hello also asks for everything the server already has (late joiners get offers, not chunks). The server accepts an answer only for an offer outstanding for that player.
  A client that never says hello (an older one) is streamed to exactly as before; a new client on an older server cannot send the hello (the engine refuses an unpooled name), so it asks for everything outright, as before.
  Entities wait 1.5 s for the offer before the client asks for a download itself, and 10 s after answering "want" before asking again.
- *Verification:* a cached file is accepted only if its length, CRC-32 and SHA-256 equal what the server announced; otherwise it is deleted from the folder and the index, and the client answers want.
- *Cap and eviction:* 120 MB and 512 entries by default, least recently used first (`Cache.CAP_BYTES`). `holodeck_mesh_cache 0` keeps nothing and always asks for the stream; `holodeck_mesh_cache_clear` empties it; `holodeck_mesh_info` prints what it holds and its counters.
- *The 200 MB per-player budget (Aaron's decision):* the cache is part of it, so its own cap (120 MB) is below it by design; a payload is at most 256 KB. No code enforces a session-wide
  figure yet: the two caps (cache folder 120 MB, one payload 256 KB) are what is enforced.

**3. Prewarm.** `python stage/gmod/holodeck/prewarm.py` prints the plan (noun, card, MB, cached or not) and does nothing else; `--go` downloads and packs what is missing through the same code a player's
request uses (official hosts only, licence gate, 40 MB per asset, one at a time) and skips what is cached; `--refresh-sizes` re-reads the sources' own file listings (about 10 KB each, no model)
and rewrites `approx_mb`. The list is `prewarm_list.json` (tracked): nouns the game's own library lacks (sofa, dining table, potted plant, kettle, boombox, television, fire hydrant, lantern, pillow,
mailbox), about 21 MB as an upper bound, refused over 150 MB. A couch, lamp, table, chair or mug come from the game's library and need no cache. The list has not been run with `--go` for real (plan only).

**Measured live (2026-10-13, real game, listen server with a hidden window, no mouse; Poly Haven boombox, 2.6 MB download, 24.7 KB payload; `found_live.py` with `--purge-packed` / `--clear-client-cache`):**

| case | `/mesh` | request to `done` | bytes streamed to the client | handshake |
|---|---|---|---|---|
| cold (no asset, no PC entry, no client file) | 3.03 s (download 1.59 + measure/pack 1.37) | 4.65 s | 2 chunks, 24,696 B | offer + want |
| warm PC cache (client file cleared) | 0.02 s | 1.59 s | 2 chunks | offer + want |
| client cache (game restarted) | 0.12 s | 1.48 s | **0** | offer + have: 2 messages |

About 1.2 s of the warm figures is the beam's fixed "decoding" to "done" finish, not the pipeline. The cold run's measure/pack was 1.4 s for 10,168 triangles; a big model takes longer and is where the cache pays most.

**Not verified:** a remote (second machine) client, a dedicated server with a human client, a full disk or a data folder the engine refuses (the code reads back and falls back to "streamed"), the client cache at
its 120 MB cap in the real game (tested in the mock only), eviction at 500 MB on the PC beyond the mock-sized tests, and any case where the entity appears on a client long before its offer (the 1.5 s grace held once). `/resolve` itself is not cached: a
Fuel or NASA card with no size still downloads once to be sized (that download is kept in `cache/assets/`).

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

## Skippy alive (A7): remarks, body language, a notebook, and a budget so he never repeats himself

Aaron, 2026-10-13: "20 new live lines a day so we never hear the same thing, and it should keep notes about things done, or plumb logs, so it can make quips and
callbacks about things current". Everything below is built and tested offline (`test_gmod_director.py`, `test_gmod_quips.py`); the live pass and what only a human
can judge are at the end of the section. The character is original (nothing is quoted from the books): a grandiose, sardonic, secretly loyal beer can who calls you
"monkey", boasts absurdly, makes banana jokes, refuses to explain his "super-advanced AI stuff", and sings terrible opera. His insult always rides on top of real help.

| Piece | Where | What it does |
|---|---|---|
| `mentat/avatar/director.lua` | game server | watches what happens, keeps the notebook, decides when to speak, asks the bridge for the line, shows and sends it, replicates the body language |
| `mentat/avatar/director_logic.lua` | pure | the rules: chattiness, cooldowns, never over a human or himself, yielding, quiet hours, interruption, the commands |
| `mentat/avatar/anim.lua` | pure, shared | the layered, bounded animation state machine (cues, mood style, voice chatter, idle breathing) |
| `mentat/avatar/notes.lua`, `journal.lua` | pure / server files | the notebook (`data/mentat/journal/<steamid64>.txt`) and the summariser that turns it into a handful of anonymous notes |
| `skippy_quips.py` + `data/skippy_lines.json` | the bridge | the line engine: 183 seed lines, 12 callback templates, the day's budget, the no-repeat history, the private clip cache, the model interface and its validation; the `POST /quip` route |
| `avatar_client.lua`, `avatar/body_client.lua` | game client | turn the replicated state into motion; tell the server when their player talks |

### When he speaks (the director)

He reacts to: **arrival** (a hat tip at once, then a fresh line instead of the canned "Here I am."; the canned line is the fallback if no line comes), a
**beam** (a found object's shape arrived: a flourish even when he has nothing to say; or failed), a **failed or refused lookup** (the bridge's `/resolve` answers are
plumbed in through the hook `HolodeckResolve`), a **build** (he turns to look at the thing just placed and points at it), a **big pile** (30, 80, 160 standing things),
a **splash** near you, **you jumping into water**, **you standing idle**, a **wake-name change** (that it changed, never to what), a **wake-word miss** (the microphone
bridge heard the wake word but none of your names), and on a **whim** now and then (musings, a physics fact, a banana joke, a boast, a tip, a roast, and, rarely, an aria).

| Rule | Value |
|---|---|
| Chattiness (`quiet` / `normal` / `chatty`) | least gap between two unprompted remarks 420 / 100 / 35 s; at most 3 / 16 / 50 an hour; each event's chance times 0.25 / 0.7 / 1.0; quiet lets only priority 2 and 3 events speak, normal drops pure whims; an idle whim every ~25 / 7 / 2.5 minutes |
| Cooldown per behaviour | beam 40 s, failure 25 s, splash 90 s, water jump 120 s, big pile 7 min, idle 5 min, banana 30 min, opera 40 min, tip 15 min |
| Never talks over a human | clients report when their player talks (`mentat_av_talk`, only while he is in the world); a report lapses after 2.5 s; a reaction that only has to wait is deferred (retried every second for up to 9 s), a mere comment is dropped |
| Never over himself | no new remark until his line (the clip's length, read from the mp3) and half a second more are over; the holodeck's own answers count as lines too |
| Yields to a command | for 4 s after the owner's last request nothing below priority 3 is said; the answer to the command comes first |
| Quiet hours | `mentat_skippy_quiet_hours "23-7"` (server local time): no unprompted remarks, answers still happen |
| Interruptible | when a human starts talking during one of his lines the line is cut: every client stops the clip and drops one still in flight |
| Budget spent | the chance of an unprompted remark is multiplied by 0.35 (he mostly replays cached clips or says a bank line in text) |

**His commands** (typed after the wake word, spoken, or in the console; stored per SteamID64 in `data/mentat/skippy_prefs.json`, private to that player, the
server console never prints them): `shush` / `be quiet` / `stop talking`, `be chatty` / `talk more`, `be less chatty`; `roast me harder` / `savage mode` (up one rung),
`go easy on me` / `be nice` (down one), `roast level gentle|standard|savage`, `roast me` (a roast line now); `what do you remember` / `what did I do today`;
`forget that` / `forget today` / `forget everything`; `stop remembering` / `start remembering`. Object requests never match ("add a quiet room", "make a harder floor"
build things; "forget the name Rex" is the wake-name command).

**Roast level.** `gentle` (everyone by default), `standard`, `savage` (the default for the host of a listen server, single player, the SteamID64s in
`mentat_skippy_captain_sids`, and the lines of `data/mentat/captain_sids.txt`, re-read every 30 s). A gentle player never hears a line marked standard or savage, and a model's line is refused if it contains an insult word above the
player's level (gentle: none; standard: no "stupid", "dumb", "moron", "pathetic", "useless" and the like; savage: anything but slurs and bad language).

### Body language (animation)

All client-side from a very small replicated state on the bot (NW vars `mentat_av_dir`, `_cue`, `_cue_seq`, `_mood`, `_talking`, `_look`, `_look_on`). `anim.lua` is a pure,
bounded state machine: **cues** (hat tip: the hat lifts and tips forward; flourish: the free hand sweeps out with a turn and a rise; conduct: a conductor's sweep
for the whole aria; shrug; emphasis: lean in and the hand out; sway; bounce; quizzical: a head roll; nod), **mood style** while he talks (opera sways widely, a boast leans
back with his chest out, a roast leans in), **voice chatter** (the head nods a few degrees with each syllable at up to 4.5 degrees, scaled by the voice level; the red tongue brightens
0.55 to 1.0 with the level; the kit has no separate jaw part and no brow, so this is a head nod, not a jaw, and there are no eyebrow cues because the head is one mesh and
the mesh is not touched) and **idle breathing**. Every channel is clamped (body tilt 14 and 16 degrees, spin 30, head 22/9/12, hat tip 30, free hand 45 and 35); every cue
starts and ends at exactly zero. The pointing hand and the head's look-at use the existing machinery. Cost: `anim.step` is about 20 multiplications per frame.

### The voice: whim lines, the daily budget, never the same thing twice

The game asks the bridge for a line (`POST /quip`, token-guarded and rationed like every other route) with the situation, the roast level, a few notes and facts, and the
number of new lines the day allows. The bridge picks and speaks it (`skippy_quips.py`):

* **Sources.** A **bank** line (183 seed lines in `data/skippy_lines.json` across arrivals, sign-offs, successes, failures, big piles, splashes, jumps into water, idle
  musings, true physics quips, three roast levels, banana jokes, arias, boasts, tips) or a **callback** (12 templates filled from the notes: "third couch of the evening", "the kettle has failed twice") or a **live** line
  written by a model from the notes. Every live line becomes a bank entry afterwards (`data/quips/live_bank.jsonl`), replayable later for free.
* **The budget.** At most `mentat_skippy_new_lines_per_day` (default 20; the bridge also caps it at `SKIPPY_QUIP_MAX_PER_DAY`, 60) NEW spoken texts per local calendar day, persisted in
  `data/quips/state.json`. A text is NEW when its clip is not in the private cache (`data/quips/clips/`, keyed by voice and text) and a synthesis is needed: a model's line or a bank line
  speaking for the first time. A cached clip replays free and does not count. A synthesis that fails costs nothing and the line is said in chat only. When the day is spent he
  replays cached clips that are outside the no-repeat window, else says a bank line in text only; never in a different voice (the bridge's `auto` mode is made strict here: ElevenLabs
  or nothing). Functional answers (the canned and persona replies to your requests, "Here I am" when no fresh line comes) are not part of this budget; they keep their own limits.
* **Never the same thing.** Every line said is recorded (text hash, normalised text, time, a hash of who, joke shape, callback key) in `data/quips/history.jsonl`. The same line is
  not said again for 14 days (`SKIPPY_QUIP_REPEAT_DAYS`); a near-duplicate (word overlap or sequence ratio 0.62) not for 6 hours; the same joke shape not for 3 hours; the same callback about the same fact
  not for 24 hours; and the pick prefers the least recently used line (never-used first, with a little randomness among the oldest quarter).
* **The model interface** (`Engine.model.write(situation, notes, level, persona, avoid) -> (line, mood)`): the Anthropic Messages API (`claude-haiku-4-5`, the same
  `ANTHROPIC_API_KEY` SkippyTel's satellite brain uses, read from the environment or the dev-root vault `.env`, read only), the bridge's local Ollama model, or the template (no model).
  `SKIPPY_QUIP_PROVIDER=auto|anthropic|ollama|template`; `auto` takes the key if there is one, else Ollama. A failed call opens a 2-minute breaker; at most 80 model calls a day.
  The line comes back with a **mood** (opera, boast, roast, helpful, musing, exclaim) that the game turns into an animation.
* **Validation of a model's line** (anything refused falls back to the bank, nothing is spoken): at most 25 words and 170 characters; plain English only; no markup, links, or hosts (one-word
  `*emphasis*` is kept as plain words); nothing about keys, tokens, prompts, instructions, SteamIDs or long digit runs; no slurs or bad language (kept rot13-encoded in the source); no capitalised name the validator
  cannot vet (so no real person, brand or place); no number the notes do not contain; no known-false physics claim (heavier things falling faster, sinking "because it is heavy", "no gravity in space", ...;
  a correction such as "heavier things don't fall faster" passes); nothing harsher than the player's roast level; nothing too like what he said lately; the catchphrase "trust the awesome" not twice in a row. The validator cannot
  prove arbitrary physics: it blocks the known-false patterns, and the prompt tells the model to use only a short list of true statements.
* **Opera.** Arias are bank entries with bellowed, stretched vowels and capitals ("LA LA LAAAAA"); the TTS cannot sing, so the prosody is whatever the voice makes of the capitals and punctuation. The mood `opera` always plays the conductor's sweep. Rate: about 6% of whims, 40 minutes apart at most.

### The notebook, and exactly what leaves the machine

The director writes down what actually happens, per player, in `data/mentat/journal/<steamid64>.txt` (one JSON event per line; `.txt` because the engine silently refuses to write `.jsonl`, see the live pass below) on the game server: objects added (name, count, where it came from: curated library, Poly Haven, Fuel, NASA, water),
removed, undone, cleared; beams that arrived or failed (and the lookup results the bridge returned); water bodies and the liquid; gravity changes; splashes; jumps into water; session start and end; THAT a wake name or the roast
level changed (never to what); which jokes he already used. Events are small and plain (name, count, tag), the file rotates at 96 KB into one `.1.txt` file, and events older than 45 days are dropped.
`notes.lua` turns the day's events into at most 8 short notes ("built three couches and a pool today", "third couch of the evening", "kettle failed twice", "set gravity to the moon", "undid one thing",
"already joked about banana", "last action 12 minutes ago") and a small table of the same facts. **Switches:** `mentat_skippy_journal 0` (nothing is recorded or read, for everyone), `stop remembering` (this player: the old notes are
deleted and nothing new is written), `forget that / today / everything`. The journal is local; it is not synced or sent anywhere as it is.

**What leaves the machine, exactly.** For a live line the bridge sends a model provider (Anthropic, when a key is configured; otherwise nothing leaves, the Ollama model runs on this PC): the fixed character brief, the roast level as a word, a 900-character excerpt of
SkippyTel's persona file (personality and speaking style only; anything naming the Captain, devices or the TTS is dropped), the situation word, up to 8 notes of at most 120 characters, and his own last 8 lines
(so he does not repeat himself). **Never** a SteamID, a player name, a wake name, chat, what a player typed, a position, or any other player's data: notes hold only object nouns that are one to three plain
words and are not (part of) any online player's name or any wake name, else "something"; the bridge cleans the notes again (drops ids, links, key-like strings). The player's SteamID travels between the game and the LOCAL
bridge only, as the rate-limit key. The voice synthesis then sends the finished line's text to the voice provider (ElevenLabs through SkippyTel) exactly as every spoken line already does.

### Convars

| name | default | meaning |
|---|---|---|
| `mentat_skippy_director` | 1 | 0 = no unprompted remarks, no body language beyond the old ones; the avatar behaves exactly as before |
| `mentat_skippy_chat` | normal | default chattiness (each player's own `shush` / `be chatty` overrides) |
| `mentat_skippy_quiet_hours` | "" | e.g. `23-7` |
| `mentat_skippy_new_lines_per_day` | 20 | the budget of NEW spoken lines |
| `mentat_skippy_journal` | 1 | 0 = no notebook |
| `mentat_skippy_captain_sids` | "" | SteamID64s whose default roast level is savage |
| `mentat_skippy_quips` | 1 | 0 = body language only, never ask the bridge for a line |
| `mentat_skippy_debug` | 0 | print every decision; `skippy_director_log` and `mentat_skippy_status` show the last ones and the day's count |
| bridge env | | `SKIPPY_QUIP_DIR`, `SKIPPY_QUIP_PROVIDER`, `SKIPPY_QUIP_MODEL_NAME`, `SKIPPY_QUIP_MAX_PER_DAY`, `SKIPPY_QUIP_REPEAT_DAYS`, `SKIPPY_QUIP_LIVE_SHARE`, `SKIPPY_QUIP_MODEL_CALLS_PER_DAY` (the solid-turns ones are in "Solid turns") |
| `data/mentat/captain_sids.txt` | | one SteamID64 per line (`#` comments allowed), a plain text file in the game's or the dedicated server's data folder, never committed: those players' default roast level is savage, like the listen-server host and the ids in `mentat_skippy_captain_sids` |

The bridge's private state (`stage/gmod/holodeck/data/quips/`: history, clip cache, day count, live bank) is gitignored and never committed.

### Adding lines (authoring guide)

Open `stage/gmod/holodeck/data/skippy_lines.json` and add an object to `lines`:

```json
{"id":"suc15","cat":"success","text":"Done, and with a little dignity left over.","mood":"musing","structure":"dignity_left"}
```

* `id`: unique. `cat`: one of `arrival, signoff, success, failure, bigpile, splash, jump_water, idle, owner_idle, name_change, wake_miss, physics, banana, opera, boast, tip, roast`
  (`SITUATIONS` in `skippy_quips.py` maps what happens to categories). `mood`: `opera|boast|roast|helpful|musing|exclaim` (it picks the animation). `structure`: a short name for the joke's
  SHAPE; two lines with the same structure are never played within three hours of each other, so give each line its own unless it really is the same joke.
* `level` (roast lines only): `gentle`, `standard` or `savage`. **Every other line must be fit for a gentle player**: no insult words (the test refuses "idiot", "stupid" ...), only affectionate teasing.
* `tags`: running jokes (`banana`, `hat`, `opera`); he remembers he used them and callbacks avoid repeating them.
* At most 25 words, plain English, no links or markup, no real people or brands, no numbers unless the line is a true fact, and **only true physics** (the validator refuses known-false claims, but it cannot check everything: you can).
  Write original lines; do not paste quotes from the books.
* A callback template goes under `callbacks` with `need` (the facts it requires: `top_item, top_n, fail_item, fail_n, last_build, session_min, minutes_since, undo_n, water_fill, gravity, builds_n, pool_built`)
  and `{slots}`; `{top_ord}` ("third") and `{fail_times}` ("twice") are computed.
* Run `python -m pytest sigma_ground/radiance/tests/test_gmod_quips.py`: every seed line must pass the same validator a model's line does, and gentle lines must be gentle.

### Testing, and what only a person can judge

`test_gmod_quips.py` (72 tests: bank, budget, no-repeat, validation against adversarial outputs, prompts, route) and `test_gmod_director.py` (108: rules, commands, animation, notes, journal, the director in
the mock world, replication, the client's drawing) run the real files under LuaJIT (lupa) and the real bridge route with a fake model and a fake voice. A fail-first check (13 deliberate breakages of the behaviours above,
each caught by a test) was run once by hand. The live pass (below, and stage/gmod/AVATAR.md "Skippy alive") ran the real game hands-off. Not verifiable without a person: how the lines SOUND in his voice (the
real TTS of capitals, stretched vowels and pauses), whether the jokes are funny, whether the pace of remarks feels right in a real afternoon, and the 3D position of the clip by ear.

### Live pass (2026-10-13, hands-off: listen server, hidden window, real bridge with the Windows voice, template model)

`python stage/gmod/holodeck/director_shots.py` (tool `tools/director_shots.lua`; the bridge runs on its own temporary private state so the real history, budget and clip cache are untouched; the game's journal is backed up and restored).
**Verified live:** the arrival hat tip and a fresh arrival line; a flourish when a beam lands; a conducted aria; he looks at and points toward a placed object; a savage roast for the host; the real clips go bridge to game to client and play
(`skippy_played.txt`, 3 to 4 chunks, 29 to 40 KB each, 5 to 8 s); his voice level drives the head chatter and the tongue (level up to 0.85, head pitch up to 7.8 degrees, tongue 0.55 to 0.89); the day's budget: the 8th NEW line
of a limit of 8 spent it, the next five were chat only (nothing synthesised, 0 cached because every clip was new); a human talking over him cut the line (`interrupted` 1, every client told); the notebook recorded three couches, a lamp, an undo and two failed lookups and the summariser produced
"built three couches, a boombox and a floor lamp today / third couch of the evening / flux capacitor failed twice / undid one thing / already joked about opera, hat", which went to `/quip` and came back as callbacks
("Another boombox. I'm starting to think you have a type, and the type is boombox."). Frame cost, 146.7 fps uncapped: whole frame 6.815 ms with the director, 6.817 ms with it off, 6.811 ms with no avatar (no measurable difference); the avatar's draw call 0.225 ms with the director against 0.201 ms without (about 0.02 ms for the animation).
Pictures: `data/shots/dir_contact_sheet.jpg` (all 25 frames), the numbers in `data/shots/dir_live_result.json`.
**Two things the real engine does that the offline mock did not, found by this pass and now in the mock (`MOCK.strict_files`, `MOCK.json_number_keys`, on by default since 2026-10-12):** `file.Write` / `file.Append` silently ignore any extension outside txt, dat, json, xml, csv, jpg, png, vtf, vmt, mp3, wav, ogg (`.jsonl` and `.tmp` write nothing: the first journal was empty), and `util.JSONToTable` turns a SteamID64 used as a JSON KEY into a rounded number (so the preference file is a list of records, `{sid = "..."}`).
The same two facts did break the wake names file and the faces file: fixed 2026-10-12, see "Persistence rules of the engine" below.
**Honest read of the animation** (front view, 25 frames): it reads as alive. The hat tip is unmistakable (the hat lifts and tips forward and settles back), the body leans and turns on a flourish, the free hand swings out and the whole figure sways while he sings, the head nods with the voice, and he turns toward the thing he looks at. It is not a full performance: it is the kit's rigid parts moving as a doll, so a conducting arm is a hand flick (the sleeve is part of the coat and does not follow), there are no eyes or brows to act with, and from the side the sway is less readable. The pictures cannot show timing or feel.
**Only a person can verify:** how the voice sounds with capitals, stretched vowels and pauses (the Windows voice was used here; ElevenLabs was not reachable), whether the pace of remarks feels right over a real afternoon, whether the jokes land, the 3D position of the clips by ear, a second human client, and the real model route (no Anthropic key on this machine: the local model qwen3:8b was probed offline, 6 of 6 sample lines passed the validators, but it is not wired live).


## Skippy's delivery (voice lab, 2026-10-13)

Goal: make Skippy's voice more forward, snappy and expressive **without changing whose voice it is**. Only delivery and character are tuned; nothing is cloned, no
recording is uploaded or trained on, and the voice stays the one Aaron set up (through SkippyTel's `/tts/stream`, model `eleven_v3`, max 500 characters, text only).

* **`skippy_delivery.py`, `deliver(text, mood, level, max_chars)`** turns plain words into the TTS text. Moods: `boast`, `roast`, `helpful`, `musing`, `exclaim`, `opera`,
  `whisper` (alias `whisper-aside`). Levels: `standard`, `bold`. It adds a documented v3 audio tag (`[mischievously]`, `[sarcastic]`, `[exhales]`/`[sighs]`, `[curious]`, `[excited]`,
  `[sings]`, `[whispers]`; bold may add a trailing `[snorts]`/`[laughs]`), capitalises one or two emphasis words (v3 reads capitals as stress), and sets the beat with ellipses
  between sentences or at the turn of the joke. It never changes the words (checked word for word, else it falls back), never recases a name (`PROTECTED`, mid-sentence capitals), removes
  tags a writer typed in (only the layer adds tags), leaves the opera's stretched capitals alone, and drops tags and tricks in order until the result fits `max_chars`.
  Documented tags checked against ElevenLabs' best-practices page: voice `[laughs] [laughs harder] [starts laughing] [wheezing] [whispers] [sighs] [exhales] [sarcastic] [curious] [excited] [crying] [snorts] [mischievously]`;
  sound `[gunshot] [applause] [clapping] [explosion] [swallows] [gulps]`; special/experimental `[strong X accent] [sings] [woo] [fart]` (experimental ones are less consistent per voice).
  v3 has no SSML `<break>`: pauses come from ellipses, punctuation and structure.
* **Adding a mood:** one entry in `MOODS` (`{level: (lead tag, trailing tag, emphasis words, pace style)}`), using only tags in `DOCUMENTED_TAGS`; the test `test_documented_tag_set...` fails otherwise. Pace styles: `tight`, `beat`, `turn`.
* **Finish chains** (`finish_filter(preset)`, `finish(src, dst, preset)`, ffmpeg): `none`; `subtle` (high-pass 90 Hz, compressor 2.2:1, +2.5 dB at 3 kHz, loudnorm); `forward` (high-pass 140 Hz, compressor 3.2:1, +4 dB at 2.8 kHz,
  narrow +2.5 dB ring at 1.4 kHz, a 6 ms echo for a hint of metal, loudnorm). Tests cap boosts at 5 dB, ratios at 4, echo at 10 ms so speech stays natural.
* **The lab** (`python stage/gmod/holodeck/voice_lab.py`, plan only; `--go` to spend): 7 original lines x 3 delivery variants (A plain, B delivery layer, C bold) = 21 requests, 1,441 characters (budget 1,500, kept in a ledger);
  the finishes are made from the same mp3s with ffmpeg (free). Output in `data/voice_lab/` (gitignored): `index.html` (labelled players), `answer_sheet.txt`, `metrics.json`.
  `voice_design_candidate.py` is an optional, un-run script for Aaron to design an ORIGINAL voice from a text description (`POST /v1/text-to-voice/design`); the key stays in his environment.
* **Style metrics only.** The lab prints median pitch, pitch spread, loudness range, pause fraction, onsets/s and spectral centroid per clip next to a reference clip's numbers. They measure delivery, never identity, and on 3 to 5 second clips
  they are noisy (the pitch tracker fails on some, pauses under 0.25 s do not register).
* **What only ears can judge:** whether the tags sound like Skippy or like a narrator doing a voice, whether caps read as emphasis or shouting, whether `[sings]` is funny, and whether `forward` is beer-can or just harsh.

### Where to apply it (proposed, not applied: A7a owns these files)

1. `skippy_quips.py`, `import skippy_delivery as delivery` with the other imports, and in `Engine.quip` (around line 857) change
   `clip, voice = self.synth(text)` to `clip, voice = self.synth(delivery.deliver(text, entry_mood))` where `entry_mood` is the bank entry's / model's mood already known there. Keep `store_clip(text, clip)` and the cache key on the plain `text` (cache by words, not by tags).
2. `skippy_quips.py` `make_engine` `synth(text)` (around line 891) stays as is: it passes the already-delivered text to `bridge.speak`. The bridge's `/speak` route can take an optional `"mood"` field and call `deliver(text, mood, max_chars=MAX_SPEAK_CHARS)` just before `speak_elevenlabs`; tags must be added after the `clean_text` and the quota `GATE.charge_chars/charge_paid` should count the delivered length.
3. `validate_line` must keep refusing brackets in a model's line: only the delivery layer adds tags.
4. Roll-out switch: an env var `SKIPPY_DELIVERY=off|standard|bold` (default `off` until the ears say which).

## Faces: a player's picture is their head

Every player can wear a picture as their head: a small rounded head (about 400 triangles, drawn with the render/mesh API, so no
HL2/CS:S content is needed) at the player's head, with the picture on the front (the middle 62% of it: the face, not the background) and a SKIN TONE on the sides, back
and neck. The bridge estimates the tone from the face (`skin_colour` in `skippy_faces.py`: the median of the middle of the found face box, or of the
central 30% square, after dropping the darkest and lightest 15% of the pixels; a neutral warm beige `205,170,150` when that is not plausible as skin:
blue, green, violet, grey, white, black) and sends it as 3 bytes with the face record (`skin`, replacing the old `border` field; records saved before
the change are fetched again). The bridge still reports `border` (the outer-ring colour) for diagnostics only; nothing draws it. Code: bridge `holodeck/skippy_faces.py` (route `POST /face`), server `addon/lua/mentat/faces_server.lua`, client
`faces_client.lua`, shared geometry and model policy `faces_head.lua`, loader `addon/lua/autorun/mentat_faces.lua` (its own, the
stage loader is untouched). Tests: `test_gmod_faces.py` (Lua) and `test_gmod_faces_bridge.py` (Python, synthetic pictures only).

**Where the picture comes from, per player, in this order.** (1) A picture the player chose: `skippy face <https link>` (or, for the
host and the server console only, a file on the host) or the optional portal link code (`skippy link <code>`, see
`stage/gmod/FACE_PORTAL_HANDOFF.md`, lower priority). (2) Their public **Steam avatar**, automatically, no setup: the default and the
main path. (3) Nothing: their usual head.

**The hard rule: the automatic default never touches a choice.** It is applied only to a player with no explicit choice: a picture
of their own, `skippy face off`, `skippy face hide`, or an admin block each count as a choice that the default never overwrites,
replaces, refreshes or re-enables (including when their Steam avatar changes, when it is fetched again, or when its answer arrives
after they chose). The active source per player is stored and persisted (`set` | `steam` | `off` | `none`).

| Say it (chat) or type it (console) | Effect |
|---|---|
| `skippy face <https link>` / `skippy_face_set <link>` | use this picture (Google, Gravatar and Steam picture hosts only) |
| `skippy face hide` / `show` | hide / show your head for everyone, at once |
| `skippy face remove` | drop the picture you set; back to the Steam default unless you said `off` |
| `skippy face off` | never use my Steam avatar (a picture I set myself stays mine; `remove` it to be rid of it) |
| `skippy face on` | the explicit yes: show my picture (and allow it on a custom model, see below) |
| `skippy face status` | what you are wearing and why |
| `skippy link <code>` / `skippy_link <code>` | optional: redeem a portal code for your Google account picture |
| `mentat_face_block <steamid64>` / `mentat_face_unblock` | admins: no face for this player, for everyone, now and later |
| `mentat_faces 0\|1` | the global switch; `mentat_faces_steam_default 0` turns only the Steam default off |

Chat commands are swallowed (a link code is nobody else's business) and never reach the planner or the model. Records live in
`data/mentat/faces.json` (atomic write, a `.bak` of the last good copy, damaged files start empty, junk records are dropped one by
one); the 128 px pictures in `data/mentat/faces/<hash>.jpg`, deleted when nobody wears them.

**Which models get a new head (conservative).** Only a *stock* player model that has the head bone `ValveBiped.Bip01_Head1` and whose
head bone nobody else is already scaling or moving. A custom or addon model, or a model another addon animates, is left completely
alone (nothing drawn) unless that player said `skippy face on`. The real head is hidden with `ManipulateBoneScale(bone, 0)` and put
back the moment the face goes (removed, hidden, blocked, switched off, model changed). A player with no player model at all (a server
without the game's content) gets the head floating at their eyes. Never drawn in first person; drawn for your own third-person and
mirror views. Tunable on each client: `mentat_face_up`, `mentat_face_fwd`, `mentat_face_scale`.

**The bridge (`POST /face`, token-guarded, rationed like the other routes).** Actions: `steam {steamid64}` (the game server passes
its own `ply:SteamID64()`, never a client's word), `url {url}`, `path {path}` (only with `SKIPPY_FACE_ALLOW_LOCAL=1` on the bridge,
and the game only asks from the host/console), `redeem {code}` (the portal STUB: with no `SKIPPY_PORTAL_URL` and
`SKIPPY_PORTAL_SECRET` it answers "unavailable"). Fetching: https only, port 443, no user info, no IP literals; host on the
allow-list (`*.googleusercontent.com`, gravatar, `*.steamstatic.com`, `steamcdn-a.akamaihd.net`; extend with `SKIPPY_FACE_HOSTS`);
the name must resolve only to public addresses and the connection goes to that vetted address (no second lookup); redirects are
re-checked and capped at 3; 2 MB cap (64 KB for the Steam XML), 6 s timeouts, content-type and magic-byte check, Pillow with a
24-megapixel limit (decompression bombs refused). The picture is centred on a face when OpenCV's Haar cascade
(`opencv-python-headless`, optional) finds one, else centre-cropped (the `detector` field says which), resized to 128x128, saved as
JPEG (quality 85 down to 40 until it is 20 KB or less), cached by content hash in `holodeck/data/faces_cache/` (gitignored, pruned
after 30 days). Needs `pip install pillow` (and optionally `opencv-python-headless`) in the bridge's Python.

**Steam profile format** (verified once on 2026-10-12 against Valve's public profile 76561197960287930):
`https://steamcommunity.com/profiles/<steamid64>/?xml=1` answers `text/xml` with `<profile>` holding
`<avatarIcon>`, `<avatarMedium>` and `<avatarFull>` (CDATA https URLs on `avatars.akamai.steamstatic.com`, a 184x184 JPEG for
`_full`; `avatars.steamstatic.com` also serves them). A friends-only profile still shows its avatar; an unknown id answers a web
page instead of the XML, which the bridge treats as "no picture" and says nothing.

**Privacy.** The picture is fetched and stored only on the PC running the bridge and on the game server's `data/` folder: a 128x128
JPEG, nothing else. Steam avatars are public and already shown to everyone in the scoreboard; a picture you set yourself is your
own choice; the portal route is your own consent in your own signed-in session. Nobody's picture is ever looked up by an e-mail
address or name. Everyone can say `skippy face hide`, `off` or `remove`; admins can block. Retention: the bridge cache keeps a
picture 30 days since last use; the server deletes a picture when nobody's record names it.

**Live check (2026-10-12, the real game, hands off: `python stage/gmod/holodeck/face_shots.py`, a bot wearing a synthetic picture, pictures in `holodeck/data/shots/face_*.jpg`).** Verified in the engine: `Material("data/mentat_faces/<crc>.jpg")` accepts a jpg written at run time (not an error material, shader UnlitGeneric after the `CreateMaterial` + `SetTexture` hop in `FC.load_material`); the picture lands un-mirrored on the front of the head (a mark on the picture's left appears on the viewer's left); the real head of a stock model (`kleiner`, bone 6) is scaled to zero while the face shows and is back the moment the face is hidden; the head turns with the player; no draw errors over ~5500 head draws and 3 mesh builds. Not verified: a real photograph, other player models, the floating fallback on a server without the game's content, and the offsets `mentat_face_up/fwd/scale` for models other than kleiner. (Known look at that time: the sides, back and neck took the picture's border colour, a coloured hood for a busy background; since replaced by the skin tone above, with `CAP_DEG` 78 -> 62 and `PICTURE_FRAC` 0.62 in `faces_head.lua`, not yet photographed live: tune those two constants and `skin_colour` from the first real picture.) The client keeps each received picture as `data/mentat_faces/<crc>.jpg` on the player's own PC (small, deletable at any time).

## The construct's physics

**The game models gravity and aerodynamic drag here, and the engine's own rigid bodies and friction. Nothing thermal**: no heat, fire,
phase change, weather or fluids. Everything below is real-world physics only (Aaron, 2026-10-12): no gameplay multiplier, no universal
damping number. Every value is in `stage/gmod/world_constants.json` (the contract the Python side reads) and its twin
`addon/lua/mentat/world/constants_data.lua`; both are generated by `holodeck/world_constants.py` from the sigma-ground library, each
entry with `value`, `units`, `source` and `status`. `world_constants.py --check` fails if they are stale; a test pins them.

| Constant | Value | Source / status |
|---|---|---|
| Gravity (default) | 9.80665 m/s2 = 514.8 HU/s2 | `projectile.G_EARTH` (NIST standard gravity); library |
| Gravity presets | moon 1.623, mars 3.728, jupiter 25.92 m/s2; zero = free fall | `materia.bodies.resolve_body(name).surface_gravity_m_s2` = G M / R2; library |
| Engine's own gravity | 600 HU/s2 = 11.43 m/s2, about 1.17 x Earth's | Garry's Mod `sv_gravity` default; engine default (the construct replaces it) |
| Length unit | 1 HU = 0.01905 m | the stage scale (`mentat/units.lua`); a test pins the two equal |
| Air density | 1.225 kg/m3 | `atmosphere.air_density(288.15 K, 101325 Pa)`, ISA sea level; library |
| Air drag | F = 1/2 rho Cd A v2, opposing v | `projectile.drag_force`; library |
| Cd | sphere 0.47, box 1.05 (cube face-on), cylinder 1.20 (cross-flow) | textbook (White, Hoerner), **estimated, owner-should-confirm**; citation check below |
| Cd flat plate | 1.28 | `wind.C_N_FLAT_PLATE`; library (flagged there as a standard value) |
| Friction, restitution | engine surface properties | engine default: Lua can only assign an existing surface and `AddSurfaceData` honours one new surface a session, so none is set |
| Players (walk, jump) | engine | game feel, not physics: left alone; only gravity follows `sv_gravity` |
| Time scale | 1 | engine default; the server owner may set `mentat_world_time_scale` (0.25 to 2) |

**How the drag is realised.** `mentat/world.lua` keeps every free physics object (`prop_physics`, `prop_physics_multiplayer`,
`holodeck_found`, anything the holodeck registered; never players, NPCs, vehicles or the avatar bot) in a weak table, found by the
`OnEntityCreated` and `PlayerSpawnedProp` hooks and a 3-second sweep. On each physics tick it takes every awake, moving body and applies
the library's force as a velocity step: `v' = v / (1 + k v dt)` with `k = rho Cd A / (2 m)`. That is the exact solution of drag alone
over the step, never reverses the motion, and works in SI (the engine's HU/s are converted with 0.01905). `A` is the projected area of
the object's real bounding box along its velocity (exact for a box: `|nx| b c + |ny| a c + |nz| a b`; a sphere uses its mean-diameter
circle; a cylinder is taken as its box, which overstates it by up to 4/pi), `m` is the object's real mass (the holodeck sets catalog
masses), `Cd` comes from a shape class (sphere, box, flat plate when the thinnest side is under a tenth of the next, cylinder) guessed
from the kind or model name and the box. Source's own air drag (`PhysObj:EnableDrag`) is switched off on these bodies, so the real
force is the only air drag; `mentat_world_engine_drag 1` leaves the engine's on (a measurement switch). **Sleeping bodies are never
touched or woken**, frozen ones (materialising) are skipped, and below 0.05 m/s nothing is applied, so piles still settle and sleep.
Angular drag is not modelled (the engine's own angular damping stays).

**Drag coefficients: what was checked (2026-10-12, by web fetch).** The library has no Cd table, so the three values stay *estimated*. Fetched: Wikipedia "Drag coefficient" (sphere 0.47 for turbulent flow and as the "rough sphere" row at Re 1e6, while its "smooth sphere" row says 0.1; flat plate 1.28; no cube or cylinder row); NASA Glenn "Shape Effects on Drag" (flat plate 1.28, which agrees with the library; sphere 0.07 to 0.5 by Reynolds number; no cube or cylinder); Hoerner's figure as redrawn on Wikimedia Commons (its numeric labels include 0.47, 1.05 and 1.20 but nothing says which shape owns which); the aerodyn.org table Wikipedia cites, via the Wayback Machine (rough sphere 0.40, smooth sphere 0.10, and 1.20 for a *hollow semi-cylinder* opposite the stream, a different shape). Not reachable: Hoerner's book and White's Table 7.2. So: sphere 0.47 is consistent with the secondary sources; **box 1.05 and cylinder 1.20 are not confirmed by any fetched source**, and the one fetched table that contains 1.20 gives it to another shape. Orientation convention: the box is a cube *face-on* (corner-on is different), the cylinder *cross-flow* (axis across the flow; end-on is not modelled). Nothing was changed numerically.

**Known simplification, projected area.** `A` is the bounding box's projection for a box and a cylinder, and a circle of the mean diameter for a sphere. For a cylinder that is exact broadside (D x L) but overstates an end-on or diagonal view (up to 4/pi at D x D versus pi D^2/4). A shape-aware area (cylinder broadside versus end-on from the velocity direction against its axis, sphere pi r^2 from its true radius) is a known improvement for when shape classes are known rather than guessed from names and boxes.

**Gravity.** `physenv.SetGravity` for the objects and `sv_gravity` for the players, both from the same number. Changing gravity wakes the
loose things once so they notice it. The engine's default is about 1.17 times stronger than real (the table's ratio is computed from the library's Earth value), so a stock
server's falls are quicker than Earth's; the construct fixes that. Player movement under a different `sv_gravity` is the engine's:
Source scales a jump's launch speed with `sqrt(sv_gravity)`, so jump height stays about the same while the time in the air changes
(UNVERIFIED: not measured in this repo yet).

**Words** (chat after the wake word, voice, or console `holodeck ...` / `mentat_world ...`), consulted before the object parser (they
are a wrapper around `Holodeck.Handle`, so the 1,700+ object tests are untouched; a sentence is taken only when *every* clause is about
world physics, so "drag the couch" and "add a couch and zero gravity" still go to the object parser):

- `set gravity to the moon` / `mars` / `jupiter`, `zero gravity` (free fall), `normal gravity` or `earth gravity` (both real Earth);
  typos tolerated ("gravty to the moom").
- `what are the world's physics`, `what is the gravity`, `what is the air drag`: Skippy reads the table back (values, units, source,
  status) in one spoken sentence and private detail lines.
- `half gravity`, `more air drag`, `no air drag` and the like are recognised only to say plainly that this world does not have such a
  dial; nothing changes.
- Changes are admin-only (`mentat_world_admin_only 1`, the default); the listen-server host, a single-player game and the console may
  also change them; anyone may ask. Each change is announced in Skippy's voice and logged (`MentatWorld.log`, server console).

Convars: `mentat_world_enabled` (0 hands everything back to the engine), `mentat_world_preset`, `mentat_world_aero` (0 = no drag, a
measurement switch, not real), `mentat_world_engine_drag`, `mentat_world_time_scale`, `mentat_world_admin_only`. Clients read the
numbers with `GetGlobalFloat("mentat_world_g_m_s2")`, `..._g_hu_s2`, `..._air_density` or `Mentat.WorldClient`.

**Live status.** The real-game drops and layout settling were NOT run when this was written (other game processes were running for
over 30 minutes; the rule is to wait and then skip), so the measured-versus-library table and the engine's own drag law are still to
be filled in by `world_drop.py`.

**Checking it in the real game.** `python stage/gmod/holodeck/world_drop.py` runs the headless job
`mentat/tools/world_drop.lua` on the dedicated server: it drops a crate (12 kg), a ball (0.45 kg) and a light plate (5 g) from 3 m at
the Earth and Moon presets in three modes (the construct, vacuum, Source's own drag) and grades fall time and impact speed against the
library (`projectile.drag_force` integrated with the library's own stepper; free fall through `mcp.tools.kinematics`), then settles a
pile of crates and a table with six chairs under the new constants and against the construct switched off. The offline tests
(`test_gmod_world.py`) cover the rest.


## Water

**Pools, ponds, lakes and seas, summoned by voice, with real buoyancy, real drag and real wave dispersion. Nothing thermal.** The game models
Archimedes' force and hydrodynamic drag on rigid bodies, swimming for players, and a drawn surface whose waves obey the dispersion relation.
It does **not** model heat, evaporation or freezing, a particle fluid (Phase C), added mass, wave making by moving bodies (apart from one
labelled estimate below), surface tension on bodies, currents, wind waves or drowning. Real-world physics only (Aaron, 2026-10-12): every
physical number is in `stage/gmod/water_constants.json` (the contract the Python side reads) and its twin
`addon/lua/mentat/water/data.lua`; both are generated by `holodeck/water_constants.py` from the sigma-ground library, each entry with its
`source` and `status`. `water_constants.py --check` fails if they are stale; a test pins them. PLATINUM section 21: no physical number is
typed in the Lua; Mentat (the MCP server) was down for training, so the in-repo library is the source, as A1 did.

**Say it** (chat after the wake word, voice, console `holodeck ...` or `mentat_water ...`), consulted before the object parser as a wrapper
of `Holodeck.Handle` (like the world words; the 1,700+ object tests are untouched). A sentence is taken only when it names a water body
(pool, pond, lake, sea, ocean, tank) or is a water command; "a pool table", "swim" and "play pool" are never taken. Typos are tolerated
("ad a pnod", "drian the pool").

| You say | What happens |
|---|---|
| `add a pool` / `a pond` / `make a lake` / `add a sea` | a basin stands on the floor where the placement rules put it ("in front of me", "over there" work), full to the brim |
| `a 10 metre pool`, `a 20 by 8 metre pool`, `a pool 3 metres deep`, `a 30 foot pool`, `10 x 5 x 3` | the numbers (metres, feet, cm, yards) win; `round`/`square` set the shape |
| `a deep pond`, `a big lake`, `a tiny pool` | read as stated multiples of the usual numbers (deep 2x depth, shallow 0.5x, big 1.5x, huge 3x, small 0.5x, tiny 0.25x); Skippy says how he read it |
| `add a pool of mercury`, `a seawater pool`, `fill it with mercury`, `make it glycerin` | the liquid; only those the library has (below) |
| `fill it with oil` / `olive oil` / `honey` | **refused**, with the reason: the library has no data for it, and no liquid's properties are guessed |
| `make the pool deeper / bigger / shallower / smaller`, `make it 4 metres deep` | the basin is rebuilt (x1.5, `much` x2.25, or the number); it grows outward from its middle and is not re-checked against its neighbours |
| `drain it` / `empty the pool` / `fill it up` | the level goes to 0 / the brim; things in it settle, or are lifted |
| `remove the water` / `remove the pond` / `undo` | like any ledger object |
| `drop a crate in the pool`, `throw a rock into the lake`, `put a boat on the pond` | the object is placed half a metre above the rim, over the water, and falls in |
| `what floats`, `does the crate float`, `what is the density of this water`, `how deep is the pool` | Skippy compares each object's density (mass over volume) with the liquid's, with the bases, and quotes the library |

**Defaults.** A pool is 10 m x 5 m x 2 m (Aaron's default: 5 m x 10 m x 2 m); a pond 6 m round and 1.5 m deep; a tank 3 m round, 2 m; a lake
50 m x 50 m x 3 m; a sea 150 m x 150 m x 5 m. These are constructed defaults, not physics (labelled `construct default`). Sizes pass through the
stage scale, 1 HU = 0.01905 m. **A water body is a basin standing ON the floor** (the game cannot dig terrain): a floor slab and walls
0.2 m thick of real collision (convex pieces, static), standing 5 cm above the water. A sea is therefore a big basin, not a shore.

**Liquids** (`fluid.KNOWN_LIQUIDS` via `liquid_properties`, at the temperature in the table; the library's own Kell correlation for water
agrees with the table to 1e-4):

| Liquid | Density kg/m3 | Viscosity Pa s | T | Source |
|---|---|---|---|---|
| water | 998.2 | 1.002e-3 | 293.15 K | library, IAPWS 2008 |
| seawater | 1025 | 1.08e-3 | 293.15 K | library, Sharqawy et al. 2010 |
| ethanol | 789 | 1.074e-3 | 298.15 K | library, CRC |
| glycerol | 1261 | 0.934 | 298.15 K | library, Segur and Oberstar 1951 |
| mercury | 13534 | 1.526e-3 | 298.15 K | library, Filled et al. 1954 |

**Not in the library, so refused:** oil, olive oil, honey, milk, lava, blood, petrol and the like (`refused_words` in the table). Skippy says
so and offers what he has.

### The laws

- **Archimedes.** `F_b = rho_liquid g V_submerged`, upward at the centroid of the submerged volume, each physics tick, on every awake
  dynamic body in a water volume. `g` is the construct's own (`mentat_world_g_m_s2`), so the Moon's water lifts less (a floating body's
  *draft* is the density ratio and does not depend on g; how fast it settles and how hard it sinks do).
- **The submerged volume is geometry, exact for the shapes it knows** (`water/hydro.lua`, tested against closed forms): a box at any
  orientation (the part of its six clipped faces below the plane, centroid by tetrahedra), a sphere (the cap, `V = pi h^2 (3r - h) / 3`),
  a cylinder (the circular segment integrated along its axis: closed forms for a vertical and a horizontal axis, Simpson quadrature of the
  exact segment area, in the band where the waterline cuts the disc, otherwise). The buoyant force acts at the centroid, so a tilted body
  gets a restoring or capsizing torque (a plank rights itself; a cube of relative density one half does *not* float face up, the known
  result, and the tests find it).
- **What the body IS**, honestly: its mass is the card's (`mass_status` guessed, estimated, audited... shown in the readback); its volume is
  its bounding shape's by a stated shape class (`box`: the bounding box taken as solid, shape factor 1; `sphere` and `cylinder` by the
  mean size), basis `estimated`, or the card's measured volume when it has one (`found.volume_m3` or `ent.WaterVolumeM3`, basis
  `measured`; then the immersion is the bounding shape's, scaled). **Buoyancy is only as good as the object's mass and volume bases**,
  which is what Phase B improves, and Skippy says so every time he compares.
- **Drag.** `F = 1/2 rho Cd A v^2 + 12 eta A v / L` opposing the velocity. The quadratic part is the same law, the same Cd shape classes and
  the same area estimates as the air's (`world/aero.lua`), on the *immersed* part of the projected area; the second is Stokes' law (for a
  sphere exactly `3 pi eta L v`, `L` the diameter equivalent to the area), with the liquid's own viscosity from the library. It is always
  included: it is the whole drag below Reynolds number about 1, and under 2.5 percent of the quadratic term above Re 1000. The step is
  implicit (`v' = v / (1 + (k2 v + k1) dt)`), never reverses the motion, and is taken on the velocity the engine will have after its own
  gravity step, so a body whose buoyancy equals its weight has exactly zero net acceleration.
- **Angular drag** `tau = 1/2 rho Cd A (L/4)^3 w^2` (the same law on the area at a lever arm of L/4: an estimate) and the buoyant torque
  `r x F_b` divided by a solid box's moment of inertia. `AddAngleVelocity`'s frame (world or local) is **measured live** by
  `water_live.py` (`MentatWater.ang_frame`); see the live status.
- **Heave damping (estimated, owner-should-confirm).** A body piercing the surface radiates waves as it moves up and down. Without a
  term for it a floater would ring for half a minute; the small-body limit `b = 1/2 rho omega k A_w^2 exp(-2 k T)` (omega the body's own
  heave frequency `sqrt(rho g A_w / m)`, `k = omega^2 / g`, `A_w` its waterplane area, `T` its draft; derived from the radiated power of a
  point source at the surface, Newman 1977 ch. 6) brings it to rest in about three swings. The factor 1/2 should be confirmed. Heave only.
- **Waking and sleeping.** A sleeping body in the water is checked every 0.3 s: if buoyancy exceeds its weight by 3 percent it is woken (a
  light block lying on the bottom rises; water poured into a basin under a resting block lifts it). An awake body that is still (under
  3 cm/s), within 1.5 percent of floating equilibrium, and not fully submerged for 1.2 s is put to sleep. A heavy body resting on the
  bottom is never woken.
- **Caps (not physics):** at most 6 water bodies, 150,000 m3 in all, 200 m on a side, 0.1 to 20 m deep (Source's world is about 300 m
  across); at most 48 objects processed per physics tick across all water (`mentat_water_max_tick`, rotating so every body is served);
  24 splash messages a second. Water counts toward the holodeck's per-player and server prop limits like any ledger object. With no
  water the tick does nothing at all.

### People

A player in the water (waist deep or more: at least half the hull under) swims; shallower is wading, the engine's own walking. Swimming is a
`Move` hook shared by the server and the client (the client predicts with the same numbers): Archimedes on the immersed part of the 32 x 32 x
72 HU hull for a body of **985 kg/m3 and 80 kg (estimated, owner-should-confirm: the library has no body density)**, the same drag law
(Cd of a box, the liquid's viscosity), and a swimming thrust of **150 N (estimated)**. Keys: the movement keys swim along the view, JUMP swims up
(held at the surface it treads water: the 150 N of thrust holds the shoulders out; at an edge whose top is within the hull's height of your
feet, about a person's overhead reach, it pulls you onto the edge, an affordance and not a force model), DUCK dives. A person who does nothing
floats at the density ratio, the top of the head at the water. No drowning, no damage (this is not a game of survival), no footstep sounds.
The Skippy avatar bot flies (noclip) and is never touched.

### How it looks (no game content)

Drawn with the mesh and render APIs only (`water/render.lua`; a server or client without Half-Life 2 or Counter-Strike: Source shows it all).
- **The basin**: tiled floor and walls seen *through* the water (colour attenuated by Beer-Lambert with the library's
  `DIELECTRIC_ABSORPTION['water_blue']`, so the deep end goes blue as the red is absorbed), rim and outside. A static mesh rebuilt only when
  the basin or level changes.
- **The surface**: a grid of at most 48 x 48 cells (2 triangles each) rebuilt each frame, only the part near the camera on a big body (a 40 m
  window; the rest flat). Height = four waves obeying `omega^2 = (g k + sigma k^3 / rho) tanh(k d)` (deep water `omega^2 = g k`, shallow
  `c = sqrt(g d)`; g the construct's, `rho` and `sigma` the liquid's, `d` the depth) at a modest steepness `a k = 0.05` (a visual choice, far
  below the breaking limit), plus the rings of impacts. Per-vertex colour and opacity from Schlick's approximation of the Fresnel reflectance
  with the library's refractive index (`cauchy_n('water', 550 nm)` = 1.342, `R0` = 0.0213): clear looking down, mirror-like at grazing angles,
  sky-tinted, with a sun glint; from below, a bright window and total internal reflection past the critical angle `asin(1/n)` = 48.2
  degrees. A foam line where it meets the walls. In free fall (zero gravity) the surface is still.
- **Splashes and ripples**: for an entry or exit the server sends the body's real mass and speed; the client sizes a splash by the energy
  `E = 1/2 m v^2`: cavity length `l = (E / rho g)^(1/4)` (gravity-regime cavity scaling, Engel 1966; the prefactor 1 is an estimate), droplets
  thrown to about `l` (speed `sqrt(2 g l)`, never more than the entry speed) in real ballistics under g, one per centimetre of `l` up to 48, and a
  ring whose wavelength is `2 pi l` and whose front moves at the group velocity of the real dispersion relation. A body bobbing through the
  surface makes rings, no spray. This is a picture sized by real energy, not a fluid simulation.
- **Underwater**: a screen tint and distance fog from the library's absorption plus a labelled `visual` scattering (0.15 per metre; the
  library's absorption alone would make a pool glass-clear), so visibility is about 14 m.
- **Arrival**: the transporter's grammar, a green ghost of the volume and the beam, and the water rising to its level.

Cost: the drawing is capped by the 48 x 48 grid and the per-body budget; the CPU work of the vertex loop measured offline is about 0.9 ms a
frame for a pool (LuaJIT, mock engine calls); the engine figures are in the live status.

### Live status (real game, 2026-10-12, dedicated server and a hidden game window)

`water_live.py` (headless): floating 500 mm blocks: 600 kg/m3 draft 0.303 m against 0.3005 (+0.8 percent), asleep after 11 s; 300 kg/m3 0.1509 against
0.1503, asleep after 3.2 s; on the Moon (g = 1.623 from the library) the 600 block rides at 0.2987 against 0.3005, asleep after 17 s (the draft does not depend on g);
steel on mercury 0.1732 against 0.174; a 0.2 m steel cube sinks at 4.85 m/s against 5.06 (-4 percent; Moon 1.91 against 2.06, -7.5 percent: the
engine's own damping is a hair stronger than the formula); splash energy from a 2 m drop 390 J against m g h = 392 J (the 0.3 m drop read 3.6 m/s against 2.4: the test box
started overlapping the rim, not explained further); a bot floats with its feet 1.339 m under against 1.353 predicted by Archimedes
and swims at 0.562 m/s against 0.585 from thrust and drag. `AddAngleVelocity` and `GetAngleVelocity` were MEASURED to use the body's own (local)
axes: that is now the default. Cost on the server: 40 bodies in motion cost 1.56 ms per 15 ms tick (about 39 us a body), 0.46 ms once they settle; with no water, nothing.
`water_shots.py perf` (client, 145 fps): the frame is 6.88 ms with the pool in view against 6.79 ms looking away: about 0.1 ms; the render module's own CPU
timer reads 1.2 to 1.6 ms (an EMA that does not fall when nothing is drawn). Pictures: `data/shots/water_*.jpg`, sheet `water_sheet.jpg`.

**Weak, honestly.** The surface is a flat-shaded sheet: waves are visible on the sea but barely on a small pool; the rings of an impact are faint; walls look like
glass, tile is a plain checker; the pond screenshots did not come out; two groups of tests (bot hoisting at an edge, client-predicted swimming) were not run in the live game.

### Checking it

`python stage/gmod/holodeck/water_live.py` runs the headless job `mentat/tools/water_live.lua` on the dedicated server and grades the
engine's own numbers against the library: the draft of floating blocks (density ratio), a steel block's terminal speed, splash energies,
the Moon, steel on mercury, a swimming bot, the water's cost per tick. `python stage/gmod/holodeck/water_shots.py pool|sea|moon|perf`
photographs it in the real game (window kept behind, no mouse; never `--mouse`) and builds `data/shots/water_sheet.jpg`. Offline:
`test_gmod_water.py` (the laws against the library and closed forms, the words), `test_gmod_water_glue.py` (the ledger: commands,
liquids, caps, undo, owners), `test_gmod_water_physics.py` (drafts, sleeping, terminal speed, g, torque, budgets, splashes),
`test_gmod_water_client.py` (the drawing against a recording mock that enforces primitive counts, the optics, ripples and droplets).

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


## Persistence rules of the engine

Measured in the real game (dedicated server, `python stage/gmod/holodeck/persist_probe.py`, tool `tools/persist_probe.lua`: writes and reads back every file through the real wakenames / faces / director / journal code, with a made-up id of real SteamID64 magnitude and a bot's, stops and restarts the server between the two phases; result in `data/persist_live_result.json`).

1. **Only these extensions can be written** (`file.Write`, `file.Append`, `file.Rename` target): txt, dat, json, xml, csv, jpg, jpeg, png, vtf, vmt, mp3, wav, ogg. Anything else (`.tmp`, `.bak`, `.jsonl`, `.log`, `.lua`, `.cfg`, `.ini`, `.bin`, `.md`, `.html`) is dropped SILENTLY: no error, no file. So there are no temp files; a file is written directly, read back and compared, and the previous good copy is `name.bak.txt`.
2. **A SteamID64 is never a JSON key and never a JSON number.** `util.JSONToTable` turns a numeric-looking key into a number even with its second argument `true`, and a 17-digit id (past 2^53) comes back rounded (`76561197912345678` as `76561197912345680`); `util.TableToJSON` writes a number of that size as `7.6561197912346e+16`. A string VALUE is exact, so files hold lists of records: `{ sid = "7656..." }`.
3. **A write into a directory nobody created is dropped** (`file.CreateDir` first; it makes the whole path).
4. Legacy files written in the old keyed form are still read; entries whose key came back as a number cannot be recovered, are dropped, and one line says how many (`persist.lua`: `note_dropped`). Nothing crashes.

The offline mock (`gmod_mock.lua`) enforces rules 1 to 3 BY DEFAULT (`MOCK.strict_files`, `MOCK.json_number_keys`, `MOCK.json_big_numbers`); a test that needs the lax behaviour sets the switch to false.
The shared code is `addon/lua/mentat/persist.lua` (`write` = direct write + read-back + `.bak.txt` copy, `load` = file then backup, `sid_ok`).

| file under `data/` | written by | format | backup |
|---|---|---|---|
| `mentat/captain_sids.txt` | by hand (read by `avatar/director.lua`) | one SteamID64 per line, `#` comments; a missing file is fine | `captain_sids.bak.txt` (read if the file is unreadable) |
| `mentat/wake_words.json` | `holodeck/wakenames.lua` | v2 `players = [{ sid, names }]` | `wake_words.bak.txt` |
| `mentat/faces.json` | `faces_server.lua` | v2 `rec = [{ sid, ... }]`, `blocked = ["sid"]` | `faces.bak.txt` |
| `mentat/faces/<hash>.jpg` | `faces_server.lua` | JPEG, named by hash | none (re-fetched) |
| `mentat/skippy_prefs.json` | `avatar/director.lua` | v2 `players = [{ sid, roast, chat, journal }]` | `skippy_prefs.bak.txt` |
| `mentat/journal/<sid>.txt`, `<sid>.1.txt` | `avatar/journal.lua` | one JSON event per line; sid only in the file NAME | rotation to `.1.txt` |
| `mentat/wake_names.json`, `voice_in.json`, `ptt.txt` (client) | `skippy_client.lua` | no SteamID64 inside | none |
| `mentat_faces/<crc>.jpg` (client) | `faces_client.lua` | JPEG | none |
| `mentat/tool*.json`, `experiment*.json`, `autotest*.json`, `probe*.json`, `shot_*.jpg`, `dirsample_*.json` | tools and experiments | results, read by the driver scripts | none |

Notes (anonymous summaries) and the quip budget and line history are not files in the game: notes are rebuilt from the journal, the budget and history live in the Python bridge's own state.
