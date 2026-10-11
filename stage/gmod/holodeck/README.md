# The holodeck (Garry's Mod)

Ask for things in plain words and real models appear, arranged the way a person would arrange them:

> **Skippy, add three couches.** → three couches in a U, open toward you, facing each other
> **Skippy, a table with six chairs.** → the smallest table that seats six, the chairs tucked in and looking at it
> **Skippy, a living room.** → couch facing the TV, coffee table between, armchairs flanking, a lamp and a plant in the corners
> **Skippy, put a lamp next to the couch. Turn the chair to face me. Remove the lamp. Undo. Clear the room.**

It never asks a clarifying question. Whatever is unclear resolves to the most sensible reading, and the reply says what it did
("there was no table, so I added one", "I don't have a dragon in my library").

## Using it

- **Chat or voice:** any sentence that starts with **Skippy** is a request. Everything else players say is left alone, so players with
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
- **Failure keeps the box.** No answer, a bad answer, a checksum or length that does not match, a payload that does not decode, a client that never gets
  the chunks, `util.Decompress` returning nothing: the entity keeps drawing the box, nothing is re-planned, the label is unchanged.
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

## Playing with your voice (one command)

```bash
bash stage/gmod/skippy_play.sh                 # the bridge, your microphone and the game, started in order and wired together
bash stage/gmod/skippy_play.sh --selftest-voice "add two couches" --stop-after     # prove the voice path with no human
```

`play.py` checks SkippyTel's Whisper, Ollama, ffmpeg, your microphone and your Tailscale address, starts the bridge behind a fresh token,
starts the game as a small listen server, connects the game to the bridge (`mentat/tools/skippy_play.lua`, run through the same
`data/mentat/tool.txt` mechanism as every other tool) and starts the microphone bridge. Then **hold X and say what you want** (no wake
word; `--key` or the `skippy_key` setting picks another key), or hold V and start with "Skippy". The key is read by the game itself,
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
