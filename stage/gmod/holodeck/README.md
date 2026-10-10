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
- **Safety on a shared server:** everyone's props belong to whoever made them (admins change anything), a cooldown between requests,
  caps per player and overall.

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
  `holodeck.lua` is only engine glue (traces for floors and walls, spawning, chat, undo, ownership).

## Testing

```bash
pip install lupa pytest                                     # lupa = LuaJIT, the VM Garry's Mod uses (optional, test-only)
python -m pytest sigma_ground/radiance/tests/test_gmod_holodeck.py        # 175 tests, about 3 s
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
| `skippy_ears.py` | each player's own PC | reads the microphone into memory, and when the game says the player is transmitting, has the sentence recognised by the LOCAL Whisper; only a sentence that starts with "Skippy" is handed to the game |
| `skippy_client.lua` | the game client | tells the ears when the player transmits; plays Skippy's voice in 3D from where he built things |

Voices: `--tts sapi` (Windows speech, free, the default), `elevenlabs` (Skippy's own voice through SkippyTel; costs API credit per
character) or `auto`. A dedicated server reaches the bridge on localhost with `-allowlocalhttp` (already in `gmod_ds_ctl.sh`);
a game that hosts its own world must use the machine's Tailscale address (the engine's flag works on servers only).

## Proximity voice

Native to the server (`mentat/voice.lua`, the `PlayerCanHearPlayersVoice` hook): a normal voice carries about 21 m and fades smoothly
over its last 40 %, crouching whispers (about 7 m), holding `+mentat_shout` shouts (about 46 m), walls muffle. `mentat_voice_mode`
`proximity | global | off`, `mentat_voice_range`, `mentat_voice_wall`. Bind a key: `bind v +voicerecord` is the game's own; `bind x +mentat_shout`.

## Where the data came from

The library's masses are the engine's own. A separate audit (`audit_physics.py`, report in `misc/HOLODECK_PHYSICS_AUDIT.md`) sets
each prop beside what Deckard compiles for the same name; it reports, it does not correct.
