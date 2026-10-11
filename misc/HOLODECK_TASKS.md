# Holodeck task list (baked 2026-10-10 from the second review)

*Source: `scratchpad/review2_result.json` (127 confirmed of 137 reported; the 9 high-severity ones were fixed in `daa90a6`). Indices in brackets are positions in the review's sorted medium/low list. Plus the open work that did not come from the review at the end.*


## A. Verify first: probably closed by the nine high-severity fixes (re-run the reviewers' repro scripts, then close or reopen)

- [ ] [0] (critic, medium) Scene exclusions are parsed but never applied: 'a living room without the tv' still builds the tv
- [ ] [45] (integration, medium) 'A living room without the tv' / 'no tv' / 'a kitchen without a stove' still builds the excluded thing
- [ ] [91] (planner, medium) Scene exclusions ('a living room without the tv', 'no tv', 'but no bed') are never applied: the excluded thing is built
- [ ] [1] (critic, medium) 'move X away from Y' ignores Y: the planner never reads req.away_from and can move the item toward Y
- [ ] [51] (integration, medium) 'Move X away from Y' moves X away from the PLAYER; the parsed reference is ignored
- [ ] [92] (planner, medium) 'move X away from Y' ignores Y: the planner moves X away from the player, which can bring it closer to Y
- [ ] [41] (integration, medium) Moving, turning or rearranging a selection that contains both a support and its load drops the load to the floor; piles collapse onto identical spots
- [ ] [46] (integration, medium) 'In a circle facing outward' / 'facing away from the tv' are ignored and the reply says 'facing the middle'
- [ ] [47] (integration, medium) Grid dimensions ('2 rows of 6', 'a 1 by 12 grid') are parsed but never used
- [ ] [70] (parser, medium) Contract fields facing_away, shape, rows, cols and away_from are emitted but ignored by the planner (and a facing_away combined with another relation faces the wrong way)
- [ ] [94] (planner, low) The parser's rows/cols, shape and facing_away fields are ignored by the planner
- [ ] [73] (parser, medium) Opinions and statements about existing props become ADD of duplicates: 'keep the couch', 'leave the lamp alone', 'I like the couch', 'I hate couches' (3 couches)
- [ ] [68] (parser, medium) Unrecognised relation words make the reference noun a NEW item: duplicate props (parser-1 backstop not implemented)
- [ ] [49] (integration, medium) 'Move the lamp to the other side / opposite side / right side of the couch' moves the couch too
- [ ] [3] (critic, medium) A plural with no 'the/all' in a removal means three: 'clear the room of barrels' removes only three of six
- [ ] [79] (parser, low) A bare plural after 'clear the room/floor of' removes only 3: 'clear the room of crates' with 7 crates removes 3

**Each needs one repro run in scratchpad/wf2/ and a regression test if it still fails. Scene exclusions, away_from, rows/cols/shape/facing_away, support+load moves, keep/leave clauses were all in the nine fixes.**


## B. P0 Destructive or unsafe reads (a remove, undo or clear nobody asked for)
Do these first. Prefer 'none' with a note over guessing a destructive reading.

- [ ] [61] (parser, medium) A spoken self-correction inside a removal ('remove the couch, no wait, the lamp') removes both
- [ ] [62] (parser, medium) 'undo what you just did', 'undo all of that', 'undo twice' and 'undo everything' do not undo: they remove the last build or clear everything
- [ ] [63] (parser, medium) A management verb whose object the parser cannot place falls back to the player's last build: 'clear the air', 'clear some space', 'clear the way', 'clean up the mess', 'remove yourself', 'turn around', 'move over'
- [ ] [64] (parser, medium) Retraction words in plain talk undo the last build: 'sorry', 'sorry about that', 'wait here', 'oops i dropped my drink', 'I meant to ask ...'
- [ ] [65] (parser, medium) 'the red one' drops the colour and removes the last build: 'remove the red one' removes the wrong things
- [ ] [66] (parser, medium) Modal questions, suggestions and conditionals execute the destructive verb: 'should I clear the room', 'maybe we should clear the room', 'we could clear the room', 'if you remove the couch I will be sad'
- [ ] [67] (parser, medium) Past-tense statements and 'for X' purposes trigger removals: 'the lamp was removed' removes the lamp, 'clear space for a table' removes the table
- [ ] [78] (parser, low) 'toss' and 'chuck' with an indefinite item remove it ('toss me a crate', 'chuck a crate')
- [ ] [16] (critic, low) Casual phrases still clear or remove: 'that is enough', 'enough already', 'clean up', 'clear the way', 'clear space around the table'
- [ ] [80] (parser, low) Sentence-initial 'no', 'not', 'nope' swallow the whole sentence unless a comma follows: 'no add a lamp' and 'no I want a lamp' do nothing
- [ ] [82] (parser, low) Negations written with ` ´ ʼ ′ instead of ' lose the 'not': "don`t remove the couch" ADDS a couch; a zero-width space splits a word
- [ ] [83] (parser, low) strip_wake treats the possessive 'Skippy's' as the wake word: "Skippy's couch is ugly" adds a couch
- [ ] [72] (parser, medium) 'less / fewer / one less chair' adds chairs, the opposite of a reduction
- [ ] [74] (parser, medium) Negation scope in add requests is too short: 'and dont add a lamp' adds the lamp, 'without the table and chairs' adds the chairs, 'minus the bed and the lamp' adds a lamp
- [ ] [77] (parser, medium) Typo repair turns plain chat words into actions: 'tell me a joke' adds a TV, 'sing me a song' rotates the last group, bare 'loose' / 'ease' / 'pure' remove
- [ ] [60] (integration, low) Reply claims props the engine refused to create

**Acceptance: a held-out slice per class (30+ sentences each), the old parser fails it, the new passes 100 %, and the strict-slice test replaces the 3 % tolerance (task T1).**


## C. P0 Undo and shared-server limits (glue)
One cluster: undo must restore the right thing for the right owner, and one player must not be able to destroy or starve others.

- [ ] [25] (glue-security, medium) make_room runs before the request is known to be buildable: other players' props are destroyed by requests that build nothing or less
- [ ] [26] (glue-security, medium) Undo of a clear/remove/move restores props onto ground that is now occupied: no overlap or player check
- [ ] [29] (glue-security, medium) An admin's undo of a clear restores other players' props past THEIR per-player cap and as the admin's creator/CPPI props
- [ ] [42] (integration, medium) Props restored by undo are created as the undoing player's, not their owner's
- [ ] [52] (integration, medium) Undo of a remove/clear puts props back into space that someone else has since filled
- [ ] [56] (integration, low) A partly restored undo throws the rest away
- [ ] [40] (integration, medium) Undo of a move/turn/rearrange leaves a load owned by another player behind
- [ ] [8] (critic, low) An admin's undo of a clear restores other players' props above their own per-player cap
- [ ] [27] (glue-security, medium) Compound sentences multiply the per-request limits by up to 4 (planner time and prop count)
- [ ] [28] (glue-security, medium) The planner and bridge allowances are first-come global buckets: a few hostile players starve everyone
- [ ] [44] (integration, medium) The planner's shared CPU allowance can be emptied by three players, locking everyone else out
- [ ] [57] (integration, low) A player at their own limit still makes the server take down other people's props
- [ ] [32] (glue-security, low) A restored/new prop is set free after 10 s even if a player is still standing inside it

**Acceptance: glue tests in test_gmod_glue.py for each; make_room only after the request is known buildable; per-player, not first-come-global, allowances; undo restores owner, cap and checks occupied ground.**


## D. P0 Server rules, keys and bridge hardening

- [ ] [30] (glue-security, medium) The server's own spawn rules are only half honoured: the gamemode's PlayerSpawnProp/PlayerSpawnObject is never consulted
- [ ] [7] (critic, low) Spawn rules: only PlayerSpawnProp hooks are consulted, not the gamemode's own rules or sbox_maxprops
- [ ] [31] (glue-security, medium) New default skippy_key 'x' collides with the documented shout key and turns shouting at players into a transcription path to the server
- [ ] [53] (integration, low) Props still being materialised when their builder leaves spawn without a creator or prop-protection owner
- [ ] [102] (tooling, medium) An unauthenticated trickle client holds every connection slot indefinitely; the 10 s Handler.timeout is per read, not per request
- [ ] [103] (tooling, medium) The persona/agent can still say 1-4 words the player typed; the 'repeats the player's words' guard needs five identical consecutive words
- [ ] [104] (tooling, medium) Output checks are ASCII/English only: model text in any other script bypasses the echo, off-topic and link checks
- [ ] [106] (tooling, low) Link filter is a 24-TLD allow-list on ASCII labels: goo.gl, is.gd, rb.gy, bit.do, youtu.be, .top, .club, .ai and any lookalike with one non-ASCII letter are spoken and shown
- [ ] [107] (tooling, low) An authenticated body of deeply nested JSON gets no answer (and a traceback) instead of 400; the code comment says the game must always get an answer
- [ ] [108] (tooling, low) clean_text runs quadratic regexes over the whole 20 KB body before any budget is spent: 18 KB of 'a.a.a.' costs 10 s of CPU per request
- [ ] [111] (tooling, low) _ps_quote doubles only ASCII apostrophes; PowerShell also treats U+2018/2019/201A/201B as quotes, so /speak breaks (and a crafted value injects) for a profile folder or voice name containing them
- [ ] [101] (tooling, medium) A hang-up that arrives while the bridge is connecting to Ollama is lost; the abandoned request keeps its model slot
- [ ] [109] (tooling, low) A token file with a UTF-8 BOM silently gives a token that never matches; a UTF-16 file crashes with a traceback

**Acceptance: gamemode spawn rules honoured; the talk key no longer collides with shout (pick another default); bridge survives slow-loris and deep JSON, text guards work for non-ASCII and lookalike links; each with a test.**


## E. P1 Planner correctness

- [ ] [86] (planner, medium) Three or more couches/loveseats/benches with a side table (or tables with extra plants) are refused on open ground
- [ ] [87] (planner, medium) The 0.05 s ordinary-player deadline refuses or randomly degrades plain requests that need a modest search
- [ ] [88] (planner, medium) Turning a square-ish or long prop that stands at a wall or in a corner is refused ('no room to turn it there') although sliding a few HU would do
- [ ] [89] (planner, medium) '<deeper thing> next to <shallower thing that stands against a wall>' always falls back to 'nearby'
- [ ] [90] (planner, medium) 'N crates stacked' is laid flat on the floor but reported as 'stacked in a pile' (MAX_PILE_H = 128 is below two standard crates)
- [ ] [17] (critic, low) Mixed pile kinds: crates/boxes mixed with other kinds are stacked under them, through walls, with a success message
- [ ] [98] (planner, low) Mixed-kind piles: upper items skip the wall/floor check, poke through walls (up to 117 HU) and can float
- [ ] [5] (critic, medium) A second 'in the corner' request cannot find a free corner: single things only ever try the nearest corner
- [ ] [6] (critic, medium) Moving a pile flattens it, and moving one crate of a pile to a corner drags its load through the wall
- [ ] [2] (critic, medium) The parser-30 fix breaks follow-ups: after 'a lamp on the table' the standing desk lamp can no longer be removed, moved or turned as 'the lamp'
- [ ] [50] (integration, medium) Bare 'the lamp' in management and 'put' verbs means only the floor lamp
- [ ] [48] (integration, medium) 'Put X next to Y' fails whenever X is deeper than a wall-flush Y
- [ ] [9] (critic, low) Two scenes in one sentence: the second is dropped silently, and 'two living rooms' builds nothing
- [ ] [93] (planner, low) A time slice (not the whole budget) ending a search yields a fallback arrangement with a false explanation and no timed-out flag
- [ ] [95] (planner, low) Rows flagged `legacy` are still planned with a stale variant index
- [ ] [96] (planner, low) 'N plants in each corner' is read as a total when N is a multiple of 4
- [ ] [97] (planner, low) Refusal replies carry notes for things that were never done
- [ ] [75] (parser, medium) '3 chairs on each side of the table' places 3 total, not 3 per side
- [ ] [69] (parser, medium) Distributive and elliptical quantities are silently miscounted ('2 tables with 4 chairs each', '3 red and 2 blue chairs', '2 blue ones')
- [ ] [71] (parser, medium) Several rooms in one request build only the first: 'a kitchen and a bedroom'
- [ ] [10] (critic, low) 'in between X and Y' adds a second Y; 'on my left/right' and 'a set/group of chairs' are ignored
- [ ] [15] (critic, low) Parser note 'so I ignored where it should go' is shown while nothing is done

**Acceptance: the reviewers' repro for each passes through lab.grade and the glue; no loosening of overlap tolerances; run live.py on the dedicated server for any placement change.**


## F. P1 Glue behaviour and honesty

- [ ] [39] (integration, medium) A compound request is cut off after the 4th clause with no word, and the dropped clauses poison 'another one'
- [ ] [76] (parser, medium) Parser/glue contract mismatches: req.truncated and its note are unreachable in the game, and parse_all returns up to 8 requests while the glue silently runs 4
- [ ] [55] (integration, low) Requests over 400 characters lose their tail with no notice (the parser's truncation note is dead code)
- [ ] [54] (integration, low) 'Add another one' / 'again' work only directly after an add
- [ ] [58] (integration, low) Undoing an add does not give 'it' back to the earlier group
- [ ] [59] (integration, low) Library change with props standing: a standing prop whose model left the library cannot be named any more
- [ ] [43] (integration, medium) Undo of a remove/clear after a library change re-registers props with a stale variant index
- [ ] [33] (glue-security, low) Undo after holodeck_reload / a content change re-spawns models the server no longer has, with a wrong or nil variant
- [ ] [34] (glue-security, low) floor_z, ceiling_z and wall_probe still use MASK_SOLID_BRUSHONLY: a requester standing on props gets things placed on the world floor below (glue-13 only half fixed)
- [ ] [35] (glue-security, low) Requester sees '[Skippy to <own name>]' instead of '[Skippy]' for many nicknames
- [ ] [36] (glue-security, low) mentat_voice_mode: the server treats any unknown value as proximity, the clients treat only the exact string 'proximity' as fading
- [ ] [37] (glue-security, low) Proximity voice is installed once; a later addon that replaces GAMEMODE.PlayerCanHearPlayersVoice silently removes it
- [ ] [4] (critic, medium) The README's headline example loses its fifth request silently: the glue runs only the first four requests of an utterance
- [ ] [114] (tooling, low) live.py grader still reports a chair/stool through a dresser or a side table (drawer unit) as clean at up to 14 HU

**Acceptance: no silent truncation of compound requests (say so or raise the cap), req.truncated reaches the player, 'another one' works after any request, replies never claim props that were refused.**


## G. P2 Deckard and sourced masses

- [ ] [19] (deckard-fidelity, medium) New exemplar aliases (and the old 'pot' alias) identify unrelated names as flowerpots, TVs and cabinets
- [ ] [20] (deckard-fidelity, medium) Head-noun tie-break resizes accessory compounds as the wrong object ('a bed frame' is a 32 cm picture frame; 'a closet door' a closet-sized door)
- [ ] [21] (deckard-fidelity, medium) Thin-board exemplar masses are not converged: a bed is 70.6 kg at the resolution the tests use and 150.6 kg at the library default; the self-check fails
- [ ] [22] (deckard-fidelity, medium) Several masses are still 4-17x the repo's own sourced ranges, and the test bands and README bless them
- [ ] [23] (deckard-fidelity, low) _hollow_pass guard is material-blind: a 3.8 cm steel door or 3 cm glass panel is now solid, and a test pins it
- [ ] [24] (deckard-fidelity, low) fidelity.py --verify-densities cannot run on this Windows machine (resolves to the WSL launcher)

**Acceptance: aliases stop matching unrelated names; thin-board masses converge across resolutions; test bands tightened to the sourced ranges; the axis fix (P4) written as its own change.**


## H. P2 Tooling

- [ ] [105] (tooling, medium) The ears give up for good after the 4th microphone dropout of the whole session; MIC_RESTARTS is a lifetime counter, not consecutive failures
- [ ] [110] (tooling, low) package_addon.sh rejects GMAD set to a program found on PATH although the header says GMAD names the program directly
- [ ] [112] (tooling, low) validate_request: WORLD_TARGETS lists '@last' but _norm_name strips '@', so it can never be emitted; 'center' is in the list but plan.lua does not know it
- [ ] [113] (tooling, low) A scene plus an explicit item of the room's own kind and count is dropped, losing its colour/size adjectives
- [ ] [115] (tooling, low) play.py --selftest-voice prints PASS without checking that anything was built or that the recognised text was the request
- [ ] [116] (tooling, low) The holodeck_bridge convar help text suggests port 8765; the bridge listens on 8777
- [ ] [14] (critic, low) New launcher play.py: the token is also written to disk, --selftest-voice can pass with nothing built, and it has no tests

**Acceptance: mic restarts count consecutive failures; self-test fails when nothing was built; token not left on disk; port text fixed; play.py gets tests.**


## I. P3 Tests and docs

- [ ] [18] (critic, low) The 750-case 'held-out' corpus is no longer held out and its test tolerates 3% silent failures
- [ ] [38] (glue-security, low) The glue test suite has blind spots and a vacuous assertion (and some code paths never run in any test)
- [ ] [99] (planner, low) lab.py grade/audit mis-flag piles of things that have a flat top, and testenv.clear cannot see an obstacle inside a footprint
- [ ] [100] (planner, low) Tests accept refusals and one assertion is a tautology, which is why the regressions above pass
- [ ] [13] (critic, low) The glue-3 test cannot see the garbage-collection bug it is named for
- [ ] [81] (parser, low) Parse.normalise (tokenise) is quadratic on long digit runs, contradicting its 'linear' claim and test (parse_all is protected by its 400-char cut)
- [ ] [84] (parser, low) 'undo and undo', 'undo, undo' and 'undo undo undo' merge into a single undo; 'undo twice' removes the last group instead
- [ ] [85] (parser, low) Digit groups with a space or a decimal comma, and 'a dozen and a half', give wrong counts: '1 000 crates' -> 'zero is nothing', '1,5 couches' -> 5, 'a dozen and a half' -> 1
- [ ] [11] (critic, low) README has X for two different things
- [ ] [12] (critic, low) README test instructions are out of date

**Acceptance: no test tolerates silent failures or accepts a refusal as success; README accurate (two uses of X, test commands).**


## K. Open work that did not come from the review
- [ ] **T1** Replace the 3 % tolerance in the parser pass-rate test with per-slice 100 % checks; stop calling the 750 cases 'held out' (they are regression cases); write a fresh unseen set of 200 by someone other than the parser's author.
- [ ] Apply the SkippyTel cloud patch (commit 6800362 in the scratch clone; needs a push to SkippyTel main or a permission rule), then verify the addon installs at the next VM boot.
- [ ] Mount Half-Life 2 and Counter-Strike: Source content on the cloud server (ask the portal chat), or run it with `+holodeck_content hl2`.
- [x] Bathroom: bath fixtures now stand 8 HU apart (was 2; the toilet tipped over in the engine). Live-checked 2026-10-10: the wall-hung prison toilet (it rolled over in the engine) is removed from the library; four bathrooms stood clean.
- [ ] 'Except the last one' / 'except what I just made': planner understanding exclude=['@last'].
- [ ] Ears: a 3-second gap in the game's key signal cut one press; widen PTT_HEARTBEAT_STALE_S if it recurs; Whisper hears 'add' as 'placed' (accepted by the parser).
- [ ] Deckard axis-orientation fix (P4) as its own change with the pinned 1700s scenes re-baked.
- [ ] Physics import P0: the law registry and an all-289 sweep (misc/SKIPPYVERSE_PHYSICS_IMPORT_PLAN.md).
- [ ] Objects on command increment 1 (web page + vehicle shelf) after the red-team fixes (misc/SKIPPYVERSE_OBJECTS_DESIGN.md); decisions needed first.
- [ ] Lazy repair I1: stop the damage in Deckard's voxel fill (misc/SKIPPYVERSE_LAZY_REPAIR_DESIGN.md); Mentat owner enters the missing materials.
- [ ] Later: a beam from the finger instead of the held gun (web client pointing ray / WebXR hand).
