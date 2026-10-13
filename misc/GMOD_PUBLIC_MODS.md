# Public Garry's Mod addons worth considering for the SkippyVerse

Research date: 2026-10-12. Read-only survey; nothing installed or launched.

**How verified.** Workshop numbers (lifetime subscribers, current subscribers, last update, GMA size) come from Valve's public
`ISteamRemoteStorage/GetPublishedFileDetails` API, fetched today. Licences, features and hook usage come from the GitHub repos
(API for licence/last push; shallow clones grepped for `PlayerCanHearPlayersVoice`, `PhysgunPickup`, `CanTool`, `weapon_physgun`,
`weapon_physcannon`). Wiki claims come from wiki.facepunch.com pages fetched today. Steam item pages themselves returned
errors or HTTP 429 when fetched, so **Workshop "required content" lists and descriptions are unverified** unless a repo README says it.
Workshop-only items with no stated licence are "all rights reserved" to the author by default; do not redistribute them.
Sizes are the Workshop GMA size (what a client downloads).

**Headline finding on your hard constraints.** In the nine repos cloned (Wire, AdvDupe2, Glide, StormFox2, DrGBase, GWater2, ACF-3,
Real CSM, simfphys) there is **no** `PlayerCanHearPlayersVoice` hook or `GM:` override anywhere (grep returned nothing). None replace the
physgun or gravgun. They only add hooks on `PhysgunPickup`/`CanTool` for their own entities (details in the Avoid list).
Closed-source Workshop items were not checked.

## 1. Building and engineering

| Addon | Author | ID / URL | Subs (lifetime / current), updated, size | What it does | Licence | Content needed | Verdict |
|---|---|---|---|---|---|---|---|
| Wiremod (+ Expression2) | Wiremod team | 160250458 <https://steamcommunity.com/sharedfiles/filedetails/?id=160250458>, repo <https://github.com/wiremod/wire> | 4.47M / 2.28M, 2026-10-05, 20.7 MB | Wire-connected gates, sensors, E2 scripting. Signal/logic level, **not** analog voltage/current (no resistor/capacitor sim found) | Apache-2.0 | None stated (unverified) | **adopt**. Small, maintained, Lua only. Only hook found is E2 `PhysgunPickup` (run-on event, passive). |
| Advanced Duplicator 2 | Wiremod team | 773402917 <https://steamcommunity.com/sharedfiles/filedetails/?id=773402917>, <https://github.com/wiremod/advdupe2> | 1.49M / 528k, 2026-08-23, 0.2 MB | Save/paste constructions | Apache-2.0 | None stated | **adopt**, but test that pastes go through your registry/undo (it spawns entities itself, not via your summon path). |
| Precision Tool [Official] | creator id 76561198014435136 | 104482086 | 2.98M / 1.31M, 2016-05-08, ~0 MB | Precision alignment/move/rotate | Not stated (all rights reserved) | None stated | **try**. Old but tiny and frozen; suits real-parts alignment. |
| Hydraulics, pneumatics, rails | - | no dedicated public addon verified | - | Sandbox already ships hydraulic/winch/rope tools (engine). Rails: Metrostroi 261801217 (89.6 MB, last update 2018-09-18; 350k subs) and Metrostroi: Project Light Rail Base 2634043997 (526.6 MB, alpha, 2026-07-30) | unverified | unverified | **skip** rails (too large, scenario-specific). Build hydraulics/pneumatics from Wire + built-in constraints. |
| SProps Workshop Edition | shadowscion | 173482196 | 512k current, 381 MB | 3,200 models | all rights reserved | - | **skip**. The page fetched shows it removed for violating Steam content guidelines and marked incompatible with GMod. |

Electronics gap: no public analog-circuit addon found. Wire + E2 would have to emulate it; a Mentat-backed solver is the better path.

## 2. Realism and physics

| Addon | Author | ID / URL | Subs, updated, size | What it does | Licence | Notes | Verdict |
|---|---|---|---|---|---|---|---|
| Glide (Styled's Vehicle Base) | StyledStrike | 3389728250 <https://github.com/StyledStrike/gmod-glide> | 932k / 624k, 2026-09-22, **167.5 MB** | Cars, bikes, boats, planes, helicopters, tank; AD2 and Wire support; no constraints for wheel physics | MIT | Only `CanTool` blocks for its own entities. A Workshop thread (seen via search, not independently confirmed) reports Glide input dead with TacRP weapons enabled | **try** if vehicles are in scope; ~168 MB client download is the cost. |
| simfphys LUA Vehicles Base | SpaxscE | 771487490 <https://github.com/SpaxscE/simfphys_base> | 4.40M / 2.14M, 2026-05-05, **139.1 MB** | Older vehicle framework | none stated in repo | Large | **skip** (pick one framework; Glide is MIT and newer). |
| ACF-3 Armored Combat Framework | ACF Team | 3248769144 <https://github.com/ACF-Team/ACF-3> | 33k / 9.9k, 2026-10-09, **346.8 MB** | Armour/penetration damage system | MIT | Adds `CanTool`/`PhysgunPickup`/`PlayerSpawnedProp` hooks. No source found describing real physical units | **skip** (too big, combat-themed). |
| GWater2 | meetric1 | 3403736992 <https://github.com/meetric1/gwater2> | 351k / 116k, 2026-09-17, 16.4 MB | GPU fluid sim (Nvidia FleX), reaction forces, rain emitter | GPL-3.0 | **Needs binary modules installed manually** (cannot be bundled on Workshop, per README/search); DirectX11 GPU. README support table: Windows x86-64 branch yes, Windows 64-bit main no, Linux Nvidia yes, Linux AMD/Intel only under proton, macOS no | **skip** for now. Manual DLLs break the "no pre-download" goal. |
| Buoyancy Tool | creator id 76561198000609376 | 104818321 | 622k, 2012-10-26 | Make props float | none stated | Engine API `PhysObj:SetBuoyancyRatio` exists; wiki notes the getter is unavailable on x86-64 beta and macOS | **skip**; set buoyancy from Lua. |
| Heavier Ragdolls | creator id 76561198257209664 | 2821640676 | 256k, 2024-12-23, ~0 MB | Ragdoll mass/gravity feel | none stated | Reported conflicts with other ragdoll addons (via search) | skip |
| Simple Prop Damage / TENSION | unverified | IDs not checked | - | Prop break; constraint-strain effects | unverified | TENSION reacts only, does not alter physics (per its page via search) | skip |

Real physical units: **no public addon found that advertises them.** The wiki gives 1 m ~ 39.37 source units and per-object gravity via
`PhysObj:EnableGravity` / `ApplyForceCenter`, so the Mentat bridge must supply units itself. Gravity/orbital tools: none found.
Fire and smoke: nothing credible found.

## 3. Visuals and stage

| Addon | Author | ID / URL | Subs, updated, size | What it does | Licence | Notes | Verdict |
|---|---|---|---|---|---|---|---|
| StormFox 2 | Nak | 2447774443 <https://github.com/Nak2/StormFox2> | 727k / 171k, 2026-10-06, 31.4 MB | Weather, day/night, sun/moon, wind, lighting, surface editor | GPL-3.0 | Only entity-specific `PhysgunPickup`/`CanPlayerUnfreeze` hooks. Reported clash with some viewmodel addons (via search). Clients render effects. Separate content pack item 2447979470 (not checked) | **try**. Fits stage lighting; the black=transparent palette fights sky/weather, so expect to disable it in AR mode. |
| StormFox 1 | Nak | 1132466603 | 770k / 146k, 2026-10-05, 34.9 MB | Predecessor | - | still updated | skip (use SF2) |
| Atmos | looter | 185609021 | 1.61M / 516k, **2019-11-13**, 1 MB | env_skypaint day/night | none stated | Threads report black/blinking skybox | skip |
| Real CSM | Xenthio (orig. Blueberry_pie) | 2817072898 <https://github.com/Xenthio/RealCSM> | 438k / 121k, 2026-06-02, **130.7 MB** | Cascaded sun shadows | BSD-3-Clause | Heavy GPU cost; `r_flashlightdepthtexture` needed; prop-shadow trouble with a ReShade addon (via search); settings reset per map without `sbox_persist` | **try** for desktop screenshots only; wrong for AR. |
| Dynamic Volumetric Fog | creator id 76561198367278639 | 3280216633 | 83k, 2025-01-17 | Fog | unverified | none verified | skip (palette doctrine) |

Volumetric clouds, SSAO, SSR, PBR/cubemap packs: **no public Garry's Mod addon found** in two search passes.

**Maps with only base content.** Required content could not be verified for any map (item pages blocked). Known/likely:
- `gm_construct` and `gm_flatgrass` ship with the game (general knowledge; not checked in a fetched source).
- Candidates from Workshop browse (sizes/dates verified, **content requirements unverified**): `gm_flatgrass_empty` 3787080776 (3.9 MB, 2026-08-20, 537 subs); `Blank` 411904602 (4.1 MB, 2015); `gm_Blank` 1124071212 (0.6 MB); `Flatgrass Classic` 104609986 (0.2 MB, 181k subs); `Flatgrass Ramped` 409296486 (8.2 MB); `Showroom maps` 229642590 (69 MB); `Studio` 1126304494 (18 MB); `InfMap2 CE` 3423422716 (46.7 MB, "infinite map base", 2026-01-08).
- Search material says many community maps need CS:S, which the Linux VM lacks. **Test each map on the VM before adopting.**

## 4. Avatars, characters, animation

| Addon | Author | ID | Subs, updated, size | What it does | Licence | Verdict |
|---|---|---|---|---|---|---|
| DrGBase NextBot base | Dragoteryx (repo Zaurzo/drgbase) | 1560118657 <https://github.com/Zaurzo/drgbase> | 4.36M / 2.75M, 2026-10-07, **0.6 MB** | NextBot/NPC framework with possession | **none stated in repo** | **try** for populating a town; tiny. Hooks: own projectiles (`PhysgunPickup`, `GravGunPickupAllowed`), `PlayerSpawnedProp`. Repo last push 2024-11-07. NPC-reaction conflicts reported. |
| Ragdoll Mover | Winded, maintained by PenolAkushari | 104575630 | 2.0M / 873k, 2025-08-15, 0.4 MB | Bone/IK posing | none stated | **adopt** for posing. Finger posing limited by vanilla tool; undo only for physbones. |
| Enhanced Faceposer | creator id 76561197960440071 | 148090220 | 676k / 255k, 2025-11-25, ~0 MB | Flex/face posing | none stated | **try**. Conflict with "Composite Bonemerge Tool" reported (via search). |
| Improved Faceposer Tool (TF2) | - | 2864740314 | 171k, 2024-04-11 | Same idea, TF2 flavour | none stated | skip (TF2 models) |
| Auto-Poser [BETA] | - | 2563582365 | 177k / 48k, 2021-08-11 | Ragdoll pose copy/paste | none stated | try |
| Enhanced PlayerModel Selector | LibertyForce | 504945881 | 3.61M / 1.94M, **2020-07-15** | Model picker | none stated | skip; localization fork 2247755443 (229k, 2026-10-01) reportedly crashes on x86-64 branch |
| Gmod Legs 3 | creator id 76561197996456215 | 112806637 | 3.3M / 1.3M, 2026-05-07, ~0 MB | Own-body legs in first person | none stated | try (AR body presence); may clash with the avatar bot rendering |
| PAC3 | creator id 76561197978977007 | 104691717 | 2.4M / 915k, 2026-10-11, 8.7 MB | Outfit/parts editor | none stated | skip (scope) |
| VRMod | Catse | 1678408548 | 890k / 290k, 2025-12-03, 1.5 MB | Experimental VR; **needs a separate binary module** (catsethecat/vrmod-module; repo URL 404'd when fetched) | none stated | **later**. SteamVR-based per search; a comment claims Quest hand tracking works (unverified). No gesture/hand-tracking GMod addon found. Viture compatibility: unknown. |

Lip sync: **no public voice-to-flex addon found.** Engine side is `Entity:SetFlexWeight` (96 controllers per wiki via search). Build it in mentat-stage from the audio level you already have.

## 5. Quality of life

| Addon | ID / source | Data | Verdict |
|---|---|---|---|
| Catmull-rom Cinematic Cameras | 198542384 | 616k / 145k, 2021-07-03, 0.5 MB | **try** for camera paths |
| Advanced Camera | 881605937 | 378k, 2022-04-09 | skip unless needed |
| Stop Motion Helper | 111895870 | 693k / 248k, 2022-02-02 | try for animation |
| Nadmod Prop Protection | 159298542 | 355k, **2020-03-04** | **skip**: ownership model overlaps your registry/undo |
| ULib / ULX | 557962238 / 557962280 | 2.8M, 2025-08-24 / 2.9M, 2024-03-24; ULX licence NOASSERTION | skip unless ranks are needed |
| GmodAdminSuite | 1595332211 | 912k, 2026-10-02, 4.8 MB | skip |
| BadCoderz | <https://github.com/ExtReMLapin/BadCoderz> (MIT, last push 2024-10-13) | finds slow addon code | **try** (dev-only) to audit addons |

Server query/performance monitors: no mature public Workshop one verified; "FProfiler" could not be found.

## 6. Content delivery

| Question | Answer | Source |
|---|---|---|
| Deliver Workshop items | Server: `+host_workshop_collection <id>` downloads and mounts the collection (nested collections too; must be public/unlisted). Also `cfg/srcds_workshop_ids.txt` (needs version 2026.09.22+). A Workshop map/gamemode matching the collection is automatically offered for client download | <https://wiki.facepunch.com/gmod/Workshop_for_Dedicated_Servers> |
| Make clients download | `resource.AddWorkshop("<id>")` forces joiners to download an addon before joining; **one ID per call, collections not allowed**; Lua-only addons need none; limit 8192 downloadable files shared across `resource.Add*` | <https://wiki.facepunch.com/gmod/resource.AddWorkshop> |
| Download limits | WorkshopDL "up to ~2gb (gma) at once"; ServerDL limited to 64 kB/s; FastDL via `sv_downloadurl` | <https://wiki.facepunch.com/gmod/Serving_Content> |
| Server without HL2/CS:S | Mounting via `cfg/mount.cfg` and `cfg/mountdepots.txt`. Wiki: without correct config, HL:S Hunters and TF2 characters cannot spawn and addons report missing content. Hosting guides add purple checkers and ERROR models, that clients need the content too, and that Linux paths are case-sensitive; mount only games you own | <https://wiki.facepunch.com/gmod/Mounting_Content_on_a_Dedicated_Server>, <https://supercraft.host/wiki/garrys-mod/missing_textures_and_errors/> |
| Runtime mount by connected clients | `steamworks.DownloadUGC(workshopID, cb)` (client/menu; workshop ID, not file ID) yields a `.gma`; `game.MountGMA(path)` mounts it (client and server). Wiki: already-loaded error **models** are reloaded, error **materials are not**. Behaviour for a client already in a map is not documented: **partly verified, needs a test** | <https://wiki.facepunch.com/gmod/steamworks.DownloadUGC>, <https://wiki.facepunch.com/gmod/game.MountGMA> |
| Server-side runtime download | `gmsv_workshop` (MIT, last push 2026-10-03) adds `steamworks.DownloadUGC`/`FileInfo` on the server; DLL into `garrysmod/lua/bin`; use `srcds_run_x64` for 64-bit | <https://github.com/WilliamVenner/gmsv_workshop> |

## Ranked shortlist: top 10 to try first

| # | Addon | Why | Risk | Install effort |
|---|---|---|---|---|
| 1 | Wiremod 160250458 | Standard construction/logic, Apache-2.0, 20.7 MB | Low | Subscribe/collection |
| 2 | Advanced Duplicator 2 773402917 | Save/paste real-parts builds, 0.2 MB | Low-med (bypasses your registry) | Subscribe |
| 3 | gmsv_workshop | Server can fetch/mount the 4,000 models on demand | Med (binary module must match branch) | One DLL |
| 4 | StormFox 2 2447774443 | Day/night and stage lighting, GPL-3.0 | Med (palette conflict, viewmodel clash) | Subscribe, 31 MB |
| 5 | Ragdoll Mover 104575630 | Posing for avatar and props | Low | Subscribe, 0.4 MB |
| 6 | DrGBase 1560118657 | Populate the town, 0.6 MB | Med (no licence, NPC reaction conflicts) | Subscribe |
| 7 | Enhanced Faceposer 148090220 | Face flex posing for the avatar | Low-med | Subscribe |
| 8 | Precision Tool 104482086 | Precise part alignment | Low (frozen 2016) | Subscribe |
| 9 | Catmull-rom Cameras 198542384 | Cinematic shots | Low | Subscribe |
| 10 | Glide 3389728250 | MIT vehicle base if vehicles are wanted | Med (167.5 MB download) | Subscribe |

Next tier: Real CSM (desktop only, 131 MB), VRMod (only if a SteamVR path is chosen), BadCoderz (audit tool).

## Avoid / known conflicts

- **SProps (173482196)**: removed for guideline violation, marked incompatible.
- **GWater2**: manual binary DLLs per machine; no Windows 64-bit-main support.
- **Nadmod PP, ULX/ULib**: overlap your ownership/undo model; Nadmod last updated 2020.
- **ACF-3 (347 MB), simfphys (139 MB), Metrostroi (90-527 MB)**: break the "no hundreds of MB" rule.
- **Atmos, original Enhanced PlayerModel Selector**: stale (2019/2020); EPS forks reportedly crash on the x86-64 branch.
- **Glide + TacRP**: reported input failure. **StormFox 2 + viewmodel addons**: reported arm reset. **Real CSM + ReShade addons**: prop shadow breakage.
- Any map requiring CS:S on the Linux VM.
- Hooks seen: Glide `CanTool`; ACF `CanTool`/`PhysgunPickup`/`PlayerSpawnedProp`; DrGBase `PlayerSpawnedProp`/`PhysgunPickup`/`GravGunPickupAllowed`; StormFox2 `PhysgunPickup`/`CanPlayerUnfreeze`; Wire E2 `PhysgunPickup`. None touch `PlayerCanHearPlayersVoice`.

## Questions for Aaron

1. Does the AR client go through SteamVR (VRMod viable) or a Viture-native path where VRMod is irrelevant?
2. What is the per-player download budget (e.g. 50 MB vs 200 MB)? It decides Glide and Real CSM.
3. Run `gmsv_workshop` and test late `game.MountGMA` on connected clients, or stay with a collection and pre-download?
4. Must Wire/E2 and AD2 builds be forced through the registry/undo (wrapping them), or is a separate "builder" mode acceptable?
5. Are weather and sky effects wanted at all given the pure-black-is-transparent doctrine, or only in desktop mode?
