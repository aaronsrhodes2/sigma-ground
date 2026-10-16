# Skippy ecosystem: whole-project handoff (2026-10-15)

Paste this to a fresh session. "SkippyCore" is also a specific sub-project (the Bosgame M5 brain box, in `SkippyTel/core`); this
document covers the whole ecosystem and gives SkippyCore its own section.

## 0. Standing rules (read first)
- Read `D:\Aaron\development\PLATINUM_RULES.md` and the CLAUDE.md files. Key ones: bash first; Mentat does the physics (never hand-compute);
  **§22: D: not C:** (everything created, stored or installed lives on D:; under ~50 GB free on C: stop and tell him); deploy and verify
  every change; commit/push straight to `main` on solo repos (§17).
- Never let a launched game grab his mouse (hands-off tests: `-nomouse -nojoy`, window behind, `GMOD_MOUSE=0`; played sessions `GMOD_MOUSE=2`).
- Real-world physics only, no fudge numbers. Data alone: a claim stays in the model only if it can be disproved and survived
  (`misc/DISPROOF_LEDGER.md`). No numerology.
- At most ~2 agents at once while heavy jobs (Mentat training, N-body) run. He said "I need to let the science finish": no new agents
  or game launches until he says go.
- Secrets live in the vault `G:\My Drive\vault\.env`; read one variable, never print or copy it.
- Pushing SkippyTel `main` deploys production. Push only when he asks. Never commit the Skippy print kit or anything derived
  (`stage/skippyverse/private/` is gitignored).
- Long jobs (> ~20 min) get a retrying driver plus `scripts/jobwatch.py` as a tracked background command.
- Load `.claude/skills/windows-shell/SKILL.md` before scripting here (Write tool for backslash text, never `git add -A`, `MSYS_NO_PATHCONV=1`
  for docker paths).

## 1. Repos and sync state (checked 2026-10-15 after `git fetch`)
| Repo | Branch | State |
|---|---|---|
| sigma-ground | master | 0 behind, **2 ahead (unpushed)**: f3c7ed6 comets as test particles, f62d5e7 Horizons comet parser. 57 dirty files belong to other sessions (misc/, radiance/web/data, fixtures); do not sweep them into commits. |
| SkippyTel | main | **Now level with origin** (rebased; my three local commits were duplicates of origin's and dropped). A stash named "pre-rebase 2026-10-15" holds CRLF copies of the Orion release plus the old settings.local; safe to drop. `.claude/settings.local.json` is the only dirty file. |
| music-collection | main | **Diverged**: 17 behind, 1 ahead (637aeb5 checkpoint). `--ff-only` refused. Needs an explicit rebase/merge decision; dormant since 2026-05-01. |
| remnant-silly | main | Up to date, 8 dirty files. |
| zPhone | main | SHELVED (the portal in a browser is the universal client). 19 commits ahead, unpushed. Do not touch or push without being asked. |
| orion-trail (game repo, github aaronsrhodes2/orion-trail) | main at 8bf1ae6 | The new version exists only on branch `claude/practical-volta-b6x8pw`; `tools/vendor_orion_trail.sh` with default main would revert the live release. Merge the branch or leave it. |

## 2. Live services (all checked)
SkippyTel containers up ~40 h: flask-api (:3003, healthy), ollama, whisper (healthy, healthcheck added), mentat, wa-trickle,
home-reporter, voice-gate, video-gate, music-gate, music-studio (healthy), uptime-kuma (:3001), dozzle (:9999). `/health` answers; default model
`qwen3:8b`. Portal at makerportal.live (Orion Trail live with 19 scenario scripts). Ollama in Docker :11434; benchmark Ollama on :11435 (single slot).
**C: has 43 GB free (96% full): under the 50 GB line.** Still on C:: Ubuntu WSL (204 GB), Docker image (250 GB), Ollama (66 GB), HF cache (39 GB).

## 3. Sub-projects
### 3a. sigma-ground / Mentat (physics + tools)
N-body vs DE440 shoot-out is the proof; fast ephemeris, dense matter, SPARC and the strip are done; comets just landed (test particles, Horizons
parser, 1P/Halley matches JPL to 0.07 km over 5 yr). Router fine-tune study: v1-v3.1 trained routers did not beat stock qwen3:8b on the narrative
benchmark; v3.1 (4B) raised the held-out math suite 194 to 309/315 of 329. Contender plan (local Wolfram-class physics+math): W0-W2 done,
trickle scorer / W3 materials DB / W4 sky tools / W5 text front door open; training LAST. Package map: `ARCHITECTURE.md`. Mentat MCP is down while training runs.

### 3b. SkippyVerse holodeck (Garry's Mod)
"Skippy, add three couches" typed or spoken, real objects appear (4,100 free models streamed on demand, beam-in transporter effect with coherence,
stall and failure wisps). Skippy is a floating bot avatar (silver can, navy coat, gold trim, bicorn hat) with his own voice (ElevenLabs
eleven_v3 via SkippyTel `/tts/stream`), per-player wake names, faces, 20 new spoken lines a day. Real-world gravity/drag/water verified to ~1-4%.
Roadmap: `misc/SKIPPYVERSE_ROADMAP.md`; docs: `stage/gmod/holodeck/README.md`, `stage/gmod/AVATAR.md`. Launch: `bash stage/gmod/skippy_play.sh`
(hold X to talk); **plug the mic in BEFORE launching**. Voice-to-text in the real game with a real mic has not been re-verified.
Admin-only join gate (Captain's SteamID64 only) is on SkippyTel `main`; a dedicated-server proposal with NPC bots is written but not applied.
Resume list (paused): A6 hands + finger beam (replaces the gun), VL2 voice presets (after he listens to `stage/gmod/holodeck/data/voice_lab/index.html`),
FR1 friends route + prompt caching, A3 stage summoning, A5 assemblies, LK2 look pass, a hands-on monkey-test session (ask first).

### 3c. Home / Alexa replacement
9 Leviton Decora Smart HomeKit devices found unpaired (need the 8-digit setup codes); Ring doorbell, garage opener, 2 Espressif devices still
to identify. Audio router (`stage/skippy_home`): here = HDMI "EPSON PJ", upstairs = "Enclave Audio" Bluetooth, "everywhere" synced. Echos to be retired.

### 3d. SkippyCore (the M5 brain box) - `SkippyTel/core`, `docs/skippycore-handoff.md`, `core/README.md`
Bosgame M5 (Ryzen AI Max+ 395, 128 GB) ordered; Ubuntu 26.04, llama.cpp Vulkan behind llama-swap, Ollama-API shim so callers keep `OLLAMA_URL`.
Three slots: chat (resident), general and coder (swap). Built and on `main`: slot tool, shim, pull deployer (2-min systemd timer, waits for green CI,
rolls back on red smoke), compose, smoke test, simulator (`make core-sim`), CI `core.yml`. Open: **register the midnight baseline-model download on
skippy-pc** (`bash tools/install_models_baseline_task.sh` dry run, then `--apply`; ~411 GB to D:\models), pin sha256s, STT/TTS slots, `--api-key` on the
llm port, chat-model choice (Qwen3-30B-A3B vs Instruct-2507), HF gated-repo token (`HF_ID` in the vault). Day-one runbook is in `core/README.md`.
Also `docs/` has a plan for BF16 training on the M5 (commit 7edc007 in sigma-ground).

## 4. Decisions waiting on the Captain
1. music-collection: rebase or merge its 17 upstream commits onto the local checkpoint?
2. Push sigma-ground's 2 comet commits? (Verify nothing else is staged first.)
3. TEMP/TMP redirect to D: (C: is under the line); the C: moves after the science finishes: quit Docker Desktop with `start-docker-clean.ps1`,
   `wsl --shutdown`, `wsl --manage Ubuntu-24.04 --move D:\wsl\Ubuntu-24.04`, Docker disk image to D:, then `OLLAMA_MODELS`/`HF_HOME` to D: with folder moves.
   (`scripts/c_cache_clean.py` is a dry-run-by-default cache cleaner; `--go` deletes.)
4. Merge the orion-trail branch into the game repo's main?
5. Uptime Kuma Whisper monitor needs his Kuma login. Stale quarksum worktrees on C: (wt-caps, wt-pr3, wt-site) need that session's cleanup.
6. Note: some origin commits are dated 2026-10-16 while the local date is 2026-10-15 (another session's clock); harmless.

## 5. First moves for a new session
1. `git fetch` in each repo and re-check section 1; read `misc/SESSION_LOG.md` tail and `misc/SKIPPYVERSE_ROADMAP.md`.
2. Check C: free space and `docker ps`.
3. Ask which track he wants; do not start games, training or agents until he says "go".
