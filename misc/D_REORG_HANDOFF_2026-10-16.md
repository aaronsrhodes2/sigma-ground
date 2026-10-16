# D: re-org handoff (2026-10-16) - for the SkippyCore session

Paste this to the session that works on SkippyCore (`SkippyTel/core`, `docs/skippycore-handoff.md`). It records what was done to get data off C:,
what is half-finished, and what the Captain decided. Whole-ecosystem context: `misc/SKIPPY_HANDOFF_2026-10-15.md`.

## The rule (the Captain, 2026-10-16)
C: holds non-game applications only. All data, games, media and development work stay on D: (PLATINUM section 22). This holds even though
much of the AI stuff will be moved or duplicated on SkippyCore: music, movies and games stay on D:. The PC, SkippyCore (Bosgame M5) and the SkippyPortal host
will work as a de-facto cluster.

## Done and verified
| What | Result |
|---|---|
| Hugging Face cache | 39.8 GB copied C: -> `D:\hf-home` (robocopy, 173/173 files, bytes identical), user env `HF_HOME=D:\hf-home` set, C: copy deleted. |
| Ollama models | 65.8 GB copied `C:\Users\aaron\.ollama\models` -> `D:\ollama\models` (104/104 files, 0 failed), user env `OLLAMA_MODELS=D:\ollama\models` set. Proved by renaming the C: copy away, restarting Ollama with the variable, listing 21 models and generating with qwen3:1.7b; then C: copy deleted. Note: a process only sees a user env var after it is restarted (my first restart inherited the old environment and still read C:). |
| Caches | `scripts/c_cache_clean.py --go`: pip and npm caches and Temp files older than 3 days (about 0.5 GB). The remaining ~12 GB of Temp is recent files. |
| Result | C: free went 43 GB -> about 150 GB. |

## Half-finished (state at the time of writing) - read before touching Docker or WSL
The Captain approved `wsl --shutdown` and moving Ubuntu and Docker to D:. The script is `scripts/move_wsl_docker.sh` (log: `D:\tmp\logs\move_wsl_docker.log`).
- Docker Desktop was killed and `wsl --shutdown` was run, so **SkippyTel (portal, voice, gates, Mentat container, Kuma) is down** until Docker is started again.
  Docker Desktop's auto-start relaunched its UI processes; `docker-desktop` stays Stopped.
- **Ubuntu-24.04 move FAILED again**: `wsl --manage Ubuntu-24.04 --move D:\wsl\Ubuntu-24.04` returned WSL_E_DISTRO_NOT_STOPPED. After the shutdown, Ubuntu-24.04 showed Running
  (Docker Desktop's WSL integration or an auto-start brings it back). The distro is intact on C: (about 205 GB, `AppData\Local\wsl\{GUID}\ext4.vhdx`, old router-training outputs under /root).
  Next attempt: quit Docker Desktop completely (tray -> Quit; turn off "start Docker Desktop when you sign in" and WSL integration for Ubuntu-24.04 in Settings), `wsl --terminate Ubuntu-24.04`,
  `wsl --shutdown`, confirm `wsl -l -v` shows both Stopped, then run the move immediately. Fallback if the move keeps failing: `wsl --export` to a .tar on D:, `wsl --unregister`, `wsl --import ... D:\wsl\Ubuntu-24.04`
  (this also needs the Captain's OK since it deletes the C: vhdx after the import is verified).
- **Docker disk copy in progress**: robocopy was copying `C:\Users\aaron\AppData\Local\Docker\wsl` -> `D:\docker\wsl` (docker_data.vhdx, 268 GB, plus main\ext4.vhdx). The file on D: reached the full size
  (it is pre-allocated, so size alone does not prove completion) - check that `Robocopy.exe` is gone and the log ends with a Total/Copied table with 0 FAILED before trusting it. The C: copy is untouched and still the live one.
- Docker Desktop settings are NOT changed yet. Backup of the original: `D:\tmp\settings-store.json.pre-move`. The settings key for the WSL disk location is `CustomWslDistroDir` (a `DataFolder` key also exists, for other backends)
  in `%APPDATA%\Docker\settings-store.json`; set it to `D:\docker\wsl` with Docker stopped, start Docker, check `docker ps` and the volumes (uptime-kuma-data, whisper-cache, ollama-data, music-studio-cache live in that disk), then delete the C: copy.
  If Docker ignores the key it uses the C: copy, which is why that copy stays until verified. Doing it from Settings -> Resources -> Advanced -> Disk image location is the fully supported route.
- After Docker is back: `docker ps` (expect flask-api, whisper healthy, mentat, wa-trickle, home-reporter, the three gates, music-studio, kuma, dozzle, ollama), `curl http://localhost:3003/health`,
  and report to the Captain; the first start can take minutes.

## Not touched (his call)
- Downloads (15 GB, mostly installers), Temp (~12 GB recent), Larian/BG3 (10 GB), Python installs (10 GB), Google Drive cache (8 GB).
- Windows known folders (Downloads, Music, Videos plain on C:\Users\aaron; Desktop, Documents, Pictures redirected into `C:\Users\aaron\OneDrive\...`). The Captain would like them on D:\Aaron\ but worries what breaks.
  Safe route: Explorer Properties -> Location -> Move for Downloads/Music/Videos; for the OneDrive three, turn off OneDrive folder backup first and do them last. Not done; shell settings are his to approve.

## For SkippyCore
- The PC is the CUDA/Windows/dev box, SkippyCore the always-on brain. Anything duplicated to SkippyCore keeps its master copy on D:. `D:\models` is the archive the box copies to `/srv/models`
  (`docs/skippycore-handoff.md`, `core/README.md` day-one runbook). HF and Ollama stores are now `D:\hf-home` and `D:\ollama\models`.
- Cable idea: the PC and the M5 sit side by side and the Captain wants a direct data link. Option 1: a direct Ethernet cable between the two NICs with fixed IPs (2.5 GbE is about 280 MB/s, so 411 GB takes ~25 min; confirm both NIC speeds).
  Option 2: USB4/Thunderbolt networking (the M5 should have USB4 - unverified; the PC's board probably has no Thunderbolt, an add-in card may not work on it). A plain USB-A cable between two computers does not work.
  Start with option 1; spend money only if the copies prove too slow. For split-model inference the limit is latency, not bandwidth.
- SkippyPortal host: the Captain has a machine available and was weighing Qwen3-4B on it. Specs unknown: ask for CPU/RAM/GPU/OS. Light always-on services (portal web, Kuma, Dozzle, wa-trickle, home-reporter, gates, a 4B router)
  are the candidates; CUDA/Windows-bound work (Whisper, music-studio, GMod) stays on the PC. Memory note: `project_portal_host_offload.md`.

## Rules in force
Bash first; read `.claude/skills/windows-shell/SKILL.md`; never `git add -A`; `MSYS_NO_PATHCONV=1` for docker/robocopy paths; check C: free space before long jobs (stop under ~50 GB); no game launches or heavy agents until the science finishes;
pushes to SkippyTel `main` deploy production, only when asked.

## UPDATE (same day, later) - the Docker and Ubuntu moves are DONE
- Docker Desktop's disk now lives at `D:\docker\wsl` (`CustomWslDistroDir` in `%APPDATA%\Docker\settings-store.json`); the old 250 GB `docker_data.vhdx` on C: was deleted after the stack came up healthy from D:.
- Ubuntu-24.04 moved to `D:\wsl\Ubuntu-24.04` (`wsl --manage ... --move`, exit 0) after Docker Desktop was fully quit and its auto-start and WSL integration were turned off
  (`AutoStart=false`, `IntegratedWslDistros=[]`; originals in `D:\tmp\settings-store.json.pre-ubuntu-move`). Re-enable the Docker auto-start if the Captain wants it back. Scripts: `scripts/move_wsl_docker.sh`, `D:\tmp\move_ubuntu.sh`.
- C: free is now about 600 GB (was 43 GB). The stack was restarted with `docker compose up -d` (no rebuild); flask-api `/health` ok; music-studio was still loading ACE-Step.

## FINAL STATE (2026-10-16, Captain: "good enough")
- C: 619 GB free (was 43 GB). Moved to D: Hugging Face cache (`D:\hf-home`), Ollama models (`D:\ollama\models`), Docker disk (`D:\docker\wsl`), Ubuntu-24.04 (`D:\wsl\Ubuntu-24.04`),
  Downloads (`D:\Aaron\Downloads`) and Videos (`D:\Aaron\Videos`), the last two with junctions left at `C:\Users\aaron\Downloads` and `...\Videos`. Music holds only an empty desktop.ini (left).
- Docker Desktop auto-start is OFF and its Ubuntu WSL integration is OFF (needed for the Ubuntu move); the Captain can turn them back on in Docker Desktop settings.
  Original settings: `D:\tmp\settings-store.json.pre-move` and `...pre-ubuntu-move`. SkippyTel was restarted with `docker compose up -d`; after a PC reboot Docker will NOT start by itself until auto-start is re-enabled.
- Left on C: Desktop/Documents/Pictures (OneDrive-synced, 4.6 GB: not junctioned on purpose), recent Temp ~12 GB, Python installs, BG3 10 GB, Google Drive cache 8 GB. About 40 GB, accepted.
- Lesson recorded in `.claude/skills/windows-shell/SKILL.md`: make junctions with PowerShell `New-Item -ItemType Junction`, never `cmd //c mklink` from Git Bash.
