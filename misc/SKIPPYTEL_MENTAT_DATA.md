# Giving the Mentat container on skippy-core its data (patch, for approval; nothing applied or deployed)

`misc/SKIPPYTEL_MENTAT_DATA.patch` is a unified diff against the SkippyTel repo (checked: `git apply --check --cached` succeeds against the index; the working tree here is CRLF, so use `git apply --ignore-whitespace` there).

## What it changes
* `core/compose.core.yml`, service `mentat`: mounts `${CORE_DATASETS:-/srv/datasets}/jpl` (106 MB: the DE440s kernel and the Horizons copies) and `.../forecast` (the sealed N-body forecast table that `body_position` serves for future dates, about 20 MB) read-only under `/datasets`, and sets `SIGMA_DATASETS=/datasets`. Nothing else on the box is exposed, nothing is writable. The forecast folder is created by `scripts/make_forecast_table.py` on the PC and has to be mirrored with `tools/sync_datasets.sh --apply` before the container is started, or the mount source will not exist.
* `mentat/requirements.txt`: adds `jplephem==2.24`, the package that reads the kernel. The image does not have it today (astropy does not pull it in), so without this the kernel cannot be read.
* The container's user (`mentat`, non-root) can read the files: they are mode 644 in directories mode 755.

## What it does NOT change, and what has to happen first
1. **The image gets sigma-ground from GitHub at a pinned commit** (`SIGMA_GROUND_REF` in `mentat/Dockerfile` and `.github/workflows/deploy.yml`, now `25466d4`). The new tools, the DE440 constants, the reduced model and `sigma_ground/field/datasets.py` exist only in local commits (6 ahead of `origin/master`). They have to be pushed (the repo is public) and the two `SIGMA_GROUND_REF` lines bumped to the new commit before a rebuild brings any of it into Mentat. I have not pushed anything.
2. **Rebuild and restart on skippy-core** with the usual SkippyTel route (the core deployer / `docker compose -f core/compose.core.yml up -d --build mentat`), then check from this PC: `ssh -o HostKeyAlias=skippy-core aaron@192.168.0.242 'docker exec skippycore-mentat-1 ls /datasets/jpl/de440s.bsp'` and `bash tools/check_mentat_deploy.sh` (set `MENTAT_CONTAINER=skippycore-mentat-1`, and it would need to run over ssh to see that container).
3. **Keep the mirror current.** `SkippyTel/tools/sync_datasets.sh` (rsync over the LAN to `192.168.0.242`) is already the way `/srv/datasets` is filled; run it with `--apply` after any new fetch.

## Seen on the way, not touched
* `core/compose.core.yml` sets `MENTAT_ROUTER_MODEL` twice in the mentat service (lines 88 and 89); the later one wins.
* If the data mount is missing, `datasets.de440_kernel()` returns `None` and the tools are meant to say "the kernel is not installed here", not fail; this is tested in `tests/test_datasets.py`.
