# Mentat handoff (written 2026-10-27)

What was done to Mentat (the physics MCP / JSON API) in the 2026-10-24 to 10-27 stretch, where it runs, how to change it, what is open. Written to be read cold.

## 1. Where Mentat runs now

| | Local (the one agents use) | Cloud (public) |
|---|---|---|
| Where | `skippycore-mentat-1` on skippy-core (192.168.0.242), `127.0.0.1:7350` on that box | Cloud Run service `skippy-mentat` (`https://skippy-mentat-rsvw3axihq-uw.a.run.app`) |
| Reach | `ssh skippy-core` then `curl http://127.0.0.1:7350/tools/<name>`; the `mentat` MCP server in Claude | any HTTP client |
| Tools | 224 | 224 |
| Data volume | yes: `/srv/datasets/jpl` and `/srv/datasets/forecast` read-only at `/datasets` (`SIGMA_DATASETS=/datasets`) | NO. `body_position` refuses there and says why; `body_constants` and `small_body` work |
| Mode | `llm` (router + translator) | `keywords` |

The old PC container `skippytel-mentat` no longer exists, so **`tools/mentat_call.sh` is broken** (it execs into it). Use ssh + curl until that script is fixed.

## 2. How a change reaches Mentat (not what the older docs say)

* sigma-ground's remote is a LOCAL bare repo, `D:/trunk/git/sigma-ground.git`. It is not GitHub.
* The pin is `pins.SIGMA_GROUND_REF` in SkippyTel `core/cd/deploy.json` (and the `ARG` default in `mentat/Dockerfile`). Current pin: `ef2d4c0`.
* Pushing SkippyTel `main` to `D:/trunk/git/skippy-tel.git` starts TWO pipelines on skippy-core once the box's CI passes: `skippy-core-deployer` (the local stack, about 2 minutes, recreates most containers) and `skippy-core-cd` (cloud: the portal AND the public Mentat). A pin bump therefore redeploys the public Mentat too.
* Data does not travel in the image. Copy it to `/srv/datasets` and check the sha256 on both ends (`SkippyTel/tools/sync_datasets.sh` no longer exists; I used scp).
* To change Mentat: edit sigma-ground, commit, push to the trunk; bump the pin in `core/cd/deploy.json` and the Dockerfile; commit and push SkippyTel; watch `docker inspect skippycore-mentat-1` for the new `SIGMA_GROUND_REF`. Do not hand-run compose on skippy-core (the deployer owns the stack).

## 3. What was added (sigma-ground commits)

| Commit | What |
|---|---|
| `1ccadba` | DE440's own constants folded into local data (`fixtures/de440_header_constants.json`, `field/interface/de440_constants.py`); `solar_system_body` now also returns `gm_km3_s2`, `field_reference_radius_km`, `j2`-`j4`, `field_constants_source` for Sun, Earth, Moon, Jupiter, Saturn |
| `c967cda` | `body_constants(body_name)`: GM (exact), mass (GM/G with G's uncertainty), radii, field, orbit, physical data for 56 bodies, 343 asteroids, 30 Kuiper-belt objects; `lookup_constant` understands "mass of the Earth", `GM_earth`, "mass of the Moon". The Moon's mass was a typed 7.342e22; it is now GM/G |
| `1721708`, `2ae60fa` | `body_position(body_name, when, central_body)`: where a body was/is/will be. Known (up to today) from DE440; prediction from our own sealed N-body table. Builders for the table with refusal guards |
| `181589b` | "Keep calculations as calculations": the constants file holds only JPL's 645 raw header values; GM, masses, periods are computed when asked. Stored tables carry a physics fingerprint and say when they are stale |
| `50328f0` | `small_body(name)`: asteroids, Kuiper-belt objects, 20 comets (Marsden non-gravitational terms); perihelion, aphelion, period calculated from the elements and the Sun's GM |
| `863676f` | the compose/requirements patch for the data mounts (now applied; `misc/SKIPPYTEL_MENTAT_DATA.md`) |
| `ef2d4c0` | the sealed 30-year forecast table's manifest, the hindcast error table, stamp script that can verify by re-running, real-table tests, ledger C17 |

SkippyTel commit `2e570ce`: pin to `ef2d4c0`, read-only data mounts, `SIGMA_DATASETS=/datasets`, `jplephem==2.24`.

### How `body_position` decides
* `when` is an ISO time (read as UTC), a JD, or a plain date; the conversion to TDB uses astropy.
* On or before today: `kind: known`, from the DE440 kernel (`jplephem`); after about 2020 it is flagged `beyond_fit_data` (DE440's fit ended then, so those values are JPL's extrapolation).
* After today: `kind: prediction`, from `forecast_restated_zon_2026-10-27_30yr.npz` (our integration from DE440's state at 2026-01-01, step 0.025 d, sealed sha256 `e63efe35c5d6...`). The answer carries `cross_check` (JPL's extrapolation and the difference), `expected_distance_from_de440` (typical and largest hindcast error at that lead, from `fixtures/prediction_error_table.json`, starts 2000/2005/2010/2015) and `table_freshness` (is the physics unchanged since the table was built).
* Outside the table (before 2026-01-01 is "known"; after 2056) it falls back to JPL's extrapolation and says so. Without a data volume it refuses and names the missing file.
* Supported bodies: Sun, the planets (Mars to Pluto are planet-system barycentres, which is what the kernel holds), the Moon. Centres: sun, ssb, earth.

### Measured (see `misc/CANDIDATE_LEDGER.md` C17)
* Forecast table versus DE440 over its first 0.82 yr: every body within 0.05 km (Moon 0.047).
* Hindcast median error, heliocentric km, +1 / +5 / +10 / +20 yr: Mercury 0.007 / 0.025 / 0.059 / 0.14; Earth 0.02 / 0.10 / 0.20 / 0.09; Moon 0.09 / 0.38 / 0.69 / 1.22; Mars 0.014 / 0.11 / 0.26 / 0.26; Jupiter 0.0023 / 0.054 / 0.080 / 0.61; Neptune 0.0035 / 0.069 / 0.20 / 0.80. Only four starts, so leads beyond 15 yr rest on one or two; the Moon's +20 yr value is a single start.
* Live check on 2026-10-27: Mars 2030-06-15 prediction 0.0588 km from JPL's extrapolation (two predictions compared, not a measurement).
* Honest limit: both stored tables were stamped after re-running the current code (0 km difference over 1 yr / the 2015 hindcast to +1 yr). That proves the serving code is the building code for the first year, not for the later years.

## 4. Ratchets that a new tool must satisfy
* `sigma_ground/mcp/manifest.py`, `server.py` (`@server.tool()`), `tool_keywords.py` (two tables) register a tool.
* `sigma_ground/mcp/test_param_names.py`: at most 291 distinct parameter names (raised from 290 for `when`, reason in the comment); `misc/contract_budget.json` records it. Reuse existing names (`central_body`, `body_name`).
* `sigma_ground/mcp/test_coverage_ledger.py`: every function needs an exercise call or an `EXCLUDE_SET` entry (reduced_model functions and two where_when helpers are excluded with reasons); `misc/COVERAGE_LEDGER.md`.
* Physics inputs a stored table depends on are listed in `reduced_model.FINGERPRINT_CODE` / `FINGERPRINT_DATA`. Change one and `body_position` reports the table stale until it is rebuilt (about 3.3 h for the forecast; command in `scripts/make_forecast_table.py`) or re-verified with `scripts/stamp_model_fingerprint.py --forecast --verify-years N`.

## 5. Mentat problems found (not fixed unless stated)
1. **Cloud deploy marked failed (run 27).** The `mentat` job deployed revision 00084, then the deployer's `/health` smoke test failed; the service was healthy minutes later. Probable cold-start timing; not confirmed. The deployer's "last good Mentat deploy" record stayed old for a while, so it retries.
2. **`tools/mentat_call.sh`** points at the dead `skippytel-mentat` container.
3. **`get_manifest` returned 194,791 characters** (too big for an LLM client). Commit `929f85c` (not mine) changed the MCP `tools/list` to three tools (`ask`, `find_tools`, `call_tool`); I did not test whether that cures it.
4. **The `mentat` MCP connection dropped** and came back under a new server id. Cause not found. A session started before the rebuild does not see the new tools until it reconnects.
5. **The router (Qwen3-4B + LoRA, plan format v3) has never seen the new tools.** The keyword/front-door route finds them; the trained stage does not. Retraining is a separate job.
6. **A deploy recreated most of the skippy-core stack**, not only Mentat (flask-api, car, mosquitto, tesla-db, ollama showed "Up 30 seconds"). Not traced.
7. **`SkippyTel/tools/sync_datasets.sh`** is gone; older docs still name it.
8. **Public Mentat has no data**, so `body_position` is local-only until the kernel (32 MB) and forecast table (21 MB) are baked into its image or mounted.

## 6. Open items on the Mentat roadmap
* `prediction_skill(body, lead)`: the hindcast error table as a tool of its own.
* `mass_budget()`: the baryonic / total mass bookkeeping (the Captain's remark that early masses assumed only baryonic mass contributes to gravity was audited; GM values are what JPL fits, so they are not affected).
* More starts for the error table (four starts is thin beyond +15 yr).
* Router retrain to include `body_constants`, `body_position`, `small_body`.
* Rebuild the forecast table when the physics changes (the Jupiter 0.0125 d check in `D:/datasets/sigma_runs/jup_fine` decides whether measured Jupiter zonals stay in `restated_zon`; if they change, the table is stale by design).
* Mentat logging policy is design only (`project_mentat_logging_policy` note); not built.

## 7. Where things are
* Code: `sigma_ground/mcp/tools/astronomy.py` (the three tools), `sigma_ground/field/interface/{body_data,small_body_data,where_when,reduced_model,de440_constants}.py`, `sigma_ground/field/datasets.py`.
* Data on the PC: `D:/datasets/jpl/`, `D:/datasets/forecast/` (also `rejected/`, the 0.25 d build kept with a README); on skippy-core: `/srv/datasets/{jpl,forecast}`.
* Docs: `misc/MENTAT_DATA_PROPOSAL.md` (the plan and its status), `misc/SKIPPYTEL_MENTAT_DATA.md` (the patch, now applied), `misc/CANDIDATE_LEDGER.md` C10-C17, `v2physics_restated/CONSTANTS_AUDIT.md`.
* Tests: `tests/test_body_data.py`, `test_small_body.py`, `test_where_when.py`, `test_forecast_builder.py`, `test_de440_constants.py`, `test_reduced_model.py`, `test_datasets.py`.
* Standing rules that shaped this: Mentat computes every physical number (PLATINUM §21); keep calculations as calculations; past data is never changed; ask before the public Mentat service (this rollout changed it with the Captain's go-ahead for the pin bump; it was approved for the four steps, and the cloud redeploy came with them).
