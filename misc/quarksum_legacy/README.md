# quarksum legacy

quarksum (`D:\Aaron\development\quarksum`) was merged into sigma-ground on 2026-10-27 and its directory moved to
`../_archive/quarksum-legacy/`. Its git history is unrelated to sigma-ground's, so it is kept as two tags rather than
merged into master:

| Tag | Commit | What |
|---|---|---|
| `quarksum-legacy/master` | d2c88ae | quarksum master, 29 commits (2026-03 to 2026-06) |
| `quarksum-legacy/nostalgic-kapitsa` | d8f832b | side branch, sessions 14-15 (J2-J4 zonal terms, PhysicsToggles, MCP benchmark) |

Every other quarksum branch pointed at one of these two commits or an ancestor.

Content check at merge time (both tips, 261 tracked files): 224 were byte-identical to blobs already in sigma-ground,
12 had the same path with older content (sigma-ground's is the newer version; the J3/J4 fixes landed as a894112 and
19ec88e), and 25 existed only in quarksum. Of those 25, the files below were copied here; the rest stay in the tags:

- Copied: the two double-slit demos (standard Born-rule sampling and Englert duality; the "SSBM" in their titles is a
  label only), operatic scene 15 (quarksum numbering), the non-observer camera note, the coffee-cup shape, and the
  March 2026 test baseline.
- Left in the tags: `theory/` (SSBM cheatsheets, poster, research PDFs, GW150914 waveform; SSBM was removed from the
  model by the 2026-10-11 data-alone rule, see `misc/DISPROOF_LEDGER.md`), `misc/SUMMARY.md` (an 84k-line
  auto-generated dump), `Dockerfile` and `MANIFEST.in` (both point at a `quarksum/` package that no longer exists),
  the old `run_all_tests.py` / `summary.py` scripts, and `.DS_Store`.

To read anything left behind: `git show quarksum-legacy/nostalgic-kapitsa:<path>`.
