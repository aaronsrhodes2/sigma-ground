---
name: windows-shell
description: Use before running shell commands, writing scripts, launching long jobs or committing on this Windows machine (Git Bash tool, Python 3.13). Lists the failure modes that have cost turns here, each with the fix at its CAUSE, plus the rule for handling a repeated scripting mistake.
---

# Windows shell: known failure modes and where each is fixed

Bash tool first (PLATINUM §20). PowerShell only for Docker Desktop processes. Anything non-trivial goes into a Python script, not a shell one-liner.

## The rule (the Captain, 2026-10-12)
When a scripting problem has a **fixable cause** (the tool, the environment, a convention, a missing helper), fix the cause and record it here, not just the script.
If the fix needs a config change (settings, git config, a package), propose it to the Captain with the reason. A mistake made twice means a cause is still open.

## Symptom -> cause -> fix
| symptom | cause | fix (where) |
|---|---|---|
| A Python file written by a bash heredoc has `"` + newline + `")` (an unterminated string) or tests fail with `IndentationError` | the Bash tool rewrites `\\n` inside heredocs into a backslash-n/newline | **Write any file that contains backslash escapes with the Write tool**, not a heredoc. After writing a script by any route run `python -m py_compile FILE` before running it |
| `UnicodeEncodeError ... cp1252`, or `UnicodeDecodeError` reading a file | Windows Python defaults to cp1252 | `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` are set in `.claude/settings.json` (new sessions); for a single command use `python -X utf8`; always `open(path, encoding="utf-8")` |
| `warning: LF will be replaced by CRLF` on every add/commit | machine-wide `core.autocrlf=true` | `.gitattributes` now has `* text=auto eol=lf` (repo-wide). Do not `git add --renormalize` the repo |
| a figure or data file will not `git add` | covered by `.gitignore` | `git add -f <file>` for deliberate fixtures. Never `git add -A` / `git add .`: other sessions leave uncommitted files here |
| one worker dies and a whole `ProcessPoolExecutor` fails (`BrokenProcessPool`) | a pool breaks when any worker is killed; Windows has killed workers silently three times | one subprocess per job, retried (see `scripts/run_shootout_parallel.py`, `scripts/fair_jpl_contest.py`); results appended per job so a rerun resumes |
| workers die at once with many processes | memory/commit pressure: each Windows worker re-imports and re-parses its inputs | pass each worker only its slice (not the whole fixture), cap workers (~10 here), keep results incremental |
| long job finishes or dies and nobody notices | nothing watching | attach `scripts/jobwatch.py` as a tracked background command (`run_in_background`); its exit wakes the session |
| `sleep N` blocked, polling loops rejected | harness rule, not Windows | wait with `until <condition>; do sleep 5; done` inside a background command, or rely on the completion notification |
| PowerShell inside a bash command mangles `$`, quotes, `%` | two shells parse the string | put it in a `.ps1` file, or use Python `psutil` (process lists, trees, kill by PID) instead |
| `PermissionError` replacing a file | an open handle (numpy `np.load`, a reader) blocks replace on Windows | read inside `with` and copy values out; replace atomically via a temp file |
| a process "is running" per the log but is not | logs lie after a crash | verify by PID and creation time (`psutil`), count children, check checkpoint mtimes |
| `multiprocessing` hangs or re-runs the script | Windows uses spawn | guard with `if __name__ == "__main__":`; no work at import time |
| paths | Git Bash `/d/Aaron/...` vs Windows `D:\Aaron\...` | Python accepts `D:/Aaron/...` everywhere; use that in scripts and docs |
| pytest leaves cache folders in a shared repo | default cacheprovider | `python -m pytest -p no:cacheprovider` |

## Before launching anything expensive
1. dry run at tiny scale (1 job, short horizon) and validate against a known value; 2. check free RAM; 3. driver retries + incremental results; 4. watchdog attached;
5. verify by PID that driver and workers exist; 6. write the live PIDs into memory.
