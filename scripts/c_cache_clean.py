"""Clear regenerable caches on C: (PLATINUM section 22: D: not C:). DRY RUN by default: prints what it would delete; deletes only with --go.

    python scripts/c_cache_clean.py          # dry run
    python scripts/c_cache_clean.py --go     # delete

What it touches: the pip cache, the npm cache, and files in the user Temp folder that are older than --days (default 3). It NEVER touches the
Claude harness scratchpad (Temp/claude), anything newer than --days, or anything it cannot delete (locked files are skipped and counted).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
TEMP = Path(os.environ.get("TEMP", LOCAL / "Temp"))
PROTECTED = {"claude"}                      # the harness scratchpad lives in Temp/claude: never touched
PROTECTED |= {"xrlinuxdriver", "viture_virtual_display", "sillytavern", "repo"}      # look like someone's working copies or driver setups: left for the Captain


def size_of(p: Path) -> int:
    total = 0
    if p.is_file():
        return p.stat().st_size
    for root, _dirs, files in os.walk(p, onerror=lambda e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def gb(n: int) -> str:
    return f"{n / 1024**3:6.2f} GB" if n >= 1024**3 else f"{n / 1024**2:6.0f} MB"


def old_temp_items(days: int) -> list[tuple[Path, int]]:
    cutoff = time.time() - days * 86400
    out = []
    for e in os.scandir(TEMP):
        if e.name.lower() in PROTECTED:
            continue
        try:
            newest = max([e.stat().st_mtime] + ([os.path.getmtime(os.path.join(r, f)) for r, _d, fs in os.walk(e.path, onerror=lambda x: None) for f in fs] if e.is_dir(follow_symlinks=False) else []))
        except (OSError, ValueError):
            continue
        if newest < cutoff:                  # nothing inside is newer than the cutoff
            out.append((Path(e.path), size_of(Path(e.path))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--days", type=int, default=3)
    a = ap.parse_args()

    free0 = shutil.disk_usage("C:/").free
    print(("DELETING" if a.go else "DRY RUN") + f"  (C: free before: {free0 / 1024**3:.1f} GB)")
    total = 0

    pip = LOCAL / "pip"
    if pip.exists():
        n = size_of(pip)
        total += n
        print(f"  pip cache   {gb(n)}  [{pip}]  -> `pip cache purge`")
        if a.go:
            r = subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True, text=True)
            print("     ", (r.stdout or r.stderr).strip().splitlines()[-1:] or "")
    npm = LOCAL / "npm-cache"
    if npm.exists():
        n = size_of(npm)
        total += n
        print(f"  npm cache   {gb(n)}  [{npm}]  -> `npm cache clean --force`")
        if a.go:
            r = subprocess.run(["npm.cmd", "cache", "clean", "--force"], capture_output=True, text=True)
            print("     ", (r.stdout or r.stderr).strip().splitlines()[-1:] or "")

    items = sorted(old_temp_items(a.days), key=lambda x: -x[1])
    t = sum(s for _p, s in items)
    total += t
    print(f"  Temp: {len(items)} items older than {a.days} days, {gb(t)} (Temp/claude is protected)")
    for p, s in items[:8]:
        print(f"      {gb(s)}  {p.name}")
    skipped = 0
    if a.go:
        for p, _s in items:
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                skipped += 1
        print(f"      locked or in use, skipped: {skipped}")

    print(f"\n  total {'removed' if a.go else 'that would be removed'}: about {gb(total)}")
    if a.go:
        print(f"  C: free after: {shutil.disk_usage('C:/').free / 1024**3:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
