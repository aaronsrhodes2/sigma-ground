"""Read-only survey of what sits on C: (PLATINUM section 22: D: not C:). Sizes only: nothing is moved, changed or deleted.

    python scripts/c_drive_survey.py

Measures the usual suspects (tool stores, Docker/WSL disk images, caches, temp, downloads) and lists the biggest top-level folders of the user
profile and AppData, so a move plan can be made from numbers.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HOME = Path(os.environ.get("USERPROFILE", str(Path.home())))
LOCAL = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))
ROAM = Path(os.environ.get("APPDATA", HOME / "AppData" / "Roaming"))


def size_of(path: Path, limit_files: int = 400_000) -> int:
    """Total bytes under path (files only, no symlink following); stops after limit_files files so a huge tree cannot hang the survey."""
    total, seen = 0, 0
    stack = [path]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                            seen += 1
                            if seen >= limit_files:
                                return -total          # negative = a lower bound
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def fmt(n: int) -> str:
    bound = ">=" if n < 0 else ""
    n = abs(n)
    return f"{bound}{n / 1024**3:7.1f} GB" if n >= 1024**3 else f"{bound}{n / 1024**2:7.0f} MB"


def main() -> int:
    total, used, free = shutil.disk_usage("C:/")
    print(f"C: {used / 1024**3:.0f} of {total / 1024**3:.0f} GB used, {free / 1024**3:.0f} GB free")
    suspects = [
        ("Docker Desktop data (WSL images)", LOCAL / "Docker"),
        ("WSL distro packages (ext4.vhdx live here)", LOCAL / "Packages"),
        ("Ollama (models, if default)", HOME / ".ollama"),
        ("Ollama app data", LOCAL / "Ollama"),
        ("Hugging Face cache", HOME / ".cache"),
        ("pip cache", LOCAL / "pip"),
        ("npm cache", LOCAL / "npm-cache"),
        ("npm global", ROAM / "npm"),
        ("Temp", LOCAL / "Temp"),
        ("Downloads", HOME / "Downloads"),
        ("Claude (projects, transcripts, memory)", HOME / ".claude"),
        ("Docker config (user)", HOME / ".docker"),
        ("Steam on C: (if any)", Path("C:/Program Files (x86)/Steam")),
        ("Google Drive cache", LOCAL / "Google"),
        ("Python installs", LOCAL / "Programs" / "Python"),
        ("Windows.old / Windows update leftovers", Path("C:/Windows.old")),
    ]
    rows = []
    for label, p in suspects:
        if p.exists():
            rows.append((size_of(p), label, p))
    rows.sort(key=lambda r: -abs(r[0]))
    print("\nKnown suspects (biggest first):")
    for s, label, p in rows:
        print(f"  {fmt(s)}  {label}  [{p}]")
    print("\nBiggest top-level folders under the profile and AppData:")
    top = []
    for base in (HOME, LOCAL, ROAM):
        try:
            for e in os.scandir(base):
                if e.is_dir(follow_symlinks=False):
                    top.append((size_of(Path(e.path), 150_000), Path(e.path)))
        except OSError:
            pass
    top.sort(key=lambda r: -abs(r[0]))
    for s, p in top[:14]:
        print(f"  {fmt(s)}  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
