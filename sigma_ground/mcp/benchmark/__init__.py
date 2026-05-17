"""Physics benchmark corpus + runners for the 3-way comparison.

Three runners exercise the same 150-question corpus:
  - run_sigma_ground.py: sigma-ground MCP + qwen2.5:7b via Ollama
  - run_wolfram.py:      Wolfram Alpha free-tier API
  - run_gemini.py:       Gemini 2.5 Pro via google-generativeai

Each runner saves results to results/<system>_run.json.
score.py compares against ground_truth.json and emits a report.

The runners auto-load environment variables (API keys) by walking up
from the current working directory looking for a .env file. The
canonical location is D:\\Aaron\\development\\.env (the dev root),
documented in D:\\Aaron\\development\\.env.reference.md.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_from_dev_root(verbose: bool = False) -> Path | None:
    """Find and load the .env file shared across the dev tree.

    Walks up from CWD looking for a .env file. Loads it via python-dotenv
    if installed; otherwise parses it manually (fallback for users who
    skipped the [benchmark] extras install).

    Existing env vars are NOT overridden -- shell-set values win.

    Returns
    -------
    Path of the loaded .env, or None if none found.
    """
    # Collect all .env candidates from CWD ancestry + package ancestry.
    candidates: list[Path] = []
    here = Path.cwd().resolve()
    for ancestor in [here, *here.parents]:
        candidates.append(ancestor / ".env")
    pkg_root = Path(__file__).resolve()
    for ancestor in pkg_root.parents:
        candidates.append(ancestor / ".env")

    # Find all existing .env files, dedup, and load shallowest-first
    # (closest to filesystem root). With override=False, the dev-root
    # .env (typically the shallowest) wins for any shared key. This
    # matters because the dev root is the AUTHORITATIVE location for
    # shared secrets; project-local .env files may exist as stubs.
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            existing.append(path)
    # Shallowest first (fewest path parts).
    existing.sort(key=lambda p: len(p.parts))

    if not existing:
        if verbose:
            print("No .env file found in CWD ancestry or package ancestry.")
        return None

    for path in existing:
        _load_env_file(path)
        if verbose:
            print(f"Loaded env vars from {path}")
    # Return the dev-root one (shallowest) for the function name to be honest.
    return existing[0]


def _load_env_file(path: Path) -> None:
    """Load a .env file. Prefer python-dotenv; fall back to manual parse."""
    try:
        from dotenv import load_dotenv
        load_dotenv(str(path), override=False)
        return
    except ImportError:
        pass
    # Manual fallback: minimal KEY=VALUE parser, # comments, no interpolation.
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

