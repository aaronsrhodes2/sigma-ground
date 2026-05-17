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
    # Walk up from CWD; also check the parent of this package as a fallback
    # so the runners work even if invoked from inside an IDE with a weird CWD.
    candidates: list[Path] = []
    here = Path.cwd().resolve()
    for ancestor in [here, *here.parents]:
        candidates.append(ancestor / ".env")
    # Also try the dev root if we can identify it from this file's path
    pkg_root = Path(__file__).resolve()
    for ancestor in pkg_root.parents:
        candidates.append(ancestor / ".env")

    for path in candidates:
        if path.is_file():
            _load_env_file(path)
            if verbose:
                print(f"Loaded env vars from {path}")
            return path
    if verbose:
        print("No .env file found in CWD ancestry or package ancestry.")
    return None


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

