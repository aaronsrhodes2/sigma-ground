"""Ask a LIVE Mentat the benchmark corpora and write the run files the daily job scores (sigma_ground/mcp/benchmark/live_ask.py).

    python tools/ask_suites.py                                    # MENTAT_URL or http://skippy-core:7350; physics, adversarial, math
    python tools/ask_suites.py --url http://127.0.0.1:17350 --suites math --workers 1
    python tools/ask_suites.py --results-dir local-cache/live_probe   # somewhere other than the daily job's results/

One worker by default: the router slot on skippy-core serves two requests at a time, and a faster run only queues behind itself.
Exit 3 when the service does not answer /health (nothing written).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark import live_ask as L  # noqa: E402


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("MENTAT_URL") or L.DEFAULT_URL)
    ap.add_argument("--suites", default="physics,adversarial,math")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--results-dir", type=Path, default=L.HERE / "results")
    a = ap.parse_args(argv)
    try:
        meta = L.run_all(a.url, a.results_dir, tuple(s for s in a.suites.split(",") if s), a.workers, a.timeout)
    except L.Unreachable as e:
        print(f"live Mentat unreachable: {e}", file=sys.stderr)
        return 3
    print(json.dumps({k: v for k, v in meta.items() if k != "suites"}))
    for name, s in meta["suites"].items():
        print(f"  {name:12s} {json.dumps(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
