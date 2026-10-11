"""Parse a raw listing FILE the caller supplies into cards.  No network access exists in this module.

    python stage/skippyverse/sources/cli.py --source polyhaven --raw listing.json [--out cards.json]

`--allow-network` is accepted so the interface is the one the design asks for, but live fetching is NOT implemented in this build:
using it exits with status 3 and does nothing. Wiring it needs Aaron's approval (downloads, API keys, accounts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skippyverse.sources import gazebo_fuel, kenney, nasa3d, polyhaven, smithsonian  # noqa: E402
from skippyverse.sources.common import NetworkNotAllowed  # noqa: E402

ADAPTERS = {"polyhaven": polyhaven, "kenney": kenney, "fuel": gazebo_fuel, "nasa": nasa3d, "smithsonian": smithsonian}


def fetch_listing(source: str, *, allow_network: bool = False):
    """Deliberately a stub: raises NetworkNotAllowed without the flag, NotImplementedError with it. Never opens a socket."""
    if source not in ADAPTERS:
        raise KeyError(source)
    if not allow_network:
        raise NetworkNotAllowed(f"{source}: live fetching needs --allow-network")
    raise NotImplementedError(f"{source}: live fetching is not implemented in this build (needs Aaron's approval)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--raw", help="JSON file with the raw metadata")
    ap.add_argument("--out")
    ap.add_argument("--include-disallowed", action="store_true")
    ap.add_argument("--allow-network", action="store_true")
    a = ap.parse_args(argv)
    if a.allow_network or not a.raw:
        try:
            fetch_listing(a.source, allow_network=a.allow_network)
        except (NetworkNotAllowed, NotImplementedError) as e:
            print(f"no network access: {e}", file=sys.stderr)
            return 3
    raw = json.loads(Path(a.raw).read_text(encoding="utf-8"))
    cards, report = ADAPTERS[a.source].parse_listing_with_report(raw, include_disallowed=a.include_disallowed)
    print(f"{len(cards)} cards, {len(report['skipped'])} skipped by the licence gate, {len(report['problems'])} problems")
    if a.out:
        Path(a.out).write_text(json.dumps(cards, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
