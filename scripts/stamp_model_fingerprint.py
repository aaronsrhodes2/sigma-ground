"""Stamp the physics fingerprint on a forecast or error table that was built BEFORE builders recorded one (the first corrected build of 2026-10-26 was already running).

A stored table is a cache of a calculation. Its fingerprint (reduced_model.model_fingerprint: the physics code, the predictor's configuration, the body parameters and the mass data) lets body_position notice that the
calculation has changed since. A late stamp is only honest if nothing the fingerprint covers changed between the START of the build and now, so this script checks that with git (no commit touching those files since the
start, no uncommitted edit to them) and refuses otherwise. The stamp records that it was made after the build.

usage: python scripts/stamp_model_fingerprint.py --forecast        # the newest entry of fixtures/forecast_manifest.json
       python scripts/stamp_model_fingerprint.py --error-table     # fixtures/prediction_error_table.json (its start is taken from --since)
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIX = _ROOT / "sigma_ground/field/interface/fixtures"


def physics_changed_since(start: datetime.datetime) -> str | None:
    """Why the fingerprint cannot be stamped late, or None if the covered files are unchanged since `start` (UTC)."""
    from sigma_ground.field.interface import reduced_model as rm
    base = Path(rm.__file__).resolve().parent
    norm = [(base / rel).resolve().relative_to(_ROOT).as_posix() for rel in rm.FINGERPRINT_CODE + rm.FINGERPRINT_DATA]
    log = subprocess.run(["git", "log", f"--since={start.strftime('%Y-%m-%dT%H:%M:%SZ')}", "--format=%h %s", "--", *norm], cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    if log:
        return "commits touched the physics files since the build started: " + log.splitlines()[0]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *norm], cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        return "uncommitted edits to the physics files: " + dirty.splitlines()[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--error-table", action="store_true")
    ap.add_argument("--since", help="UTC time the build started (ISO), required for --error-table")
    a = ap.parse_args()
    from sigma_ground.field.interface import reduced_model as rm
    now = datetime.datetime.now(datetime.timezone.utc)
    if a.forecast:
        f = FIX / "forecast_manifest.json"
        doc = json.loads(f.read_text(encoding="utf-8"))
        e = doc["tables"][-1]
        created = datetime.datetime.strptime(e["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        start = created - datetime.timedelta(seconds=float(e["wall_seconds"]))
        why = physics_changed_since(start)
        if why:
            raise SystemExit("NOT STAMPED: " + why)
        e["model_fingerprint"] = rm.model_fingerprint(e["predictor"])
        e["model_fingerprint_note"] = f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build; no commit or edit touched the covered physics files between the build's start ({start:%Y-%m-%dT%H:%M:%SZ}) and the stamp"
        f.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print("stamped", e["file"], e["model_fingerprint"][:16])
    if a.error_table:
        if not a.since:
            raise SystemExit("--since is required for --error-table")
        start = datetime.datetime.fromisoformat(a.since.replace("Z", "+00:00"))
        why = physics_changed_since(start)
        if why:
            raise SystemExit("NOT STAMPED: " + why)
        f = FIX / "prediction_error_table.json"
        doc = json.loads(f.read_text(encoding="utf-8"))
        doc["metadata"]["model_fingerprint"] = rm.model_fingerprint(doc["metadata"]["predictor"])
        doc["metadata"]["model_fingerprint_note"] = f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build; the covered physics files were unchanged since {start:%Y-%m-%dT%H:%M:%SZ}"
        f.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print("stamped the error table", doc["metadata"]["model_fingerprint"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
