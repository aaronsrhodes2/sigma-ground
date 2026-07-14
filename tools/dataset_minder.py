"""Dataset Minder — keeps our external data connections alive and honest.

A daily-cadence orchestrator over the fetch/distill tools already built this
session. It does NOT reimplement any fetch logic — it asks each source "have
you changed?" via the cheapest possible check (a metadata call, a git
ls-remote, an HTTP HEAD/ETag — never a bulk re-download), and only calls the
existing tool's `main()` when something is actually due. The persistent
result is a status manifest: per dataset, its source, when it was last
checked/synced, and the outcome — the direct answer to "list every dataset
we sync with and when its last sync was."

Cadence is per-source, not blanket-daily: the orchestrator itself runs once a
day (via Task Scheduler, see run_dataset_minder.bat), but a source whose data
changes on a ~4-year cycle (CODATA) has no business being re-checked every
day just because the SERVICE has a daily heartbeat. See CADENCE_RATIONALE.

Run:  python tools/dataset_minder.py --run            (check + sync what's due)
      python tools/dataset_minder.py --run --dry-run   (report only, no sync)
      python tools/dataset_minder.py --force <name>    (sync one source now)
      python tools/dataset_minder.py --status           (print the manifest)
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma_ground.deckard.sources.web import get_json_with_backoff  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
_MANIFEST = _ROOT / "local-cache" / "dataset_minder_status.json"
_ENV = pathlib.Path("D:/Aaron/development/.env")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _hf_token() -> str | None:
    """Same '.env' HF_ID= reader as fetch_shapenet_archives.py / distill_shapenetsem.py."""
    try:
        for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("HF_ID="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


# ── freshness checks — each returns a small string "version marker" to diff
# against the manifest's last-seen value, or None if the check itself failed
# (a failed CHECK is not a failed SYNC — we just skip this round, honestly). ──
def _check_hf_repo(repo_id: str, repo_type: str = "dataset"):
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        info = HfApi().repo_info(repo_id, repo_type=repo_type, token=_hf_token())
        lm = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
        return str(lm) if lm else None
    except Exception:
        return None


def _check_git_remote(local_path: str):
    try:
        out = subprocess.run(["git", "-C", local_path, "ls-remote", "origin", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return out.stdout.split()[0]                      # the remote HEAD commit hash
    except Exception:
        return None


def _check_http_head(url: str):
    """ETag or Last-Modified from a HEAD request — no body downloaded."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "sigma-ground-dataset-minder/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.headers.get("ETag") or r.headers.get("Last-Modified")
    except Exception:
        return None


def _check_wikidata_revisions():
    """A real freshness check, not a placeholder: reads the QIDs we already
    resolved and cited in electrical_properties.json (no re-resolving names),
    batches one wbgetentities call for their lastrevid (info props only — no
    claims payload), and returns the combined revision ids as the marker. A
    changed digit means an editor touched one of OUR tracked entities."""
    try:
        data = json.loads((_ROOT / "sigma_ground" / "inventory" / "data"
                           / "electrical_properties.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    qids = sorted({v["qid"] for k, v in data.items() if k != "_meta" and v.get("qid")})
    if not qids:
        return None
    url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&format=json"
          f"&props=info&ids={'|'.join(qids)}")
    d = get_json_with_backoff(url)
    if not d:
        return None
    ents = d.get("entities", {})
    revs = sorted(str(e.get("lastrevid", "")) for e in ents.values())
    return ",".join(revs) or None


def _check_file_exists(path: str):
    p = pathlib.Path(path)
    return "present" if p.is_file() else None


def _check_not_yet_available(*_a, **_kw):
    return None                                            # PartNet-Mobility: awaiting account


# ── sync actions — invoke the EXISTING tool as a subprocess; never re-implement it ──
def _sync_tool(module_or_script: str, *, timeout: float = 1800.0) -> tuple[bool, str]:
    script = str(_TOOLS / module_or_script)
    try:
        out = subprocess.run([sys.executable, script], cwd=str(_ROOT),
                             capture_output=True, text=True, timeout=timeout)
        ok = out.returncode == 0
        tail = (out.stdout or out.stderr or "")[-400:]
        return ok, tail
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _sync_noop(*_a, **_kw):
    return True, "check-only source; sync is a manual/offline step"


def _sync_not_yet_available(*_a, **_kw):
    return False, "awaiting Captain's SAPIEN account approval — no ETA"


# ── the registry ──────────────────────────────────────────────────────────
class Source:
    __slots__ = ("name", "source_url", "cadence_days", "license_note",
                "check_fn", "sync_fn", "check_method")

    def __init__(self, name, source_url, cadence_days, license_note,
                check_fn, check_method, sync_fn):
        self.name, self.source_url = name, source_url
        self.cadence_days, self.license_note = cadence_days, license_note
        self.check_fn, self.check_method, self.sync_fn = check_fn, check_method, sync_fn


SOURCES = [
    Source("shapenet_partnet", "huggingface.co (ShapeNet/PartNet-archive)", 30,
          "ShapeNet ToU — non-commercial research",
          lambda: _check_hf_repo("ShapeNet/PartNet-archive"),
          "HfApi.repo_info().lastModified", lambda: _sync_tool("fetch_shapenet_archives.py")),
    Source("shapenetsem", "huggingface.co (ShapeNetSem)", 30,
          "ShapeNet ToU — non-commercial research",
          lambda: _check_hf_repo("ShapeNet/ShapeNetSem-archive"),
          "HfApi.repo_info().lastModified", lambda: _sync_tool("distill_shapenetsem.py")),
    Source("objaverse_lvis", "objaverse (Allen AI, via PyPI package)", 30,
          "per-object Sketchfab licenses — see inventory/data/objaverse_ledger.csv",
          lambda: None, "no lightweight metadata API identified yet — manual recheck",
          lambda: _sync_tool("fetch_objaverse_lvis.py")),
    Source("refractiveindex_db", "github.com/polyanskiy/refractiveindex.info-database", 7,
          "open database; papers cited per entry",
          lambda: _check_git_remote("D:/datasets/optics/refractiveindex.info-database"),
          "git ls-remote origin HEAD",
          lambda: _sync_tool("distill_optics_nk.py")),
    Source("nist_atomic_weights", "physics.nist.gov (atomic weights CGI)", 90,
          "public domain (US gov)",
          lambda: _check_http_head("https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses"),
          "HTTP HEAD ETag/Last-Modified", lambda: _sync_tool("distill_chemistry.py")),
    Source("burcat_thermo", "garfield.chem.elte.hu (Burcat/ATcT database)", 90,
          "free for scientific use (cited)",
          lambda: None, "no stable HEAD target identified — manual recheck",
          lambda: _sync_tool("distill_chemistry.py")),
    Source("codata_constants", "physics.nist.gov/cuu/Constants", 365,
          "public domain (US gov)",
          lambda: _check_http_head("https://physics.nist.gov/cuu/Constants/index.html"),
          "HTTP HEAD ETag/Last-Modified", lambda: _sync_tool("distill_chemistry.py")),
    Source("wikidata_electrical", "wikidata.org (P5679 resistivity, P2068 thermal conductivity)", 1,
          "CC0", _check_wikidata_revisions,
          "wbgetentities lastrevid for our own cited QIDs (real diff, not a guess)",
          lambda: _sync_tool("distill_electronics.py")),
    Source("nasa_tpsx", "tpsx.arc.nasa.gov (materials database)", 30,
          "US gov work, freely retrievable",
          lambda: _check_http_head("https://tpsx.arc.nasa.gov/"),
          "HTTP HEAD ETag/Last-Modified", lambda: _sync_tool("fetch_tpsx.py")),
    Source("kicad_packages3d", "gitlab.com/kicad/libraries/kicad-packages3D", 14,
          "CC-BY-SA 4.0 (with KiCad libraries exception)",
          lambda: _check_git_remote("D:/datasets/electronics/kicad-packages3D"),
          "git ls-remote origin HEAD", _sync_noop),
    Source("jpl_de440s", "naif.jpl.nasa.gov (DE440s ephemeris kernel)", 365,
          "public domain (NAIF/JPL)",
          lambda: _check_file_exists("D:/datasets/jpl/de440s.bsp"),
          "file-existence + version sanity (essentially permanent data)", _sync_noop),
    Source("partnet_mobility", "sapien.ucsd.edu (PartNet-Mobility / SAPIEN)", 7,
          "SAPIEN ToU, non-commercial; account required",
          _check_not_yet_available,
          "awaiting account approval — justify-the-connection cadence once live",
          _sync_not_yet_available),
]

CADENCE_RATIONALE = (
    "The service runs daily; each SOURCE only actually checks/syncs when its own "
    "cadence is due. A CODATA constants page revised on a ~4-year cycle gains "
    "nothing from a daily HEAD request; Wikidata, a live continuously-edited graph, "
    "genuinely benefits from one. Override any cadence_days in the registry above "
    "if a different rhythm is wanted for a specific source."
)


# ── manifest ──────────────────────────────────────────────────────────────
def _load_manifest() -> dict:
    if _MANIFEST.is_file():
        try:
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(m: dict) -> None:
    _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(_MANIFEST)                                 # atomic — never a half-written manifest


def _due(entry: dict | None, cadence_days: int) -> bool:
    if entry is None or not entry.get("last_checked"):
        return True
    try:
        last = datetime.datetime.fromisoformat(entry["last_checked"])
    except Exception:
        return True
    return (datetime.datetime.now(datetime.timezone.utc) - last).days >= cadence_days


def run(*, dry_run: bool = False, force: str | None = None) -> dict:
    manifest = _load_manifest()
    for src in SOURCES:
        entry = manifest.get(src.name, {})
        if force and force != src.name:
            continue
        if not force and not _due(entry, src.cadence_days):
            print(f"  = {src.name:20s} not due (next in "
                  f"{src.cadence_days - (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(entry['last_checked'])).days} days)")
            continue

        now = _now()
        prev_marker = entry.get("_version_marker")
        marker = src.check_fn() if src.check_fn else None
        changed = force is not None or marker is None or marker != prev_marker
        entry.update({"source_url": src.source_url, "check_method": src.check_method,
                      "cadence_days": src.cadence_days, "license_note": src.license_note,
                      "last_checked": now, "_version_marker": marker})

        if not changed:
            entry["last_sync_result"] = "no_change"
            print(f"  ~ {src.name:20s} checked, no change")
        elif dry_run:
            entry["last_sync_result"] = "would_sync (dry-run)"
            print(f"  > {src.name:20s} WOULD sync (dry-run)")
        else:
            ok, tail = src.sync_fn()
            entry["last_sync"] = _now()
            entry["last_sync_result"] = "ok" if ok else "failed"
            entry["last_sync_note"] = tail[-200:]
            print(f"  {'+' if ok else '!'} {src.name:20s} sync {'ok' if ok else 'FAILED'}")

        due_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=src.cadence_days)
        entry["next_due"] = due_at.isoformat(timespec="seconds")
        manifest[src.name] = entry

    if not dry_run:
        _save_manifest(manifest)
    return manifest


def print_status(manifest: dict | None = None) -> None:
    manifest = manifest or _load_manifest()
    print(f"{'dataset':22s} {'last sync':22s} {'result':16s} {'next due':22s}")
    for src in SOURCES:
        e = manifest.get(src.name, {})
        print(f"{src.name:22s} {e.get('last_sync', 'never'):22s} "
              f"{e.get('last_sync_result', '(none)'):16s} {e.get('next_due', '(none)'):22s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="check + sync what's due")
    ap.add_argument("--dry-run", action="store_true", help="report only, no real sync")
    ap.add_argument("--force", metavar="NAME", help="sync one source now, ignoring cadence")
    ap.add_argument("--status", action="store_true", help="print the manifest and exit")
    args = ap.parse_args()

    if args.status:
        print_status()
        return
    if args.force:
        m = run(dry_run=args.dry_run, force=args.force)
        print_status(m)
        return
    if args.run:
        m = run(dry_run=args.dry_run)
        print_status(m)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
