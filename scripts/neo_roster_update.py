"""The near-miss roster (misc/CANDIDATE_LEDGER.md C33): numbered asteroids that JPL's close-approach list puts within 0.05 au of Earth in the coming year join the
daily scoreboard's tracked set, and stay (a body is never removed: its prediction is meant to age).

    python scripts/neo_roster_update.py [--date YYYY-MM-DD] [--dist-max-au 0.05] [--days 365] [--dry-run]

Sources, each response saved under D:/datasets/mpc_live/neo_roster/<date>/ and never overwritten:
* JPL CNEOS close-approach API (ssd-api.jpl.nasa.gov/cad.api): the candidates. Only a roster source; it never filters what we score.
* JPL SBDB API (sbdb.api?sstr=<number>): the orbit's condition code, arc and observation count for the roster record, and the fitted non-gravitational parameters
  (A1/A2/A3 with sigmas, the law's ALN/NK/NM/R0), appended to fixtures/sbdb_nongrav_asteroids.json so the C31 rule (apply only above 3 sigma) sees them; a body
  without an A2 goes on that file's checked_without_A2 list.

Unnumbered designations (most of the list: 140 of 151 on 2026-11-04) are recorded in the day's candidates file but not tracked: the scorer is keyed on the number
(MPC observations, Horizons COMMAND). Tracking them is the roster's open item.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
ROSTER = FIX / "mpc_live" / "neo_roster.json"
NONGRAV = FIX / "sbdb_nongrav_asteroids.json"
DATA = Path("D:/datasets/mpc_live/neo_roster")
RAW_SBDB = Path("D:/datasets/jpl/sbdb_nongrav")
CAD = "https://ssd-api.jpl.nasa.gov/cad.api"
SBDB = "https://ssd-api.jpl.nasa.gov/sbdb.api"
NONGRAV_FIELDS = ["full_name", "pdes", "name", "epoch", "a", "e", "i", "om", "w", "ma", "A1", "A2", "A3", "DT", "A1_sigma", "A2_sigma", "A3_sigma", "ALN", "NK", "NM", "NN", "R0", "soln_date", "other_model_pars"]


def fetch_json(url: str, out: Path, log, delay_s: float = 1.0) -> dict | None:
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sigma-ground neo_roster_update"}), timeout=120) as r:
                body = r.read().decode("utf-8")
            d = json.loads(body)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(body, encoding="utf-8")
            time.sleep(delay_s)
            return d
        except Exception as e:                                        # noqa: BLE001  a refused request is retried, then recorded as missing; never guessed around
            log(f"  {url[:90]}...: {e!r} (attempt {k + 1}/3)"); time.sleep(5 * (k + 1))
    return None


def load_roster() -> dict:
    if ROSTER.exists():
        return json.loads(ROSTER.read_text(encoding="utf-8"))
    return {"metadata": {"what": "numbered asteroids added to the daily scoreboard from JPL's close-approach list (within 0.05 au of Earth in the coming year at the time of adding); never removed",
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, "bodies": []}


def candidates(cad: dict) -> tuple[list[dict], list[dict]]:
    """(numbered, unnumbered) rows of the close-approach response as dicts."""
    f = cad["fields"]
    rows = [dict(zip(f, r)) for r in cad.get("data", [])]
    return [r for r in rows if r["des"].strip().isdigit()], [r for r in rows if not r["des"].strip().isdigit()]


def sbdb_record(number: int, day_dir: Path, log) -> dict | None:
    d = fetch_json(f"{SBDB}?sstr={number}&phys-par=0", day_dir / f"sbdb_{number}.json", log)
    if not d or "orbit" not in d:
        return None
    o = d["orbit"]
    el = {e["name"]: e["value"] for e in o.get("elements", [])}
    mp = {p["name"]: p for p in (o.get("model_pars") or [])}
    name = d["object"].get("shortname") or d["object"]["fullname"]
    return {"number": number, "full_name": d["object"]["fullname"].strip(), "name": name.strip(), "condition_code": o.get("condition_code"), "data_arc_days": o.get("data_arc"),
            "n_obs_used": o.get("n_obs_used"), "soln_date": o.get("soln_date"), "elements": el, "model_pars": mp, "epoch": o.get("epoch")}


def nongrav_row(rec: dict) -> list | None:
    """A fixtures/sbdb_nongrav_asteroids.json row for a body whose solution carries A2, else None."""
    mp = rec["model_pars"]
    if "A2" not in mp:
        return None
    g = lambda k: mp[k]["value"] if k in mp else None
    s = lambda k: mp[k].get("sigma") if k in mp else None
    el = rec["elements"]
    other = {k: v["value"] for k, v in mp.items() if k not in ("A1", "A2", "A3", "DT", "ALN", "NK", "NM", "NN", "R0")} or None
    return [rec["full_name"], str(rec["number"]), rec["name"], rec["epoch"], el.get("a"), el.get("e"), el.get("i"), el.get("om"), el.get("w"), el.get("ma"),
            g("A1"), g("A2"), g("A3"), g("DT"), s("A1"), s("A2"), s("A3"), g("ALN"), g("NK"), g("NM"), g("NN"), g("R0"), rec["soln_date"], other]


def update_nongrav(rec: dict, log, dry_run: bool) -> str:
    d = json.loads(NONGRAV.read_text(encoding="utf-8"))
    assert d["fields"] == NONGRAV_FIELDS, "the non-grav fixture's fields changed; update NONGRAV_FIELDS"
    i = d["fields"].index("pdes")
    if any(str(r[i]) == str(rec["number"]) for r in d["data"]):
        return "already in the fixture"
    if rec["full_name"] in d["metadata"].get("checked_without_A2", []):
        return "already checked, no A2"
    row = nongrav_row(rec)
    if row is None:
        d["metadata"].setdefault("checked_without_A2", []).append(rec["full_name"])
        d["metadata"]["checked_without_A2"].sort()
        verdict = "no A2 in JPL's solution: checked_without_A2"
    else:
        d["data"].append(row)
        verdict = f"A2 {row[11]} +- {row[15]} au/d^2 appended"
    d["metadata"]["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    d["metadata"]["roster_additions"] = "scripts/neo_roster_update.py appends rows for near-miss roster bodies from the same per-body SBDB request"
    if not dry_run:
        NONGRAV.write_text(json.dumps(d, indent=1), encoding="utf-8")
        raw = RAW_SBDB / f"sbdb_{rec['number']}.json"
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps({"object": {"fullname": rec["full_name"]}, "orbit": {"model_pars": list(rec["model_pars"].values()), "soln_date": rec["soln_date"]}}, indent=1), encoding="utf-8")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=_date.today().isoformat())
    ap.add_argument("--dist-max-au", type=float, default=0.05)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    day_dir = DATA / a.date
    d0 = _date.fromisoformat(a.date)
    url = CAD + "?" + urllib.parse.urlencode({"dist-max": a.dist_max_au, "date-min": d0.isoformat(), "date-max": (d0 + timedelta(days=a.days)).isoformat(), "fullname": "true", "sort": "date"})
    cad = fetch_json(url, day_dir / "cad.json", log)
    if cad is None:
        log("close-approach list not fetched; roster unchanged"); return 1
    numbered, unnumbered = candidates(cad)
    log(f"close approaches within {a.dist_max_au} au, {a.date} + {a.days} d: {len(numbered)} numbered, {len(unnumbered)} unnumbered (not tracked)")
    roster = load_roster()
    have = {b["number"] for b in roster["bodies"]}
    added = []
    for r in numbered:
        n = int(r["des"])
        if n in have:
            continue
        rec = sbdb_record(n, day_dir, log)
        if rec is None:
            log(f"  {n}: SBDB record not fetched; not added today"); continue
        ng = update_nongrav(rec, log, a.dry_run)
        entry = {"number": n, "name": rec["name"], "full_name": rec["full_name"], "added_on": a.date, "approach_date": r["cd"], "approach_jd": float(r["jd"]), "approach_dist_au": float(r["dist"]),
                 "v_rel_km_s": float(r["v_rel"]), "h_mag": float(r["h"]) if r["h"] else None, "t_sigma_f": r["t_sigma_f"], "orbit_id": r["orbit_id"],
                 "condition_code": rec["condition_code"], "data_arc_days": rec["data_arc_days"], "n_obs_used": rec["n_obs_used"], "soln_date": rec["soln_date"],
                 "nongrav": ng, "source": {"cad": str(day_dir / "cad.json"), "sbdb": str(day_dir / f"sbdb_{n}.json")}}
        roster["bodies"].append(entry); have.add(n); added.append(entry)
        log(f"  + {n} {rec['name']}: approach {r['cd']} at {float(r['dist']):.5f} au, H {r['h']}, condition code {rec['condition_code']}, arc {rec['data_arc_days']} d; non-grav: {ng}")
    (day_dir / "candidates.json").parent.mkdir(parents=True, exist_ok=True)
    (day_dir / "candidates.json").write_text(json.dumps({"numbered": numbered, "unnumbered": unnumbered, "added": [e["number"] for e in added]}, indent=1), encoding="utf-8")
    if added and not a.dry_run:
        roster["metadata"]["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ROSTER.parent.mkdir(parents=True, exist_ok=True)
        ROSTER.write_text(json.dumps(roster, indent=1), encoding="utf-8")
    log(f"roster: {len(roster['bodies'])} bodies ({len(added)} added today){' [dry run, nothing written]' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
