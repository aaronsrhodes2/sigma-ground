"""Prove the coverage ledger is complete and summarise it.

    python stage/gmod/coverage/check_coverage.py              # validate + print the tables
    python stage/gmod/coverage/check_coverage.py --json       # machine-readable summary
    python stage/gmod/coverage/check_coverage.py --refresh    # re-freeze the snapshot from the live Mentat container, then validate

Validation fails loudly if a tool or verb exists in the snapshot but not the ledger (a new Mentat tool
nobody has decided how to test), or the reverse, or an entry is internally inconsistent.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger as L  # noqa: E402

SNAPSHOT = HERE / "mentat_toolbelt_snapshot.json"
IN_SCOPE = ("ENGINE", "PORT", "SHOW", "PREDICT", "BLOCKED")        # classes a scenario covers or must cover


def load_snapshot():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def validate(snap=None):
    """Return a list of problems (empty means the ledger is complete and consistent)."""
    snap = snap or load_snapshot()
    problems = []
    for label, table, names in (("tool", L.TOOLS, [t["name"] for t in snap["tools"]]),
                                ("verb", L.VERBS, [v["name"] for v in snap["verbs"]])):
        have, want = set(table), set(names)
        for n in sorted(want - have):
            problems.append(f"{label} '{n}' is in the Mentat snapshot but has no ledger entry")
        for n in sorted(have - want):
            problems.append(f"{label} '{n}' is in the ledger but not in the Mentat snapshot")
        for n, e in table.items():
            if e["cls"] not in L.CLASSES:
                problems.append(f"{label} '{n}': unknown class {e['cls']!r}")
            if e["cls"] == "OUT":
                if e["kind"] not in L.OUT_KINDS:
                    problems.append(f"{label} '{n}': OUT needs a kind from {L.OUT_KINDS}")
                if e["scenarios"]:
                    problems.append(f"{label} '{n}': OUT entries must not name a scenario")
            else:
                if e["kind"]:
                    problems.append(f"{label} '{n}': only OUT entries carry a kind")
                if not e["scenarios"]:
                    problems.append(f"{label} '{n}': {e['cls']} needs at least one scenario")
            for s in e["scenarios"]:
                if s not in L.SCENARIOS:
                    problems.append(f"{label} '{n}': unknown scenario {s}")
    used = {s for table in (L.TOOLS, L.VERBS) for e in table.values() for s in e["scenarios"]}
    for s in L.SCENARIOS:
        if s not in used:
            problems.append(f"scenario {s} ({L.SCENARIOS[s]['name']}) covers nothing")
    return problems


def summary():
    out = {"tools": len(L.TOOLS), "verbs": len(L.VERBS)}
    for label, table in (("tools", L.TOOLS), ("verbs", L.VERBS)):
        out[f"{label}_by_class"] = dict(Counter(e["cls"] for e in table.values()))
        out[f"{label}_out_by_kind"] = dict(Counter(e["kind"] for e in table.values() if e["cls"] == "OUT"))

    both = {**{("tool", n): e for n, e in L.TOOLS.items()}, **{("verb", n): e for n, e in L.VERBS.items()}}
    per = defaultdict(lambda: {"primary": 0, "any": 0})
    for e in both.values():
        for i, s in enumerate(e["scenarios"]):
            per[s]["any"] += 1
            if i == 0:
                per[s]["primary"] += 1
    out["per_scenario"] = {s: dict(per[s], name=L.SCENARIOS[s]["name"], kind=L.SCENARIOS[s]["kind"],
                                   effort_days=L.SCENARIOS[s]["effort_days"]) for s in sorted(L.SCENARIOS)}

    # Greedy build order over the in-scope entries (everything a scenario can cover now):
    # always take the scenario that newly covers the most entries.
    todo = {k for k, e in both.items() if e["cls"] in ("ENGINE", "PORT", "SHOW", "PREDICT")}
    total = len(todo)
    order, covered = [], set()
    avail = {s for s in L.SCENARIOS if L.SCENARIOS[s]["kind"] not in ("BLOCKED", "LOOKUP")}
    while covered != todo and avail:
        best = max(sorted(avail), key=lambda s: len({k for k in todo - covered if s in both[k]["scenarios"]}))
        gain = {k for k in todo - covered if best in both[k]["scenarios"]}
        if not gain:
            break
        covered |= gain
        order.append((best, L.SCENARIOS[best]["name"], len(gain), round(100 * len(covered) / total, 1)))
        avail.discard(best)
    out["greedy_order"] = order
    out["in_scope_entries"] = total
    out["effort_days_total"] = sum(v["effort_days"] or 0 for v in L.SCENARIOS.values())
    return out


def refresh():
    """Re-freeze the snapshot from the live container (needs Docker and skippytel-mentat)."""
    code = (
        "import json,urllib.request\n"
        "b='http://localhost:8080'\n"
        "t=json.load(urllib.request.urlopen(b+'/tools',timeout=30))\n"
        "r=urllib.request.Request(b+'/tools/list_simulation_scenarios',data=b'{}',headers={'content-type':'application/json'})\n"
        "v=json.load(urllib.request.urlopen(r,timeout=60))\n"
        "print(json.dumps({'tools':t,'verbs':v}))\n")
    raw = subprocess.run(["docker", "exec", "-i", "skippytel-mentat", "python", "-c", code],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    live = json.loads(raw)
    cat = (live["verbs"].get("result") or live["verbs"])["inputs"]["catalog"]

    def first(d):
        d = re.sub(r"\s+", " ", d).strip()
        m = re.match(r"(.{20,200}?[.:;])(\s|$)", d)
        return m.group(1) if m else d[:200]

    snap = {"_comment": "Frozen from the live Mentat HTTP API; regenerate with check_coverage.py --refresh.",
            "tools": [{"name": t["name"], "domain": t["domain"], "summary": first(t["description"])}
                      for t in sorted(live["tools"], key=lambda t: (t["domain"], t["name"]))],
            "verbs": [{"name": k, "summary": first(v["description"]), "slots": v.get("slots", [])}
                      for k, v in sorted(cat.items())]}
    SNAPSHOT.write_text(json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    return snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    if args.refresh:
        snap = refresh()
        print(f"snapshot refreshed: {len(snap['tools'])} tools, {len(snap['verbs'])} verbs")
    problems = validate()
    if problems:
        print("LEDGER PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    s = summary()
    if args.json:
        print(json.dumps(s, indent=1))
        return 0
    print(f"ledger complete: {s['tools']} tools + {s['verbs']} simulation verbs, no gaps\n")
    for label in ("tools", "verbs"):
        print(f"{label} by class:", {k: v for k, v in sorted(s[f'{label}_by_class'].items())})
        print(f"{label} out of scale by kind:", {k: v for k, v in sorted(s[f'{label}_out_by_kind'].items())})
    print("\nper scenario (entries where it is the primary / any listed scenario):")
    for sid, d in s["per_scenario"].items():
        st = L.SCENARIOS[sid].get("status")
        print(f"  {sid} {d['name']:34s} {d['kind']:9s} primary {d['primary']:3d}  any {d['any']:3d}  ~{d['effort_days']} d" + (f"   [{st}]" if st else ""))
    print(f"\ngreedy build order over {s['in_scope_entries']} in-scope entries (new coverage, cumulative %):")
    for sid, name, gain, cum in s["greedy_order"]:
        print(f"  {sid} {name:34s} +{gain:3d}  -> {cum}%")
    print(f"\ntotal estimated effort for all buildable scenarios: ~{s['effort_days_total']} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
