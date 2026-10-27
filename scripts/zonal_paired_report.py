"""Paired comparison of two predictors on the SAME windows of the full 33-body shoot-out: the control (restated_ast_ring_kbo_c22, table zonals) against restated_zon (measured Jupiter and Saturn zonals).

Only the zonal values differ (same code, fixture r4, step, windows), so any difference is the zonals. For every body and lead the median heliocentric position error of both, the number of windows in which
the new one is better, and the ratio. Written to ZONAL_PAIRED.md / .json.

usage: python scripts/zonal_paired_report.py CONTROL_RUN_DIR_OR_FILES NEW_RUN.json [--control NAME] [--new NAME] [--out DIR]
       python scripts/zonal_paired_report.py --control-files A.json B.json --new-files C.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import shootout_full_restate_report as R  # noqa: E402

LEADS = (1, 3, 5, 10, 20)


def compare(control: dict, new: dict, windows: list[str]) -> list[dict]:
    bodies = sorted({b for w in windows for b in new.get(w, {})})
    rows = []
    for b in bodies:
        for lead in LEADS:
            c = [control.get(w, {}).get(b, {}).get(lead) for w in windows]
            n = [new.get(w, {}).get(b, {}).get(lead) for w in windows]
            pairs = [(x, y) for x, y in zip(c, n) if x is not None and y is not None]
            if not pairs:
                continue
            mc, mn = R.med([x for x, _ in pairs]), R.med([y for _, y in pairs])
            rows.append({"body": b, "lead": lead, "control_km": mc, "new_km": mn, "ratio": mn / mc if mc else None, "windows_better": sum(y < x for x, y in pairs), "windows": len(pairs)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control-files", nargs="+", type=Path, required=True)
    ap.add_argument("--new-files", nargs="+", type=Path, required=True)
    ap.add_argument("--control", default="restated_ast_ring_kbo_c22")
    ap.add_argument("--new", default="restated_zon")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="ZONAL_PAIRED", help="output file stem (a new label keeps the earlier tables)")
    a = ap.parse_args()
    control = R.load(a.control_files, {a.control}).get(a.control, {})
    new = R.load(a.new_files, {a.new}).get(a.new, {})
    windows = sorted(set(control) & set(new))
    rows = compare(control, new, windows)
    lines = [f"# Paired comparison ( {len(windows)} windows: {', '.join(windows)})", "",
             f"Control `{a.control}` against `{a.new}`, same step (0.025 d) and windows. What differs is whatever differs between the two runs (the label's README section says: for ZONAL_PAIRED only Jupiter's and Saturn's zonals; for PAIRED_R4_TO_R8 also the fixture, r4 against r8 with 23 more moons and the current Neptune and Triton). Median heliocentric position error, km.", "",
             "| body | lead | control | new | ratio | windows better |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["lead"] != 5, r["ratio"] if r["ratio"] is not None else 9)):
        lines.append(f"| {r['body']} | +{r['lead']} yr | {R.fmt(r['control_km'])} | {R.fmt(r['new_km'])} | {r['ratio']:.3f} | {r['windows_better']}/{r['windows']} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{a.label}.md").write_text(text + "\n", encoding="utf-8")
        (a.out / f"{a.label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
