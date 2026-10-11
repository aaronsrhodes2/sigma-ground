"""Measure the resolver on a labelled query set.

    python stage/skippyverse/metrics.py [--queries data/queries_seed.json] [--index build/index] [--failures]

Positives (`expect` = a name): top-1 = the chosen kind is a kind carrying that name or alias; top-3 = it is among the chosen kind
and the next two candidates/options. `placeable` additionally requires the resolver to have said yes (exact/alias/synonym/closest)
rather than refusing on size. Negatives (`expect` null): `false_accept` = a confident wrong object (exact/alias/synonym);
`closest_offered` = an honest 'the closest I have is X' / 'I read that as X' (reported separately: it is a wrong object only if
you would not have wanted X); `ambiguous` = a short list of kinds, nothing chosen; `refused` = not found.

The seed set was written by the builder, in one sitting, as a smoke test. The real 200-query set is Aaron's to write.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skippyverse.index import Index, build_local, load_index  # noqa: E402
from skippyverse.resolve import resolve  # noqa: E402
from skippyverse.text import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
PLACEABLE = {"exact", "alias", "synonym", "closest"}
CONFIDENT = {"exact", "alias", "synonym"}


def kinds_named(index: Index, name: str) -> set[str]:
    return {norm(c["kind"]) for c, role in index.by_term(norm(name)) if role in ("name", "alias")}


def evaluate(index: Index, queries: list[dict], audience: str = "local") -> dict:
    pos = [q for q in queries if q.get("expect")]
    neg = [q for q in queries if not q.get("expect")]
    by_tag: dict[str, Counter] = defaultdict(Counter)
    fails, unlabelled = [], []
    for q in pos:
        want = kinds_named(index, q["expect"])
        if not want:
            unlabelled.append(q["q"])
        r = resolve(q["q"], index, audience=audience)
        top = [r["kind"]] if r["kind"] else []
        top = (top + [k for k in (r["options"] + r["candidates"]) if k not in top])[:3]
        t1 = bool(r["kind"]) and r["kind"] in want
        t3 = any(k in want for k in top)
        c = by_tag[q["tag"]]
        c["n"] += 1
        c["top1"] += t1
        c["top3"] += t3
        c["placeable"] += t1 and r["status"] in PLACEABLE
        if not t3 or not t1:
            fails.append({"q": q["q"], "tag": q["tag"], "expect": q["expect"], "got": r["kind"], "status": r["status"], "top": top})
    neg_c: Counter = Counter()
    neg_bad = []
    for q in neg:
        r = resolve(q["q"], index, audience=audience)
        st = r["status"]
        neg_c["n"] += 1
        if st in CONFIDENT:
            neg_c["false_accept"] += 1
            neg_bad.append({"q": q["q"], "got": r["kind"], "status": st})
        elif st == "closest":
            neg_c["closest_offered"] += 1
            neg_bad.append({"q": q["q"], "got": r["kind"], "status": st, "sentence": r["sentence"]})
        elif st in ("ambiguous", "category"):
            neg_c["ambiguous"] += 1
        else:
            neg_c["refused"] += 1
    n = sum(c["n"] for c in by_tag.values())
    tot = {k: sum(c[k] for c in by_tag.values()) for k in ("top1", "top3", "placeable")}
    return {
        "audience": audience, "n_positive": n, "n_negative": neg_c["n"],
        "top1": tot["top1"] / n if n else None, "top3": tot["top3"] / n if n else None, "placeable": tot["placeable"] / n if n else None,
        "by_tag": {t: {"n": c["n"], "top1": c["top1"] / c["n"], "top3": c["top3"] / c["n"]} for t, c in by_tag.items()},
        "negatives": dict(neg_c),
        "false_accept_rate": neg_c["false_accept"] / neg_c["n"] if neg_c["n"] else None,
        "closest_offered_rate": neg_c["closest_offered"] / neg_c["n"] if neg_c["n"] else None,
        "labels_not_in_index": unlabelled, "failures": fails, "negative_offers": neg_bad,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "data" / "queries_seed.json"))
    ap.add_argument("--index")
    ap.add_argument("--audience", default="local", choices=["local", "web"])
    ap.add_argument("--failures", action="store_true")
    a = ap.parse_args(argv)
    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))["queries"]
    idx = load_index(a.index) if a.index else build_local()
    m = evaluate(idx, queries, a.audience)
    f = lambda x: "n/a" if x is None else f"{x:.1%}"          # noqa: E731
    print(f"queries: {m['n_positive']} positive, {m['n_negative']} negative (SEED set, written by the builder; not Aaron's 200)")
    print(f"top-1 {f(m['top1'])}   top-3 {f(m['top3'])}   placeable-and-right {f(m['placeable'])}")
    for t, c in sorted(m["by_tag"].items()):
        print(f"  {t:9s} n={c['n']:3d}  top-1 {f(c['top1'])}  top-3 {f(c['top3'])}")
    nn = m["negatives"]
    print(f"negatives: false_accept {nn.get('false_accept', 0)}/{nn['n']} ({f(m['false_accept_rate'])})  "
          f"closest_offered {nn.get('closest_offered', 0)}  ambiguous {nn.get('ambiguous', 0)}  refused {nn.get('refused', 0)}")
    if m["labels_not_in_index"]:
        print("labels whose name is not in the index:", m["labels_not_in_index"])
    if a.failures:
        print("\nFAILURES (positives):")
        for x in m["failures"]:
            print("  ", x)
        print("\nNEGATIVES that got an object:")
        for x in m["negative_offers"]:
            print("  ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
