"""Assemble the router v2 training data.

  1. v1 records (router_dataset.py, already verified and rewritten by the teacher) are
     converted: the new system prompt, `args` -> `tool_args`, and every number written
     without a zero-padded exponent (Ollama's grammar cannot write 2.4e-06; see
     router_plan.py). Their train/val split is kept. No new teacher pass for them.
  2. Recipe seeds (router_recipes.py: multi-step chains and the weak families, every plan
     executed against the real server) are added, split by group like v1.
  3. With --teacher-model, each recipe seed gets teacher rewrites that must keep every
     number, unit and named key (router_dataset.check_rewrite), stay clear of the
     benchmark (BenchmarkGuard) and of the deterministic classifiers. Accepted rewrites
     are appended to rewrites.jsonl as they come, so an interrupted pass resumes.

    python tools/build_router_v2.py --v1 local-cache/router_dataset \
        --recipes local-cache/router_recipes --out local-cache/router_dataset_v2 \
        [--teacher-model qwen2.5:14b --teacher-url http://127.0.0.1:11435 --variants 3]

Writes train.jsonl, val.jsonl and manifest.json. Run with the repo's Python.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp import param_names as PN  # noqa: E402
from sigma_ground.mcp.benchmark import router_dataset as D  # noqa: E402
from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402
from sigma_ground.mcp.benchmark import router_quantities as QR  # noqa: E402

PADDED_EXPONENT = re.compile(r"\d[eE][-+]?0\d")


PLAN = "v2"            # plan format and system prompt the records carry; assemble() sets it
_SIGNATURES = None     # tool -> real parameters (tool_census, read from server.py); needed to relabel for plan v4


def _signatures() -> dict:
    global _SIGNATURES
    if _SIGNATURES is None:
        sys.path.insert(0, str(REPO / "tools"))
        import tool_census
        _SIGNATURES = tool_census.signatures()
    return _SIGNATURES


_VARIABLE_KEYS = ("x_var", "k_var", "t_var", "s_var")


def label_names_missing_variable(question: str, plan: dict) -> str | None:
    """'k_var=w' when a gold plan names a transform variable the question never shows (the router could not know it), else None."""
    for s in (plan["steps"] if isinstance(plan.get("steps"), list) else [plan]):
        args = s.get("tool_args") or s.get("args") or {}
        for key in _VARIABLE_KEYS:
            v = args.get(key)
            if isinstance(v, str) and v and not re.search(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", question):
                return f"{key}={v}"
    return None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(question: str, plan: dict, meta: dict) -> dict:
    if PLAN == "v4":     # quantities quoted as the question wrote them, parameters under their canonical names
        plan, quoted = QR.relabel_plan(question, plan, _signatures())
        meta = {**meta, "quoted": quoted} if quoted else meta
    return {"messages": [
        {"role": "system", "content": RP.system_prompt(PLAN)},
        {"role": "user", "content": question},
        {"role": "assistant", "content": RP.dumps_plan(plan)}],
        "meta": meta}


def convert_v1(rec: dict) -> dict:
    """A v1 record in the v2 format; the question and the meta are untouched."""
    sys_m, user_m, asst_m = rec["messages"]
    plan = RP.to_v2(json.loads(asst_m["content"]))
    return record(user_m["content"], plan, {**rec["meta"], "origin": "v1"})


def seed_to_record(seed: dict) -> dict:
    return record(seed["question"], seed["plan"],
                  {"group": seed["group"], **seed["meta"], "origin": "recipe"})


def args_of(seed: dict) -> list:
    return [D.Arg(a["name"], a["value"], a["text"], a["name"], a["unit_text"], a["key_text"])
            for a in seed["args_meta"]]


def validate(rec: dict) -> None:
    """Raises AssertionError on anything the model must never be trained on."""
    sys_m, user_m, asst_m = rec["messages"]
    assert sys_m["content"] == RP.system_prompt(PLAN)
    assert not PADDED_EXPONENT.search(asst_m["content"]), asst_m["content"]
    plan = json.loads(asst_m["content"])
    for s in (plan["steps"] if isinstance(plan.get("steps"), list) else [plan]):
        if "decline" not in s:
            assert "args" not in s, f"a v1 key is left: {asst_m['content']}"
            assert "tool_args" in s, f"no tool_args: {asst_m['content']}"
    c = RP.canonical(plan)
    assert c is not None, asst_m["content"]
    if c["kind"] == "steps":
        RP.check_refs(c["steps"])
        if PLAN == "v4":
            for s in c["steps"]:
                real = [p["name"] for p in _signatures().get(s["tool"], [])]
                old = set(PN.rename_map(s["tool"], real)) & set(s["args"])
                assert not old, f"v4 labels use canonical parameter names, but {s['tool']} has {sorted(old)}"


def teacher_pass(seeds: list[dict], teacher: Callable[[str, int], str], variants: int,
                 cache: Path, guard, taken: set, log=print) -> tuple[list[dict], Counter]:
    """Rewrites for the recipe seeds. -> (records, counts). Appends to `cache`."""
    done: dict[int, list[str]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["i"]] = row["rewrites"]
        # The cache is indexed by seed. If the seeds moved (a recipe inserted before the cached
        # ones), a rewrite would be attached to another question's plan -- a wrong label. A
        # rewrite was checked against its own seed's numbers when it was made, so check it again.
        for i, rewrites in done.items():
            if i >= len(seeds):
                raise RuntimeError(f"rewrite cache has seed {i}, but only {len(seeds)} seeds exist")
            for rw in rewrites:
                why = D.check_rewrite(seeds[i]["question"], rw, args_of(seeds[i]))
                if why:
                    raise RuntimeError(f"rewrite cache does not line up with the seeds at index {i} "
                                       f"({why}); keep the old seeds first or delete the cache")
        log(f"  resuming: {len(done)} of {len(seeds)} seeds already rewritten (cache lines up)")
    counts: Counter = Counter()
    out: list[dict] = []
    t0 = time.time()
    with cache.open("a", encoding="utf-8") as f:
        for i, seed in enumerate(seeds):
            if i in done:
                rewrites = done[i]
            elif seed["meta"].get("no_rewrite"):
                rewrites = []
                f.write(json.dumps({"i": i, "rewrites": rewrites}, ensure_ascii=False) + "\n")
            else:
                rewrites = []
                for k in range(variants):
                    counts["attempts"] += 1
                    try:
                        rw = D.clean_rewrite(teacher(seed["question"], 1000 * i + k))
                    except Exception:
                        counts["teacher error"] += 1
                        continue
                    why = D.check_rewrite(seed["question"], rw, args_of(seed))
                    if why:
                        counts["failed the rewrite check"] += 1
                        continue
                    rewrites.append(rw)
                f.write(json.dumps({"i": i, "rewrites": rewrites}, ensure_ascii=False) + "\n")
                f.flush()
            for rw in rewrites:
                key = D._norm(rw)
                if key in taken:
                    counts["duplicate"] += 1
                    continue
                if D.classifier_verdict(rw, {"tool": seed["plan"].get("tool")
                                             or seed["plan"]["steps"][-1]["tool"]}) == "conflict":
                    counts["classifier conflict"] += 1
                    continue
                verified = seed["meta"].get("verified") or [{}]
                if guard is not None and guard.check(rw, verified[-1].get("tool"),
                                                     verified[-1].get("value")):
                    counts["benchmark overlap"] += 1
                    continue
                taken.add(key)
                counts["accepted"] += 1
                out.append(record(rw, seed["plan"], {
                    "group": seed["group"], **seed["meta"], "source": "rewrite",
                    "origin": "recipe", "rewrite_of": seed["question"]}))
            if (i + 1) % 200 == 0:
                log(f"  rewrites: {i + 1}/{len(seeds)} seeds, {counts['accepted']} kept "
                    f"({time.time() - t0:.0f}s)")
    return out, counts


def assemble(v1_dir: Path, recipes_dir, out_dir: Path, teacher=None, variants: int = 3,
             guard=None, val_pct: int = 10, log=print, plan: str = "v2") -> dict:
    """`recipes_dir`: one folder of seeds.jsonl or several, read in the order given -- keep the
    old seeds first and the teacher-rewrite cache (rewrites.jsonl, indexed by seed) stays valid."""
    global PLAN
    PLAN = plan
    out_dir.mkdir(parents=True, exist_ok=True)
    taken: set = set()
    train: list[dict] = []
    val: list[dict] = []

    for name, bucket in (("train", train), ("val", val)):
        for rec in read_jsonl(v1_dir / f"{name}.jsonl"):
            new = convert_v1(rec)
            validate(new)
            taken.add(D._norm(new["messages"][1]["content"]))
            bucket.append(new)
    v1_counts = {"train": len(train), "val": len(val)}

    dirs = [recipes_dir] if isinstance(recipes_dir, Path) else list(recipes_dir)
    seeds = [sd for d in dirs for sd in read_jsonl(d / "seeds.jsonl")]
    kept_seeds = []
    dropped_label_inconsistent = 0
    for seed in seeds:
        key = D._norm(seed["question"])
        if key in taken:
            continue                       # the same question already exists in v1
        taken.add(key)
        kept_seeds.append(seed)                  # its slot stays: the rewrite cache is indexed by position
        if label_names_missing_variable(seed["question"], seed["plan"]):
            dropped_label_inconsistent += 1      # no record: the label names a variable the question never shows
            continue
        rec = seed_to_record(seed)
        validate(rec)
        (val if D.split_of(seed["group"], val_pct) == "val" else train).append(rec)

    rewrite_counts: Counter = Counter()
    if teacher is not None:
        rewrites, rewrite_counts = teacher_pass(kept_seeds, teacher, variants,
                                                out_dir / "rewrites.jsonl", guard, taken, log)
        for rec in rewrites:
            validate(rec)
            (val if D.split_of(rec["meta"]["group"], val_pct) == "val" else train).append(rec)

    # a question must live in one split only, and a group must not straddle the split
    groups = {"train": {r["meta"]["group"] for r in train}, "val": {r["meta"]["group"] for r in val}}
    assert not groups["train"] & groups["val"], "a group is in both splits"
    qs_train = {D._norm(r["messages"][1]["content"]) for r in train}
    assert not any(D._norm(r["messages"][1]["content"]) in qs_train for r in val), "a question is in both splits"

    for name, rows in (("train", train), ("val", val)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def tally(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "by_source": dict(Counter(r["meta"].get("source") for r in rows)),
            "by_origin": dict(Counter(r["meta"].get("origin") for r in rows)),
            "calls_per_plan": dict(Counter(
                len(RP.canonical(json.loads(r["messages"][2]["content"]))["steps"])
                if RP.canonical(json.loads(r["messages"][2]["content"]))["kind"] == "steps" else 0
                for r in rows)),
            "recipe_families": dict(Counter(r["meta"].get("family") for r in rows if r["meta"].get("family"))),
            **({"records_with_quoted_quantities": sum(1 for r in rows if r["meta"].get("quoted")),
                "quoted_quantities": sum(r["meta"].get("quoted", 0) for r in rows)} if plan == "v4" else {}),
        }
    v1_manifest = {}
    if (v1_dir / "manifest.json").exists():
        v1_manifest = json.loads((v1_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": plan, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system_prompt": RP.system_prompt(plan), "recipe_dirs": [str(d) for d in dirs],
        "v1_records": v1_counts, "v1_tool_surface_hash": v1_manifest.get("tool_surface_hash"),
        "recipe_seeds": len(kept_seeds), "seeds_without_a_record_label_inconsistent": dropped_label_inconsistent,
        "teacher": bool(teacher), "variants": variants if teacher else 0,
        "rewrite_counts": dict(rewrite_counts),
        "train": tally(train), "val": tally(val),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", type=Path, default=REPO / "local-cache" / "router_dataset")
    p.add_argument("--recipes", type=Path, action="append",
                   help="folder of seeds.jsonl; repeat to add more, old seeds first (default: router_recipes)")
    p.add_argument("--plan", choices=["v2", "v3", "v4"], default="v2",
                   help="v3 = v2 + \"$n.field\" references and the v3 system prompt")
    p.add_argument("--out", type=Path, default=REPO / "local-cache" / "router_dataset_v2")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--teacher-url", default="http://127.0.0.1:11434")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--no-guard", action="store_true")
    a = p.parse_args()
    teacher = D.ollama_teacher(a.teacher_url, a.teacher_model) if a.teacher_model else None
    guard = None if a.no_guard else D.BenchmarkGuard.from_repo()
    m = assemble(a.v1, a.recipes or REPO / "local-cache" / "router_recipes", a.out, teacher,
                 a.variants, guard, plan=a.plan)
    print(json.dumps({k: m[k] for k in ("v1_records", "recipe_seeds", "rewrite_counts")}, indent=2))
    print("train:", json.dumps(m["train"]))
    print("val:  ", json.dumps(m["val"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
