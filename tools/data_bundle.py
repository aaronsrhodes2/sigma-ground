"""Pack the router's training data into ONE verified file that can be carried to another machine, and prove it rebuilds there.

The data lives in gitignored `local-cache/`, so a new machine (the Bosgame M5, plan: misc/M5_TRAINING_PLAN.md) has none of it. This writes a zip whose
manifest holds a sha256 for every file, the repo commit it was built with, and the exact rebuild command with the hashes that command must reproduce.

    python tools/data_bundle.py create router_v31 --out local-cache/bundles/router_v31_2026-10-15.zip
    python tools/data_bundle.py verify  local-cache/bundles/router_v31_2026-10-15.zip
    python tools/data_bundle.py extract local-cache/bundles/router_v31_2026-10-15.zip --to local-cache      # refuses to overwrite a different file
    python tools/data_bundle.py rebuild-check local-cache/bundles/router_v31_2026-10-15.zip                 # CPU only, no teacher, no GPU

Three tiers: `train` (what training and evaluation read), `rebuild` (seeds, the teacher-rewrite cache, the v1 data and caches: enough to regenerate the
dataset byte for byte, or to regenerate it after the labels change, with no teacher model), `reference` (the results the new runs are compared with).
Weights, run directories and GGUF files are never included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
DEAD_TEACHER = "http://127.0.0.1:9"                                 # nothing listens: a rebuild that needs a teacher fails loudly

SEEDS = ["router_recipes", "router_recipes_v3", "router_declines", "router_math_v31", "router_params_v31"]      # order matters: the rewrite cache is indexed by seed

PROFILES: dict[str, dict] = {
    "router_v31": {
        "description": "Mentat router v3.1 data (plan v3 format, math + parameter-coverage + materials-DB seeds); the data the 2026-10-14 4B run trained on",
        "train": ["router_dataset_v31/train.jsonl", "router_dataset_v31/val.jsonl", "router_dataset_v31/manifest.json",
                  "router_hardval_v31/hard.jsonl", "router_hardval_v31/report.json", "router_math_suite.json"],
        "rebuild": ["router_dataset_v31/rewrites.jsonl", "router_dataset/manifest.json", "router_dataset/train.jsonl", "router_dataset/val.jsonl",
                    "router_hardval/hard.jsonl", "router_hardval/report.json", "router_hardval/verified.json"]
                   + [f"{d}/{f}" for d in SEEDS for f in ("manifest.json", "seeds.jsonl")] + ["router_math_v31/oracle_report.json"],
        "reference": ["router_v31/4b/comparison.md", "eval_math_v31_4b_on_master_tools3.json", "lint_v31_on_master.json",
                      "direct_p1_100.json", "direct_phi4-mini-reasoning_100.json", "direct_qwen3_4b_100.json", "direct_qwen3_8b_100_retry.json"],
        "rebuild_out": "router_dataset_v31",
        "rebuild_expect_files": ["train.jsonl", "val.jsonl", "rewrites.jsonl"],
        "rebuild_args": ["--plan", "v3", "--v1", "{root}/router_dataset"] + [a for d in SEEDS for a in ("--recipes", "{root}/" + d)]
                        + ["--out", "{out}", "--teacher-model", "qwen2.5:14b", "--teacher-url", DEAD_TEACHER, "--variants", "3"],
        "rebuild_seed_file": ("router_dataset_v31/rewrites.jsonl", "rewrites.jsonl"),         # the cache is copied into the output folder first, as the pipeline does
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes_stream(zf: zipfile.ZipFile, name: str) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with zf.open(name) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "tools_dirty": [l[3:] for l in run("status", "--porcelain", "--", "tools", "sigma_ground/mcp").splitlines() if l[:2].strip() in ("M", "A", "MM")][:20]}


# ── create ─────────────────────────────────────────────────────────────────────────────────────────

def create_bundle(root: Path, name: str, profile: dict, out: Path) -> dict:
    files = []
    for tier in ("train", "rebuild", "reference"):
        for rel in profile.get(tier, []):
            p = root / rel
            if not p.is_file():
                raise SystemExit(f"{name}: {tier} file missing: {p}")
            files.append({"path": rel, "tier": tier, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                          **({"lines": count_lines(p)} if rel.endswith(".jsonl") else {})})
    paths = [f["path"] for f in files]
    if len(set(paths)) != len(paths):
        raise SystemExit(f"{name}: a file is listed twice")
    by = {f["path"]: f for f in files}
    seed_src, seed_dst = profile["rebuild_seed_file"]
    manifest = {"bundle": name, "description": profile["description"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "repo": git_state(),
                "python": sys.version.split()[0], "files": files,
                "totals": {t: {"files": sum(f["tier"] == t for f in files), "bytes": sum(f["bytes"] for f in files if f["tier"] == t)} for t in ("train", "rebuild", "reference")},
                "rebuild": {"command": ["python", "tools/build_router_v2.py"] + profile["rebuild_args"], "out": profile["rebuild_out"],
                            "seed_cache": [seed_src, seed_dst], "teacher": "none: the rewrite cache covers every seed",
                            "expect": {f: by[f"{profile['rebuild_out']}/{f}"]["sha256"] for f in profile["rebuild_expect_files"]}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=1))
        for f in files:
            zf.write(root / f["path"], f["path"])
    tmp.replace(out)
    manifest["_zip_bytes"] = out.stat().st_size
    return manifest


# ── verify / extract ───────────────────────────────────────────────────────────────────────────────

def read_manifest(zf: zipfile.ZipFile) -> dict:
    return json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))


def verify_zip(path: Path) -> list[str]:
    """Problems found (empty = the bundle is exactly what its manifest says)."""
    problems = []
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            problems.append(f"corrupt member: {bad}")
        m = read_manifest(zf)
        names = set(zf.namelist()) - {MANIFEST_NAME}
        listed = {f["path"] for f in m["files"]}
        problems += [f"in the zip but not in the manifest: {n}" for n in sorted(names - listed)]
        problems += [f"in the manifest but not in the zip: {n}" for n in sorted(listed - names)]
        for f in m["files"]:
            if f["path"] in names:
                digest, size = sha256_bytes_stream(zf, f["path"])
                if digest != f["sha256"] or size != f["bytes"]:
                    problems.append(f"{f['path']}: sha256/size differs from the manifest")
    return problems


def extract_zip(path: Path, to: Path, force: bool = False, tiers: tuple[str, ...] = ("train", "rebuild", "reference")) -> dict:
    problems = verify_zip(path)
    if problems:
        raise SystemExit("bundle failed verification, nothing written:\n  " + "\n  ".join(problems))
    wrote, same, refused = [], [], []
    with zipfile.ZipFile(path) as zf:
        m = read_manifest(zf)
        for f in m["files"]:
            if f["tier"] not in tiers:
                continue
            dst = to / f["path"]
            if dst.exists():
                if sha256_file(dst) == f["sha256"]:
                    same.append(f["path"])
                    continue
                if not force:
                    refused.append(f["path"])
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(f["path"]) as src, open(dst, "wb") as out:
                out.write(src.read())
            wrote.append(f["path"])
    if refused:
        raise SystemExit("these exist and differ (use --force to replace):\n  " + "\n  ".join(refused))
    return {"wrote": wrote, "already_identical": same}


# ── rebuild check ──────────────────────────────────────────────────────────────────────────────────

def rebuild_check(path: Path, keep: bool = False) -> dict:
    """Extract the bundle to a scratch folder and regenerate the dataset with no teacher; compare with the manifest's hashes."""
    with tempfile.TemporaryDirectory(prefix="bundle_rebuild_") as tmp:
        root = Path(tmp) / "local-cache"
        extract_zip(path, root, tiers=("train", "rebuild"))
        with zipfile.ZipFile(path) as zf:
            m = read_manifest(zf)
        out = root / "_rebuilt"
        out.mkdir(parents=True)
        src, dst = m["rebuild"]["seed_cache"]
        (out / dst).write_bytes((root / src).read_bytes())
        args = [a.replace("{root}", str(root)).replace("{out}", str(out)) for a in m["rebuild"]["command"][2:]]
        t0 = time.time()
        r = subprocess.run([sys.executable, "-X", "utf8", str(REPO / m["rebuild"]["command"][1]), *args], cwd=REPO, capture_output=True, text=True, timeout=3600)
        result = {"exit": r.returncode, "seconds": round(time.time() - t0, 1), "files": {}}
        for fname, want in m["rebuild"]["expect"].items():
            p = out / fname
            got = sha256_file(p) if p.exists() else None
            result["files"][fname] = {"identical": got == want, "expected": want[:16], "got": (got or "missing")[:16]}
        result["ok"] = r.returncode == 0 and all(v["identical"] for v in result["files"].values())
        if not result["ok"]:
            result["stderr_tail"] = (r.stderr or r.stdout)[-1500:]
        if keep:
            result["kept_at"] = str(out)
            import shutil
            shutil.copytree(out, REPO / "local-cache" / "_rebuild_kept", dirs_exist_ok=True)
        return result


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("profile", choices=sorted(PROFILES))
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--root", type=Path, default=REPO / "local-cache")
    v = sub.add_parser("verify")
    v.add_argument("bundle", type=Path)
    e = sub.add_parser("extract")
    e.add_argument("bundle", type=Path)
    e.add_argument("--to", type=Path, default=REPO / "local-cache")
    e.add_argument("--force", action="store_true")
    e.add_argument("--tiers", default="train,rebuild,reference")
    r = sub.add_parser("rebuild-check")
    r.add_argument("bundle", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "create":
        m = create_bundle(a.root, a.profile, PROFILES[a.profile], a.out)
        t = m["totals"]
        print(f"{a.out}: {m['_zip_bytes'] / 2**20:.1f} MiB zipped; " + "; ".join(f"{k} {v['files']} files {v['bytes'] / 2**20:.1f} MiB" for k, v in t.items())
              + f"\nrepo {m['repo']['commit'][:10]} ({m['repo']['branch']}); {len(m['files'])} files hashed")
        if m["repo"]["tools_dirty"]:
            print("WARNING: uncommitted changes under tools/ or sigma_ground/mcp/ at build time:", m["repo"]["tools_dirty"])
        return 0
    if a.cmd == "verify":
        problems = verify_zip(a.bundle)
        print("OK: every file matches its manifest hash" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
        return 1 if problems else 0
    if a.cmd == "extract":
        res = extract_zip(a.bundle, a.to, a.force, tuple(a.tiers.split(",")))
        print(f"wrote {len(res['wrote'])} files, {len(res['already_identical'])} already identical, into {a.to}")
        return 0
    res = rebuild_check(a.bundle)
    for fname, v in res["files"].items():
        print(f"  {'IDENTICAL' if v['identical'] else 'DIFFERENT':10s} {fname}  expected {v['expected']}  got {v['got']}")
    print(f"rebuild {'OK' if res['ok'] else 'FAILED'} in {res['seconds']} s (exit {res['exit']}, no teacher, CPU only)")
    if not res["ok"]:
        print(res.get("stderr_tail", ""))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
