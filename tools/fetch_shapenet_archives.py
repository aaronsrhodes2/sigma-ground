"""Download the gated ShapeNet archives to local disk (access-approved account).

Sequential, resumable (hf resume on partial files), priority-ordered: the
metadata-bearing Sem zip first, then Core v2/v1, then PartNet's split data_v0
volumes, then the ML segmentation bundles, and finally the processed
ShapeNetCore repo. Raw archives land OUTSIDE any git repo (never committed,
never redistributed — ShapeNet Terms of Use, see docs/DATA_SOURCES.md).

    python tools/fetch_shapenet_archives.py
"""
from __future__ import annotations

import pathlib

_ENV = pathlib.Path("D:/Aaron/development/.env")
_DEST = pathlib.Path("D:/Aaron/datasets/shapenet")

# (repo, filename) in download priority order; None filename = whole repo snapshot
_FILES = [
    ("ShapeNetSem-archive",  "ShapeNetSem.zip"),         # 12.2 GB — dims/weights metadata
    ("ShapeNetCore-archive", "ShapeNetCore.v2.zip"),     # 26.8 GB — improved meshes
    ("ShapeNetCore-archive", "ShapeNetCore.v1.zip"),     # 32.2 GB — original + splits
    ("PartNet-archive",      "data_v0_chunk.zip"),       # split-zip last volume
] + [("PartNet-archive", f"data_v0_chunk.z{i:02d}") for i in range(1, 11)] + [
    ("PartNet-archive",      "ins_seg_h5.zip"),          # 21.2 GB — ML point clouds
    ("PartNet-archive",      "sem_seg_h5.zip"),          #  8.6 GB — ML point clouds
    ("ShapeNetCore",         None),                      # 24 GB — processed repo snapshot
]


def _token() -> str | None:
    try:
        for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("HF_ID="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    tok = _token()
    done = failed = 0
    for repo, fn in _FILES:
        dest = _DEST / repo
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if fn is None:
                print(f"[snapshot] ShapeNet/{repo} -> {dest}", flush=True)
                snapshot_download(f"ShapeNet/{repo}", repo_type="dataset",
                                  token=tok, local_dir=str(dest))
            else:
                print(f"[file] ShapeNet/{repo}/{fn} -> {dest}", flush=True)
                hf_hub_download(f"ShapeNet/{repo}", fn, repo_type="dataset",
                                token=tok, local_dir=str(dest))
            done += 1
            print(f"  OK ({done} done)", flush=True)
        except Exception as e:                            # keep going; rerun resumes
            failed += 1
            print(f"  FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"\nfinished: {done} ok, {failed} failed (rerun to resume any failures)",
          flush=True)


if __name__ == "__main__":
    main()
