#!/usr/bin/env bash
# Download a public GGUF baseline model resumably, then verify its size and sha256 against the Hugging Face file listing.
#
#   bash tools/fetch_baseline.sh mradermacher/P1-30B-A3B-GGUF P1-30B-A3B.Q4_K_M.gguf local-cache/baselines/P1-30B-A3B
#
# Free public weights only (misc/BASELINES.md records the licence of each); a failed or partial download is resumed by running it again.
set -euo pipefail
repo="${1:?hf repo, e.g. mradermacher/P1-30B-A3B-GGUF}"; file="${2:?file in the repo}"; dest="${3:?destination folder}"
mkdir -p "$dest"
meta="$(curl -fsS "https://huggingface.co/api/models/$repo/tree/main" | python -X utf8 -c "
import sys, json
for f in json.load(sys.stdin):
    if f['path'] == sys.argv[1]:
        print(f['size'], f['lfs']['oid'])" "$file")"
[[ -n "$meta" ]] || { echo "no such file $file in $repo" >&2; exit 2; }
size="${meta% *}"; sha="${meta#* }"
echo "$(date '+%F %T') downloading $repo/$file ($size bytes) -> $dest"
curl -fL --retry 20 --retry-delay 10 --retry-all-errors -C - -o "$dest/$file.part" "https://huggingface.co/$repo/resolve/main/$file"
got="$(stat -c %s "$dest/$file.part")"
[[ "$got" == "$size" ]] || { echo "size mismatch: $got != $size (run again to resume)" >&2; exit 3; }
echo "$(date '+%F %T') verifying sha256 ..."
actual="$(sha256sum "$dest/$file.part" | cut -d' ' -f1)"
[[ "$actual" == "$sha" ]] || { echo "sha256 mismatch: $actual != $sha" >&2; exit 4; }
mv "$dest/$file.part" "$dest/$file"
echo "$(date '+%F %T') ok: $dest/$file ($size bytes, sha256 $sha)"
