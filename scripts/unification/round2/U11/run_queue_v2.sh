#!/bin/bash
# v2 queue: each line (name + args), at most 3 concurrent processes, unmodified engine, repo root as cwd
D=/c/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11
cd /d/Aaron/development/sigma-ground
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_one() {
  name=$1; shift
  python -X utf8 $D/u11_sph_v2.py "$@" --steps 2000 --record_every 100 --K_line 0.477464829275686 --out $D/v2/run_${name}_v2.json > $D/v2/log_${name}_v2.txt 2>&1
  echo "finished $name $(date +%H:%M:%S)"
}
export -f run_one; export D
cat $D/v2/queue_v2.txt | xargs -P 3 -L 1 bash -c 'run_one $0 "$@"'
echo ALLDONE $(date +%H:%M:%S)
