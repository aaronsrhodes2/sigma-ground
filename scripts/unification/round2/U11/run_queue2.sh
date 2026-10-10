#!/bin/bash
# runs each queue line (name + args) with at most 3 concurrent processes
D=/c/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11
cd /d/Aaron/development/sigma-ground
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_one() {
  name=$1; shift
  python -X utf8 $D/u11_sph.py "$@" --steps 2000 --max_steps 8000 --record_every 100 --K_line 0.477464829275686 --out $D/run_$name.json > $D/log_$name.txt 2>&1
  echo "finished $name"
}
export -f run_one; export D
cat $1 | xargs -P 2 -L 1 bash -c 'run_one $0 "$@"'
