#!/usr/bin/env bash
# U5 run queue: at most 3 concurrent worker processes (process hygiene rule 7)
cd /d/Aaron/development/sigma-ground
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf '%s\n' "A 600 50" "B 600 50" "C 600 50" "E 600 50" "D 600 50" "A 300 50" "C 300 50" |   xargs -P 3 -L 1 bash -c 'python -X utf8 "C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U5/u5_run.py" "C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U5" $0 $1 $2 > "C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U5/runs/log_$0_$1.txt" 2>&1'
echo QUEUE_DONE
