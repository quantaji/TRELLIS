#!/bin/bash
set -euo pipefail

LANES=10
NBATCH=36
A=/home/quanta/TRELLIS/guangda_scripts/omni_partverse_preprocess_02_render.sbatch
B=/home/quanta/TRELLIS/guangda_scripts/omni_partverse_preprocess_04_extract_dino_feature.sbatch

for lane in $(seq 0 $((LANES-1))); do
  prev=""  # 这个 lane 上一个B的jobid
  for batch in $(seq 0 $((NBATCH-1))); do
    r=$((lane + batch*LANES))

    dep=()
    [[ -n "$prev" ]] && dep=(--dependency=afterany:$prev)  

    a=$(sbatch --parsable "${dep[@]}" --export=ALL,RANK=$r,WORLD_SIZE=360 "$A")
    b=$(sbatch --parsable --dependency=afterok:$a --export=ALL,RANK=$r,WORLD_SIZE=360 "$B")

    prev=$b
    echo "lane=$lane batch=$batch rank=$r A=$a B=$b"
  done
done

# sbatch --export=ALL,RANK=7,WORLD_SIZE=360 /home/quanta/TRELLIS/guangda_scripts/omni_partverse_preprocess_04_extract_dino_feature.sbatch
