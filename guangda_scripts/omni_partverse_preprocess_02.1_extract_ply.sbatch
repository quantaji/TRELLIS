#!/bin/bash
#SBATCH --job-name=omnipart_partverse_preproces_02_render
#SBATCH --output=omnipart_partverse_preproces_02_render_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-359

set -euo pipefail

module load apptainer/1.3.5 cuda/12.9

# Host paths
SRC_DIR_HOST="/scratch/quanta/partverse/"
PRE_DIR_HOST="/scratch/quanta/partverse_omnipart_preprocessing_temp"
TGT_DIR_HOST="/scratch/quanta/partverse_omnipart_preprocessed"
PY_SCRIPT_HOST="/home/quanta/TRELLIS"
SIF_IMG="/home/quanta/trellis.sif"

mkdir -p "$TGT_DIR_HOST"
mkdir -p "$PRE_DIR_HOST"

CONDA_ENV_NAME="trellis"
CONDA_ROOT="/miniconda3"

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD_SIZE=360

apptainer exec  --nv \
  --bind "${SRC_DIR_HOST}:/mnt/src" \
  --bind "${PRE_DIR_HOST}:/mnt/pre" \
  --bind "${TGT_DIR_HOST}:/mnt/tgt" \
  --bind "/home/quanta:/home/quanta" \
  "$SIF_IMG" \
  bash -lc "
    set -euo pipefail

    export PATH=${CONDA_ROOT}/bin:\$PATH
    eval \"\$(conda shell.bash hook)\"
    conda activate ${CONDA_ENV_NAME}

    python ${PY_SCRIPT_HOST}/dataset_toolkits/render.py Partverse \
        --output_dir /mnt/pre --max_workers 1 --rank ${RANK} --world_size ${WORLD_SIZE} --skip_render
  "
