#!/bin/bash
#SBATCH --job-name=metadata
#SBATCH --output=omnipart_partverse_preproces_00_metadata_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

module load apptainer/1.3.5

# Host paths
SRC_DIR_HOST="/scratch/quanta/partverse/"
PRE_DIR_HOST="/scratch/quanta/partverse_omnipart_preprocessing_temp"
TGT_DIR_HOST="/scratch/quanta/partverse_omnipart_preprocessed"
PY_SCRIPT_HOST="/home/quanta/TRELLIS/"
SIF_IMG="/home/quanta/trellis.sif"

mkdir -p "$TGT_DIR_HOST"
mkdir -p "$PRE_DIR_HOST"

CONDA_ENV_NAME="trellis"
CONDA_ROOT="/miniconda3"

apptainer exec \
  --bind "${SRC_DIR_HOST}:/mnt/src" \
  --bind "${PRE_DIR_HOST}:/mnt/pre" \
  --bind "${TGT_DIR_HOST}:/mnt/tgt" \
  --bind "/home/quanta:/home/quanta" \
  "$SIF_IMG" \
  bash -lc "
    set -eo pipefail

    export PATH=${CONDA_ROOT}/bin:\$PATH
    eval \"\$(conda shell.bash hook)\"
    conda activate ${CONDA_ENV_NAME}

    python ${PY_SCRIPT_HOST}/dataset_toolkits/build_metadata.py Partverse \
        --partverse_dir /mnt/src \
        --output_dir /mnt/pre
  "
