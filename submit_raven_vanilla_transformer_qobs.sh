#!/bin/bash -l
#SBATCH --job-name=camels_vit_qobs
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@600
#SBATCH --chdir=/u/xshan/Research/GT/cs7643_DL/courseProject/CAMELS_data_load
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -uo pipefail

PROJECT_DIR="/u/xshan/Research/GT/cs7643_DL/courseProject/CAMELS_data_load"
SCRIPT_PATH="${PROJECT_DIR}/submit_raven_vanilla_transformer_qobs.sh"
PYTHON_SCRIPT="${PROJECT_DIR}/train_vanilla_transformer_qobs.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_NAME="${RUN_NAME:-raven_default}"
ROUND="${ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
CONFIG_JSON="${CONFIG_JSON:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/model_artifacts}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DONE_FLAG="${RUN_DIR}/DONE"

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/ptmp/${USER}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Found DONE flag at ${DONE_FLAG}. Nothing to run."
  exit 0
fi

echo "JOBID=${SLURM_JOB_ID} ROUND=${ROUND} MAX_ROUNDS=${MAX_ROUNDS}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

CMD=(
  "${PYTHON_BIN}" "${PYTHON_SCRIPT}"
  --device cuda
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --num-workers 4
  --pin-memory
)

if [[ -n "${CONFIG_JSON}" ]]; then
  CMD+=(--config-json "${CONFIG_JSON}")
fi

echo "Launching training command: ${CMD[*]}"
"${CMD[@]}"
exit_code=$?

echo "Training command exited with code ${exit_code}"

if [[ -f "${DONE_FLAG}" ]]; then
  echo "Training completed successfully. DONE flag found."
  exit 0
fi

if [[ "${ROUND}" -ge "${MAX_ROUNDS}" ]]; then
  echo "Reached MAX_ROUNDS=${MAX_ROUNDS} without DONE flag."
  exit "${exit_code}"
fi

next_round=$((ROUND + 1))
echo "DONE flag not found. Resubmitting round ${next_round} after job ${SLURM_JOB_ID}."

sbatch --dependency=afterany:${SLURM_JOB_ID} \
  --export=ALL,ROUND="${next_round}",MAX_ROUNDS="${MAX_ROUNDS}",RUN_NAME="${RUN_NAME}",CONFIG_JSON="${CONFIG_JSON}",DATA_DIR="${DATA_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",PYTHON_BIN="${PYTHON_BIN}" \
  "${SCRIPT_PATH}"

exit "${exit_code}"
