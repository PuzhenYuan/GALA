#!/usr/bin/env bash
# Multi-process RoboCasa GR1 simulation evaluation.
# Each worker runs one single-GPU inference server and evaluates a disjoint task shard.
#
# Usage:
#   GPU_IDS=0,1,2,3,4,5,6,7 PROCS_PER_GPU=3 bash examples/run_eval_parallel.sh <model_path> <eval_type> [note]
#
# Environment variables:
#   PROCS_PER_GPU     Number of eval workers per GPU. Default: 3.
#   NUM_EVAL_PROCS    Number of parallel eval workers. Default: number of GPU_IDS * PROCS_PER_GPU.
#   GPU_IDS           Comma-separated or space-separated GPU ids. Default: 0,1,2,3,4,5,6,7.
#   PORT_BASE         First inference server port. Default: 5810.
#   EVAL_TAG          Base tag (default: _gala_parallel). Workers write under `${base_eval_dir}/p{rank}of{NUM_EVAL_PROCS}`.
#   N_ENVS            Passed through to run_eval.sh.
#   N_EPISODES        Passed through to run_eval.sh.
#   N_ACTION_STEPS    Actions executed before requesting a new action chunk. Default: 16.
#   EVAL_SAMPLE_STEP  Number of flow-matching denoising steps per inference. Default: 4.
#   INFER_SAMPLE_SHIFT Rational shift: >1 noise-dense, <1 action-dense. Default: 1.0.
#   EVAL_NOTE         Optional output-name suffix; third positional argument takes priority.
#   EVAL_MAX_TASKS    Optional cap before sharding, passed through to run_eval.sh.
#   DATA_CONFIG       Passed through to run_eval.sh.
#   PYTHON_BIN        Passed through to run_eval.sh.
#   EVAL_PARALLEL_DRY_RUN  Print worker assignment and create no eval workers. Default: 0.
#   RESUME_EVAL       Skip shards whose client logs already contain all assigned success rates. Default: 0.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH=${1:?"Usage: bash examples/run_eval_parallel.sh <model_path> <eval_type> [note]"}
EVAL_TYPE=${2:?"Eval type: id"}
EVAL_NOTE=${3:-${EVAL_NOTE:-}}

export PYTHONDONTWRITEBYTECODE=1
export USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PROJECT_ROOT}/.cache/numba}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${EVAL_PARALLEL_DRY_RUN:-0}" != "1" ]]; then
    MODEL_PATH="$("${PYTHON_BIN:-python3}" scripts/resolve_checkpoint.py "${MODEL_PATH}")"
fi


GPU_IDS_RAW="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPU_ID_LIST <<< "${GPU_IDS_RAW}"

PROCS_PER_GPU="${PROCS_PER_GPU:-3}"
NUM_EVAL_PROCS="${NUM_EVAL_PROCS:-$((${#GPU_ID_LIST[@]} * PROCS_PER_GPU))}"
PORT_BASE="${PORT_BASE:-5810}"
BASE_EVAL_TAG="${EVAL_TAG:-_gala_parallel}"
EVAL_PARALLEL_DRY_RUN="${EVAL_PARALLEL_DRY_RUN:-0}"
RESUME_EVAL="${RESUME_EVAL:-0}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
EVAL_SAMPLE_STEP="${EVAL_SAMPLE_STEP:-4}"
INFER_SAMPLE_SHIFT="${INFER_SAMPLE_SHIFT:-1.0}"

EVAL_NOTE="${EVAL_NOTE#_}"
if [[ -n "${EVAL_NOTE}" && ! "${EVAL_NOTE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: note may contain only letters, digits, '.', '_' and '-': ${EVAL_NOTE}" >&2
  exit 1
fi
NOTE_SUFFIX=""
if [[ -n "${EVAL_NOTE}" ]]; then
  NOTE_SUFFIX="_${EVAL_NOTE}"
fi
OUTPUT_EVAL_TAG="${BASE_EVAL_TAG}${NOTE_SUFFIX}"

if (( NUM_EVAL_PROCS < 1 )); then
  echo "Error: NUM_EVAL_PROCS must be >= 1, got ${NUM_EVAL_PROCS}" >&2
  exit 1
fi
if (( PROCS_PER_GPU < 1 )); then
  echo "Error: PROCS_PER_GPU must be >= 1, got ${PROCS_PER_GPU}" >&2
  exit 1
fi
if (( ${#GPU_ID_LIST[@]} < 1 )); then
  echo "Error: GPU_IDS must provide at least one GPU id" >&2
  exit 1
fi
if (( N_ACTION_STEPS < 1 )); then
  echo "Error: N_ACTION_STEPS must be >= 1, got ${N_ACTION_STEPS}" >&2
  exit 1
fi
if [[ ! "${EVAL_SAMPLE_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: EVAL_SAMPLE_STEP must be a positive integer, got ${EVAL_SAMPLE_STEP}" >&2
  exit 1
fi

case "$EVAL_TYPE" in
  id) BASE_EVAL_SUBDIR="evaluation_sim_id_${N_ENVS:-1}envs${OUTPUT_EVAL_TAG}" ;;
  *)
    echo "Error: Unknown EVAL_TYPE '${EVAL_TYPE}'." >&2
    exit 1 ;;
esac
BASE_EVAL_DIR="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}/${BASE_EVAL_SUBDIR}"
AGG_CLIENT_LOG="${BASE_EVAL_DIR}/test_robocasa_gr1_client_merged.log"
AGG_RESULTS_JSON="${BASE_EVAL_DIR}/results.json"

total_tasks_by_type() {
  case "$1" in
    id) echo 24 ;;
    *) echo 0 ;;
  esac
}
TOTAL_TASKS="$(total_tasks_by_type "${EVAL_TYPE}")"
if [[ -n "${EVAL_MAX_TASKS:-}" && "${EVAL_MAX_TASKS}" -lt "${TOTAL_TASKS}" ]]; then
  TOTAL_TASKS="${EVAL_MAX_TASKS}"
fi
if (( TOTAL_TASKS > 0 && NUM_EVAL_PROCS > TOTAL_TASKS )); then
  echo "Capping NUM_EVAL_PROCS from ${NUM_EVAL_PROCS} to TOTAL_TASKS=${TOTAL_TASKS}."
  NUM_EVAL_PROCS="${TOTAL_TASKS}"
fi

echo "=============================================="
echo "Parallel RoboCasa GR1 Evaluation"
echo "  MODEL_PATH:      ${MODEL_PATH}"
echo "  EVAL_TYPE:       ${EVAL_TYPE}"
echo "  TOTAL_TASKS:      ${TOTAL_TASKS}"
echo "  PROCS_PER_GPU:    ${PROCS_PER_GPU}"
echo "  NUM_EVAL_PROCS:  ${NUM_EVAL_PROCS}"
echo "  GPU_IDS:         ${GPU_ID_LIST[*]}"
echo "  PORT_BASE:       ${PORT_BASE}"
echo "  BASE_EVAL_TAG:   ${BASE_EVAL_TAG}"
echo "  EVAL_NOTE:       ${EVAL_NOTE:-<none>}"
echo "  N_ACTION_STEPS:  ${N_ACTION_STEPS:-16}"
echo "  EVAL_SAMPLE_STEP: ${EVAL_SAMPLE_STEP}"
echo "  INFER_SAMPLE_SHIFT: ${INFER_SAMPLE_SHIFT}"
echo "  RESUME_EVAL:     ${RESUME_EVAL}"
echo "  BASE_EVAL_DIR:   ${BASE_EVAL_DIR}"
echo "=============================================="

mkdir -p "${BASE_EVAL_DIR}"

pids=()
for ((rank = 0; rank < NUM_EVAL_PROCS; rank++)); do
  gpu_id="${GPU_ID_LIST[$((rank % ${#GPU_ID_LIST[@]}))]}"
  port="$((PORT_BASE + rank))"
  shard_name="p${rank}of${NUM_EVAL_PROCS}"
  shard_tag="${OUTPUT_EVAL_TAG}/${shard_name}"

  assigned_tasks=()
  for ((task_idx = rank; task_idx < TOTAL_TASKS; task_idx += NUM_EVAL_PROCS)); do
    assigned_tasks+=("${task_idx}")
  done

  echo "[worker ${rank}/${NUM_EVAL_PROCS}] gpu=${gpu_id} gpu_slot=$((rank / ${#GPU_ID_LIST[@]})) port=${port} dir=${BASE_EVAL_DIR}/${shard_name} task_indices=${assigned_tasks[*]:-(none)}"
  shard_client_log="${BASE_EVAL_DIR}/${shard_name}/test_robocasa_gr1_client.log"
  if [[ "${RESUME_EVAL}" == "1" && -f "${shard_client_log}" ]]; then
    completed_rates="$(grep -c -i '^Success rate:' "${shard_client_log}" || true)"
    if (( completed_rates >= ${#assigned_tasks[@]} )); then
      echo "[worker ${rank}/${NUM_EVAL_PROCS}] resume: complete (${completed_rates}/${#assigned_tasks[@]} rates), skipping"
      continue
    fi
    echo "[worker ${rank}/${NUM_EVAL_PROCS}] resume: incomplete (${completed_rates}/${#assigned_tasks[@]} rates), rerunning"
  fi
  if [[ "${EVAL_PARALLEL_DRY_RUN}" == "1" ]]; then
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PORT="${port}"
    export EVAL_TAG="${shard_tag}"
    export EVAL_TASK_OFFSET="${rank}"
    export EVAL_TASK_STRIDE="${NUM_EVAL_PROCS}"
    export EVAL_WORKER_RANK="${rank}"
    export EVAL_WORKER_TOTAL="${NUM_EVAL_PROCS}"
    export EVAL_NOTE=""
    export N_ACTION_STEPS
    export EVAL_SAMPLE_STEP
    export INFER_SAMPLE_SHIFT
    bash examples/run_eval.sh "${MODEL_PATH}" "${EVAL_TYPE}"
  ) > /dev/null &
  pids+=("$!")
done

if [[ "${EVAL_PARALLEL_DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. No eval workers were started."
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "One or more eval workers failed. Check stderr and the client/server logs under ${BASE_EVAL_DIR}/p*of${NUM_EVAL_PROCS}/" >&2
  exit "${status}"
fi

echo "All eval workers completed."

: > "${AGG_CLIENT_LOG}"
for ((rank = 0; rank < NUM_EVAL_PROCS; rank++)); do
  shard_dir="${BASE_EVAL_DIR}/p${rank}of${NUM_EVAL_PROCS}"
  shard_log="${shard_dir}/test_robocasa_gr1_client.log"
  if [[ -f "${shard_log}" ]]; then
    {
      echo "================ ${shard_dir} ================"
      cat "${shard_log}"
      echo
    } >> "${AGG_CLIENT_LOG}"
  else
    echo "Warning: missing shard client log: ${shard_log}" >&2
  fi
done

"${PYTHON_BIN:-python3}" scripts/compute_success_rate.py -i "${AGG_CLIENT_LOG}" -o "${AGG_RESULTS_JSON}"
echo "Merged client log: ${AGG_CLIENT_LOG}"
echo "Aggregated results: ${AGG_RESULTS_JSON}"
