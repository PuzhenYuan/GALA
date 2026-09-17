#!/bin/bash
# GALA RoboCasa GR1 simulation evaluation.
# Usage:
#   bash examples/run_eval.sh <model_path> <eval_type> [note]
#
# EVAL_TYPE:
#   id                         - In-distribution training tasks
#                                (6 PnPClose + 18 PosttrainPnPNovel SplitA)
#
# Environment variables:
#   ACTION_REPRESENTATION  absolute (default).
#   DATA_CONFIG            Optional explicit override. Its action semantics must match the checkpoint.
#   PYTHON_BIN             (default: python3)
#   N_ENVS                 (default: 1)
#   N_EPISODES             (default: 50)
#   N_ACTION_STEPS         Actions executed before requesting a new action chunk. Default: 16.
#   EVAL_SAMPLE_STEP       Number of flow-matching denoising steps per inference. Default: 4.
#   INFER_SAMPLE_SHIFT     Rational shift: >1 noise-dense, <1 action-dense. Default: 1.0.
#   EVAL_MAX_TASKS         (default: all tasks)
#   EVAL_TASK_OFFSET       (default: 0; shard task index offset)
#   EVAL_TASK_STRIDE       (default: 1; shard task index stride)
#   EVAL_WORKER_RANK       Optional worker rank label for parallel eval logs.
#   EVAL_WORKER_TOTAL      Optional worker count label for parallel eval logs.
#   CUDA_VISIBLE_DEVICES   (default: 0)
#   PORT                   (default: 5800 + first GPU id)
#   EVAL_TAG               (default: _run1)
#   EVAL_NOTE              Optional output-name suffix; third positional argument takes priority.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${1:?"Usage: bash examples/run_eval.sh <model_path> <eval_type> [note]"}
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


ACTION_REPRESENTATION=${ACTION_REPRESENTATION:-absolute}
case "$ACTION_REPRESENTATION" in
  absolute)
    DEFAULT_DATA_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only
    ;;
  *)
    echo "Error: ACTION_REPRESENTATION must be absolute, got '$ACTION_REPRESENTATION'" >&2
    exit 2
    ;;
esac
DATA_CONFIG=${DATA_CONFIG:-$DEFAULT_DATA_CONFIG}
PYTHON_BIN=${PYTHON_BIN:-python3}
N_ENVS=${N_ENVS:-1}
N_EPISODES=${N_EPISODES:-50}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
EVAL_SAMPLE_STEP=${EVAL_SAMPLE_STEP:-4}
INFER_SAMPLE_SHIFT=${INFER_SAMPLE_SHIFT:-1.0}
EVAL_MAX_TASKS=${EVAL_MAX_TASKS:-}
EVAL_TASK_OFFSET=${EVAL_TASK_OFFSET:-0}
EVAL_TASK_STRIDE=${EVAL_TASK_STRIDE:-1}
EVAL_WORKER_RANK=${EVAL_WORKER_RANK:-0}
EVAL_WORKER_TOTAL=${EVAL_WORKER_TOTAL:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
_CVD_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
PORT="${PORT:-$((5800 + _CVD_GPU))}"
EVAL_TAG=${EVAL_TAG:-_run1}
export CUDA_VISIBLE_DEVICES

if (( N_ACTION_STEPS < 1 )); then
    echo "Error: N_ACTION_STEPS must be >= 1, got ${N_ACTION_STEPS}" >&2
    exit 1
fi
if [[ ! "${EVAL_SAMPLE_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: EVAL_SAMPLE_STEP must be a positive integer, got ${EVAL_SAMPLE_STEP}" >&2
    exit 1
fi

EVAL_NOTE="${EVAL_NOTE#_}"
if [[ -n "${EVAL_NOTE}" && ! "${EVAL_NOTE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: note may contain only letters, digits, '.', '_' and '-': ${EVAL_NOTE}" >&2
    exit 1
fi
NOTE_SUFFIX=""
if [[ -n "${EVAL_NOTE}" ]]; then
    NOTE_SUFFIX="_${EVAL_NOTE}"
fi

case "$EVAL_TYPE" in
  id)
    EVAL_SUBDIR="evaluation_sim_id_${N_ENVS}envs${EVAL_TAG}${NOTE_SUFFIX}"
    task_names=(
        gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  *)
    echo "Error: Unknown EVAL_TYPE '$EVAL_TYPE'."
    echo "Valid: id"
    exit 1 ;;
esac

echo "=============================================="
echo "GALA RoboCasa GR1 Evaluation"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  EVAL_TYPE:  $EVAL_TYPE  ($EVAL_SUBDIR)"
echo "  DATA_CONFIG:$DATA_CONFIG"
echo "  ACTION_REPRESENTATION: $ACTION_REPRESENTATION"
echo "  PYTHON_BIN: $PYTHON_BIN"
echo "  N_ENVS:     $N_ENVS"
echo "  N_EPISODES: $N_EPISODES"
echo "  ACTION STEPS BEFORE RE-INFER: $N_ACTION_STEPS"
echo "  DENOISING SAMPLE STEPS: $EVAL_SAMPLE_STEP"
echo "  INFER SAMPLE SHIFT: $INFER_SAMPLE_SHIFT"
echo "  EVAL_NOTE:  ${EVAL_NOTE:-<none>}"
echo "  PORT:       $PORT  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  TASKS:      ${#task_names[@]}"
echo "  WORKER:     ${EVAL_WORKER_RANK}/${EVAL_WORKER_TOTAL}"
echo "=============================================="

if [[ -n "$EVAL_MAX_TASKS" ]]; then
    task_names=("${task_names[@]:0:${EVAL_MAX_TASKS}}")
fi

if (( EVAL_TASK_STRIDE < 1 )); then
    echo "Error: EVAL_TASK_STRIDE must be >= 1, got ${EVAL_TASK_STRIDE}" >&2
    exit 1
fi
if (( EVAL_TASK_OFFSET < 0 || EVAL_TASK_OFFSET >= EVAL_TASK_STRIDE )); then
    echo "Error: EVAL_TASK_OFFSET must be in [0, EVAL_TASK_STRIDE), got ${EVAL_TASK_OFFSET}/${EVAL_TASK_STRIDE}" >&2
    exit 1
fi
if (( EVAL_TASK_STRIDE > 1 )); then
    sharded_task_names=()
    for i in "${!task_names[@]}"; do
        if (( i % EVAL_TASK_STRIDE == EVAL_TASK_OFFSET )); then
            sharded_task_names+=("${task_names[$i]}")
        fi
    done
    task_names=("${sharded_task_names[@]}")
fi

echo "  TASK_SHARD: offset=${EVAL_TASK_OFFSET} stride=${EVAL_TASK_STRIDE} selected=${#task_names[@]}"
for task_name in "${task_names[@]}"; do
    echo "  TASK_ASSIGNMENT: worker=${EVAL_WORKER_RANK}/${EVAL_WORKER_TOTAL} gpu=${CUDA_VISIBLE_DEVICES} port=${PORT} task=${task_name}"
done

EVAL_DIR="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}/${EVAL_SUBDIR}"
mkdir -p "$EVAL_DIR"

CLIENT_LOG="${EVAL_DIR}/test_robocasa_gr1_client.log"
SERVER_LOG="${EVAL_DIR}/test_robocasa_gr1_server.log"
RESULTS_JSON="${EVAL_DIR}/results.json"

export PYTHONDONTWRITEBYTECODE=1
export USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PROJECT_ROOT}/.cache/numba}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

nohup "$PYTHON_BIN" -u scripts/inference_service.py --server \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --data_config "$DATA_CONFIG" \
    --denoising_steps "$EVAL_SAMPLE_STEP" \
    --infer_sample_shift "$INFER_SAMPLE_SHIFT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Server PID: $SERVER_PID (log: $SERVER_LOG)"
"$PYTHON_BIN" scripts/wait_for_server.py "$PORT" "$SERVER_PID" "${SERVER_START_TIMEOUT:-600}"

{
failed=0
for task_name in "${task_names[@]}"; do
    echo "Executing command for: $task_name WORKER: ${EVAL_WORKER_RANK}/${EVAL_WORKER_TOTAL} GPU: ${CUDA_VISIBLE_DEVICES} PORT: ${PORT} MODEL_PATH: ${MODEL_PATH}"
    retry=0; max_retries=5; ok=0
    while [ $retry -lt $max_retries ] && [ $ok -eq 0 ]; do
        if "$PYTHON_BIN" -u scripts/simulation_service.py --client \
            --env_name "$task_name" \
            --video_dir "${EVAL_DIR}/videos/$task_name" \
            --max_episode_steps 720 \
            --n_envs "$N_ENVS" \
            --n_episodes "$N_EPISODES" \
            --n_action_steps "$N_ACTION_STEPS" \
            --port "$PORT"; then
            echo "Successfully executed: $task_name"
            ok=1
        else
            retry=$((retry+1))
            echo "Retry $retry/$max_retries for $task_name"
            sleep 5
        fi
    done
    if [ $ok -eq 0 ]; then echo "FAILED: $task_name"; failed=1; fi
    echo -e "\n==================================================\n"
done
exit "$failed"
} 2>&1 | tee "$CLIENT_LOG"

sleep "${POST_EVAL_WAIT_SECONDS:-60}"
"$PYTHON_BIN" scripts/compute_success_rate.py -i "$CLIENT_LOG" -o "$RESULTS_JSON"
echo "Done: $RESULTS_JSON"
