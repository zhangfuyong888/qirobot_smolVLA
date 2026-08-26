#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPISODES=100
EPISODE_TIMEOUT_S=300
RESET_SETTLE_S=2.0
RECORD_EVERY_N=6
RANDOM_SEED=42
HEADLESS=true
OVERWRITE=false
RESUME=false
OUTPUT_DIR=""
HDF5_FILE=""
LEROBOT_OUTPUT_ROOT=""
REPO_ID=""
MAX_FAILED_ATTEMPTS=""
CONTINUE_ON_FAILURE=false
ALLOW_SKIPPED_GRID_CELLS=false

usage() {
    cat <<'EOF'
Usage:
  bash run.sh collect-convert [options]

Collect and convert only; this command never starts training.

Options:
  --episodes N              Successful episodes to collect. Default: 100
  --episode-timeout-s S     Retry an attempt after S wall seconds. Default: 300
  --reset-settle-s S        Physics settling time after reset. Default: 2.0
  --record-every-n N        Save one frame every N simulation steps. Default: 6 (20 Hz)
  --random-seed N           Collection randomization seed. Default: 42
  --headless, --no-render   Hide GUI but keep all camera sensors rendering. Default
  --render                  Show the Isaac Sim window while collecting
  --output-dir PATH         Directory for this run's HDF5 file
  --hdf5-file PATH          Exact HDF5 output path; overrides --output-dir
  --output-root PATH        Parent directory for the converted LeRobotDataset
  --repo-id ID              Converted dataset name/repo id; default comes from task config
  --overwrite               Replace an existing converted dataset
  --resume                  Append to --hdf5-file until --episodes total successes
  --max-failed-attempts N   Abort if failures exceed N (default: 1000 safety cap)
  --continue-on-failure      Keep retrying rejected/timeout attempts without a failure-count cap;
                             failed attempts are logged and never committed to HDF5
  --allow-skipped-grid-cells
                            Permit conversion when a grid cell was skipped
  -h, --help                Show this help

Example:
  bash run.sh collect-convert --episodes 200 --headless --overwrite
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes|--num-episodes) EPISODES="$2"; shift 2 ;;
        --episode-timeout-s) EPISODE_TIMEOUT_S="$2"; shift 2 ;;
        --reset-settle-s) RESET_SETTLE_S="$2"; shift 2 ;;
        --record-every-n) RECORD_EVERY_N="$2"; shift 2 ;;
        --random-seed) RANDOM_SEED="$2"; shift 2 ;;
        --headless|--no-render) HEADLESS=true; shift ;;
        --render) HEADLESS=false; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --hdf5-file) HDF5_FILE="$2"; shift 2 ;;
        --output-root) LEROBOT_OUTPUT_ROOT="$2"; shift 2 ;;
        --repo-id) REPO_ID="$2"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --resume) RESUME=true; shift ;;
        --max-failed-attempts) MAX_FAILED_ATTEMPTS="$2"; shift 2 ;;
        --continue-on-failure) CONTINUE_ON_FAILURE=true; shift ;;
        --allow-skipped-grid-cells) ALLOW_SKIPPED_GRID_CELLS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown collect-convert option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$RESUME" == true && -z "$HDF5_FILE" ]]; then
    echo "--resume requires --hdf5-file PATH so the existing collection is unambiguous" >&2
    exit 2
fi

if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episodes must be a positive integer" >&2
    exit 2
fi
if ! [[ "$RECORD_EVERY_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "--record-every-n must be a positive integer" >&2
    exit 2
fi
if [[ "$RECORD_EVERY_N" -ne 6 ]]; then
    echo "--record-every-n must be 6 for the configured 120 Hz simulation -> 20 Hz dataset contract" >&2
    exit 2
fi
if [[ "$CONTINUE_ON_FAILURE" == true && -n "$MAX_FAILED_ATTEMPTS" ]]; then
    echo "Use either --continue-on-failure or --max-failed-attempts, not both" >&2
    exit 2
fi
if [[ "$CONTINUE_ON_FAILURE" == false && -z "$MAX_FAILED_ATTEMPTS" ]]; then
    MAX_FAILED_ATTEMPTS=1000
fi
if [[ "$CONTINUE_ON_FAILURE" == false ]] && ! [[ "$MAX_FAILED_ATTEMPTS" =~ ^[0-9]+$ ]]; then
    echo "--max-failed-attempts must be a non-negative integer" >&2
    exit 2
fi

CONFIG_PYTHON="${S4_ISAACLAB_PREFIX:-$HOME/miniconda3/envs/env_isaaclab}/bin/python"
if [[ ! -x "$CONFIG_PYTHON" ]]; then
    echo "IsaacLab environment Python not found: $CONFIG_PYTHON" >&2
    exit 2
fi

mapfile -t CONFIG_VALUES < <("$CONFIG_PYTHON" - <<'PY'
from s4_pipeline.config import load_project_config

cfg = load_project_config()
print(cfg.dataset.staging_root)
print(cfg.dataset.lerobot_root)
print(cfg.dataset.repo_id)
print(cfg.dataset.task_id)
PY
)
STAGING_ROOT="${CONFIG_VALUES[0]}"
if [[ -z "$LEROBOT_OUTPUT_ROOT" ]]; then
    LEROBOT_OUTPUT_ROOT="${CONFIG_VALUES[1]}"
fi
if [[ -z "$REPO_ID" ]]; then
    # Local conversion/training uses the final dataset directory name. A
    # namespace such as "local/foo" is metadata, not an extra path level.
    REPO_ID="${CONFIG_VALUES[2]##*/}"
fi
TASK_ID="${CONFIG_VALUES[3]}"

if [[ -z "$OUTPUT_DIR" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="$STAGING_ROOT/collect_convert_$RUN_ID"
fi
if [[ -z "$HDF5_FILE" ]]; then
    HDF5_FILE="$OUTPUT_DIR/${TASK_ID}_scripted.hdf5"
else
    OUTPUT_DIR="$(dirname "$HDF5_FILE")"
fi
FAILURE_LOG="${HDF5_FILE%.hdf5}_failures.jsonl"
FAILURE_SUMMARY="${HDF5_FILE%.hdf5}_failure_summary.json"
mkdir -p "$OUTPUT_DIR"

RECORD_ARGS=(
    --output "$HDF5_FILE"
    --episodes "$EPISODES"
    --episode-timeout-s "$EPISODE_TIMEOUT_S"
    --reset-settle-s "$RESET_SETTLE_S"
    --record-every-n "$RECORD_EVERY_N"
    --random-seed "$RANDOM_SEED"
    --failure-log "$FAILURE_LOG"
    --failure-summary "$FAILURE_SUMMARY"
)
if [[ "$CONTINUE_ON_FAILURE" == false ]]; then
    RECORD_ARGS+=(--max-failed-attempts "$MAX_FAILED_ATTEMPTS")
fi
if [[ "$HEADLESS" == true ]]; then
    RECORD_ARGS+=(--headless)
fi
if [[ "$RESUME" == true ]]; then
    RECORD_ARGS+=(--resume)
fi

echo "========================================"
echo "  S4 collect -> validate -> convert"
echo "  Training:       disabled"
echo "  Episodes:       $EPISODES successful episodes"
echo "  HDF5:           $HDF5_FILE"
echo "  Record stride:  $RECORD_EVERY_N simulation steps"
echo "  Random seed:    $RANDOM_SEED"
echo "  Attempt timeout:${EPISODE_TIMEOUT_S}s"
echo "  Reset settle:   ${RESET_SETTLE_S}s"
echo "  Failure report: $FAILURE_SUMMARY"
if [[ "$CONTINUE_ON_FAILURE" == true ]]; then
    echo "  Failure budget: disabled (rejected attempts are logged and discarded)"
else
    echo "  Failure budget: $MAX_FAILED_ATTEMPTS cumulative attempts"
fi
echo "  Headless:       $HEADLESS"
echo "  Dataset repo:   $REPO_ID"
echo "  Dataset parent: $LEROBOT_OUTPUT_ROOT"
echo "========================================"

echo "[1/4] Collecting successful HDF5 demonstrations"
bash run.sh record "${RECORD_ARGS[@]}"

echo "[2/4] Validating HDF5 contract"
CHECK_ARGS=(
    "$HDF5_FILE" --hdf5
    --expected-episodes "$EPISODES"
    --failure-summary "$FAILURE_SUMMARY"
)
if [[ "$CONTINUE_ON_FAILURE" == false ]]; then
    CHECK_ARGS+=(--max-failed-attempts "$MAX_FAILED_ATTEMPTS")
fi
if [[ "$ALLOW_SKIPPED_GRID_CELLS" == true ]]; then
    CHECK_ARGS+=(--allow-skipped-grid-cells)
fi
bash run.sh dataset-check "${CHECK_ARGS[@]}"

CONVERT_ARGS=(
    --root-path "$HDF5_FILE"
    --output-root "$LEROBOT_OUTPUT_ROOT"
    --repo-id "$REPO_ID"
)
if [[ "$OVERWRITE" == true ]]; then
    CONVERT_ARGS+=(--overwrite)
fi

echo "[3/4] Converting HDF5 to LeRobotDataset"
bash run.sh convert "${CONVERT_ARGS[@]}"

DATASET_DIR="$LEROBOT_OUTPUT_ROOT/${REPO_ID##*/}"
echo "[4/4] Validating converted LeRobotDataset"
bash run.sh dataset-check "$DATASET_DIR" --expected-episodes "$EPISODES"

echo "[DONE] Collection and conversion completed; training was not started."
echo "[DONE] HDF5: $HDF5_FILE"
echo "[DONE] LeRobotDataset: $DATASET_DIR"
