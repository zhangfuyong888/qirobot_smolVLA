#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

HDF5_FILE=""
EXPECTED_EPISODES=""
FAILURE_SUMMARY=""
MAX_FAILED_ATTEMPTS=""
ALLOW_SKIPPED_GRID_CELLS=false
OUTPUT_ROOT=""
REPO_ID=""
OVERWRITE_DATASET=false
TRAIN_CONFIG=""
TRAIN_STEPS=""
TRAIN_BATCH_SIZE=""
TRAIN_SAVE_FREQ=""
OVERWRITE_TRAINING_OUTPUT=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage:
  bash run.sh convert-train --hdf5-file PATH --expected-episodes N [options]

Safely run an existing recording through:
  HDF5 check -> LeRobot conversion -> LeRobotDataset check -> SmolVLA training

Data:
  --hdf5-file PATH               Existing HDF5 recording (required)
  --expected-episodes N          Required successful episode count
  --failure-summary PATH         Optional collection failure-summary JSON
  --max-failed-attempts N        Maximum failures allowed by the summary
  --allow-skipped-grid-cells     Permit skipped cells recorded in the summary
  --output-root PATH             LeRobotDataset parent (default: active config)
  --repo-id ID                   Dataset repo/name (default: active config)
  --overwrite-dataset            Explicitly replace the converted target

Training:
  --config PATH                  Training YAML (default: active task config)
  --steps N                      Override final total training steps
  --batch-size N                 Override batch size
  --save-freq N                  Override checkpoint frequency
  --overwrite-training-output    Explicitly replace an old training run

Other:
  --dry-run                      Print the guarded commands without executing them
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hdf5-file) HDF5_FILE="$2"; shift 2 ;;
        --expected-episodes) EXPECTED_EPISODES="$2"; shift 2 ;;
        --failure-summary) FAILURE_SUMMARY="$2"; shift 2 ;;
        --max-failed-attempts) MAX_FAILED_ATTEMPTS="$2"; shift 2 ;;
        --allow-skipped-grid-cells) ALLOW_SKIPPED_GRID_CELLS=true; shift ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --repo-id) REPO_ID="$2"; shift 2 ;;
        --overwrite-dataset) OVERWRITE_DATASET=true; shift ;;
        --config) TRAIN_CONFIG="$2"; shift 2 ;;
        --steps) TRAIN_STEPS="$2"; shift 2 ;;
        --batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --save-freq) TRAIN_SAVE_FREQ="$2"; shift 2 ;;
        --overwrite-training-output) OVERWRITE_TRAINING_OUTPUT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown convert-train option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$HDF5_FILE" ]] || { echo "--hdf5-file is required" >&2; exit 2; }
[[ -f "$HDF5_FILE" ]] || { echo "HDF5 file does not exist: $HDF5_FILE" >&2; exit 2; }
if ! [[ "$EXPECTED_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "--expected-episodes must be a positive integer" >&2
    exit 2
fi
if [[ -n "$MAX_FAILED_ATTEMPTS" ]] && ! [[ "$MAX_FAILED_ATTEMPTS" =~ ^[0-9]+$ ]]; then
    echo "--max-failed-attempts must be a non-negative integer" >&2
    exit 2
fi
if [[ -n "$MAX_FAILED_ATTEMPTS" && -z "$FAILURE_SUMMARY" ]]; then
    echo "--max-failed-attempts requires --failure-summary" >&2
    exit 2
fi
if [[ -n "$FAILURE_SUMMARY" && ! -f "$FAILURE_SUMMARY" ]]; then
    echo "Failure summary does not exist: $FAILURE_SUMMARY" >&2
    exit 2
fi

CONFIG_PYTHON="${S4_ISAACLAB_PREFIX:-$HOME/miniconda3/envs/env_isaaclab}/bin/python"
[[ -x "$CONFIG_PYTHON" ]] || { echo "IsaacLab Python not found: $CONFIG_PYTHON" >&2; exit 2; }
mapfile -t CONFIG_VALUES < <("$CONFIG_PYTHON" - <<'PY'
from s4_pipeline.config import load_project_config
from s4_pipeline.paths import SMOLVLA_CONFIG_PATH
c = load_project_config()
print(c.dataset.lerobot_root)
print(c.dataset.repo_id)
print(SMOLVLA_CONFIG_PATH)
PY
)
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="${CONFIG_VALUES[0]}"
[[ -n "$REPO_ID" ]] || REPO_ID="${CONFIG_VALUES[1]}"
[[ -n "$TRAIN_CONFIG" ]] || TRAIN_CONFIG="${CONFIG_VALUES[2]}"
[[ -f "$TRAIN_CONFIG" ]] || { echo "Training config does not exist: $TRAIN_CONFIG" >&2; exit 2; }

DATASET_NAME="${REPO_ID##*/}"
DATASET_DIR="$OUTPUT_ROOT/$DATASET_NAME"
TRAIN_DATASET="$(python3 scripts/config_value.py training dataset --config "$TRAIN_CONFIG")"
TRAIN_DATASET_ROOT="$(python3 scripts/config_value.py training dataset_root --config "$TRAIN_CONFIG")"
if [[ "$TRAIN_DATASET_ROOT/$TRAIN_DATASET" != "$DATASET_DIR" ]]; then
    echo "Training config does not consume the conversion target." >&2
    echo "  converted: $DATASET_DIR" >&2
    echo "  training:  $TRAIN_DATASET_ROOT/$TRAIN_DATASET" >&2
    exit 2
fi
if [[ -e "$DATASET_DIR" && "$OVERWRITE_DATASET" != true ]]; then
    echo "Converted dataset already exists: $DATASET_DIR" >&2
    echo "Use --overwrite-dataset only when replacing it intentionally." >&2
    exit 2
fi

run_cmd() {
    printf '[COMMAND]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != true ]]; then
        "$@"
    fi
}

CHECK_HDF5=(bash run.sh dataset-check "$HDF5_FILE" --hdf5 --expected-episodes "$EXPECTED_EPISODES")
if [[ -n "$FAILURE_SUMMARY" ]]; then
    CHECK_HDF5+=(--failure-summary "$FAILURE_SUMMARY")
fi
if [[ -n "$MAX_FAILED_ATTEMPTS" ]]; then
    CHECK_HDF5+=(--max-failed-attempts "$MAX_FAILED_ATTEMPTS")
fi
[[ "$ALLOW_SKIPPED_GRID_CELLS" == true ]] && CHECK_HDF5+=(--allow-skipped-grid-cells)

CONVERT=(bash run.sh convert --root-path "$HDF5_FILE" --output-root "$OUTPUT_ROOT"
    --repo-id "$REPO_ID" --control-mode bimanual)
[[ "$OVERWRITE_DATASET" == true ]] && CONVERT+=(--overwrite)

TRAIN=(bash run.sh train --config "$TRAIN_CONFIG")
[[ "$OVERWRITE_TRAINING_OUTPUT" == true ]] && TRAIN+=(--no-resume --overwrite-output)
[[ -n "$TRAIN_STEPS" ]] && TRAIN+=(--steps "$TRAIN_STEPS")
[[ -n "$TRAIN_BATCH_SIZE" ]] && TRAIN+=(--batch-size "$TRAIN_BATCH_SIZE")
[[ -n "$TRAIN_SAVE_FREQ" ]] && TRAIN+=(--save-freq "$TRAIN_SAVE_FREQ")

echo "[1/4] Checking source HDF5"
run_cmd "${CHECK_HDF5[@]}"
echo "[2/4] Converting 20-stage recording with the active language contract"
run_cmd "${CONVERT[@]}"
echo "[3/4] Checking converted LeRobotDataset"
run_cmd bash run.sh dataset-check "$DATASET_DIR" --expected-episodes "$EXPECTED_EPISODES"
echo "[4/4] Starting SmolVLA training"
run_cmd "${TRAIN[@]}"

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN COMPLETE] no check, conversion, or training command was executed"
else
    echo "[COMPLETE] conversion, validation, and training exited successfully"
fi
