#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-}"
if [[ "${1:-}" == "--"* || -z "${1:-}" ]]; then
    CONFIG="$(python3 -c 'from s4_pipeline.paths import SMOLVLA_CONFIG_PATH; print(SMOLVLA_CONFIG_PATH)')"
else
    shift
fi

OVERWRITE_OUTPUT=false
RESUME_OVERRIDE=""
STEPS_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
SAVE_FREQ_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash run.sh train [CONFIG] [--resume|--no-resume] [--overwrite-output] [--steps N] [--batch-size N] [--save-freq N]"
            exit 0
            ;;
        --resume)
            RESUME_OVERRIDE="true"
            shift
            ;;
        --no-resume)
            RESUME_OVERRIDE="false"
            shift
            ;;
        --overwrite-output)
            OVERWRITE_OUTPUT=true
            shift
            ;;
        --steps)
            STEPS_OVERRIDE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ_OVERRIDE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown train option: $1" >&2
            echo "Usage: bash run.sh train [config] [--resume|--no-resume] [--overwrite-output] [--steps N] [--batch-size N] [--save-freq N]" >&2
            exit 2
            ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
fi

cfg() { python3 scripts/config_value.py training "$1" --config "$CONFIG"; }
DATASET=$(cfg dataset)
DATASET_ROOT=$(cfg dataset_root)
OUTPUT_DIR=$(cfg output_dir)
STEPS=$(cfg steps)
BATCH_SIZE=$(cfg batch_size)
NUM_WORKERS=$(cfg num_workers)
DEVICE=$(cfg device)
SEED=$(cfg seed)
CHUNK_SIZE=$(cfg chunk_size)
MAX_STATE_DIM=$(cfg max_state_dim)
MAX_ACTION_DIM=$(cfg max_action_dim)
SAVE_FREQ=$(cfg save_freq)
N_OBS_STEPS=$(cfg n_obs_steps)
RESIZE_IMAGES=$(cfg resize_imgs_with_padding)
FREEZE_VISION=$(cfg freeze_vision_encoder)
TRAIN_EXPERT=$(cfg train_expert_only)
TRAIN_STATE_PROJ=$(cfg train_state_proj)
LOAD_VLM=$(cfg load_vlm_weights)
LANGUAGE_CONTRACT_VERSION=$(cfg language_contract_version)
if [ -n "$STEPS_OVERRIDE" ]; then
    STEPS="$STEPS_OVERRIDE"
fi
if [ -n "$BATCH_SIZE_OVERRIDE" ]; then
    BATCH_SIZE="$BATCH_SIZE_OVERRIDE"
fi
if [ -n "$SAVE_FREQ_OVERRIDE" ]; then
    SAVE_FREQ="$SAVE_FREQ_OVERRIDE"
fi
for value_name in STEPS BATCH_SIZE SAVE_FREQ; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer: $value" >&2
        exit 2
    fi
done
RESUME=$(cfg resume)
if [ -n "$RESUME_OVERRIDE" ]; then
    RESUME="$RESUME_OVERRIDE"
fi
VLM_PATH=$(cfg vlm_model_name)
OPT_LR=$(cfg optimizer_lr)
OPT_WD=$(cfg optimizer_weight_decay)
OPT_CLIP=$(cfg optimizer_grad_clip_norm)

echo "========================================"
echo "  Local S4 SmolVLA training"
echo "  Config:  $CONFIG"
echo "  Dataset: $DATASET_ROOT/$DATASET"
echo "  Output:  $OUTPUT_DIR"
echo "  Resume:  $RESUME"
echo "  Steps:   $STEPS"
echo "  Batch:   $BATCH_SIZE"
echo "  Save:    every $SAVE_FREQ steps"
echo "  Seed:    $SEED"
echo "========================================"

if [ ! -d "$DATASET_ROOT/$DATASET" ]; then
    echo "Dataset does not exist yet: $DATASET_ROOT/$DATASET" >&2
    echo "Run: bash run.sh convert --root-path <hdf5 file or dir>" >&2
    exit 2
fi

DATASET_CONTRACT="$DATASET_ROOT/$DATASET/meta/s4_contract.json"
if [ ! -f "$DATASET_CONTRACT" ]; then
    echo "Dataset language contract is missing: $DATASET_CONTRACT" >&2
    echo "Run dataset conversion and dataset-check before training." >&2
    exit 2
fi
ACTUAL_LANGUAGE_CONTRACT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("language_contract_version", ""))' "$DATASET_CONTRACT")
if [ "$ACTUAL_LANGUAGE_CONTRACT" != "$LANGUAGE_CONTRACT_VERSION" ]; then
    echo "Dataset language contract=$ACTUAL_LANGUAGE_CONTRACT, expected=$LANGUAGE_CONTRACT_VERSION" >&2
    exit 2
fi

if [ "${S4_TEST_SKIP_TRAIN_DATASET_CHECK:-0}" != "1" ]; then
    echo "[PREFLIGHT] Validating the complete LeRobotDataset before training"
    python3 scripts/dataset_check.py "$DATASET_ROOT/$DATASET"
fi

if [ "$OVERWRITE_OUTPUT" = true ]; then
    if [ "$RESUME" = true ]; then
        echo "--overwrite-output cannot be used together with --resume" >&2
        exit 2
    fi
    if [ -e "$OUTPUT_DIR" ] || [ -L "$OUTPUT_DIR" ]; then
        echo "[INFO] Removing existing training output: $OUTPUT_DIR"
        python3 scripts/safe_remove_train_output.py \
            --output "$OUTPUT_DIR" \
            --allowed-root "${S4_OUTPUT_ROOT:-$PWD/outputs}/train" \
            --project-root "$PWD" \
            --data-root "${S4_DATA_ROOT:-$PWD/datasets}" >/dev/null
    fi
fi

if [ "$RESUME" != true ] && { [ -e "$OUTPUT_DIR" ] || [ -L "$OUTPUT_DIR" ]; }; then
    echo "Fresh training requires a non-existent output path: $OUTPUT_DIR" >&2
    echo "LeRobot creates the directory itself. Use --resume for the same run or" >&2
    echo "--overwrite-output for an intentional fresh run." >&2
    exit 2
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HUGGINGFACE_HUB_OFFLINE="${HUGGINGFACE_HUB_OFFLINE:-1}"
PROJECT_CACHE="${S4_CACHE_ROOT:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$PROJECT_CACHE/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [ "$RESUME" = true ]; then
    OUTPUT_CONTRACT="$OUTPUT_DIR/s4_dataset_contract.json"
    if [ ! -f "$OUTPUT_CONTRACT" ] || ! cmp -s "$DATASET_CONTRACT" "$OUTPUT_CONTRACT"; then
        echo "Cannot resume: training output does not match the active dataset language contract." >&2
        exit 2
    fi
    RESUME_CHECKPOINT="$OUTPUT_DIR/checkpoints/last"
    RESUME_CONFIG="$RESUME_CHECKPOINT/pretrained_model/train_config.json"
    RESUME_STATE="$RESUME_CHECKPOINT/training_state/training_step.json"
    if [ ! -f "$RESUME_CONFIG" ] || [ ! -f "$RESUME_STATE" ]; then
        echo "Cannot resume: no complete checkpoint found under $RESUME_CHECKPOINT" >&2
        echo "A resumable checkpoint must contain pretrained_model/train_config.json and training_state/." >&2
        echo "Available checkpoints:" >&2
        find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' 2>/dev/null | sort >&2 || true
        exit 2
    fi
    RESUME_STEP=$(python3 -c "import json; print(json.load(open('$RESUME_STATE'))['step'])")
    echo "[RESUME] checkpoint: $RESUME_CHECKPOINT"
    echo "[RESUME] saved step: $RESUME_STEP; target total steps: $STEPS"
    if [ "$STEPS" -le "$RESUME_STEP" ]; then
        echo "Resume target --steps ($STEPS) must be greater than saved step ($RESUME_STEP)." >&2
        exit 2
    fi
    exec python3 scripts/supervise_training.py --output-dir "$OUTPUT_DIR" -- lerobot-train \
        --config_path="$RESUME_CONFIG" \
        --resume=true \
        --output_dir="$OUTPUT_DIR" \
        --steps="$STEPS" \
        --batch_size="$BATCH_SIZE" \
        --save_freq="$SAVE_FREQ" \
        --num_workers="$NUM_WORKERS" \
        --persistent_workers=false \
        --dataset.video_backend=pyav
fi

OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
OUTPUT_NAME="$(basename "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT"
CONTRACT_STAGE="$(mktemp "$OUTPUT_PARENT/.${OUTPUT_NAME}.s4_dataset_contract.XXXXXX")"
cp "$DATASET_CONTRACT" "$CONTRACT_STAGE"

python3 scripts/supervise_training.py \
    --output-dir "$OUTPUT_DIR" \
    --contract-stage "$CONTRACT_STAGE" \
    -- lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id="$DATASET" \
    --dataset.root="$DATASET_ROOT/$DATASET" \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --log_freq=100 \
    --env_eval_freq=0 \
    --save_freq="$SAVE_FREQ" \
    --num_workers="$NUM_WORKERS" \
    --persistent_workers=false \
    --seed="$SEED" \
    --resume="$RESUME" \
    --policy.device="$DEVICE" \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps="$CHUNK_SIZE" \
    --policy.n_obs_steps="$N_OBS_STEPS" \
    --policy.max_state_dim="$MAX_STATE_DIM" \
    --policy.max_action_dim="$MAX_ACTION_DIM" \
    --policy.resize_imgs_with_padding="$RESIZE_IMAGES" \
    --policy.freeze_vision_encoder="$FREEZE_VISION" \
    --policy.train_expert_only="$TRAIN_EXPERT" \
    --policy.train_state_proj="$TRAIN_STATE_PROJ" \
    --policy.load_vlm_weights="$LOAD_VLM" \
    --policy.vlm_model_name="$VLM_PATH" \
    --policy.optimizer_lr="$OPT_LR" \
    --policy.optimizer_weight_decay="$OPT_WD" \
    --policy.optimizer_grad_clip_norm="$OPT_CLIP" \
    --policy.push_to_hub=false

if [ ! -d "$OUTPUT_DIR" ] || [ -L "$OUTPUT_DIR" ]; then
    echo "Training exited normally but did not create a real output directory: $OUTPUT_DIR" >&2
    exit 2
fi
