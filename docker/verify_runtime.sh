#!/usr/bin/env bash
set -euo pipefail

profile="full"
usage() {
    cat <<'EOF'
Usage: s4-verify-runtime [--profile rollout|train|full]

  rollout  CUDA, Vulkan, Isaac renderer, RGB camera and checkpoint pipeline
  train    CUDA, exact training dependencies and complete dataset validation
  full     Both profiles (default)
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || { echo "Missing value for --profile" >&2; exit 2; }
            profile="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown verification option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
if [[ "$profile" != "rollout" && "$profile" != "train" && "$profile" != "full" ]]; then
    echo "Invalid profile: $profile (expected rollout, train or full)" >&2
    exit 2
fi
run_rollout=false
run_train=false
[[ "$profile" == "rollout" || "$profile" == "full" ]] && run_rollout=true
[[ "$profile" == "train" || "$profile" == "full" ]] && run_train=true

project_root="${S4_PROJECT_ROOT:-/workspace/smolVLA/s4_smolvla_isaaclab}"
conda_root="${S4_CONDA_ROOT:-/opt/conda}"
isaaclab_root="${ISAACLAB_ROOT:-/workspace/smolVLA/IsaacLab}"
isaaclab_prefix="$conda_root/envs/env_isaaclab"
rollout_checkpoint="${S4_ROLLOUT_CHECKPOINT:-$project_root/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model}"
vulkan_icd="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
isaac_env_helper="${S4_ISAAC_ENV_HELPER:-/usr/local/lib/s4/isaac_env.sh}"
isaac_camera_verify="$project_root/scripts/verify_isaac_camera.py"

KIT_FAILURE_PATTERNS=(
    'ERROR_INCOMPATIBLE_DRIVER'
    'Failed to create any GPU devices'
    'GPU Foundation is not initialized'
    'no valid foundation interface'
    'vkCreateInstance failed'
)

require_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "[FAIL] missing required path: $path" >&2
        exit 1
    fi
    echo "[OK] $path"
}

check_kit_log() {
    local log_file="$1"
    local pattern
    for pattern in "${KIT_FAILURE_PATTERNS[@]}"; do
        if grep -q "$pattern" "$log_file"; then
            echo "[FAIL] Isaac Kit log contains: $pattern" >&2
            cat "$log_file" >&2
            exit 1
        fi
    done
}

require_path "$project_root/run.sh"
require_path "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}/src/lerobot"
require_path "${SMOLVLA_MODEL_ROOT:-$project_root/models}/HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
require_path "${S4_DATA_ROOT:-$project_root/datasets}/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire/meta/s4_contract.json"
if [[ "$run_rollout" == true ]]; then
    require_path "$isaaclab_root/isaaclab.sh"
    require_path "${S4_SCENE_ASSET_ROOT:-$project_root/local_assets/isaac/5.1}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    require_path "$rollout_checkpoint/config.json"
    require_path "$vulkan_icd"
    require_path "$isaac_env_helper"
    require_path "$isaac_camera_verify"
fi

/usr/local/bin/s4-sanitize-release-paths \
    --project-root "$project_root" \
    --lerobot-root "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}" \
    --isaaclab-root "$isaaclab_root" \
    --data-root "${S4_DATA_ROOT:-$project_root/datasets}" \
    --model-root "${SMOLVLA_MODEL_ROOT:-$project_root/models}" \
    --output-root "${S4_OUTPUT_ROOT:-$project_root/outputs}" \
    --check

cd "$project_root"

# Level 1: CUDA
"$conda_root/bin/conda" run -n smolvla python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not visible inside the container"
assert torch.cuda.device_count() >= 1, torch.cuda.device_count()
assert torch.version.cuda == "12.8", torch.version.cuda
device_name = torch.cuda.get_device_name(0)
print("[OK] CUDA PASS:", device_name, "cuda", torch.version.cuda)
PY

if [[ "$run_train" == true ]]; then
    verify_train_gpus="${S4_VERIFY_TRAIN_GPUS:-${S4_DOCKER_SELECTED_GPU_COUNT:-1}}"
    if ! [[ "$verify_train_gpus" =~ ^[1-9][0-9]*$ ]]; then
        echo "[FAIL] S4_VERIFY_TRAIN_GPUS must be a positive integer: $verify_train_gpus" >&2
        exit 2
    fi
    verify_output_dir="$("$conda_root/bin/conda" run -n smolvla python scripts/config_value.py training output_dir)"
    "$conda_root/bin/conda" run -n smolvla python \
        scripts/training_runtime_preflight.py \
        --device cuda \
        --num-gpus "$verify_train_gpus" \
        --output-dir "$verify_output_dir"
    if (( verify_train_gpus > 1 )); then
        verify_ddp_port="${S4_VERIFY_DDP_PORT:-29600}"
        if ! [[ "$verify_ddp_port" =~ ^[1-9][0-9]{0,4}$ ]] || (( verify_ddp_port > 65535 )); then
            echo "[FAIL] S4_VERIFY_DDP_PORT must be in [1, 65535]: $verify_ddp_port" >&2
            exit 2
        fi
        verify_gpu_ids="$(seq -s, 0 $((verify_train_gpus - 1)))"
        "$conda_root/bin/conda" run -n smolvla accelerate launch \
            --multi_gpu \
            --num_processes "$verify_train_gpus" \
            --main_process_port "$verify_ddp_port" \
            --gpu_ids "$verify_gpu_ids" \
            scripts/verify_accelerate_launch.py \
            --expected-processes "$verify_train_gpus"
    fi
    "$conda_root/bin/conda" run -n smolvla python \
        scripts/dataset_check.py \
        "${S4_DATA_ROOT:-$project_root/datasets}/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire"
    echo "[OK] LeRobot training dependencies, launcher and dataset"
fi

if [[ "$run_rollout" != true ]]; then
    echo "[OK] training runtime verification passed"
    exit 0
fi

"$conda_root/bin/conda" run -n env_isaaclab python - <<'PY'
import importlib.metadata as metadata
import isaaclab

assert metadata.version("isaacsim") == "5.1.0.0"
print("[OK] env_isaaclab / Isaac Sim", metadata.version("isaacsim"))
PY

# Level 2: Vulkan
if ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1'; then
    echo "[FAIL] libvulkan.so.1 is missing from the release image" >&2
    exit 1
fi

vulkan_out=""
vulkan_rc=0
vulkan_out="$(vulkaninfo --summary 2>&1)" || vulkan_rc=$?
printf '%s\n' "$vulkan_out"
if [[ "$vulkan_rc" -ne 0 ]]; then
    echo "[FAIL] Vulkan initialization failed" >&2
    exit 1
fi
grep -q 'vendorID.*0x10de' <<<"$vulkan_out" || {
    echo "[FAIL] Vulkan vendorID is not NVIDIA (0x10de)" >&2
    exit 1
}
grep -q 'driverName.*NVIDIA' <<<"$vulkan_out" || {
    echo "[FAIL] Vulkan driverName is not NVIDIA" >&2
    exit 1
}
if grep -qi 'llvmpipe' <<<"$vulkan_out"; then
    echo "[FAIL] Vulkan using llvmpipe" >&2
    exit 1
fi
echo "[OK] NVIDIA Vulkan renderer"

bash run.sh doctor --strict

export S4_ROLLOUT_CHECKPOINT="$rollout_checkpoint"
export SMOLVLA_MODEL_ROOT="${SMOLVLA_MODEL_ROOT:-$project_root/models}"
"$conda_root/bin/conda" run -n smolvla python - <<'PY'
import os
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors

checkpoint = Path(os.environ["S4_ROLLOUT_CHECKPOINT"])
vlm_path = Path(os.environ["SMOLVLA_MODEL_ROOT"]) / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
config.vlm_model_name = str(vlm_path)
make_pre_post_processors(
    config,
    pretrained_path=str(checkpoint),
    preprocessor_overrides={
        "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
        "device_processor": {"device": "cpu"},
    },
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)
print("[OK] checkpoint tokenizer and processor pipeline")
PY

# Level 3 + 4: Isaac headless renderer and camera RGB frame
# Use exactly the same native-library setup as run.sh rollout/record/teleop.
# shellcheck disable=SC1090
source "$isaac_env_helper"
s4_setup_isaac_env
kit_log="$(mktemp)"
trap 'rm -f "$kit_log"' EXIT

set +e
PYTHONFAULTHANDLER=1 "$isaaclab_root/isaaclab.sh" -p \
    "$isaac_camera_verify" >"$kit_log" 2>&1
kit_rc=$?
set -e

cat "$kit_log"
check_kit_log "$kit_log"

if [[ "$kit_rc" -ne 0 ]]; then
    echo "[FAIL] Isaac Sim headless renderer exited with status $kit_rc" >&2
    exit 1
fi

grep -q '\[OK\] Isaac Sim headless renderer' "$kit_log" || {
    echo "[FAIL] Isaac Sim headless renderer did not report success" >&2
    exit 1
}
grep -q '\[OK\] Isaac Sim camera RGB frame' "$kit_log" || {
    echo "[FAIL] Isaac Sim camera RGB frame test did not report success" >&2
    exit 1
}

echo "[OK] $profile runtime verification passed"
