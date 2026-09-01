#!/usr/bin/env bash
set -euo pipefail

export S4_CONDA_ROOT="${S4_CONDA_ROOT:-/opt/conda}"
export S4_ISAACLAB_ENV="${S4_ISAACLAB_ENV:-env_isaaclab}"
export S4_SMOLVLA_ENV="${S4_SMOLVLA_ENV:-smolvla}"
export S4_PROJECT_ROOT="${S4_PROJECT_ROOT:-/workspace/smolVLA/s4_smolvla_isaaclab}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/smolVLA/IsaacLab}"
export LEROBOT_ROOT="${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}"
export S4_SCENE_ASSET_ROOT="${S4_SCENE_ASSET_ROOT:-${S4_PROJECT_ROOT}/local_assets/isaac/5.1}"
export ISAAC_ASSET_ROOT="${ISAAC_ASSET_ROOT:-${S4_SCENE_ASSET_ROOT}}"
export SMOLVLA_MODEL_ROOT="${SMOLVLA_MODEL_ROOT:-${S4_PROJECT_ROOT}/models}"
export S4_DATA_ROOT="${S4_DATA_ROOT:-${S4_PROJECT_ROOT}/datasets}"
export S4_OUTPUT_ROOT="${S4_OUTPUT_ROOT:-${S4_PROJECT_ROOT}/outputs}"
export S4_CACHE_ROOT="${S4_CACHE_ROOT:-/workspace/runtime/cache}"
export S4_ROLLOUT_CHECKPOINT="${S4_ROLLOUT_CHECKPOINT:-${S4_OUTPUT_ROOT}/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export S4_KIT_OFFLINE="${S4_KIT_OFFLINE:-1}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-graphics,compute,utility}"
export PATH="${S4_CONDA_ROOT}/bin:${PATH}"
export PYTHONUNBUFFERED=1

if [[ ! -r "$VK_ICD_FILENAMES" ]]; then
    echo "[FAIL] Vulkan ICD not readable: $VK_ICD_FILENAMES" >&2
    exit 1
fi

gpu_selection="${NVIDIA_VISIBLE_DEVICES:-${S4_GPUS:-all}}"
if [[ "$gpu_selection" != "all" ]]; then
    echo "[S4] host GPU selection: NVIDIA_VISIBLE_DEVICES=$gpu_selection (container cuda indices start at 0)"
fi

mkdir -p "$S4_DATA_ROOT" "$S4_OUTPUT_ROOT" "$S4_CACHE_ROOT"
mkdir -p /workspace/runtime/isaac-cache /workspace/runtime/compute-cache
mkdir -p /workspace/runtime/isaac-logs /workspace/runtime/isaac-config
mkdir -p /workspace/runtime/xdg-runtime
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/workspace/runtime/xdg-runtime}"

# Named volumes survive image rebuilds and may contain checkpoints created on
# the workstation or by an older container layout. Keep only the JSON metadata
# portable; model weights, datasets and training state are never modified.
/usr/local/bin/s4-sanitize-release-paths \
    --project-root "$S4_PROJECT_ROOT" \
    --lerobot-root "$LEROBOT_ROOT" \
    --isaaclab-root "$ISAACLAB_ROOT" \
    --data-root "$S4_DATA_ROOT" \
    --model-root "$SMOLVLA_MODEL_ROOT" \
    --output-root "$S4_OUTPUT_ROOT"

cd "$S4_PROJECT_ROOT"
exec "$@"
