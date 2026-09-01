#!/usr/bin/env bash
set -euo pipefail

project_root="${S4_PROJECT_ROOT:-/workspace/smolVLA/s4_smolvla_isaaclab}"
conda_root="${S4_CONDA_ROOT:-/opt/conda}"
isaaclab_root="${ISAACLAB_ROOT:-/workspace/smolVLA/IsaacLab}"
isaaclab_prefix="$conda_root/envs/env_isaaclab"
rollout_checkpoint="${S4_ROLLOUT_CHECKPOINT:-$project_root/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model}"
vulkan_icd="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"

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
require_path "$isaaclab_root/isaaclab.sh"
require_path "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}/src/lerobot"
require_path "${S4_SCENE_ASSET_ROOT:-$project_root/local_assets/isaac/5.1}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
require_path "${SMOLVLA_MODEL_ROOT:-$project_root/models}/HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
require_path "${S4_DATA_ROOT:-$project_root/datasets}/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire/meta/s4_contract.json"
require_path "$rollout_checkpoint/config.json"
require_path "$vulkan_icd"

/usr/local/bin/s4-sanitize-release-paths \
    --project-root "$project_root" \
    --lerobot-root "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}" \
    --isaaclab-root "$isaaclab_root" \
    --data-root "${S4_DATA_ROOT:-$project_root/datasets}" \
    --model-root "${SMOLVLA_MODEL_ROOT:-$project_root/models}" \
    --output-root "${S4_OUTPUT_ROOT:-$project_root/outputs}" \
    --check

# Level 1: CUDA
"$conda_root/bin/conda" run -n smolvla python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not visible inside the container"
assert torch.cuda.device_count() >= 1, torch.cuda.device_count()
assert torch.version.cuda == "12.8", torch.version.cuda
device_name = torch.cuda.get_device_name(0)
print("[OK] CUDA PASS:", device_name, "cuda", torch.version.cuda)
PY

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

cd "$project_root"
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
isaaclab_pyver="$("$isaaclab_prefix/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
isaaclab_cmeel="$isaaclab_prefix/lib/python${isaaclab_pyver}/site-packages/cmeel.prefix"
export LD_LIBRARY_PATH="$isaaclab_prefix/lib:$isaaclab_cmeel/lib:${LD_LIBRARY_PATH:-}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/workspace/runtime/xdg-runtime}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
mkdir -p "$XDG_RUNTIME_DIR" "${S4_CACHE_ROOT:-/workspace/runtime/cache}"
kit_log="$(mktemp)"
trap 'rm -f "$kit_log"' EXIT

set +e
CONDA_PREFIX="$isaaclab_prefix" PATH="$isaaclab_prefix/bin:$PATH" \
"$isaaclab_root/isaaclab.sh" -p -c "$(cat <<'PY'
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg

sim_utils.create_new_stage()
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)
sim_utils.update_stage()

camera_cfg = TiledCameraCfg(
    height=128,
    width=128,
    prim_path="/World/Camera",
    update_period=0,
    data_types=["rgb"],
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 4.0),
        rot=(0.0, 0.0, 1.0, 0.0),
        convention="ros",
    ),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    ),
)
camera = TiledCamera(camera_cfg)
sim.reset()

for _ in range(5):
    sim.step()

for _ in range(5):
    sim.step()
    camera.update(0.01)

rgb = camera.data.output["rgb"]
assert rgb is not None, "camera RGB output is None"
assert rgb.numel() > 0, "camera RGB output is empty"
shape = tuple(int(x) for x in rgb.shape)
assert len(shape) == 4, shape
assert shape[0] >= 1 and shape[1] > 0 and shape[2] > 0, shape
assert shape[3] in (3, 4), shape
print("[OK] Isaac Sim headless renderer")
print(f"[OK] Isaac Sim camera RGB frame {shape}")
simulation_app.close()
PY
)" >"$kit_log" 2>&1
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

echo "[OK] complete runtime verification passed"
