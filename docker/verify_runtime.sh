#!/usr/bin/env bash
set -euo pipefail

project_root="${S4_PROJECT_ROOT:-/workspace/smolVLA/s4_smolvla_isaaclab}"
conda_root="${S4_CONDA_ROOT:-/opt/conda}"
isaaclab_root="${ISAACLAB_ROOT:-/workspace/smolVLA/IsaacLab}"
isaaclab_prefix="$conda_root/envs/env_isaaclab"
rollout_checkpoint="${S4_ROLLOUT_CHECKPOINT:-$project_root/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model}"

require_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "[FAIL] missing required path: $path" >&2
        exit 1
    fi
    echo "[OK] $path"
}

require_path "$project_root/run.sh"
require_path "$isaaclab_root/isaaclab.sh"
require_path "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}/src/lerobot"
require_path "${S4_SCENE_ASSET_ROOT:-$project_root/local_assets/isaac/5.1}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
require_path "${SMOLVLA_MODEL_ROOT:-$project_root/models}/HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
require_path "${S4_DATA_ROOT:-$project_root/datasets}/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire/meta/s4_contract.json"
require_path "$rollout_checkpoint/config.json"

# Refuse a release whose saved checkpoint/evaluation metadata still points to
# a particular workstation. This catches both future checkpoints and stale
# named-volume contents before policy startup.
/usr/local/bin/s4-sanitize-release-paths \
    --project-root "$project_root" \
    --lerobot-root "${LEROBOT_ROOT:-/workspace/smolVLA/lerobot}" \
    --isaaclab-root "$isaaclab_root" \
    --data-root "${S4_DATA_ROOT:-$project_root/datasets}" \
    --model-root "${SMOLVLA_MODEL_ROOT:-$project_root/models}" \
    --output-root "${S4_OUTPUT_ROOT:-$project_root/outputs}" \
    --check

"$conda_root/bin/conda" run -n env_isaaclab python - <<'PY'
import importlib.metadata as metadata
import isaaclab
assert metadata.version("isaacsim") == "5.1.0.0"
print("[OK] env_isaaclab / Isaac Sim", metadata.version("isaacsim"))
PY

"$conda_root/bin/conda" run -n smolvla python - <<'PY'
import torch
import lerobot
assert torch.cuda.is_available(), "CUDA is not visible inside the container"
assert torch.version.cuda == "12.8", torch.version.cuda
print("[OK] smolvla / torch", torch.__version__, "cuda", torch.version.cuda)
PY

cd "$project_root"
bash run.sh doctor --strict

# Check the saved processor pipeline as well as the policy weights.  Historical
# checkpoints can contain workstation-absolute tokenizer paths even when the
# model config itself has already been remapped for the release image.
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

# Import-only checks do not load the native Kit/MaterialX stack.  Start one
# headless-rendering frame so missing X11 libraries and C++ ABI mismatches are
# caught before a real collection or rollout job is launched.
export LD_LIBRARY_PATH="$isaaclab_prefix/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
CONDA_PREFIX="$isaaclab_prefix" PATH="$isaaclab_prefix/bin:$PATH" \
"$isaaclab_root/isaaclab.sh" -p -c \
    'from isaaclab.app import AppLauncher; launcher = AppLauncher(headless=True, enable_cameras=True); launcher.app.update(); launcher.app.close(); print("[OK] Isaac Sim headless-rendering smoke test")'

echo "[OK] complete runtime verification passed"
