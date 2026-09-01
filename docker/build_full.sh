#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
image_tag="${1:-s4-smolvla:full-v3}"

if [[ ! -x "$workspace_root/IsaacLab/isaaclab.sh" ]]; then
    echo "Missing $workspace_root/IsaacLab. Run: bash docker/prepare_workspace.sh \"${ISAACLAB_SOURCE:-$HOME/IsaacLab}\"" >&2
    exit 2
fi

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required release file: $path" >&2
        exit 2
    fi
}

# A full release image must be independently usable for strict validation,
# training and rollout.  Fail before the expensive build if one of the bundled
# resource classes is absent.
require_file "$workspace_root/s4_smolvla_isaaclab/local_assets/isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
require_file "$workspace_root/s4_smolvla_isaaclab/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/model.safetensors"
require_file "$workspace_root/s4_smolvla_isaaclab/datasets/lerobot_data/s4_drawer_insert_close_v4_12phase_serial_acquire/meta/s4_contract.json"
require_file "$workspace_root/s4_smolvla_isaaclab/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/checkpoints/350000/pretrained_model/config.json"
require_file "$workspace_root/s4_smolvla_isaaclab/outputs/train/smolvla_drawer_insert_close_v4_12phase_serial_acquire/s4_dataset_contract.json"
require_file "$workspace_root/docker/vendor/kit-exts/isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311/config/extension.toml"
require_file "$workspace_root/docker/vendor/kit-exts/omni.kit.pip_archive-d38fa9ecd1fb6df4/config/extension.toml"
require_file "$workspace_root/docker/sanitize_release_paths.py"
require_file "$workspace_root/docker/vulkan/nvidia_icd_headless.json"

shopt -s nullglob
dataset_contracts=("$workspace_root"/s4_smolvla_isaaclab/datasets/lerobot_data/*/meta/s4_contract.json)
checkpoint_configs=("$workspace_root"/s4_smolvla_isaaclab/outputs/train/*/checkpoints/*/pretrained_model/config.json)
shopt -u nullglob
if [[ ${#dataset_contracts[@]} -eq 0 ]]; then
    echo "No LeRobotDataset contract found under s4_smolvla_isaaclab/datasets/lerobot_data." >&2
    exit 2
fi
if [[ ${#checkpoint_configs[@]} -eq 0 ]]; then
    echo "No complete pretrained_model config found under s4_smolvla_isaaclab/outputs/train." >&2
    exit 2
fi
if ! grep -Fxq '**/.env' "$workspace_root/.dockerignore"; then
    echo "Refusing to build: .dockerignore must exclude workstation .env files." >&2
    exit 2
fi

project_dirty=false
lerobot_dirty=false
if [[ -n "$(git -C "$workspace_root" status --porcelain --untracked-files=normal)" ]]; then
    project_dirty=true
fi
if [[ -n "$(git -C "$workspace_root/lerobot" status --porcelain --untracked-files=normal)" ]]; then
    lerobot_dirty=true
fi

{
    echo "project_commit=$(git -C "$workspace_root/s4_smolvla_isaaclab" rev-parse HEAD)"
    echo "lerobot_commit=$(git -C "$workspace_root/lerobot" rev-parse HEAD)"
    echo "isaaclab_commit=$(cat "$workspace_root/docker/ISAACLAB_COMMIT")"
    echo "project_dirty=$project_dirty"
    echo "lerobot_dirty=$lerobot_dirty"
    echo "image_tag=$image_tag"
    echo "image_release=full-v3"
    echo "cuda_userspace=12.8"
    echo "isaacsim=5.1.0"
    echo "vulkan_mode=headless-egl"
    echo "vulkan_icd_library=libEGL_nvidia.so.0"
    echo "required_arch=x86_64"
    echo "build_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$workspace_root/docker/release-manifest.env"

echo "[Docker release] building $image_tag from $workspace_root"
build_network_args=(--network=host)
for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    proxy_value="${!proxy_var-}"
    if [[ -n "$proxy_value" ]]; then
        build_network_args+=(--build-arg "$proxy_var=$proxy_value")
    fi
done

# The host-network mode is only used while building.  It lets a local proxy
# (for example 127.0.0.1:7890) serve Conda/PyPI downloads; neither the proxy
# variables nor host networking are retained by the finished image.
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    "${build_network_args[@]}" \
    --file "$workspace_root/docker/Dockerfile" \
    --tag "$image_tag" \
    "$workspace_root"

docker image inspect "$image_tag" --format 'built {{.RepoTags}} ({{.Size}} bytes)'
