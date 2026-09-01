#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
compose_file="$workspace_root/docker/compose.yaml"

image_tag="${S4_IMAGE:-s4-smolvla:full-v3}"
gpu_spec="${S4_GPUS:-all}"
use_compose=false
interactive=false

usage() {
    cat <<'EOF'
Usage: docker/run.sh [OPTIONS] [--] [COMMAND...]

Run the S4 SmolVLA release container with optional host GPU selection.

Options:
  -g, --gpus DEVICES   Host GPU indices to expose (default: all).
                       Examples: 0 | 0,1,2,3 | all
  -i, --image TAG      Docker image tag (default: s4-smolvla:full-v3)
  -c, --compose        Use "docker compose run" (named volumes + runtime/)
  -t, --tty            Allocate a TTY (default for bash, off for verify)
  -h, --help           Show this help

Environment:
  S4_GPUS              Same as --gpus (CLI wins if both are set)
  S4_IMAGE             Same as --image

Inside the container, visible GPUs are always renumbered from cuda:0.
Single-GPU runs use Docker `--gpus device=N` (not only NVIDIA_VISIBLE_DEVICES).

Examples:
  bash docker/run.sh --gpus 0 verify
  bash docker/run.sh --gpus 0 s4-verify-runtime
  bash docker/run.sh --gpus 0,1,2,3 --compose bash
  S4_GPUS=3 bash docker/run.sh bash
  bash docker/run.sh --gpus 4,5,6,7 --compose bash -lc \
    'bash run.sh train --num-gpus 4 --gpu-ids 0,1,2,3 --batch-size 4'
EOF
}

normalize_gpus() {
    local spec="${1// /}"
    if [[ "$spec" == "all" ]]; then
        printf '%s' "all"
        return 0
    fi
    if [[ ! "$spec" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid GPU spec: $1 (use all, 0, or 0,1,2,3)" >&2
        exit 2
    fi
    printf '%s' "$spec"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)
            [[ $# -ge 2 ]] || {
                echo "Missing value for $1" >&2
                exit 2
            }
            gpu_spec="$2"
            shift 2
            ;;
        -i|--image)
            [[ $# -ge 2 ]] || {
                echo "Missing value for $1" >&2
                exit 2
            }
            image_tag="$2"
            shift 2
            ;;
        -c|--compose)
            use_compose=true
            shift
            ;;
        -t|--tty)
            interactive=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        verify|s4-verify-runtime)
            if [[ $# -eq 1 ]]; then
                set -- s4-verify-runtime
                break
            fi
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    set -- bash
fi

if [[ "$1" == "verify" ]]; then
    shift
    set -- s4-verify-runtime "$@"
fi

gpu_spec="$(normalize_gpus "$gpu_spec")"

if [[ "$gpu_spec" == "all" ]]; then
    docker_gpus="all"
else
    docker_gpus="device=$gpu_spec"
fi

if [[ "$1" == "bash" && "$interactive" == false ]]; then
    interactive=true
fi

runtime_env=(
    "S4_GPUS=$gpu_spec"
    "NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility"
    "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json"
    "XDG_RUNTIME_DIR=/workspace/runtime/xdg-runtime"
    "ACCEPT_EULA=Y"
    "OMNI_KIT_ACCEPT_EULA=Y"
    "PRIVACY_CONSENT=Y"
    "S4_KIT_OFFLINE=1"
)

echo "[docker/run.sh] image=$image_tag gpus=$gpu_spec docker_gpus=$docker_gpus mode=$([[ "$use_compose" == true ]] && echo compose || echo run) command=$*"

if [[ "$use_compose" == true ]]; then
    export S4_GPUS="$gpu_spec"
    compose_cmd=(docker compose -f "$compose_file" run --rm)
    if [[ "$gpu_spec" != "all" ]]; then
        compose_cmd+=(--gpus "$docker_gpus")
    fi
    if [[ "$interactive" == true ]]; then
        compose_cmd+=(-it)
    fi
    compose_cmd+=(s4-smolvla "$@")
    exec "${compose_cmd[@]}"
fi

docker_cmd=(
    docker run --rm
    --gpus "$docker_gpus"
    --ipc=host
    --shm-size=64g
    --network=host
)

if [[ "$interactive" == true ]]; then
    docker_cmd+=(-it)
fi

for env_pair in "${runtime_env[@]}"; do
    docker_cmd+=(-e "$env_pair")
done

docker_cmd+=(
    -v "$workspace_root/docker/runtime:/workspace/runtime"
    -v docker_s4-datasets:/workspace/smolVLA/s4_smolvla_isaaclab/datasets
    -v docker_s4-outputs:/workspace/smolVLA/s4_smolvla_isaaclab/outputs
    "$image_tag"
    "$@"
)
exec "${docker_cmd[@]}"
