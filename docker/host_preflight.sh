#!/usr/bin/env bash
# Read-only host contract check plus one disposable Docker GPU probe.
set -euo pipefail

gpu_test_image="${S4_HOST_PREFLIGHT_GPU_IMAGE:-s4-cuda-base:12.8.1}"
gpu_index="0"
min_disk_gb="180"
skip_container_test=false

usage() {
    cat <<'EOF'
Usage: bash docker/host_preflight.sh [options]

Options:
  --gpu N              Physical GPU used by the disposable Docker probe (default: 0)
  --gpu-image IMAGE    Existing local CUDA image (default: s4-cuda-base:12.8.1)
  --min-disk-gb N      Required free workspace disk in GiB (default: 180)
  --skip-container-test  Skip docker run --gpus probe
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) gpu_index="$2"; shift 2 ;;
        --gpu-image) gpu_test_image="$2"; shift 2 ;;
        --min-disk-gb) min_disk_gb="$2"; shift 2 ;;
        --skip-container-test) skip_container_test=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo "Invalid --gpu: $gpu_index" >&2; exit 2; }
[[ "$min_disk_gb" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --min-disk-gb: $min_disk_gb" >&2; exit 2; }

workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
arch="$(uname -m)"
[[ "$arch" == "x86_64" ]] || { echo "[FAIL] architecture=$arch; release requires x86_64" >&2; exit 1; }
echo "[OK] architecture $arch"

for command_name in docker nvidia-smi nvidia-ctk; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "[FAIL] required host command is missing: $command_name" >&2
        exit 1
    }
done
docker info >/dev/null
echo "[OK] Docker daemon"

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader)
(( ${#gpu_rows[@]} > 0 )) || { echo "[FAIL] nvidia-smi found no GPU" >&2; exit 1; }
printf '[OK] NVIDIA GPUs (%d)\n' "${#gpu_rows[@]}"
printf '  %s\n' "${gpu_rows[@]}"
(( gpu_index < ${#gpu_rows[@]} )) || {
    echo "[FAIL] requested GPU $gpu_index, available indices 0..$((${#gpu_rows[@]} - 1))" >&2
    exit 1
}

for device in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-modeset; do
    [[ -e "$device" ]] || { echo "[FAIL] missing NVIDIA device: $device" >&2; exit 1; }
done
compgen -G '/dev/dri/renderD*' >/dev/null || {
    echo "[FAIL] no /dev/dri/renderD* device; NVIDIA DRM render nodes are unavailable" >&2
    exit 1
}
echo "[OK] NVIDIA and DRM device nodes"

cdi_output="$(nvidia-ctk cdi list 2>&1)" || {
    echo "[FAIL] NVIDIA CDI enumeration failed" >&2
    printf '%s\n' "$cdi_output" >&2
    exit 1
}
grep -q 'nvidia.com/gpu' <<<"$cdi_output" || {
    echo "[FAIL] NVIDIA CDI listed no GPU devices" >&2
    exit 1
}
echo "[OK] NVIDIA Container Toolkit CDI"

available_kb="$(df -Pk "$workspace_root" | awk 'NR==2 {print $4}')"
required_kb=$((min_disk_gb * 1024 * 1024))
(( available_kb >= required_kb )) || {
    echo "[FAIL] free workspace disk is below ${min_disk_gb}GiB: ${available_kb}KiB" >&2
    exit 1
}
memory_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
echo "[OK] resources disk_free_kib=$available_kb memory_kib=$memory_kb"

if [[ "$skip_container_test" != true ]]; then
    docker image inspect "$gpu_test_image" >/dev/null 2>&1 || {
        echo "[FAIL] GPU probe image is not local: $gpu_test_image" >&2
        echo "       Pull/tag it explicitly, or pass --gpu-image; preflight never pulls images." >&2
        exit 1
    }
    probe="$(docker run --rm --gpus "device=$gpu_index" "$gpu_test_image" \
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>&1)" || {
        echo "[FAIL] Docker --gpus device=$gpu_index probe failed" >&2
        printf '%s\n' "$probe" >&2
        exit 1
    }
    [[ "$(printf '%s\n' "$probe" | wc -l)" -eq 1 ]] || {
        echo "[FAIL] single-GPU container probe exposed more than one GPU" >&2
        printf '%s\n' "$probe" >&2
        exit 1
    }
    echo "[OK] Docker GPU isolation: $probe"
fi

echo "HOST READY"
