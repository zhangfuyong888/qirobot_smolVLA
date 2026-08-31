#!/usr/bin/env bash
# Stage the two Kit extensions IsaacLab 0.54.2 needs beyond the Isaac Sim 5.1
# pip extscache.  They are copied from a verified first-run cache and remain
# ignored by Git under docker/vendor/.
set -euo pipefail

workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="${1:-$HOME/.local/share/ov/data/exts/v2}"
target_root="$workspace_root/docker/vendor/kit-exts"
extensions=(
    "isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311"
    "omni.kit.pip_archive-d38fa9ecd1fb6df4"
)

if [[ -e "$target_root" ]]; then
    echo "Refusing to overwrite existing Kit extension bundle: $target_root" >&2
    exit 2
fi

for extension in "${extensions[@]}"; do
    if [[ ! -f "$source_root/$extension/config/extension.toml" ]]; then
        echo "Missing verified Kit extension: $source_root/$extension" >&2
        exit 2
    fi
done

mkdir -p "$target_root"
for extension in "${extensions[@]}"; do
    rsync -rlptD --no-o --no-g "$source_root/$extension/" "$target_root/$extension/"
done

echo "[Docker release] staged offline Kit extensions in $target_root"
