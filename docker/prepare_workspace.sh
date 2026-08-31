#!/usr/bin/env bash
# Copy (never move) the verified external IsaacLab checkout into this workspace
# so Docker can build from one self-contained workspace alone.
set -euo pipefail

workspace_root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="${1:-${ISAACLAB_SOURCE:-$HOME/IsaacLab}}"
target_root="$workspace_root/IsaacLab"

if [[ ! -x "$source_root/isaaclab.sh" ]]; then
    echo "IsaacLab source is invalid or missing isaaclab.sh: $source_root" >&2
    exit 2
fi
if [[ -e "$target_root" ]]; then
    echo "Refusing to overwrite existing workspace snapshot: $target_root" >&2
    echo "Inspect it first; remove it manually only if you intentionally want a new snapshot." >&2
    exit 2
fi

source_commit="$(git -C "$source_root" rev-parse HEAD)"
echo "[Docker release] copying IsaacLab $source_commit"
# Some shared or non-POSIX workspaces reject chown/chgrp metadata even when
# ordinary files are writable. Preserve content, symlinks, modes and times but
# deliberately keep the destination owner/group. Python bytecode is generated
# at runtime and is not part of the release snapshot.
rsync -rlptD --no-o --no-g --exclude=.git --exclude=__pycache__ "$source_root/" "$target_root/"

printf '%s\n' "$source_commit" > "$workspace_root/docker/ISAACLAB_COMMIT"
echo "[Docker release] workspace snapshot ready: $target_root"
