#!/usr/bin/env bash
# Canonical Isaac/IsaacLab native runtime environment for release containers.
# Source this file, then call s4_setup_isaac_env. It is safe to call repeatedly.

s4_prepend_path_once() {
    local variable="$1"
    local value="$2"
    [[ -d "$value" ]] || return 0
    local current="${!variable:-}"
    case ":$current:" in
        *":$value:"*) return 0 ;;
    esac
    if [[ -n "$current" ]]; then
        printf -v "$variable" '%s:%s' "$value" "$current"
    else
        printf -v "$variable" '%s' "$value"
    fi
    export "$variable"
}

s4_setup_isaac_env() {
    local conda_root="${S4_CONDA_ROOT:-/opt/conda}"
    local env_name="${S4_ISAACLAB_ENV:-env_isaaclab}"
    local prefix="${S4_ISAACLAB_PREFIX:-$conda_root/envs/$env_name}"
    local python="$prefix/bin/python"
    if [[ ! -x "$python" ]]; then
        echo "[S4][ISAAC-ENV][FAIL] missing Isaac Python: $python" >&2
        return 1
    fi

    local pyver
    pyver="$($python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local site_packages="$prefix/lib/python${pyver}/site-packages"
    local cmeel="$site_packages/cmeel.prefix"

    s4_prepend_path_once PATH "$prefix/bin"
    s4_prepend_path_once PYTHONPATH "$cmeel/lib/python${pyver}/site-packages"
    # Native Python extensions and Conda-forge's C++ runtime must precede the
    # Ubuntu host libraries injected into the container.
    s4_prepend_path_once LD_LIBRARY_PATH "$cmeel/lib"
    s4_prepend_path_once LD_LIBRARY_PATH "$prefix/lib"

    export CONDA_PREFIX="$prefix"
    export S4_ISAACLAB_PREFIX="$prefix"
    export PYTHONUNBUFFERED=1
    export ISAAC_LOCAL_ASSET_ROOT="${S4_SCENE_ASSET_ROOT:-${ISAAC_ASSET_ROOT:-}}"
    export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/workspace/runtime/xdg-runtime}"
    mkdir -p "$XDG_RUNTIME_DIR" "${S4_CACHE_ROOT:-/workspace/runtime/cache}"
}
