#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export S4_PROJECT_ROOT="${S4_PROJECT_ROOT:-$PROJECT_ROOT}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
export ISAAC_ASSET_ROOT="${ISAAC_ASSET_ROOT:-$HOME/isaacsim_assets/Assets/Isaac/5.1}"
PROJECT_SCENE_ASSET_ROOT="$PROJECT_ROOT/local_assets/isaac/5.1"
if [[ -z "${S4_SCENE_ASSET_ROOT:-}" ]]; then
    if [[ -f "$PROJECT_SCENE_ASSET_ROOT/Isaac/Environments/Simple_Warehouse/warehouse.usd" ]]; then
        export S4_SCENE_ASSET_ROOT="$PROJECT_SCENE_ASSET_ROOT"
    else
        # Compatibility fallback for existing workstations. Fresh clones should
        # unpack the separately distributed local_assets bundle instead.
        export S4_SCENE_ASSET_ROOT="$ISAAC_ASSET_ROOT"
    fi
fi
export LEROBOT_ROOT="${LEROBOT_ROOT:-$(dirname "$PROJECT_ROOT")/lerobot}"
export SMOLVLA_MODEL_ROOT="${SMOLVLA_MODEL_ROOT:-$PROJECT_ROOT/models}"
export S4_DATA_ROOT="${S4_DATA_ROOT:-$PROJECT_ROOT/datasets}"
export S4_OUTPUT_ROOT="${S4_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
export S4_CACHE_ROOT="${S4_CACHE_ROOT:-$PROJECT_ROOT/.cache}"
export S4_ISAACLAB_ENV="${S4_ISAACLAB_ENV:-env_isaaclab}"
export S4_SMOLVLA_ENV="${S4_SMOLVLA_ENV:-smolvla}"

CONDA_EXE_PATH="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
CONDA_ROOT="${S4_CONDA_ROOT:-${CONDA_EXE_PATH%/bin/conda}}"
if [[ -z "$CONDA_ROOT" ]]; then
    CONDA_ROOT="$HOME/miniconda3"
fi
export S4_ISAACLAB_PREFIX="${S4_ISAACLAB_PREFIX:-$CONDA_ROOT/envs/$S4_ISAACLAB_ENV}"
export S4_SMOLVLA_PREFIX="${S4_SMOLVLA_PREFIX:-$CONDA_ROOT/envs/$S4_SMOLVLA_ENV}"
export S4_SMOLVLA_PYTHON="${S4_SMOLVLA_PYTHON:-$S4_SMOLVLA_PREFIX/bin/python}"
ISAACLAB="$ISAACLAB_ROOT/isaaclab.sh"

ISAAC_LOCAL_KIT_ARGS="--/persistent/isaac/asset_root/default=$S4_SCENE_ASSET_ROOT --/persistent/isaac/asset_root/cloud=$S4_SCENE_ASSET_ROOT --/persistent/isaac/asset_root/nvidia=$S4_SCENE_ASSET_ROOT --/persistent/isaac/asset_root/timeout=1"
ISAAC_LOCAL_KIT_ARGS+=" --/exts/isaacsim.asset.browser/folders/0=file:$S4_SCENE_ASSET_ROOT/Isaac/Environments --/exts/isaacsim.asset.browser/folders/1=file:$S4_SCENE_ASSET_ROOT/Isaac/Props --/exts/isaacsim.asset.browser/folders/2=file:$S4_SCENE_ASSET_ROOT/Isaac/Robots --/exts/isaacsim.asset.browser/data/timeout=1 --/exts/isaacsim.asset.browser/visible_after_startup=false"
ISAAC_LOCAL_KIT_ARGS+=" --/exts/isaacsim.gui.content_browser/folders/0=file:$S4_SCENE_ASSET_ROOT/Isaac/Environments --/exts/isaacsim.gui.content_browser/folders/1=file:$S4_SCENE_ASSET_ROOT/Isaac/Props --/exts/isaacsim.gui.content_browser/folders/2=file:$S4_SCENE_ASSET_ROOT/Isaac/Robots --/exts/isaacsim.gui.content_browser/timeout=1"

use_isaaclab_env() {
    local pyver
    pyver="$($S4_ISAACLAB_PREFIX/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local cmeel="$S4_ISAACLAB_PREFIX/lib/python$pyver/site-packages/cmeel.prefix"
    export PATH="$S4_ISAACLAB_PREFIX/bin:$PATH"
    export PYTHONPATH="$cmeel/lib/python$pyver/site-packages:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$cmeel/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONUNBUFFERED=1
    export ISAAC_LOCAL_ASSET_ROOT="$S4_SCENE_ASSET_ROOT"
}

use_smolvla_env() {
    export PATH="$S4_SMOLVLA_PREFIX/bin:$PATH"
    export CONDA_PREFIX="$S4_SMOLVLA_PREFIX"
    export PYTHONUNBUFFERED=1
}

print_context() {
    "$S4_ISAACLAB_PREFIX/bin/python" - <<'PY'
from s4_pipeline.config import load_project_config
from s4_pipeline.paths import active_task_id
cfg = load_project_config()
print(f"[S4] task={active_task_id()} dataset={cfg.dataset.repo_id} fps={cfg.dataset.fps}Hz action={cfg.features.action_dim}D")
PY
}

record_command() {
    local output="" episodes="1" timeout="300" block="blue" resume=0
    local -a app_args=() script_args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output) output="$2"; shift 2 ;;
            --episodes|--num-episodes) episodes="$2"; shift 2 ;;
            --block) block="$2"; shift 2 ;;
            --episode-timeout-s) timeout="$2"; shift 2 ;;
            --resume) resume=1; shift ;;
            --headless|--no-render) app_args+=(--headless); shift ;;
            --render) shift ;;
            *) script_args+=("$1"); shift ;;
        esac
    done
    if [[ -z "$output" ]]; then
        output="$($S4_ISAACLAB_PREFIX/bin/python - <<'PY'
from s4_pipeline.config import load_project_config
cfg = load_project_config()
print(cfg.dataset.staging_root / f"{cfg.dataset.task_id}_scripted.hdf5")
PY
)"
    fi
    use_isaaclab_env
    if [[ "$resume" -eq 1 ]]; then
        script_args+=(--resume)
    fi
    "$ISAACLAB" -p scripts/record_dataset.py --enable_cameras "${app_args[@]}" \
        --kit_args "$ISAAC_LOCAL_KIT_ARGS" --record-output "$output" \
        --record-episodes "$episodes" --record-episode-timeout-s "$timeout" \
        --auto-grasp --auto-grasp-block "$block" "${script_args[@]}"
}

rollout_command() {
    local -a args=()
    local deterministic=0
    local want_randomize=0
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--deterministic" ]]; then
            deterministic=1
            args+=(--no-randomize-task --seed 42)
        elif [[ "$1" == "--success-rate" ]]; then
            # Convenience: bash run.sh rollout --success-rate 20 ...
            shift
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "--success-rate requires an episode count, e.g. --success-rate 20" >&2
                exit 2
            fi
            want_randomize=1
            args+=(--episodes "$1" --randomize-task --seed 42)
        elif [[ "$1" == "--headless" ]]; then
            args+=(--headless)
        else
            args+=("$1")
        fi
        shift
    done
    if [[ "$deterministic" -eq 1 && "$want_randomize" -eq 1 ]]; then
        echo "Use either --deterministic or --success-rate, not both." >&2
        exit 2
    fi
    use_isaaclab_env
    "$ISAACLAB" -p scripts/eval_policy.py --enable_cameras --kit_args "$ISAAC_LOCAL_KIT_ARGS" "${args[@]}"
}

usage() {
    cat <<'EOF'
Usage: bash run.sh <command> [options]

Core commands:
  doctor [--strict]                  Check paths, environments and contracts
  prepare-assets [--source-root DIR] Build ignored local scene-asset bundle
  list-tasks                         List registered tasks
  activate-task TASK                Select a task in .local/active_task
  sim [IsaacLab options]             Start the active task scene
  teleop [options]                   Control both arms with Meta Quest 3
  teleop-cert [--ip ADDRESS]         Generate the local HTTPS certificate
  record [--episodes N] [--resume]    Record/continue successful HDF5 demonstrations
  collect-convert [options]           Collect, validate and convert; never train
  collect-train [options]             Guarded collect, convert, check, then train
  convert-train [options]             Guarded existing HDF5 -> check -> convert -> check -> train
  convert [--overwrite]              Convert HDF5 to LeRobotDataset
  dataset-check [PATH]               Validate dataset and optional checkpoint
  validate-workspace [options]       Dense offline IK/singularity audit for grasp region
  train [--resume]                   Train SmolVLA in the smolvla environment
  preview                            Offline checkpoint preview
  rollout [--deterministic|--success-rate N]
                                         Online IsaacLab rollout / success-rate eval
  diagnose ACTIONS.csv               Summarize rollout control diagnostics
  clean [--dry-run|--yes]            Inspect/remove generated artifacts

Rollout notes:
  --deterministic     seed=42, disable task randomization (regression)
  --success-rate N    seed=42, N episodes; honors YAML can_xy/distractor enables
  Pass-through flags: --episodes, --seed (default 42), --randomize-task/--no-randomize-task,
  --can-x-range, --can-y-range, --distractor-cans/--no-distractor-cans,
  --output-dir, --output-video, --summary-json, --save-videos/--no-save-videos,
  --save-diagnostics/--no-save-diagnostics
  Record pass-through: --can-xy-randomization/--no-can-xy-randomization,
  --distractor-cans/--no-distractor-cans (default: scripted.yaml enabled flags)
  Outputs default to one folder per run under outputs/eval/:
  rollout_<timestamp>_<det|randN>_ckpt<step>/{rollout|epXXX}.avi + *_actions.* + summary.json

Compatibility aliases: inspect-config, record-hdf5, convert-lerobot,
train-smolvla, preview-smolvla, visualize-smolvla, eval-smolvla,
record-parallel, pipeline, clean-generated.
EOF
}

case "${1:-help}" in
    help|-h|--help) usage ;;
    doctor) shift; "$S4_ISAACLAB_PREFIX/bin/python" scripts/doctor.py "$@" ;;
    prepare-assets)
        shift
        use_isaaclab_env
        PYVER="$($S4_ISAACLAB_PREFIX/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        USD_LIBS_ROOT="$(find "$S4_ISAACLAB_PREFIX/lib/python$PYVER/site-packages/isaacsim/extscache" -maxdepth 1 -type d -name 'omni.usd.libs-*' -print -quit)"
        if [[ -z "$USD_LIBS_ROOT" ]]; then
            echo "Could not locate Isaac Sim omni.usd.libs under $S4_ISAACLAB_PREFIX" >&2
            exit 2
        fi
        PYTHONPATH="$USD_LIBS_ROOT:${PYTHONPATH:-}" \
        LD_LIBRARY_PATH="$S4_ISAACLAB_PREFIX/lib:$USD_LIBS_ROOT/bin:${LD_LIBRARY_PATH:-}" \
        PXR_PLUGINPATH_NAME="$USD_LIBS_ROOT/bin/usd" \
            "$S4_ISAACLAB_PREFIX/bin/python" scripts/prepare_local_assets.py "$@"
        ;;
    inspect-config) shift; python3 scripts/inspect_project.py "$@" ;;
    list-tasks) shift; python3 scripts/inspect_tasks.py "$@" ;;
    activate-task) shift; python3 scripts/activate_task.py "$@" ;;
    sim)
        shift; print_context; use_isaaclab_env
        "$ISAACLAB" -p scripts/record_dataset.py --enable_cameras --kit_args "$ISAAC_LOCAL_KIT_ARGS" \
            --print-layout --continuous --show-tcp-frames --show-drawer-handle-frame \
            --show-wrist-camera-frustums --wrist-camera-frustum-depth 0.8 \
            --wrist-camera-frustum-line-width 8 --print-tcp-pose --tcp-print-period 0.5 "$@"
        ;;
    teleop)
        shift; print_context; use_isaaclab_env
        "$ISAACLAB" -p teleoperation/isaaclab_teleop.py --enable_cameras \
            --kit_args "$ISAAC_LOCAL_KIT_ARGS" "$@"
        ;;
    teleop-cert)
        shift; "$S4_ISAACLAB_PREFIX/bin/python" -m teleoperation.certificate "$@"
        ;;
    record|record-hdf5) shift; print_context; record_command "$@" ;;
    collect-convert) shift; bash scripts/collect_convert.sh "$@" ;;
    collect-train) shift; bash scripts/collect_convert_check_train.sh "$@" ;;
    convert-train) shift; bash scripts/convert_check_train.sh "$@" ;;
    record-parallel) shift; python3 scripts/record_parallel.py "$@" ;;
    convert|convert-lerobot) shift; print_context; use_smolvla_env; "$S4_SMOLVLA_PYTHON" scripts/convert_lerobot.py "$@" ;;
    dataset-check) shift; print_context; use_smolvla_env; "$S4_SMOLVLA_PYTHON" scripts/dataset_check.py "$@" ;;
    validate-workspace)
        shift
        use_isaaclab_env
        "$S4_ISAACLAB_PREFIX/bin/python" scripts/validate_drawer_grasp_workspace.py "$@"
        ;;
    train|train-smolvla) shift; print_context; use_smolvla_env; exec bash scripts/train_smolvla_local.sh "$@" ;;
    preview|preview-smolvla) shift; print_context; use_smolvla_env; "$S4_SMOLVLA_PYTHON" scripts/preview_policy.py "$@" ;;
    visualize-smolvla) shift; use_smolvla_env; "$S4_SMOLVLA_PYTHON" scripts/visualize_policy.py "$@" ;;
    rollout|eval-smolvla) shift; print_context; rollout_command "$@" ;;
    diagnose) shift; use_smolvla_env; "$S4_SMOLVLA_PYTHON" scripts/diagnose_rollout.py "$@" ;;
    clean|clean-generated) shift; python3 scripts/clean_generated.py "$@" ;;
    pipeline) shift; use_smolvla_env; bash scripts/pipeline_collect_convert_train.sh "$@" ;;
    joint-debug) shift; use_isaaclab_env; "$ISAACLAB" -p scripts/joint_debug.py --enable_cameras "$@" ;;
    control) shift; python3 scripts/control_arm.py "$@" ;;
    reach-block) shift; python3 scripts/control_arm.py reach-block "$@" ;;
    *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
