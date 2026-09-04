#!/bin/bash
set -euo pipefail

STACK_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$STACK_ROOT/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}"
HOST_PYTHON="${S4_SMOLVLA_PYTHON:-${CONDA_ROOT:+$CONDA_ROOT/envs/smolvla/bin/python}}"
ROBOT_PYTHON="${S4_HW_TELEOP_PYTHON:-${CONDA_ROOT:+$CONDA_ROOT/envs/s4_hardware_teleop/bin/python}}"
COMMAND="${1:-help}"
shift || true
cd "$PROJECT_ROOT"
PROJECT_CACHE="${S4_CACHE_ROOT:-$PROJECT_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$PROJECT_CACHE/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

case "$COMMAND" in
  raw-check|convert|dataset-check|train|checkpoint-check|serve)
    if [[ -z "$HOST_PYTHON" || ! -x "$HOST_PYTHON" ]]; then
      echo "SmolVLA host Python not found; set S4_SMOLVLA_PYTHON" >&2
      exit 2
    fi
    exec "$HOST_PYTHON" -m real_vla_stack.cli "$COMMAND" "$@"
    ;;
  rollout)
    if [[ -z "$ROBOT_PYTHON" || ! -x "$ROBOT_PYTHON" ]]; then
      echo "Hardware Python not found; set S4_HW_TELEOP_PYTHON" >&2
      exit 2
    fi
    exec "$ROBOT_PYTHON" -m real_vla_stack.robot.rollout.main "$@"
    ;;
  help|-h|--help)
    echo "Usage: bash real_vla_stack/run.sh {raw-check|convert|dataset-check|train|checkpoint-check|serve|rollout} [options]"
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    exit 2
    ;;
esac
