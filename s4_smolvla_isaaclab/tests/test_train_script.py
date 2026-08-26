import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_smolvla_local.sh"
TRAIN_CONFIG = ROOT / "configs" / "tasks" / "drawer_insert_close.smolvla.yaml"
DATASET_NAME = "s4_drawer_insert_close_v4_12phase_serial_acquire"


def _run_fake_training(tmp_path: Path, mode: str) -> tuple[subprocess.CompletedProcess[str], Path, dict]:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    dataset_root = data_root / "lerobot_data" / DATASET_NAME
    contract = {
        "schema_version": "s4_bimanual_v1",
        "action_semantics": "absolute_joint_target",
        "state_dim": 26,
        "action_dim": 26,
        "fps": 20,
        "language_contract_version": "drawer_12phase_v4_serial_acquire",
    }
    contract_path = dataset_root / "meta" / "s4_contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_train = fake_bin / "lerobot-train"
    fake_train.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
output=""
for arg in "$@"; do
    case "$arg" in --output_dir=*) output="${arg#--output_dir=}" ;; esac
done
[[ -n "$output" ]]
if [[ -e "$output" || -L "$output" ]]; then
    echo "output existed before fake LeRobot started" >&2
    exit 91
fi
if [[ "${FAKE_TRAIN_MODE}" != "no_output" ]]; then
    mkdir -p "$output/checkpoints/last/pretrained_model" "$output/checkpoints/last/training_state"
    printf '{}\n' > "$output/checkpoints/last/pretrained_model/train_config.json"
    printf '{"step": 5}\n' > "$output/checkpoints/last/training_state/training_step.json"
fi
if [[ "${FAKE_TRAIN_MODE}" == "fail" ]]; then
    exit 7
fi
if [[ "${FAKE_TRAIN_MODE}" == "wait_for_contract" ]]; then
    for _ in $(seq 1 30); do
        [[ -f "$output/s4_dataset_contract.json" ]] && exit 0
        sleep 0.1
    done
    echo "contract was not published while training was running" >&2
    exit 92
fi
""",
        encoding="utf-8",
    )
    fake_train.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "S4_DATA_ROOT": str(data_root),
            "S4_OUTPUT_ROOT": str(output_root),
            "S4_CACHE_ROOT": str(tmp_path / "cache"),
            "SMOLVLA_MODEL_ROOT": str(tmp_path / "models"),
            "FAKE_TRAIN_MODE": mode,
            "S4_TEST_SKIP_TRAIN_DATASET_CHECK": "1",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(TRAIN_SCRIPT),
            str(TRAIN_CONFIG),
            "--steps",
            "10",
            "--save-freq",
            "5",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    output_dir = output_root / "train" / "smolvla_drawer_insert_close_v4_12phase_serial_acquire"
    return result, output_dir, contract


def test_fresh_training_keeps_output_absent_until_lerobot_starts(tmp_path: Path):
    result, output_dir, contract = _run_fake_training(tmp_path, "success")

    assert result.returncode == 0, result.stderr
    assert json.loads((output_dir / "s4_dataset_contract.json").read_text()) == contract
    assert not list(output_dir.parent.glob(".*.s4_dataset_contract.*"))


def test_failed_training_preserves_exit_code_and_installs_provenance(tmp_path: Path):
    result, output_dir, contract = _run_fake_training(tmp_path, "fail")

    assert result.returncode == 7
    assert json.loads((output_dir / "s4_dataset_contract.json").read_text()) == contract
    assert not list(output_dir.parent.glob(".*.s4_dataset_contract.*"))

    fake_train = tmp_path / "bin" / "lerobot-train"
    fake_train.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" | grep -q -- '--resume=true'
""",
        encoding="utf-8",
    )
    fake_train.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path / 'bin'}:{env['PATH']}",
            "S4_DATA_ROOT": str(tmp_path / "data"),
            "S4_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "S4_CACHE_ROOT": str(tmp_path / "cache"),
            "SMOLVLA_MODEL_ROOT": str(tmp_path / "models"),
            "S4_TEST_SKIP_TRAIN_DATASET_CHECK": "1",
        }
    )
    resumed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), str(TRAIN_CONFIG), "--resume", "--steps", "10"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr


def test_success_without_output_is_rejected_and_staging_is_cleaned(tmp_path: Path):
    result, output_dir, _contract = _run_fake_training(tmp_path, "no_output")

    assert result.returncode == 2
    assert not output_dir.exists()
    assert "did not create a real output directory" in result.stderr
    assert not list(output_dir.parent.glob(".*.s4_dataset_contract.*"))


def test_contract_is_published_while_training_process_is_running(tmp_path: Path):
    result, output_dir, contract = _run_fake_training(tmp_path, "wait_for_contract")

    assert result.returncode == 0, result.stderr
    assert json.loads((output_dir / "s4_dataset_contract.json").read_text()) == contract


@pytest.mark.parametrize("kind", ("empty_directory", "symlink"))
def test_fresh_training_rejects_any_existing_output_path(tmp_path: Path, kind: str):
    data_root = tmp_path / "data"
    dataset_root = data_root / "lerobot_data" / DATASET_NAME / "meta"
    dataset_root.mkdir(parents=True)
    (dataset_root / "s4_contract.json").write_text(
        json.dumps({"language_contract_version": "drawer_12phase_v4_serial_acquire"}), encoding="utf-8"
    )
    output_root = tmp_path / "outputs"
    output_dir = output_root / "train" / "smolvla_drawer_insert_close_v4_12phase_serial_acquire"
    output_dir.parent.mkdir(parents=True)
    if kind == "empty_directory":
        output_dir.mkdir()
    else:
        target = tmp_path / "outside"
        target.mkdir()
        output_dir.symlink_to(target, target_is_directory=True)

    env = os.environ.copy()
    env.update(
        {
            "S4_DATA_ROOT": str(data_root),
            "S4_OUTPUT_ROOT": str(output_root),
            "S4_CACHE_ROOT": str(tmp_path / "cache"),
            "SMOLVLA_MODEL_ROOT": str(tmp_path / "models"),
            "S4_TEST_SKIP_TRAIN_DATASET_CHECK": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), str(TRAIN_CONFIG), "--steps", "10"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert "requires a non-existent output path" in result.stderr
