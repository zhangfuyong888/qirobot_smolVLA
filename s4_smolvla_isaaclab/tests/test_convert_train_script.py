import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_check_train.sh"


def _run_pipeline(
    tmp_path: Path, mode: str, extra_args: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    hdf5 = tmp_path / "old_20phase.hdf5"
    hdf5.touch()
    summary = tmp_path / "failure_summary.json"
    summary.write_text("{}", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_COMMAND_LOG"
case "$FAKE_PIPELINE_MODE:$*" in
  fail_hdf5:*dataset-check*--hdf5*) exit 9 ;;
esac
exit 0
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "S4_DATA_ROOT": str(tmp_path / "data"),
            "S4_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "S4_CACHE_ROOT": str(tmp_path / "cache"),
            "SMOLVLA_MODEL_ROOT": str(tmp_path / "models"),
            "FAKE_COMMAND_LOG": str(log),
            "FAKE_PIPELINE_MODE": mode,
        }
    )
    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPT),
            "--hdf5-file",
            str(hdf5),
            "--expected-episodes",
            "300",
            "--failure-summary",
            str(summary),
            "--max-failed-attempts",
            "5",
            *extra_args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    commands = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, commands


def test_existing_hdf5_pipeline_runs_all_guarded_stages_in_order(tmp_path: Path):
    result, commands = _run_pipeline(tmp_path, "success")

    assert result.returncode == 0, result.stderr
    assert len(commands) == 4
    assert "dataset-check" in commands[0] and "--hdf5" in commands[0]
    assert "convert" in commands[1] and "--control-mode bimanual" in commands[1]
    assert "dataset-check" in commands[2] and "--hdf5" not in commands[2]
    assert "train --config" in commands[3]


def test_existing_hdf5_pipeline_never_converts_or_trains_after_failed_source_check(tmp_path: Path):
    result, commands = _run_pipeline(tmp_path, "fail_hdf5")

    assert result.returncode == 9
    assert len(commands) == 1
    assert "dataset-check" in commands[0] and "--hdf5" in commands[0]

