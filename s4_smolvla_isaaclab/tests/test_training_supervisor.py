import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "supervise_training.py"


def _wait_for_file(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _process_is_live(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    try:
        return stat.read_text(encoding="utf-8").split()[2] != "Z"
    except FileNotFoundError:
        return False


def _wait_process_gone(pid: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_live(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"process remained alive: pid={pid}")


def _supervisor_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "S4_TRAIN_SUPERVISOR_POLL_SECONDS": "0.02",
            "S4_TRAIN_SIGNAL_GRACE_SECONDS": "0.2",
            "S4_TRAIN_TERM_GRACE_SECONDS": "0.2",
        }
    )
    return env


def test_sigint_to_supervisor_stops_stubborn_trainer_and_worker(tmp_path: Path):
    pid_file = tmp_path / "pids.txt"
    fake = tmp_path / "fake_train.sh"
    fake.write_text(
        """#!/usr/bin/env bash
trap '' INT TERM HUP
(trap '' INT TERM HUP; while :; do sleep 1; done) &
worker=$!
printf '%s %s\n' "$$" "$worker" > "$PID_FILE"
while :; do sleep 1; done
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = _supervisor_env()
    env["PID_FILE"] = str(pid_file)
    process = subprocess.Popen(
        [sys.executable, str(SUPERVISOR), "--output-dir", str(tmp_path / "output"), "--", str(fake)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_file(pid_file)
    trainer_pid, worker_pid = map(int, pid_file.read_text().split())
    os.kill(process.pid, signal.SIGINT)
    _stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 130, stderr
    _wait_process_gone(trainer_pid)
    _wait_process_gone(worker_pid)


def test_failed_trainer_cleans_orphan_worker_and_preserves_status(tmp_path: Path):
    pid_file = tmp_path / "worker.txt"
    fake = tmp_path / "fake_fail.sh"
    fake.write_text(
        """#!/usr/bin/env bash
(trap '' INT TERM HUP; while :; do sleep 1; done) &
printf '%s\n' "$!" > "$PID_FILE"
exit 7
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = _supervisor_env()
    env["PID_FILE"] = str(pid_file)
    result = subprocess.run(
        [sys.executable, str(SUPERVISOR), "--output-dir", str(tmp_path / "output"), "--", str(fake)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    worker_pid = int(pid_file.read_text())

    assert result.returncode == 7, result.stderr
    _wait_process_gone(worker_pid)


def test_supervisor_publishes_contract_and_removes_stage(tmp_path: Path):
    output = tmp_path / "output"
    stage = tmp_path / "contract.stage"
    stage.write_text('{"version": 1}\n', encoding="utf-8")
    fake = tmp_path / "fake_success.sh"
    fake.write_text(
        """#!/usr/bin/env bash
mkdir -p "$OUTPUT_DIR"
for _ in $(seq 1 50); do
    [[ -f "$OUTPUT_DIR/s4_dataset_contract.json" ]] && exit 0
    sleep 0.02
done
exit 9
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = _supervisor_env()
    env["OUTPUT_DIR"] = str(output)
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output-dir",
            str(output),
            "--contract-stage",
            str(stage),
            "--",
            str(fake),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "s4_dataset_contract.json").read_text() == '{"version": 1}\n'
    assert not stage.exists()
