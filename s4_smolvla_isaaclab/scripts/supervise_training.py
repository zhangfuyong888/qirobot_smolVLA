#!/usr/bin/env python3
"""Run training in an isolated process group and reliably reap its workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


POLL_SECONDS = float(os.environ.get("S4_TRAIN_SUPERVISOR_POLL_SECONDS", "0.2"))
SIGNAL_GRACE_SECONDS = float(os.environ.get("S4_TRAIN_SIGNAL_GRACE_SECONDS", "5.0"))
TERM_GRACE_SECONDS = float(os.environ.get("S4_TRAIN_TERM_GRACE_SECONDS", "3.0"))


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def terminate_process_group(process: subprocess.Popen, initial_signal: int = signal.SIGTERM) -> None:
    """Terminate the trainer and every DataLoader descendant in its process group."""
    pgid = process.pid
    if not _group_exists(pgid):
        process.poll()
        return
    _signal_group(pgid, initial_signal)
    if _wait_until(lambda: not _group_exists(pgid), SIGNAL_GRACE_SECONDS):
        process.poll()
        return
    if initial_signal != signal.SIGTERM:
        _signal_group(pgid, signal.SIGTERM)
        if _wait_until(lambda: not _group_exists(pgid), TERM_GRACE_SECONDS):
            process.poll()
            return
    _signal_group(pgid, signal.SIGKILL)
    _wait_until(lambda: not _group_exists(pgid), TERM_GRACE_SECONDS)
    process.poll()


def publish_contract(contract_stage: Path | None, output_dir: Path) -> bool:
    if contract_stage is None or not contract_stage.is_file():
        return False
    if output_dir.is_symlink() or not output_dir.is_dir():
        return False
    destination = output_dir / "s4_dataset_contract.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".s4_dataset_contract.", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target, contract_stage.open("rb") as source:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def normalize_return_code(return_code: int) -> int:
    return 128 + (-return_code) if return_code < 0 else return_code


def run(command: list[str], output_dir: Path, contract_stage: Path | None = None) -> int:
    pending_signal: int | None = None
    contract_published = False

    def receive_signal(signum, _frame) -> None:
        nonlocal pending_signal
        if pending_signal is None:
            pending_signal = int(signum)

    previous_handlers = {
        signum: signal.signal(signum, receive_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            if not contract_published:
                contract_published = publish_contract(contract_stage, output_dir)
            if pending_signal is not None:
                terminate_process_group(process, pending_signal)
                break
            time.sleep(POLL_SECONDS)
        return_code = process.wait()
        # A killed/crashed trainer may leave multiprocessing workers behind.
        if _group_exists(process.pid):
            terminate_process_group(process, signal.SIGTERM)
        if not contract_published:
            contract_published = publish_contract(contract_stage, output_dir)
        if pending_signal is not None:
            return 128 + pending_signal
        return normalize_return_code(return_code)
    finally:
        if process is not None and _group_exists(process.pid):
            terminate_process_group(process, signal.SIGTERM)
        if not contract_published:
            publish_contract(contract_stage, output_dir)
        if contract_stage is not None:
            contract_stage.unlink(missing_ok=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-stage", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a training command is required after --")
    return run(command, args.output_dir.expanduser(), args.contract_stage)


if __name__ == "__main__":
    raise SystemExit(main())
