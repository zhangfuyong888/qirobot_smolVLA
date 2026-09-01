from pathlib import Path

import pytest

from hardware_teleop.safety import (
    SDK_MODE5_MERGE_MARKER,
    TeleopFaultLatch,
    find_verified_mode5_sdk_process,
)
from hardware_teleop.pink_main import _close_failed_initialization


class _FakeBridge:
    def __init__(self) -> None:
        self.hold_calls = 0
        self.close_calls = 0

    def hold_current_arms(self) -> None:
        self.hold_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_fault_latch_requires_both_grips_released_and_healthy_state() -> None:
    latch = TeleopFaultLatch()
    assert latch.trip("qp failed") is True
    assert latch.trip("second failure") is False
    assert latch.reason == "qp failed"
    assert latch.trip_count == 1

    assert not latch.clear_if_released(
        left_clutch=True, right_clutch=False, state_feed_ok=True
    )
    assert not latch.clear_if_released(
        left_clutch=False, right_clutch=False, state_feed_ok=False
    )
    assert latch.clear_if_released(
        left_clutch=False, right_clutch=False, state_feed_ok=True
    )
    assert not latch.active


def test_fault_latch_can_trip_again_after_recovery() -> None:
    latch = TeleopFaultLatch()
    latch.trip("first")
    latch.clear_if_released(
        left_clutch=False, right_clutch=False, state_feed_ok=True
    )
    assert latch.trip("second") is True
    assert latch.reason == "second"
    assert latch.trip_count == 2


def test_failed_initialization_never_emits_mode5_hold() -> None:
    bridge = _FakeBridge()
    _close_failed_initialization(bridge, command_output_enabled=True)
    assert bridge.hold_calls == 0
    assert bridge.close_calls == 1


def test_failed_shadow_initialization_only_closes_subscription_bridge() -> None:
    bridge = _FakeBridge()
    _close_failed_initialization(bridge, command_output_enabled=False)
    assert bridge.hold_calls == 0
    assert bridge.close_calls == 1


def _fake_process(proc_root: Path, pid: int, executable: Path, payload: bytes) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "comm").write_text("sn_loco_server\n", encoding="utf-8")
    executable.write_bytes(payload)
    (process / "exe").symlink_to(executable)


def test_sdk_mode5_merge_gate_accepts_verified_running_binary(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    executable = tmp_path / "sn_loco_server"
    _fake_process(proc_root, 123, executable, b"prefix" + SDK_MODE5_MERGE_MARKER + b"suffix")
    pid, resolved = find_verified_mode5_sdk_process(proc_root)
    assert pid == 123
    assert resolved == executable


def test_sdk_mode5_merge_gate_rejects_old_binary(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    executable = tmp_path / "sn_loco_server_old"
    _fake_process(proc_root, 456, executable, b"old SDK without merge")
    with pytest.raises(RuntimeError, match="marker=missing"):
        find_verified_mode5_sdk_process(proc_root)
