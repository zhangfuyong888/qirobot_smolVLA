from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from real_vla.collection.quality import evaluate_episode
from real_vla.collection.state_machine import CollectionEvent, CollectionState, CollectionStateMachine
from real_vla.config_loader import load_collection_config
from real_vla.input.quest_buttons import QuestButtonDecoder
from real_vla.robot.gripper_adapter import BinaryGripper, GRASP, OPEN
from real_vla.robot.home_manager import HomeManager
from teleoperation.protocol import ControllerFrame, ControllerSample


ROOT = Path(__file__).resolve().parents[1]
REAL_VLA = ROOT / "real_vla"
FORBIDDEN = {
    "isaaclab",
    "isaacsim",
    "omni",
    "torch",
    "s4_pipeline",
    "scripts.record_dataset",
    "scripts.eval_policy",
    "teleoperation.isaaclab_teleop",
}


def _sample(buttons: tuple[float, ...], valid: bool = True) -> ControllerSample:
    return ControllerSample(
        valid=valid,
        position=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        trigger=0.0,
        squeeze=0.0,
        buttons=buttons,
    )


def _frame(left_buttons=(), right_buttons=()) -> ControllerFrame:
    return ControllerFrame(
        session_id="test",
        sequence=1,
        client_time_ms=0.0,
        reference_space="local-floor",
        left=_sample(left_buttons),
        right=_sample(right_buttons),
        received_monotonic=0.0,
    )


def test_collection_config_loads() -> None:
    config = load_collection_config(ROOT / "real_vla/config/collection.yaml")
    assert config.schema_version == "s4_real_vla_v1"
    assert config.active_arm == "right"
    assert config.active_wrist_name == "wrist_right"
    assert len(config.cameras.enabled_streams("right")) == 2
    assert config.robot.arm_dim == 7
    assert config.task.text.startswith("Grasp the drawer")


def test_real_vla_python_does_not_import_simulation_stack() -> None:
    imported: set[str] = set()
    for path in REAL_VLA.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                imported.add(node.module)
    hits = sorted(name for name in imported if name in FORBIDDEN or name.split(".")[0] in FORBIDDEN)
    assert hits == []


def test_quest_buttons_are_rising_edge_and_y_is_hold() -> None:
    decoder = QuestButtonDecoder(discard_hold_s=0.6)
    first = decoder.update(_frame(right_buttons=(0, 0, 0, 0, 1, 0)), 0.0)
    second = decoder.update(_frame(right_buttons=(0, 0, 0, 0, 1, 0)), 0.1)
    assert first.a_rising is True
    assert second.a_rising is False
    decoder.update(_frame(left_buttons=(0, 0, 0, 0, 0, 1)), 1.0)
    held = decoder.update(_frame(left_buttons=(0, 0, 0, 0, 0, 1)), 1.59)
    assert held.y_held is False
    fired = decoder.update(_frame(left_buttons=(0, 0, 0, 0, 0, 1)), 1.61)
    assert fired.y_held is True
    again = decoder.update(_frame(left_buttons=(0, 0, 0, 0, 0, 1)), 1.8)
    assert again.y_held is False


def test_gripper_hysteresis_is_binary() -> None:
    gripper = BinaryGripper()
    assert gripper.update(0.5) == OPEN
    assert gripper.update(0.66) == GRASP
    assert gripper.update(0.50) == GRASP
    assert gripper.update(0.34) == OPEN


def test_state_machine_abxy_flow() -> None:
    machine = CollectionStateMachine()
    assert machine.on_startup_ok().state == CollectionState.HOMING
    assert machine.on_home_arrived().state == CollectionState.READY
    start = machine.on_buttons(
        type("B", (), {"a_rising": True, "b_rising": False, "x_rising": False, "y_held": False})(),
        disk_ok=True,
    )
    assert start.state == CollectionState.HOMING_TO_RECORD
    assert start.event == CollectionEvent.PRE_RECORD_HOME
    recording = machine.on_home_arrived()
    assert recording.event == CollectionEvent.START
    assert recording.state == CollectionState.RECORDING
    end = machine.on_buttons(
        type("B", (), {"a_rising": False, "b_rising": True, "x_rising": False, "y_held": False})()
    )
    assert end.event == CollectionEvent.END
    machine.on_home_arrived()
    review = machine.on_writer_finalized()
    assert review.state == CollectionState.REVIEW
    saved = machine.on_buttons(
        type("B", (), {"a_rising": False, "b_rising": False, "x_rising": True, "y_held": False})()
    )
    assert saved.event == CollectionEvent.SAVE
    assert saved.state == CollectionState.READY


def test_home_manager_reaches_tolerance() -> None:
    home = np.array([0.4, -0.6, -0.2, -1.2, -0.2, -0.5, -0.1], dtype=np.float32)
    start = np.zeros(26, dtype=np.float32)
    manager = HomeManager(
        home_left_arm=home,
        home_right_arm=home,
        tolerance_rad=0.03,
        stable_time_s=0.0,
        duration_s=0.2,
        max_joint_step_rad=0.2,
        control_dt=1.0 / 30.0,
    )
    manager.request_home(start)
    command = start.copy()
    for _ in range(40):
        command = manager.step()
    assert manager.is_home(command, now_s=10.0) is False
    assert manager.is_home(command, now_s=10.0) is True


def test_home_manager_accepts_command_when_measured_sags() -> None:
    from s4_robot.control_mapping import ACTION_SLICES

    home = np.array([0.4377, 0.6804, -0.2325, -1.1934, 0.1818, -0.5301, -0.1291], dtype=np.float32)
    start = np.zeros(26, dtype=np.float32)
    start[ACTION_SLICES.left_arm] = home
    start[ACTION_SLICES.right_arm] = np.array(
        [0.4377, -0.6804, -0.2325, -1.1934, -0.1818, -0.5301, -0.1291], dtype=np.float32
    )
    manager = HomeManager(
        home_left_arm=home,
        home_right_arm=start[ACTION_SLICES.right_arm],
        tolerance_rad=0.03,
        stable_time_s=0.1,
        duration_s=6.0,
        max_joint_step_rad=0.2,
        control_dt=1.0 / 30.0,
    )
    manager.request_home(start, now_s=0.0)
    manager.step()
    measured = start.copy()
    measured[ACTION_SLICES.left_arm][1] += 0.0374
    assert manager.is_home(measured, now_s=0.05) is False
    assert manager.arrived_by == "command"
    assert manager.is_home(measured, now_s=0.16) is True


def test_quality_flags_writer_drops() -> None:
    from real_vla.config_loader import QualityConfig

    quality = QualityConfig(
        camera_gap_warning_ms=80,
        camera_gap_invalid_ms=100,
        robot_state_invalid_ms=100,
        min_duration_s=0.1,
        min_camera_frames=2,
    )
    ts = np.array([0, 33_000_000, 66_000_000], dtype=np.int64)
    result = evaluate_episode(
        quality=quality,
        duration_s=1.0,
        arm_q=np.zeros((3, 7)),
        action_q=np.zeros((3, 7)),
        state_ts=ts,
        action_ts=ts,
        camera_ts={"head": ts, "wrist_right": ts},
        camera_seq={"head": np.array([1, 2, 3]), "wrist_right": np.array([1, 2, 3])},
        writer_drops={"head": 2, "wrist_right": 0},
        video_ok={"head": True, "wrist_right": True},
    )
    assert result.valid is True
    assert result.warning is True


def test_episode_writer_roundtrip(tmp_path: Path) -> None:
    from real_vla.cameras.camera_device import CameraReader
    from real_vla.collection.episode_writer import EpisodeWriter
    from real_vla.collection.schema import CameraFrame, PolicyState, PublishedCommand
    from real_vla.config_loader import CameraStreamConfig

    config = load_collection_config(ROOT / "real_vla/config/collection.yaml")
    session = tmp_path / "session"
    session.mkdir()
    writer = EpisodeWriter(config, session, ROOT)
    stream = CameraStreamConfig(
        name="head",
        enabled=True,
        serial="x",
        model="fake",
        width=16,
        height=16,
        fps=30,
    )

    class FakeSource:
        def read(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    reader = CameraReader(stream, source=FakeSource())
    wrist = CameraStreamConfig(
        name="wrist_right",
        enabled=True,
        serial="y",
        model="fake",
        width=16,
        height=16,
        fps=30,
    )
    wrist_reader = CameraReader(wrist, source=FakeSource())
    writer.start_episode(1, {"head": reader, "wrist_right": wrist_reader})
    writer.record_state(PolicyState(timestamp_ns=10, arm_q=np.zeros(7), gripper_state=0.0))
    writer.record_action(
        PublishedCommand(
            timestamp_ns=11,
            arm_target_q=np.zeros(7),
            gripper_target=0.0,
            hand_command_6d=np.zeros(6),
            quest_trigger=0.0,
            limited=False,
            motion_allowed=True,
        )
    )
    writer.accept_camera_frame(
        CameraFrame(timestamp_ns=12, capture_seq=1, image_bgr=np.zeros((16, 16, 3), dtype=np.uint8), name="head")
    )
    writer.accept_camera_frame(
        CameraFrame(
            timestamp_ns=13,
            capture_seq=1,
            image_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            name="wrist_right",
        )
    )
    writer.stop_accepting(50)
    writer.finalize_async()
    assert writer.wait_finalized(timeout_s=10.0)
    if writer._finalize_error:
        pytest.fail(writer._finalize_error)
    saved = writer.save()
    assert saved.is_dir()
    assert (saved / "meta.json").is_file()
    assert (saved / "trajectory.npz").is_file() or (saved / "trajectory.h5").is_file()


def test_begin_recording_is_cheap_after_prepare(tmp_path: Path) -> None:
    from real_vla.cameras.camera_device import CameraReader
    from real_vla.collection.episode_writer import EpisodeWriter
    from real_vla.collection.schema import PolicyState
    from real_vla.config_loader import CameraStreamConfig

    config = load_collection_config(ROOT / "real_vla/config/collection.yaml")
    session = tmp_path / "session"
    session.mkdir()
    writer = EpisodeWriter(config, session, ROOT)
    stream = CameraStreamConfig(
        name="head",
        enabled=True,
        serial="x",
        model="fake",
        width=16,
        height=16,
        fps=30,
    )

    class FakeSource:
        def read(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    reader = CameraReader(stream, source=FakeSource())
    writer.prepare_episode(1, {"head": reader})
    assert writer.is_prepared
    assert writer.recover_incomplete() == []
    assert writer.active_dir is not None and writer.active_dir.is_dir()
    started = time.monotonic()
    writer.begin_recording()
    elapsed = time.monotonic() - started
    assert writer._recording is True
    assert elapsed < 0.05
    writer.record_state(PolicyState(timestamp_ns=10, arm_q=np.zeros(7), gripper_state=0.0))
    time.sleep(0.05)
    assert (writer.active_dir / "robot_state.bin").stat().st_size > 0
    writer.stop_accepting(1)
    writer.finalize_async()
    assert writer.wait_finalized(timeout_s=10.0)
