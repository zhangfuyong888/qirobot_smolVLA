#!/usr/bin/env python
"""Real-robot collection: Quest teleop + ABXY episodes. No IsaacLab / LeRobot mix-in."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hardware_teleop.config_loader import (  # noqa: E402
    HardwareHandsConfig,
    HardwareStartupConfig,
    load_hardware_teleop_config,
)
from hardware_teleop.hooks import TeleopHooks, TeleopTick, TickRequest  # noqa: E402
from hardware_teleop.pink_main import build_parser as build_teleop_parser  # noqa: E402
from hardware_teleop.pink_main import run_pink_hardware_teleop  # noqa: E402
from s4_robot.control_mapping import ACTION_SLICES  # noqa: E402
from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS  # noqa: E402

from real_vla import SCHEMA_VERSION  # noqa: E402
from real_vla.cameras.camera_manager import CameraManager  # noqa: E402
from real_vla.collection.episode_writer import (  # noqa: E402
    EpisodeWriter,
    recover_orphaned_sessions,
)
from real_vla.collection.recorder import Recorder  # noqa: E402
from real_vla.collection.schema import PolicyState, PublishedCommand  # noqa: E402
from real_vla.collection.state_machine import CollectionEvent, CollectionState, CollectionStateMachine  # noqa: E402
from real_vla.config_loader import CollectionConfig, load_collection_config  # noqa: E402
from real_vla.input.quest_buttons import QuestButtonDecoder  # noqa: E402
from real_vla.robot.gripper_adapter import OPEN, BinaryGripper, gripper_to_hand6  # noqa: E402
from real_vla.robot.home_manager import HomeManager  # noqa: E402


def _arm_q7_from_targets(targets: dict[str, float], active_arm: str) -> np.ndarray | None:
    names = LEFT_ARM_JOINTS if active_arm == "left" else RIGHT_ARM_JOINTS
    if not all(name in targets for name in names):
        return None
    return np.asarray([float(targets[name]) for name in names], dtype=np.float64)


def _arm_q7_from_action(action: np.ndarray, active_arm: str) -> np.ndarray:
    sl = ACTION_SLICES.left_arm if active_arm == "left" else ACTION_SLICES.right_arm
    return np.asarray(action[sl], dtype=np.float64).copy()


class CollectionHooks(TeleopHooks):
    def __init__(
        self,
        config: CollectionConfig,
        cameras: CameraManager,
        writer: EpisodeWriter,
        hands_cfg: HardwareHandsConfig,
        startup_cfg: HardwareStartupConfig,
    ) -> None:
        self.config = config
        self.cameras = cameras
        self.writer = writer
        self.hands_cfg = hands_cfg
        self.recorder = Recorder(writer)
        self.machine = CollectionStateMachine()
        self.buttons = QuestButtonDecoder(
            a_index=config.buttons.a_index,
            b_index=config.buttons.b_index,
            x_index=config.buttons.x_index,
            y_index=config.buttons.y_index,
            press_threshold=config.buttons.press_threshold,
            discard_hold_s=config.buttons.discard_hold_s,
        )
        self.gripper = BinaryGripper(
            open_threshold=config.gripper.open_threshold,
            grasp_threshold=config.gripper.grasp_threshold,
        )
        self.home = HomeManager(
            home_left_arm=startup_cfg.home_left_arm,
            home_right_arm=startup_cfg.home_right_arm,
            tolerance_rad=config.home.tolerance_rad,
            stable_time_s=config.home.stable_time_s,
            duration_s=startup_cfg.duration_s,
            max_joint_step_rad=startup_cfg.max_joint_step_rad,
            control_dt=1.0 / config.robot.control_hz,
        )
        self._last_status = ""
        self._started = False
        self._stop_requested = False
        self._last_hud: dict[str, str] | None = None
        self._last_button_log = ""
        self._last_disk_check_s = 0.0
        self._disk_free_gb = float("inf")
        self._io_thread: threading.Thread | None = None
        self._haptic_cues: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()

    def begin_tick(self, frame, now_s: float) -> TickRequest:
        if self._stop_requested:
            return TickRequest(stop=True)
        if not self._started:
            self.machine.on_startup_ok()
            self._started = True
            print("[REAL-VLA] HOMING  wait; Grip ignored until READY", flush=True)

        edges = self.buttons.update(frame, now_s)
        self._log_buttons(frame, edges, now_s)
        if (
            self.machine.state == CollectionState.READY
            and edges.a_rising
            and not self.writer.is_prepared
        ):
            print(
                f"[REAL-VLA] A ignored: recorder not ready ({self.writer.prepare_error or 'preparing'})",
                flush=True,
            )
            edges = replace(edges, a_rising=False)
        if now_s - self._last_disk_check_s >= 1.0:
            self._disk_free_gb = self.writer.disk_gb()
            self._last_disk_check_s = now_s
        disk_ok = self._disk_free_gb >= self.config.storage.min_free_disk_gb
        if (
            self.machine.state == CollectionState.RECORDING
            and self._disk_free_gb < self.config.storage.critical_free_disk_gb
        ):
            self.writer.mark_invalid("recording aborted because disk space became critical")
            transition = self.machine.abort_recording("LOW DISK SPACE")
            self._on_transition(transition, now_s)
        else:
            save_ok = self.writer.quality is not None and self.writer.quality.valid
            transition = self.machine.on_buttons(
                edges,
                disk_ok=disk_ok,
                save_ok=save_ok,
            )
            self._on_transition(transition, now_s)

        allow_teleop = self.machine.state in {
            CollectionState.READY,
            CollectionState.RECORDING,
        }
        home_states = {
            CollectionState.HOMING,
            CollectionState.HOMING_TO_RECORD,
            CollectionState.RETURNING_HOME,
        }
        command_override = None
        if self.machine.state in home_states:
            if self.home.active:
                command_override = self.home.step()
        hand_triggers: tuple[float, float] | None
        if self.machine.state == CollectionState.RECORDING:
            left_analog = 0.0
            right_analog = 0.0
            if frame is not None:
                if frame.left.valid:
                    left_analog = float(frame.left.trigger)
                if frame.right.valid:
                    right_analog = float(frame.right.trigger)
            active_trigger = left_analog if self.config.active_arm == "left" else right_analog
            gripper = self.gripper.update(active_trigger)
            if self.config.active_arm == "left":
                hand_triggers = (float(gripper), right_analog)
            else:
                hand_triggers = (left_analog, float(gripper))
        elif self.machine.state in home_states:
            self.gripper.reset(OPEN)
            hand_triggers = (float(OPEN), float(OPEN))
        else:
            self.gripper.reset(OPEN)
            hand_triggers = None
        hud = self._hud()
        hud_out = None
        try:
            haptic, haptic_id = self._haptic_cues.get_nowait()
        except queue.Empty:
            haptic = ""
            haptic_id = ""
        if haptic:
            hud_out = {**hud, "haptic": haptic, "haptic_id": haptic_id}
        elif hud != self._last_hud or now_s - getattr(self, "_last_hud_send", 0.0) >= 0.5:
            hud_out = hud
        if hud_out is not None:
            self._last_hud = hud
            self._last_hud_send = now_s
        return TickRequest(
            allow_teleop=allow_teleop,
            command_override=command_override,
            hand_triggers=hand_triggers,
            publish_arms=True,
            hud=hud_out,
        )

    def end_tick(self, tick: TeleopTick) -> None:
        if self.machine.state in {
            CollectionState.RECORDING,
            CollectionState.RETURNING_HOME,
        } and self.recorder.enabled:
            self._record_tick(tick)

        if self.machine.state in {
            CollectionState.HOMING,
            CollectionState.HOMING_TO_RECORD,
            CollectionState.RETURNING_HOME,
        }:
            if not self.home.active:
                self.home.request_home(tick.command_action, now_s=tick.monotonic_s)
            elif self.home.is_home(
                tick.actual_action,
                tick.monotonic_s,
                require_measured=True,
            ):
                if self.home.arrived_by == "timeout":
                    reason = "return to Home was not confirmed by measured lowstate"
                    if self.machine.state == CollectionState.RETURNING_HOME:
                        self.writer.mark_invalid(reason)
                    else:
                        print(f"[REAL-VLA] {reason}; collection stopped", flush=True)
                        self._stop_requested = True
                        return
                transition = self.machine.on_home_arrived()
                self._on_transition(transition, tick.monotonic_s)

        if self.writer.wait_finalized(timeout_s=0.0) and self.machine.state == CollectionState.RETURNING_HOME:
            transition = self.machine.on_writer_finalized()
            self._on_transition(transition, tick.monotonic_s)

        if tick.monotonic_s - getattr(self, "_last_report", 0.0) >= 0.5:
            self._last_report = tick.monotonic_s
            extra = ""
            if self.machine.state in {
                CollectionState.HOMING,
                CollectionState.HOMING_TO_RECORD,
                CollectionState.RETURNING_HOME,
            }:
                extra = f" {self.home.status_line()} via={self.home.arrived_by or '-'}"
            print(
                f"[REAL-VLA] {self.machine.state.value} {self.cameras.health_line()} "
                f"disk={self._disk_free_gb:.1f}GB{extra}",
                flush=True,
            )

    def _record_tick(self, tick: TeleopTick) -> None:
        gripper = self.gripper.state
        state_valid = not (
            tick.state_feed_stale
            or tick.fault_active
            or tick.output_relinquished
        )
        if not state_valid:
            reason = tick.fault_reason or "robot state/control output became invalid"
            self.writer.mark_invalid(reason)
        input_valid = not tick.input_stale or (
            self.machine.state == CollectionState.RETURNING_HOME
        )
        if not input_valid:
            self.writer.mark_invalid("Quest input became stale during teleoperation")
        state = PolicyState(
            timestamp_ns=tick.timestamp_ns,
            arm_q=_arm_q7_from_action(tick.actual_action, self.config.active_arm),
            gripper_state=gripper,
            state_age_s=tick.state_age_s,
            valid=state_valid,
            phase=(
                1
                if self.machine.state == CollectionState.RETURNING_HOME
                else 0
            ),
        )
        self.recorder.on_state(state)
        arm_target = _arm_q7_from_targets(tick.published_arm, self.config.active_arm)
        published = arm_target is not None and tick.command_published
        requested = _arm_q7_from_action(tick.command_action, self.config.active_arm)
        if arm_target is None:
            arm_target = requested
        limited = published and not np.allclose(arm_target, requested, atol=1.0e-9)
        if not published:
            self.writer.mark_invalid("arm command was not published during recording")
        command = PublishedCommand(
            timestamp_ns=tick.timestamp_ns,
            arm_target_q=arm_target,
            gripper_target=gripper,
            hand_command_6d=np.asarray(
                gripper_to_hand6(
                    self.hands_cfg, side=self.config.active_arm, gripper=gripper
                ),
                dtype=np.float64,
            ),
            quest_trigger=(
                tick.left_trigger
                if self.config.active_arm == "left"
                else tick.right_trigger
            ),
            limited=limited,
            motion_allowed=bool(
                published
                and tick.command_transport_ok
                and (
                    tick.left_active
                    or tick.right_active
                    or self.machine.state == CollectionState.RETURNING_HOME
                )
            ),
            published=published,
            fault_active=tick.fault_active,
            input_valid=input_valid,
        )
        self.recorder.on_action(command)

    def _hud(self) -> dict[str, str]:
        state = self.machine.state
        if state == CollectionState.HOMING:
            return {
                "title": "HOMING",
                "detail": "Wait. Grip ignored until READY.",
                "kind": "ready",
            }
        if state == CollectionState.HOMING_TO_RECORD:
            return {
                "title": "HOMING TO RECORD",
                "detail": "Returning to home. Grip ignored. Recording starts at home.",
                "kind": "ready",
            }
        if state == CollectionState.READY:
            if not self.writer.is_prepared:
                return {
                    "title": "READY",
                    "detail": "Grip teleop OK. Recorder preparing; wait before A.",
                    "kind": "ready",
                }
            return {
                "title": "READY",
                "detail": f"Use {self.config.active_arm} Grip. A homes, then starts recording.",
                "kind": "ready",
            }
        if state == CollectionState.RECORDING:
            return {
                "title": "RECORDING",
                "detail": "Grip moves the active arm. B records the automatic return home.",
                "kind": "ready",
            }
        if state == CollectionState.RETURNING_HOME:
            return {
                "title": "RETURNING HOME",
                "detail": "Recording automatic return home. Keep the area clear.",
                "kind": "ready",
            }
        if state == CollectionState.REVIEW:
            return {
                "title": "REVIEW",
                "detail": "X saves. Hold Y 0.6s to discard.",
                "kind": "ready",
            }
        return {"title": state.value, "detail": "", "kind": "ready"}

    def _log_buttons(self, frame, edges, now_s: float) -> None:
        del now_s
        if edges.a_rising or edges.b_rising or edges.x_rising or edges.y_rising or edges.y_held:
            left = () if frame is None or not frame.left.valid else frame.left.buttons
            right = () if frame is None or not frame.right.valid else frame.right.buttons
            line = (
                f"[REAL-VLA] buttons A={int(edges.a)} B={int(edges.b)} "
                f"X={int(edges.x)} Y={int(edges.y)} heldY={int(edges.y_held)} "
                f"state={self.machine.state.value} "
                f"btnL={_fmt_buttons(left)} btnR={_fmt_buttons(right)}"
            )
            if line != self._last_button_log:
                self._last_button_log = line
                print(line, flush=True)

    def _on_transition(self, transition, now_s: float) -> None:
        del now_s
        if transition.message:
            print(f"[REAL-VLA] {transition.message}", flush=True)
        if transition.event is None:
            return
        if transition.event == CollectionEvent.HOME_DONE and self.home.arrived_by:
            print(
                f"[REAL-VLA] home accepted via {self.home.arrived_by} "
                f"({self.home.status_line()})",
                flush=True,
            )
        if transition.event == CollectionEvent.START:
            self.writer.begin_recording()
            self.recorder.start()
            self._queue_haptic("recording")
            print(f"[REAL-VLA] recording episode_{self.writer.episode_id:06d}", flush=True)
        elif transition.event in {CollectionEvent.END, CollectionEvent.ABORT_RECORDING}:
            if transition.event == CollectionEvent.END:
                self._queue_haptic("ending")
                return
            self.recorder.stop()
            self.writer.stop_accepting(time.monotonic_ns())
            self.writer.finalize_async()
        elif transition.event == CollectionEvent.RETURN_HOME_DONE:
            if self.recorder.enabled:
                self.recorder.stop()
                self.writer.stop_accepting(time.monotonic_ns())
                self.writer.finalize_async()
        elif transition.event == CollectionEvent.WRITER_DONE or (
            transition.state == CollectionState.REVIEW
        ):
            print(self.writer.review_summary(), flush=True)
        elif transition.event == CollectionEvent.SAVE:
            print("[REAL-VLA] saving in background...", flush=True)
            self._spawn_io("save")
        elif transition.event == CollectionEvent.DISCARD:
            print("[REAL-VLA] discarding in background...", flush=True)
            self._spawn_io("discard")
        elif transition.event == CollectionEvent.LOW_DISK:
            print("[REAL-VLA] A ignored: LOW DISK SPACE", flush=True)

    def shutdown(self) -> None:
        """Finish any active writer without issuing additional robot motion."""
        if self._io_thread is not None and self._io_thread.is_alive():
            self._io_thread.join(timeout=20.0)
        if self.recorder.enabled:
            self.writer.mark_invalid("collection process stopped before B/home completion")
            self.recorder.stop()
            self.writer.stop_accepting(time.monotonic_ns())
            self.writer.finalize_async()
        elif self.writer._finalize_thread is None:
            self.writer.cancel_prepared()
        if self.writer.active_dir is not None and self.writer._finalize_thread is not None:
            if not self.writer.wait_finalized(timeout_s=40.0):
                print("[REAL-VLA] writer shutdown timed out; pending episode kept", flush=True)

    def _spawn_io(self, action: str) -> None:
        if self._io_thread is not None and self._io_thread.is_alive():
            print("[REAL-VLA] save/discard already in progress", flush=True)
            return
        self._io_thread = threading.Thread(
            target=self._io_worker,
            args=(action,),
            name=f"real-vla-{action}",
            daemon=True,
        )
        self._io_thread.start()

    def _queue_haptic(self, cue: str) -> None:
        self._haptic_cues.put((cue, f"{time.monotonic_ns()}-{cue}"))

    def _io_worker(self, action: str) -> None:
        try:
            if action == "save":
                path = self.writer.save()
                print(f"[REAL-VLA] saved {path}", flush=True)
                self._queue_haptic("saved")
            else:
                self.writer.discard()
                print("[REAL-VLA] discarded pending episode", flush=True)
                self._queue_haptic("discarded")
            next_id = self.writer.next_episode_id()
            self.writer.prepare_episode(next_id, self.cameras.readers)
            if self.writer.is_prepared:
                print(
                    f"[REAL-VLA] recorder ready episode_{next_id:06d}  A: START",
                    flush=True,
                )
            else:
                print(
                    f"[REAL-VLA] recorder prepare failed: {self.writer.prepare_error}",
                    flush=True,
                )
        except Exception:
            traceback.print_exc()


def _fmt_buttons(values: tuple[float, ...]) -> str:
    if not values:
        return "[]"
    return "[" + ",".join(f"{item:.2f}" for item in values[:8]) + "]"


def _create_session_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"session_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "session.json").write_text(
        json.dumps(
            {
                "created": stamp,
                "schema_version": SCHEMA_VERSION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _acquire_collection_lock(root: Path):
    handle = (root / ".collection.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
    handle.flush()
    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect real-robot VLA episodes with Quest ABXY.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--session-dir", type=Path, default=None)
    teleop_parser = build_teleop_parser()
    args, remaining = parser.parse_known_args(argv)
    teleop_args = teleop_parser.parse_args(remaining)
    collection = load_collection_config(args.config)
    teleop_args.hardware_config = collection.hardware_teleop_config
    # Startup homing is joint-space and can recover an elbow that is outside
    # Pink's URDF range. Skipping it seeds IK from the measured pose and can
    # crash before HomeManager ever runs.
    teleop_args.skip_homing = False
    teleop_args.enabled_arms = collection.active_arm
    if not teleop_args.shadow and not teleop_args.arm_output:
        print("[REAL-VLA] refusing live collection without explicit --arm-output", flush=True)
        return 2

    session_root = collection.storage.root
    session_root.mkdir(parents=True, exist_ok=True)
    collection_lock = _acquire_collection_lock(session_root)
    if collection_lock is None:
        print("[REAL-VLA] refusing to start: another collector holds the lock", flush=True)
        return 2
    free_gb = shutil.disk_usage(session_root).free / (1024 ** 3)
    if free_gb < collection.storage.min_free_disk_gb:
        print(
            f"[REAL-VLA] WARNING low disk {free_gb:.1f}GB "
            f"(min {collection.storage.min_free_disk_gb:.1f}GB)",
            flush=True,
        )
    if free_gb < collection.storage.critical_free_disk_gb:
        print("[REAL-VLA] refusing to start: critical disk space", flush=True)
        collection_lock.close()
        return 2

    recovered = recover_orphaned_sessions(collection.storage.root)
    if recovered:
        print(f"[REAL-VLA] recovered {len(recovered)} orphaned pending episodes", flush=True)
    session_dir = args.session_dir or _create_session_dir(collection.storage.root)
    print(f"[REAL-VLA] session={session_dir}", flush=True)
    print(
        f"[REAL-VLA] task={collection.task.text!r} "
        f"record_arm={collection.active_arm} teleop_arms={collection.active_arm}",
        flush=True,
    )
    config = load_hardware_teleop_config(teleop_args.hardware_config)
    cameras = CameraManager(collection.cameras, active_arm=collection.active_arm)
    writer = EpisodeWriter(collection, session_dir, PROJECT_ROOT)
    hooks = CollectionHooks(
        collection,
        cameras,
        writer,
        hands_cfg=config.hands,
        startup_cfg=config.startup,
    )
    try:
        cameras.start()
        print(f"[REAL-VLA] cameras={list(cameras.names)} warmup done", flush=True)
        first_id = hooks.writer.next_episode_id()
        hooks.writer.prepare_episode(first_id, cameras.readers)
        if not hooks.writer.is_prepared:
            print(
                f"[REAL-VLA] recorder prepare failed: {hooks.writer.prepare_error}",
                flush=True,
            )
            return 1
        print(
            f"[REAL-VLA] recorder ready episode_{first_id:06d}  "
            f"READY: {collection.active_arm}-arm teleop  A: home then record  B: record return home",
            flush=True,
        )
        if teleop_args.ik_backend is not None:
            config = replace(config, ik=replace(config.ik, backend=str(teleop_args.ik_backend)))
        run_pink_hardware_teleop(config, teleop_args, hooks=hooks)
    except KeyboardInterrupt:
        print("\n[REAL-VLA] interrupted", flush=True)
        return 130
    except BaseException:
        print("[FATAL] real VLA collection failed:", flush=True)
        traceback.print_exc()
        return 1
    finally:
        hooks.shutdown()
        try:
            cameras.close()
        except KeyboardInterrupt:
            print("[REAL-VLA] camera close interrupted", flush=True)
        if writer.active_dir is not None and not writer.wait_finalized(timeout_s=0.0):
            print("[REAL-VLA] pending episode left in pending/ for recovery", flush=True)
        collection_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
