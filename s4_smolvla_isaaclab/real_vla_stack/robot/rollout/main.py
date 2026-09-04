from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.ros.robot_bridge import HardwareRobotBridge
from real_vla.cameras.camera_manager import CameraManager
from real_vla.config_loader import load_collection_config
from real_vla.robot.gripper_adapter import BinaryGripper
from real_vla.robot.home_manager import HomeManager
from real_vla.robot.s4_adapter import S4Adapter

from ...common.config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from ...common.errors import PolicyStaleError
from ...common.protocol import ObservationRequest
from .action_buffer import ActionBuffer
from .logger import RolloutLogger
from .observation import encode_jpeg, snapshot_observation
from .policy_client import PolicyClient
from .safety import validate_policy_chunk


def _run_home(adapter: S4Adapter, bridge: HardwareRobotBridge, hardware, logger: RolloutLogger) -> None:
    control_dt = 1.0 / float(hardware.hardware.control_rate_hz)
    manager = HomeManager(
        home_left_arm=hardware.startup.home_left_arm,
        home_right_arm=hardware.startup.home_right_arm,
        tolerance_rad=hardware.startup.position_tolerance_rad,
        duration_s=hardware.startup.duration_s,
        max_joint_step_rad=hardware.startup.max_joint_step_rad,
        control_dt=control_dt,
    )
    manager.request_home(adapter.read_bimanual())
    deadline = time.monotonic() + hardware.startup.duration_s + 10.0
    while manager.active and time.monotonic() < deadline:
        bridge.spin_once(timeout_sec=0.0)
        command = manager.step()
        adapter.publish(command, gripper_target=0.0, quest_trigger=0.0, allow_motion=True)
        if manager.is_home(adapter.read_bimanual()):
            logger.event("home", arrived_by=manager.arrived_by)
            return
        time.sleep(control_dt)
    raise RuntimeError(f"return-home did not complete: {manager.status_line()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="S4 real policy rollout; shadow unless config and CLI both enable live")
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-runtime-s", type=float)
    args = parser.parse_args()
    cfg = load_pipeline_config(args.config)
    configured_live = str(cfg.robot["rollout"]["mode"]) == "live"
    if args.live != configured_live:
        if args.live:
            raise RuntimeError("live rollout requires rollout.mode: live in robot YAML and --live")
        configured_live = False
    live = bool(args.live and configured_live)
    collection = load_collection_config()
    hardware = load_hardware_teleop_config(collection.hardware_teleop_config)
    cameras = CameraManager(collection.cameras, active_arm=cfg.contract.active_arm)
    if cameras.names != cfg.contract.camera_sources:
        raise RuntimeError(f"rollout cameras={cameras.names}, contract={cfg.contract.camera_sources}")
    bridge = HardwareRobotBridge(
        hardware.hardware,
        hardware.hands,
        gravity_cfg=hardware.gravity,
        project_root=hardware.project_root,
        check_arm_command_publishers=hardware.startup.check_arm_command_publishers,
        command_output_enabled=live,
    )
    adapter = S4Adapter(bridge, active_arm=cfg.contract.active_arm, hands_cfg=hardware.hands)
    endpoint = f"tcp://{cfg.robot['network']['policy_server']}:{cfg.robot['network']['policy_port']}"
    client = PolicyClient(endpoint, int(cfg.robot["network"]["request_timeout_ms"]))
    buffer = ActionBuffer(
        float(cfg.robot["rollout"]["policy_hz"]),
        int(cfg.robot["rollout"]["execute_horizon"]),
        float(cfg.robot["freshness"]["max_chunk_age_ms"]),
    )
    session_id = uuid.uuid4().hex
    run_root = Path.home() / "real_rollouts" / f"rollout_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = RolloutLogger(run_root, {"session_id": session_id, "live": live, "contract_sha256": cfg.contract.sha256})
    gripper = BinaryGripper(
        float(cfg.robot["gripper"]["open_threshold"]), float(cfg.robot["gripper"]["grasp_threshold"])
    )
    response_lock = threading.Lock()
    latest_response: list[tuple | None] = [None]
    request_active = threading.Event()
    request_id = 0
    timeouts = 0

    def infer(observation, head, wrist):
        nonlocal timeouts
        try:
            response, rtt = client.request(observation, head, wrist)
            with response_lock:
                latest_response[0] = (response, rtt, time.monotonic_ns())
            timeouts = 0
        except Exception as exc:
            timeouts += 1
            logger.event("policy_error", error=str(exc), consecutive=timeouts)
        finally:
            request_active.clear()

    cameras.start()
    started = time.monotonic()
    next_policy = started
    next_control = started
    max_runtime = float(args.max_runtime_s or cfg.robot["rollout"]["max_episode_s"])
    try:
        state_deadline = time.monotonic() + 5.0
        while not bridge.state_ready and time.monotonic() < state_deadline:
            bridge.spin_once(timeout_sec=0.05)
        if not bridge.state_ready:
            raise RuntimeError("robot state was not ready within 5 seconds")
        if live and bool(cfg.robot["rollout"]["start_from_home"]):
            _run_home(adapter, bridge, hardware, logger)
            started = time.monotonic()
            next_policy = started
            next_control = started
        while time.monotonic() - started < max_runtime:
            bridge.spin_once(timeout_sec=0.0)
            if not bridge.state_ready:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            measured_bimanual = adapter.read_bimanual()
            measured_q = adapter.read_arm_q7(measured_bimanual)
            if now >= next_policy and not request_active.is_set():
                images, image_ts = snapshot_observation(
                    cameras,
                    max_age_ms=cfg.contract.max_camera_age_ms,
                    max_skew_ms=cfg.contract.max_cross_camera_skew_ms,
                    now_ns=now_ns,
                )
                state = cfg.contract.make_state(measured_q, gripper.state)
                observation = ObservationRequest(
                    cfg.contract.sha256, session_id, request_id, now_ns, cfg.contract.task, state, image_ts
                )
                request_id += 1
                request_active.set()
                threading.Thread(
                    target=infer,
                    args=(
                        observation,
                        encode_jpeg(images[0], int(cfg.host["server"]["jpeg_quality"])),
                        encode_jpeg(images[1], int(cfg.host["server"]["jpeg_quality"])),
                    ),
                    daemon=True,
                ).start()
                next_policy = now + 1.0 / float(cfg.robot["rollout"]["policy_hz"])
            with response_lock:
                available = latest_response[0]
                latest_response[0] = None
            if available is not None:
                response, rtt_ms, received_ns = available
                safe_chunk = validate_policy_chunk(
                    response.action_chunk,
                    measured_q7=measured_q,
                    max_target_jump_rad=float(cfg.robot["safety"]["max_policy_target_jump_rad"]),
                    max_tracking_error_rad=float(cfg.robot["safety"]["max_policy_tracking_error_rad"]),
                )
                buffer.replace(safe_chunk, request_id=response.request_id, received_at_ns=received_ns)
                logger.event("policy_response", request_id=response.request_id, rtt_ms=rtt_ms, inference_ms=response.inference_ms)
            if timeouts > int(cfg.robot["freshness"]["max_consecutive_timeouts"]):
                raise PolicyStaleError("too many consecutive policy timeouts")
            if live:
                if buffer.chunk is None:
                    if now - started > float(cfg.robot["network"]["connect_timeout_ms"]) / 1000.0:
                        raise PolicyStaleError("policy server did not provide an initial action chunk")
                    time.sleep(0.005)
                    continue
                action = buffer.sample(now_ns)
                gripper_target = gripper.update(float(action[7]))
                command = adapter.overlay_active_arm(measured_bimanual, action[:7])
                published = adapter.publish(command, gripper_target=gripper_target, quest_trigger=0.0, allow_motion=True)
                logger.event("command", request_id=buffer.request_id, target=published.as_8d().tolist(), limited=published.limited)
            if now < next_control:
                time.sleep(next_control - now)
            next_control = max(next_control + 1.0 / float(cfg.robot["rollout"]["control_hz"]), time.monotonic())
        if live and bool(cfg.robot["rollout"]["return_home_on_finish"]):
            _run_home(adapter, bridge, hardware, logger)
        logger.event("complete", reason="max_episode_s")
        return 0
    except BaseException as exc:
        logger.event("abort", reason=str(exc))
        if live:
            try:
                if bool(cfg.robot["rollout"]["return_home_on_abort"]):
                    _run_home(adapter, bridge, hardware, logger)
                else:
                    bridge.hold_current_and_relinquish(f"rollout abort: {exc}")
            except Exception as home_exc:
                logger.event("home_failed", reason=str(home_exc))
                bridge.hold_current_and_relinquish(f"rollout abort/home failure: {home_exc}")
        raise
    finally:
        cameras.close()
        client.close()
        bridge.close()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
