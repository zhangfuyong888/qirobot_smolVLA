from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.command_route import LEG_DEPLOY_COMMAND_TOPIC, LEG_DEPLOY_PUBLISHER
from hardware_teleop.ros.robot_bridge import HardwareRobotBridge
from hardware_teleop.safety import find_verified_arm_replay_sdk_process
from real_vla.cameras.camera_manager import CameraManager
from real_vla.config_loader import load_collection_config
from real_vla.robot.gripper_adapter import BinaryGripper
from real_vla.robot.home_manager import HomeManager
from real_vla.robot.s4_adapter import S4Adapter

from ...common.config import DEFAULT_PIPELINE_CONFIG, load_pipeline_config
from ...common.errors import (
    CommandOutputRelinquishedError,
    CommandRouteConflictError,
    PolicyStaleError,
    RobotStateStaleError,
    RobotTrackingError,
)
from ...common.protocol import ObservationRequest
from .action_buffer import ActionBuffer
from .logger import RolloutLogger
from .observation import encode_jpeg, snapshot_observation
from .policy_client import AsyncPolicyClient
from .safety import validate_policy_chunk


def _ensure_robot_health(bridge: HardwareRobotBridge, hardware, *, live: bool) -> None:
    state_age_s = bridge.last_state_age_s
    if bridge.is_state_feed_stale(hardware.hardware.max_state_age_s):
        raise RobotStateStaleError(
            f"robot state age {state_age_s:.3f}s exceeds "
            f"{hardware.hardware.max_state_age_s:.3f}s"
        )
    if not live:
        return
    if bridge.output_relinquished:
        raise CommandOutputRelinquishedError(
            f"hardware command output was relinquished: {bridge.release_reason}"
        )
    if bridge.is_arm_command_graph_conflicted():
        raise CommandRouteConflictError(
            "arm command publisher conflict: "
            + ", ".join(bridge.command_publisher_conflicts)
        )


def _ensure_command_tracking(
    adapter: S4Adapter,
    measured_q: np.ndarray,
    max_error_rad: float,
    *,
    live: bool,
) -> None:
    if not live:
        return
    published = adapter.last_published()
    if published is None or not published.motion_allowed:
        return
    error = float(
        np.max(
            np.abs(
                np.asarray(published.arm_target_q, dtype=np.float64)
                - np.asarray(measured_q, dtype=np.float64)
            )
        )
    )
    limit = float(max_error_rad)
    if error > limit:
        raise RobotTrackingError(
            f"command tracking error {error:.3f}rad exceeds {limit:.3f}rad"
        )


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
        _ensure_robot_health(bridge, hardware, live=True)
        _ensure_command_tracking(
            adapter,
            adapter.read_arm_q7(adapter.read_bimanual()),
            hardware.ik.max_proximal_tracking_error_rad,
            live=True,
        )
        command = manager.step()
        adapter.publish(command, gripper_target=0.0, quest_trigger=0.0, allow_motion=True)
        if manager.is_home(adapter.read_bimanual()):
            logger.event("home", arrived_by=manager.arrived_by)
            return
        time.sleep(control_dt)
    raise RuntimeError(f"return-home did not complete: {manager.status_line()}")


def _run_live_policy_preflight(
    *,
    cfg,
    cameras,
    bridge: HardwareRobotBridge,
    adapter: S4Adapter,
    hardware,
    gripper: BinaryGripper,
    client: AsyncPolicyClient,
    session_id: str,
    logger: RolloutLogger,
) -> None:
    now_ns = time.monotonic_ns()
    measured_q = adapter.read_arm_q7(adapter.read_bimanual())
    images, image_ts = snapshot_observation(
        cameras,
        max_age_ms=cfg.contract.max_camera_age_ms,
        max_skew_ms=cfg.contract.max_cross_camera_skew_ms,
        now_ns=now_ns,
    )
    observation = ObservationRequest(
        cfg.contract.sha256,
        f"{session_id}-preflight",
        0,
        now_ns,
        cfg.contract.task,
        cfg.contract.make_state(measured_q, gripper.state),
        image_ts,
    )
    quality = int(cfg.host["server"]["jpeg_quality"])
    if not client.submit(
        observation,
        encode_jpeg(images[0], quality),
        encode_jpeg(images[1], quality),
    ):
        raise RuntimeError("policy client refused the live preflight request")
    deadline = time.monotonic() + float(cfg.robot["network"]["connect_timeout_ms"]) / 1000.0
    while time.monotonic() < deadline:
        bridge.spin_once(timeout_sec=0.0)
        _ensure_robot_health(bridge, hardware, live=True)
        result = client.poll()
        if result is None:
            time.sleep(0.005)
            continue
        if result.error is not None:
            raise PolicyStaleError(f"live policy preflight failed: {result.error}")
        response = result.response
        assert response is not None and result.rtt_ms is not None
        observation_age_ms = (
            result.received_at_ns - result.observation.robot_timestamp_ns
        ) / 1.0e6
        if observation_age_ms < 0 or observation_age_ms > float(
            cfg.robot["freshness"]["max_chunk_age_ms"]
        ):
            raise PolicyStaleError(
                f"live policy preflight age {observation_age_ms:.1f}ms is unsafe"
            )
        if int(response.policy_fps) != cfg.contract.dataset_fps:
            raise PolicyStaleError(
                f"live policy preflight fps={response.policy_fps}, "
                f"expected={cfg.contract.dataset_fps}"
            )
        validate_policy_chunk(
            response.action_chunk,
            measured_q7=measured_q,
            max_target_jump_rad=float(
                cfg.robot["safety"]["max_policy_target_jump_rad"]
            ),
            max_tracking_error_rad=float(
                cfg.robot["safety"]["max_policy_tracking_error_rad"]
            ),
        )
        logger.event(
            "live_policy_preflight",
            rtt_ms=result.rtt_ms,
            inference_ms=response.inference_ms,
            observation_age_ms=observation_age_ms,
        )
        return
    raise PolicyStaleError("policy server did not pass live preflight before timeout")


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
    if live and hardware.startup.require_sdk_arm_replay:
        sdk_pid, sdk_executable = find_verified_arm_replay_sdk_process(
            approved_sha256=hardware.startup.approved_sdk_sha256,
        )
        print(
            f"[REAL-VLA] verified SDK arm-only replay pid={sdk_pid} "
            f"executable={sdk_executable}",
            flush=True,
        )
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
        forbidden_command_publishers=(
            ((LEG_DEPLOY_COMMAND_TOPIC, LEG_DEPLOY_PUBLISHER),) if live else ()
        ),
    )
    adapter = S4Adapter(bridge, active_arm=cfg.contract.active_arm, hands_cfg=hardware.hands)
    endpoint = f"tcp://{cfg.robot['network']['policy_server']}:{cfg.robot['network']['policy_port']}"
    client = AsyncPolicyClient(endpoint, int(cfg.robot["network"]["request_timeout_ms"]))
    buffer = ActionBuffer(
        float(cfg.robot["rollout"]["policy_hz"]),
        int(cfg.robot["rollout"]["execute_horizon"]),
        float(cfg.robot["freshness"]["max_chunk_age_ms"]),
    )
    session_id = uuid.uuid4().hex
    run_root = (
        Path.home()
        / "real_rollouts"
        / f"rollout_{time.strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"
    )
    logger = RolloutLogger(run_root, {"session_id": session_id, "live": live, "contract_sha256": cfg.contract.sha256})
    gripper = BinaryGripper(
        float(cfg.robot["gripper"]["open_threshold"]), float(cfg.robot["gripper"]["grasp_threshold"])
    )
    request_id = 0
    timeouts = 0
    live_motion_enabled = False
    started = time.monotonic()
    next_policy = started
    next_control = started
    max_runtime = float(args.max_runtime_s or cfg.robot["rollout"]["max_episode_s"])
    try:
        cameras.start()
        state_deadline = time.monotonic() + 5.0
        while not bridge.state_ready and time.monotonic() < state_deadline:
            bridge.spin_once(timeout_sec=0.05)
        if not bridge.state_ready:
            raise RuntimeError("robot state was not ready within 5 seconds")
        _ensure_robot_health(bridge, hardware, live=live)
        if live:
            _run_live_policy_preflight(
                cfg=cfg,
                cameras=cameras,
                bridge=bridge,
                adapter=adapter,
                hardware=hardware,
                gripper=gripper,
                client=client,
                session_id=session_id,
                logger=logger,
            )
            live_motion_enabled = True
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
            _ensure_robot_health(bridge, hardware, live=live)
            measured_bimanual = adapter.read_bimanual()
            measured_q = adapter.read_arm_q7(measured_bimanual)
            _ensure_command_tracking(
                adapter,
                measured_q,
                float(cfg.robot["safety"]["max_command_tracking_error_rad"]),
                live=live,
            )

            result = client.poll()
            if result is not None:
                if result.error is not None:
                    timeouts += 1
                    logger.event(
                        "policy_error",
                        request_id=result.observation.request_id,
                        error=str(result.error),
                        consecutive=timeouts,
                    )
                else:
                    response = result.response
                    assert response is not None and result.rtt_ms is not None
                    observation_age_ms = (
                        result.received_at_ns - result.observation.robot_timestamp_ns
                    ) / 1.0e6
                    if observation_age_ms < 0 or observation_age_ms > float(
                        cfg.robot["freshness"]["max_chunk_age_ms"]
                    ):
                        raise PolicyStaleError(
                            f"policy response observation age {observation_age_ms:.1f}ms exceeds "
                            f"{float(cfg.robot['freshness']['max_chunk_age_ms']):.1f}ms"
                        )
                    if int(response.policy_fps) != cfg.contract.dataset_fps:
                        raise PolicyStaleError(
                            f"policy response fps={response.policy_fps}, expected={cfg.contract.dataset_fps}"
                        )
                    safe_chunk = validate_policy_chunk(
                        response.action_chunk,
                        measured_q7=measured_q,
                        max_target_jump_rad=float(
                            cfg.robot["safety"]["max_policy_target_jump_rad"]
                        ),
                        max_tracking_error_rad=float(
                            cfg.robot["safety"]["max_policy_tracking_error_rad"]
                        ),
                    )
                    buffer.replace(
                        safe_chunk,
                        request_id=response.request_id,
                        received_at_ns=result.received_at_ns,
                        source_at_ns=result.observation.robot_timestamp_ns,
                    )
                    timeouts = 0
                    event = {
                        "request_id": response.request_id,
                        "rtt_ms": result.rtt_ms,
                        "inference_ms": response.inference_ms,
                        "observation_age_ms": observation_age_ms,
                    }
                    if not live:
                        event["execute_chunk"] = buffer.chunk.tolist()
                    logger.event("policy_response", **event)

            if timeouts > int(cfg.robot["freshness"]["max_consecutive_timeouts"]):
                raise PolicyStaleError("too many consecutive policy timeouts")

            if now >= next_policy and not client.busy:
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
                head_jpeg = encode_jpeg(
                    images[0], int(cfg.host["server"]["jpeg_quality"])
                )
                wrist_jpeg = encode_jpeg(
                    images[1], int(cfg.host["server"]["jpeg_quality"])
                )
                snapshot_interval = max(
                    int(cfg.robot.get("logging", {}).get("shadow_snapshot_every_n_requests", 10)),
                    0,
                )
                if not live and snapshot_interval and request_id % snapshot_interval == 0:
                    logger.save_observation(
                        request_id=request_id,
                        state=state.tolist(),
                        image_timestamps_ns=image_ts,
                        head_jpeg=head_jpeg,
                        wrist_jpeg=wrist_jpeg,
                    )
                if client.submit(observation, head_jpeg, wrist_jpeg):
                    request_id += 1
                    next_policy = now + 1.0 / float(cfg.robot["rollout"]["policy_hz"])
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
            unsafe_to_move = (not live_motion_enabled) or isinstance(
                exc,
                (
                    RobotStateStaleError,
                    CommandRouteConflictError,
                    CommandOutputRelinquishedError,
                    RobotTrackingError,
                ),
            )
            try:
                if unsafe_to_move:
                    bridge.relinquish_without_arm_hold(f"unsafe rollout abort: {exc}")
                elif bool(cfg.robot["rollout"]["return_home_on_abort"]):
                    _ensure_robot_health(bridge, hardware, live=True)
                    _run_home(adapter, bridge, hardware, logger)
                else:
                    bridge.hold_current_and_relinquish(f"rollout abort: {exc}")
            except Exception as home_exc:
                logger.event("home_failed", reason=str(home_exc))
                bridge.relinquish_without_arm_hold(
                    f"rollout abort/home failure: {home_exc}"
                )
        raise
    finally:
        active_failure = sys.exc_info()[0] is not None
        cleanup_errors: list[tuple[str, BaseException]] = []
        for label, close in (
            ("cameras", cameras.close),
            ("policy client", client.close),
            ("robot bridge", bridge.close),
            ("logger", logger.close),
        ):
            try:
                close()
            except BaseException as cleanup_exc:
                cleanup_errors.append((label, cleanup_exc))
        if cleanup_errors:
            details = "; ".join(f"{label}: {exc}" for label, exc in cleanup_errors)
            if active_failure:
                print(f"[REAL-VLA][CLEANUP] {details}", file=sys.stderr, flush=True)
            else:
                raise RuntimeError(f"rollout cleanup failed: {details}") from cleanup_errors[0][1]


if __name__ == "__main__":
    raise SystemExit(main())
