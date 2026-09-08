from __future__ import annotations

import argparse
from collections import deque
import math
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
    ContractError,
    PolicyStaleError,
    RobotStateStaleError,
    RobotTrackingError,
)
from ...common.protocol import ObservationRequest
from .action_buffer import ActionBuffer, sample_policy_chunk
from .command_filter import JointCommandFilter
from .execution_sync import ExecutionSyncGuard
from .logger import RolloutLogger
from .observation import encode_jpeg, snapshot_observation
from .policy_client import AsyncPolicyClient
from .safety import validate_execution_target, validate_policy_chunk


class RollingLatencyEstimator:
    def __init__(self, *, window_size: int, percentile: float) -> None:
        if int(window_size) <= 0:
            raise ValueError("RTC latency window size must be positive")
        if not 0.0 < float(percentile) <= 100.0:
            raise ValueError("RTC latency percentile must be in (0, 100]")
        self._samples: deque[float] = deque(maxlen=int(window_size))
        self.percentile = float(percentile)

    def add(self, observation_age_ms: float) -> None:
        value = float(observation_age_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("RTC observation age must be finite and non-negative")
        self._samples.append(value)

    @property
    def estimate_ms(self) -> float:
        if not self._samples:
            return 0.0
        return float(np.percentile(np.asarray(self._samples), self.percentile))

    def delay_steps(self, policy_hz: float, *, maximum: int) -> int:
        value = math.ceil(self.estimate_ms * float(policy_hz) / 1000.0)
        return max(0, min(int(value), int(maximum)))


def _configured_checkpoint(cfg) -> str:
    value = cfg.host.get("deployment", {}).get("checkpoint")
    if value is None or str(value).strip().lower() == "latest":
        raise RuntimeError("rollout requires an explicit deployment checkpoint")
    return str(Path(str(value)).expanduser().resolve())


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


def _wait_for_fresh_initial_state(
    bridge: HardwareRobotBridge,
    hardware,
    *,
    timeout_s: float = 5.0,
) -> None:
    """Require a newly processed state frame, not merely historical readiness."""
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    while time.monotonic() < deadline:
        bridge.spin_once(timeout_sec=0.05)
        if bridge.state_ready and not bridge.is_state_feed_stale(
            hardware.hardware.max_state_age_s
        ):
            return
    if not bridge.state_ready:
        raise RuntimeError("robot state was not ready within 5 seconds")
    raise RobotStateStaleError(
        f"robot state did not become fresh within {float(timeout_s):.1f} seconds"
    )


def _ensure_command_tracking(
    adapter: S4Adapter,
    measured_q: np.ndarray,
    max_error_rad: float,
    *,
    live: bool,
) -> np.ndarray | None:
    if not live:
        return None
    published = adapter.last_published()
    if published is None or not published.motion_allowed:
        return None
    error_per_joint = np.abs(
        np.asarray(published.arm_target_q, dtype=np.float64)
        - np.asarray(measured_q, dtype=np.float64)
    )
    error = float(np.max(error_per_joint))
    limit = float(max_error_rad)
    if error > limit:
        raise RobotTrackingError(
            f"command tracking error {error:.3f}rad exceeds {limit:.3f}rad"
        )
    return error_per_joint


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
        if manager.is_home(adapter.read_bimanual(), require_measured=True):
            logger.event(
                "home",
                arrived_by=manager.arrived_by,
                measured_error_rad=manager.last_measured_error,
                command_error_rad=manager.last_command_error,
            )
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
    expected_checkpoint: str,
) -> None:
    measured_q = adapter.read_arm_q7(adapter.read_bimanual())
    images, image_ts, observation_ns = snapshot_observation(
        cameras,
        max_age_ms=cfg.contract.max_camera_age_ms,
        max_skew_ms=cfg.contract.max_cross_camera_skew_ms,
    )
    observation = ObservationRequest(
        cfg.contract.sha256,
        f"{session_id}-preflight",
        0,
        observation_ns,
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
        camera_age_at_observation_ms = [
            (result.observation.robot_timestamp_ns - timestamp_ns) / 1.0e6
            for timestamp_ns in result.observation.image_timestamps_ns
        ]
        camera_capture_to_response_ms = [
            (result.received_at_ns - timestamp_ns) / 1.0e6
            for timestamp_ns in result.observation.image_timestamps_ns
        ]
        if observation_age_ms < 0 or observation_age_ms > float(
            cfg.robot["freshness"]["max_response_age_ms"]
        ):
            raise PolicyStaleError(
                f"live policy preflight age {observation_age_ms:.1f}ms is unsafe"
            )
        if int(response.policy_fps) != cfg.contract.dataset_fps:
            raise PolicyStaleError(
                f"live policy preflight fps={response.policy_fps}, "
                f"expected={cfg.contract.dataset_fps}"
            )
        expected_rtc = bool(cfg.host["server"]["rtc"]["enabled"])
        if response.rtc_enabled != expected_rtc:
            raise PolicyStaleError(
                f"policy server RTC enabled={response.rtc_enabled}, expected={expected_rtc}"
            )
        if str(Path(response.checkpoint).expanduser().resolve()) != expected_checkpoint:
            raise PolicyStaleError(
                f"policy server checkpoint={response.checkpoint!r}, expected={expected_checkpoint!r}"
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
            # When rollout starts from the reviewed deterministic home
            # trajectory, this preflight chunk is never executed and was
            # inferred from the pre-home pose. The first chunk inferred after
            # homing still receives the strict measured-state tracking check.
            enforce_initial_tracking=not bool(
                cfg.robot["rollout"]["start_from_home"]
            ),
        )
        logger.event(
            "live_policy_preflight",
            rtt_ms=result.rtt_ms,
            inference_ms=response.inference_ms,
            observation_age_ms=observation_age_ms,
            camera_age_at_observation_ms=camera_age_at_observation_ms,
            camera_capture_to_response_ms=camera_capture_to_response_ms,
            initial_tracking_deferred=bool(
                cfg.robot["rollout"]["start_from_home"]
            ),
            checkpoint=response.checkpoint,
            rtc_enabled=response.rtc_enabled,
        )
        return
    raise PolicyStaleError("policy server did not pass live preflight before timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description="S4 real policy rollout; shadow unless config and CLI both enable live")
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run live safety gates, then exit before Home or policy motion",
    )
    parser.add_argument("--max-runtime-s", type=float)
    args = parser.parse_args()
    cfg = load_pipeline_config(args.config)
    expected_checkpoint = _configured_checkpoint(cfg)
    configured_live = str(cfg.robot["rollout"]["mode"]) == "live"
    if args.live != configured_live:
        if args.live:
            raise RuntimeError("live rollout requires rollout.mode: live in robot YAML and --live")
        configured_live = False
    live = bool(args.live and configured_live)
    if args.preflight_only and not live:
        raise RuntimeError("--preflight-only requires a live robot config and --live")
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
        float(cfg.robot["safety"]["chunk_blend_duration_ms"]),
    )
    command_filter = JointCommandFilter(
        control_hz=float(cfg.robot["rollout"]["control_hz"]),
        max_velocity_rad_s=cfg.robot["safety"]["max_rollout_joint_velocity_rad_s"],
        max_acceleration_rad_s2=cfg.robot["safety"]["max_rollout_joint_acceleration_rad_s2"],
    )
    execution_sync = ExecutionSyncGuard(
        policy_lag_rad=float(cfg.robot["safety"]["rtc_reset_policy_lag_rad"]),
        contact_tracking_error_rad=float(
            cfg.robot["safety"]["contact_tracking_error_rad"]
        ),
        trigger_cycles=int(cfg.robot["safety"]["execution_sync_trigger_cycles"]),
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
    policy_rejections = 0
    total_policy_rejections = 0
    rtc_delay_estimate_steps = 0
    rtc_latency = RollingLatencyEstimator(
        window_size=int(cfg.robot["freshness"].get("rtc_latency_window_size", 30)),
        percentile=float(cfg.robot["freshness"].get("rtc_latency_percentile", 95)),
    )
    previous_accepted_request_id = -1
    rtt_samples: list[float] = []
    observation_age_samples: list[float] = []
    inference_samples: list[float] = []
    command_count = 0
    rollout_limited_count = 0
    rollout_limited_per_joint = np.zeros(7, dtype=np.int64)
    velocity_limited_count = 0
    velocity_limited_per_joint = np.zeros(7, dtype=np.int64)
    acceleration_limited_count = 0
    acceleration_limited_per_joint = np.zeros(7, dtype=np.int64)
    hardware_limited_count = 0
    gripper_transition_count = 0
    gripper_raw_max = float("-inf")
    tracking_error_samples: list[np.ndarray] = []
    last_command_request_id = -1
    deployed_checkpoint = ""
    stale_hold_request_id: int | None = None
    sync_hold_q: np.ndarray | None = None
    last_policy_execution_lag_rad: float | None = None
    rtc_resync_count = 0
    stale_response_count = 0
    live_motion_enabled = False
    started = time.monotonic()
    next_policy = started
    next_control = started
    max_runtime = float(args.max_runtime_s or cfg.robot["rollout"]["max_episode_s"])
    try:
        cameras.start()
        _wait_for_fresh_initial_state(bridge, hardware, timeout_s=5.0)
        _ensure_robot_health(bridge, hardware, live=live)
        camera_warmup_s = float(cfg.robot["rollout"].get("camera_warmup_s", 0.0))
        warmup_deadline = time.monotonic() + camera_warmup_s
        while time.monotonic() < warmup_deadline:
            bridge.spin_once(timeout_sec=0.01)
            _ensure_robot_health(bridge, hardware, live=live)
            time.sleep(0.005)
        logger.event("camera_warmup_complete", duration_s=camera_warmup_s)
        started = time.monotonic()
        next_policy = started
        next_control = started
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
                expected_checkpoint=expected_checkpoint,
            )
            if args.preflight_only:
                logger.event("complete", reason="live_preflight_only_no_motion")
                print(
                    "[REAL-VLA] live preflight PASS; no motion command was published",
                    flush=True,
                )
                return 0
            live_motion_enabled = True
        if live and bool(cfg.robot["rollout"]["start_from_home"]):
            _run_home(adapter, bridge, hardware, logger)
            started = time.monotonic()
            next_policy = started
            next_control = started
        if live:
            bridge.spin_once(timeout_sec=0.0)
            command_filter.reset(adapter.read_arm_q7(adapter.read_bimanual()))
        while time.monotonic() - started < max_runtime:
            # The next deadline is based on the previous actual publication,
            # never on a missed ideal deadline. This deliberately skips late
            # ticks instead of issuing catch-up command bursts.
            before_tick = time.monotonic()
            if before_tick < next_control:
                time.sleep(next_control - before_tick)
            bridge.spin_once(timeout_sec=0.0)
            if not bridge.state_ready:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            _ensure_robot_health(bridge, hardware, live=live)
            measured_bimanual = adapter.read_bimanual()
            measured_q = adapter.read_arm_q7(measured_bimanual)
            tracking_error = _ensure_command_tracking(
                adapter,
                measured_q,
                float(cfg.robot["safety"]["max_command_tracking_error_rad"]),
                live=live,
            )
            tracking_error_max = (
                None if tracking_error is None else float(np.max(tracking_error))
            )
            if execution_sync.observe(
                policy_execution_lag_rad=last_policy_execution_lag_rad,
                command_tracking_error_rad=tracking_error_max,
                gripper_closed=bool(gripper.state >= 0.5),
            ):
                published_for_hold = adapter.last_published()
                sync_hold_q = (
                    measured_q.copy()
                    if published_for_hold is None
                    or not published_for_hold.motion_allowed
                    else np.asarray(
                        published_for_hold.arm_target_q, dtype=np.float64
                    ).copy()
                )
                next_policy = now
                logger.event(
                    "execution_sync_hold",
                    reason=execution_sync.reason,
                    policy_execution_lag_rad=last_policy_execution_lag_rad,
                    command_tracking_error_rad=tracking_error_max,
                    gripper_closed=bool(gripper.state >= 0.5),
                    hold_q=sync_hold_q.tolist(),
                )

            result = client.poll()
            if result is not None:
                if result.error is not None:
                    timeouts += 1
                    next_policy = now
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
                    camera_age_at_observation_ms = [
                        (
                            result.observation.robot_timestamp_ns
                            - timestamp_ns
                        )
                        / 1.0e6
                        for timestamp_ns in result.observation.image_timestamps_ns
                    ]
                    camera_capture_to_response_ms = [
                        (result.received_at_ns - timestamp_ns) / 1.0e6
                        for timestamp_ns in result.observation.image_timestamps_ns
                    ]
                    if observation_age_ms < 0 or observation_age_ms > float(
                        cfg.robot["freshness"]["max_response_age_ms"]
                    ):
                        timeouts += 1
                        stale_response_count += 1
                        next_policy = now
                        logger.event(
                            "stale_rejected",
                            request_id=response.request_id,
                            observation_age_ms=observation_age_ms,
                            max_response_age_ms=float(
                                cfg.robot["freshness"]["max_response_age_ms"]
                            ),
                            consecutive=timeouts,
                        )
                        if timeouts > int(
                            cfg.robot["freshness"]["max_consecutive_timeouts"]
                        ):
                            raise PolicyStaleError(
                                "too many consecutive stale/timeout policy responses"
                            )
                        # The old accepted chunk remains active. Retry on a clean
                        # request without ever inserting this stale response.
                        next_control = time.monotonic()
                        continue
                    if int(response.policy_fps) != cfg.contract.dataset_fps:
                        raise PolicyStaleError(
                            f"policy response fps={response.policy_fps}, expected={cfg.contract.dataset_fps}"
                        )
                    expected_rtc = bool(cfg.host.get("server", {}).get("rtc", {}).get("enabled", False))
                    if response.rtc_enabled != expected_rtc:
                        raise PolicyStaleError(
                            f"policy server RTC enabled={response.rtc_enabled}, expected={expected_rtc}"
                        )
                    if (
                        result.observation.rtc_reset_history
                        and not response.rtc_history_reset
                    ):
                        raise PolicyStaleError(
                            "policy server did not acknowledge requested RTC history reset"
                        )
                    if str(Path(response.checkpoint).expanduser().resolve()) != expected_checkpoint:
                        raise PolicyStaleError(
                            f"policy server checkpoint={response.checkpoint!r}, "
                            f"expected={expected_checkpoint!r}"
                        )
                    rtc_latency.add(observation_age_ms)
                    rtc_delay_estimate_steps = rtc_latency.delay_steps(
                        cfg.contract.dataset_fps,
                        maximum=int(cfg.host["server"]["rtc"]["execution_horizon"]),
                    )
                    timeouts = 0
                    policy_gripper_preclip = np.asarray(response.action_chunk[:, 7]).copy()
                    response_tracking_limit = float(
                        cfg.robot["safety"][
                            "resync_target_error_rad"
                            if result.observation.rtc_reset_history
                            else "max_policy_tracking_error_rad"
                        ]
                    )
                    try:
                        safe_chunk = validate_policy_chunk(
                            response.action_chunk,
                            # Chunk step zero belongs to the request observation,
                            # not to the later response-receipt state.
                            measured_q7=result.observation.state[:7],
                            max_target_jump_rad=float(
                                cfg.robot["safety"]["max_policy_target_jump_rad"]
                            ),
                            max_tracking_error_rad=response_tracking_limit,
                        )
                        execution_target = sample_policy_chunk(
                            safe_chunk[
                                : int(cfg.robot["rollout"]["execute_horizon"])
                            ],
                            policy_hz=float(cfg.robot["rollout"]["policy_hz"]),
                            source_at_ns=result.observation.robot_timestamp_ns,
                            now_ns=result.received_at_ns,
                        )
                        validate_execution_target(
                            execution_target[:7],
                            measured_q7=measured_q,
                            max_tracking_error_rad=response_tracking_limit,
                        )
                    except ContractError as action_exc:
                        policy_rejections += 1
                        total_policy_rejections += 1
                        execution_sync.force_resync("unsafe_policy_chunk")
                        published_for_hold = adapter.last_published()
                        sync_hold_q = (
                            measured_q.copy()
                            if published_for_hold is None
                            or not published_for_hold.motion_allowed
                            else np.asarray(
                                published_for_hold.arm_target_q,
                                dtype=np.float64,
                            ).copy()
                        )
                        # Retry on the next control iteration instead of waiting
                        # for the normal replan interval while the old plan ages.
                        next_policy = now
                        logger.event(
                            "policy_rejected",
                            request_id=response.request_id,
                            reason=str(action_exc),
                            consecutive=policy_rejections,
                            rtt_ms=result.rtt_ms,
                            observation_age_ms=observation_age_ms,
                            rtc_reset_pending=True,
                            hold_q=sync_hold_q.tolist(),
                        )
                        if policy_rejections > int(
                            cfg.robot["freshness"]["max_consecutive_policy_rejections"]
                        ):
                            raise ContractError(
                                "too many consecutive unsafe policy chunks: "
                                f"last={action_exc}"
                            ) from action_exc
                    else:
                        previous_request_id = buffer.request_id
                        published_before_replace = adapter.last_published()
                        transition_from_q7 = (
                            measured_q
                            if published_before_replace is None
                            or not published_before_replace.motion_allowed
                            else published_before_replace.arm_target_q
                        )
                        buffer.replace(
                            safe_chunk,
                            request_id=response.request_id,
                            received_at_ns=result.received_at_ns,
                            source_at_ns=result.observation.robot_timestamp_ns,
                            transition_from_q7=transition_from_q7,
                        )
                        previous_accepted_request_id = response.request_id
                        if result.observation.rtc_reset_history:
                            execution_sync.acknowledge_resync()
                            sync_hold_q = None
                            rtc_resync_count += 1
                            logger.event(
                                "execution_sync_recovered",
                                request_id=response.request_id,
                                execution_target_error_rad=float(
                                    np.max(np.abs(execution_target[:7] - measured_q))
                                ),
                            )
                        deployed_checkpoint = response.checkpoint
                        rtt_samples.append(float(result.rtt_ms))
                        observation_age_samples.append(float(observation_age_ms))
                        inference_samples.append(float(response.inference_ms))
                        if response.rtc_enabled:
                            logger.event(
                                "rtc_chunk",
                                old_request_id=previous_request_id,
                                new_request_id=response.request_id,
                                delay_steps=response.rtc_inference_delay_steps,
                                old_remaining_steps=response.rtc_prev_leftover_steps,
                                old_raw_remaining_steps=response.rtc_prev_raw_remaining_steps,
                                source_request_id=response.rtc_source_request_id,
                                elapsed_policy_position=response.rtc_elapsed_policy_position,
                                leftover_start_index=response.rtc_leftover_start_index,
                                first_target_jump_rad=float(
                                    np.max(np.abs(execution_target[:7] - transition_from_q7))
                                ),
                                first_target_jump_per_joint=np.abs(
                                    execution_target[:7] - transition_from_q7
                                ).tolist(),
                                rtc_guided=response.rtc_prev_leftover_steps > 0,
                                rtc_history_reset=response.rtc_history_reset,
                                ack_simulated=not live,
                            )
                        policy_rejections = 0
                        stale_hold_request_id = None
                        event = {
                            "request_id": response.request_id,
                            "rtt_ms": result.rtt_ms,
                            "inference_ms": response.inference_ms,
                            "observation_age_ms": observation_age_ms,
                            "camera_age_at_observation_ms": camera_age_at_observation_ms,
                            "camera_capture_to_response_ms": camera_capture_to_response_ms,
                            "rtc_enabled": response.rtc_enabled,
                            "rtc_inference_delay_steps": response.rtc_inference_delay_steps,
                            "rtc_execution_horizon": response.rtc_execution_horizon,
                            "rtc_prev_leftover_steps": response.rtc_prev_leftover_steps,
                            "rtc_prev_raw_remaining_steps": response.rtc_prev_raw_remaining_steps,
                            "rtc_source_request_id": response.rtc_source_request_id,
                            "rtc_elapsed_policy_position": response.rtc_elapsed_policy_position,
                            "rtc_leftover_start_index": response.rtc_leftover_start_index,
                            "rtc_history_reset": response.rtc_history_reset,
                            "rtc_reset_requested": result.observation.rtc_reset_history,
                            "execution_lag_at_request_rad": result.observation.execution_lag_rad,
                            "rtc_latency_window_p95_ms": rtc_latency.estimate_ms,
                            "rtc_next_delay_steps": rtc_delay_estimate_steps,
                            "raw_chunk_length": response.raw_chunk_length,
                            "physical_chunk_length": int(response.action_chunk.shape[0]),
                            "checkpoint": response.checkpoint,
                            "policy_gripper_preclip_min": float(np.min(policy_gripper_preclip)),
                            "policy_gripper_preclip_max": float(np.max(policy_gripper_preclip)),
                        }
                        if not live:
                            event["execute_chunk"] = buffer.chunk.tolist()
                        logger.event("policy_response", **event)

            if timeouts > int(cfg.robot["freshness"]["max_consecutive_timeouts"]):
                raise PolicyStaleError("too many consecutive policy timeouts")

            if now >= next_policy and not client.busy:
                images, image_ts, observation_ns = snapshot_observation(
                    cameras,
                    max_age_ms=cfg.contract.max_camera_age_ms,
                    max_skew_ms=cfg.contract.max_cross_camera_skew_ms,
                )
                state = cfg.contract.make_state(measured_q, gripper.state)
                observation = ObservationRequest(
                    cfg.contract.sha256,
                    session_id,
                    request_id,
                    observation_ns,
                    cfg.contract.task,
                    state,
                    image_ts,
                    rtc_delay_estimate_steps,
                    previous_accepted_request_id,
                    execution_sync.reset_pending,
                    float(last_policy_execution_lag_rad or 0.0),
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
                    next_policy = now + float(
                        cfg.robot["rollout"]["replan_interval_steps"]
                    ) / float(cfg.robot["rollout"]["policy_hz"])
            if live:
                if buffer.chunk is None:
                    if now - started > float(cfg.robot["network"]["connect_timeout_ms"]) / 1000.0:
                        raise PolicyStaleError("policy server did not provide an initial action chunk")
                    time.sleep(0.005)
                    continue
                sample_ns = time.monotonic_ns()
                chunk_age_ms = buffer.age_ms(sample_ns)
                if chunk_age_ms > float(cfg.robot["freshness"]["max_chunk_age_ms"]):
                    raise PolicyStaleError(
                        f"policy recovery age {chunk_age_ms:.1f}ms exceeds "
                        f"{float(cfg.robot['freshness']['max_chunk_age_ms']):.1f}ms"
                    )
                if execution_sync.hold and sync_hold_q is not None:
                    action = np.concatenate(
                        [
                            np.asarray(sync_hold_q, dtype=np.float32),
                            [float(gripper.state)],
                        ]
                    )
                elif chunk_age_ms > float(
                    cfg.robot["freshness"]["max_motion_chunk_age_ms"]
                ):
                    published_hold = adapter.last_published()
                    hold_q = (
                        measured_q
                        if published_hold is None or not published_hold.motion_allowed
                        else published_hold.arm_target_q
                    )
                    action = np.concatenate(
                        [np.asarray(hold_q, dtype=np.float32), [float(gripper.state)]]
                    )
                    if stale_hold_request_id != buffer.request_id:
                        logger.event(
                            "policy_stale_hold",
                            request_id=buffer.request_id,
                            chunk_age_ms=chunk_age_ms,
                        )
                        stale_hold_request_id = buffer.request_id
                else:
                    action = buffer.sample(sample_ns)
                filtered_q = command_filter.step(action[:7])
                rollout_limited = not np.allclose(filtered_q, action[:7], atol=1.0e-9)
                limited_joints = command_filter.limited_joints
                velocity_limited_joints = command_filter.velocity_limited_joints
                acceleration_limited_joints = (
                    command_filter.acceleration_limited_joints
                )
                gripper_before = float(gripper.state)
                policy_gripper_raw = float(action[7])
                gripper_target = gripper.update(policy_gripper_raw)
                gripper_after = float(gripper.state)
                command = adapter.overlay_active_arm(measured_bimanual, filtered_q)
                published_before_command = adapter.last_published()
                continuity_reference = (
                    measured_q
                    if published_before_command is None
                    or not published_before_command.motion_allowed
                    else np.asarray(published_before_command.arm_target_q, dtype=np.float64)
                )
                published = adapter.publish(command, gripper_target=gripper_target, quest_trigger=0.0, allow_motion=True)
                is_chunk_boundary = buffer.request_id != last_command_request_id
                policy_jump = np.abs(np.asarray(action[:7]) - continuity_reference)
                filtered_jump = np.abs(np.asarray(filtered_q) - continuity_reference)
                published_jump = np.abs(
                    np.asarray(published.arm_target_q, dtype=np.float64) - continuity_reference
                )
                last_policy_execution_lag_rad = float(
                    np.max(
                        np.abs(
                            np.asarray(action[:7], dtype=np.float64)
                            - np.asarray(published.arm_target_q, dtype=np.float64)
                        )
                    )
                )
                command_count += 1
                rollout_limited_count += int(rollout_limited)
                rollout_limited_per_joint += limited_joints.astype(np.int64)
                velocity_limited_count += int(np.any(velocity_limited_joints))
                velocity_limited_per_joint += velocity_limited_joints.astype(np.int64)
                acceleration_limited_count += int(
                    np.any(acceleration_limited_joints)
                )
                acceleration_limited_per_joint += (
                    acceleration_limited_joints.astype(np.int64)
                )
                hardware_limited_count += int(published.limited)
                gripper_transition_count += int(gripper_before != gripper_after)
                gripper_raw_max = max(gripper_raw_max, policy_gripper_raw)
                if tracking_error is not None:
                    tracking_error_samples.append(tracking_error.copy())
                logger.event(
                    "command",
                    request_id=buffer.request_id,
                    target=published.as_8d().tolist(),
                    policy_target=action.tolist(),
                    rollout_limited=rollout_limited,
                    measured_q=np.asarray(measured_q).tolist(),
                    command_tracking_error_per_joint=(
                        None if tracking_error is None else tracking_error.tolist()
                    ),
                    command_tracking_error_max=(
                        None if tracking_error is None else float(np.max(tracking_error))
                    ),
                    rollout_limited_joints=limited_joints.tolist(),
                    velocity_limited_joints=velocity_limited_joints.tolist(),
                    acceleration_limited_joints=(
                        acceleration_limited_joints.tolist()
                    ),
                    policy_execution_lag_rad=last_policy_execution_lag_rad,
                    execution_sync_hold=execution_sync.hold,
                    rtc_reset_pending=execution_sync.reset_pending,
                    hardware_limited=published.limited,
                    chunk_boundary=is_chunk_boundary,
                    boundary_policy_jump_per_joint=policy_jump.tolist(),
                    boundary_filtered_jump_per_joint=filtered_jump.tolist(),
                    boundary_published_jump_per_joint=published_jump.tolist(),
                    policy_gripper_raw=policy_gripper_raw,
                    gripper_state_before=gripper_before,
                    gripper_state_after=gripper_after,
                    gripper_transition=gripper_before != gripper_after,
                    published_gripper_target=float(gripper_target),
                )
                last_command_request_id = buffer.request_id
            next_control = time.monotonic() + 1.0 / float(
                cfg.robot["rollout"]["control_hz"]
            )
        if live and bool(cfg.robot["rollout"]["return_home_on_finish"]):
            _run_home(adapter, bridge, hardware, logger)
        logger.event(
            "complete",
            reason="max_episode_s",
            checkpoint=deployed_checkpoint,
            mean_rtt_ms=float(np.mean(rtt_samples)) if rtt_samples else None,
            p95_rtt_ms=float(np.percentile(rtt_samples, 95)) if rtt_samples else None,
            mean_inference_ms=float(np.mean(inference_samples)) if inference_samples else None,
            mean_observation_age_ms=float(np.mean(observation_age_samples)) if observation_age_samples else None,
            rollout_limited_ratio=rollout_limited_count / command_count if command_count else None,
            rollout_limited_ratio_per_joint=(
                (rollout_limited_per_joint / command_count).tolist() if command_count else None
            ),
            velocity_limited_ratio=(
                velocity_limited_count / command_count if command_count else None
            ),
            velocity_limited_ratio_per_joint=(
                (velocity_limited_per_joint / command_count).tolist()
                if command_count
                else None
            ),
            acceleration_limited_ratio=(
                acceleration_limited_count / command_count if command_count else None
            ),
            acceleration_limited_ratio_per_joint=(
                (acceleration_limited_per_joint / command_count).tolist()
                if command_count
                else None
            ),
            hardware_limited_ratio=hardware_limited_count / command_count if command_count else None,
            mean_tracking_error_per_joint=(
                np.mean(np.stack(tracking_error_samples), axis=0).tolist()
                if tracking_error_samples
                else None
            ),
            p95_tracking_error_per_joint=(
                np.percentile(np.stack(tracking_error_samples), 95, axis=0).tolist()
                if tracking_error_samples
                else None
            ),
            p95_tracking_error_max=(
                float(np.percentile(np.max(np.stack(tracking_error_samples), axis=1), 95))
                if tracking_error_samples
                else None
            ),
            gripper_raw_max=gripper_raw_max if np.isfinite(gripper_raw_max) else None,
            gripper_transitions=gripper_transition_count,
            replans=len(rtt_samples),
            rejected_chunks=total_policy_rejections,
            stale_responses=stale_response_count,
            rtc_resyncs=rtc_resync_count,
        )
        return 0
    except BaseException as exc:
        logger.event(
            "abort",
            reason=str(exc),
            checkpoint=deployed_checkpoint,
            command_count=command_count,
            mean_observation_age_ms=(
                float(np.mean(observation_age_samples))
                if observation_age_samples
                else None
            ),
            rtc_latency_window_p95_ms=rtc_latency.estimate_ms,
            rollout_limited_ratio=(
                rollout_limited_count / command_count if command_count else None
            ),
            rollout_limited_ratio_per_joint=(
                (rollout_limited_per_joint / command_count).tolist()
                if command_count
                else None
            ),
            velocity_limited_ratio=(
                velocity_limited_count / command_count if command_count else None
            ),
            acceleration_limited_ratio=(
                acceleration_limited_count / command_count if command_count else None
            ),
            hardware_limited_ratio=(
                hardware_limited_count / command_count if command_count else None
            ),
            p95_tracking_error_max=(
                float(
                    np.percentile(
                        np.max(np.stack(tracking_error_samples), axis=1), 95
                    )
                )
                if tracking_error_samples
                else None
            ),
            gripper_raw_max=(
                gripper_raw_max if np.isfinite(gripper_raw_max) else None
            ),
            gripper_transitions=gripper_transition_count,
            replans=len(rtt_samples),
            rejected_chunks=total_policy_rejections,
            stale_responses=stale_response_count,
            rtc_resyncs=rtc_resync_count,
            execution_sync_hold=execution_sync.hold,
            execution_sync_reason=execution_sync.reason or None,
        )
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
