#!/usr/bin/env python
"""Quest hardware teleoperation using vendored Pink without Isaac Sim."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hardware_teleop.config_loader import (  # noqa: E402
    HardwareTeleopConfig,
    load_hardware_teleop_config,
)
from hardware_teleop.hand_mapping import command_hand_trigger  # noqa: E402
from hardware_teleop.ik import create_pure_hardware_ik_backend  # noqa: E402
from hardware_teleop.joint_mapping import (  # noqa: E402
    apply_arm_q14,
    bimanual_to_arm_q14,
)
from hardware_teleop.ros import HardwareRobotBridge, RosImportError  # noqa: E402
from hardware_teleop.replay import PinkStateRecorder  # noqa: E402
from hardware_teleop.safety import (  # noqa: E402
    TeleopFaultLatch,
    find_verified_arm_replay_sdk_process,
)
from hardware_teleop.hooks import TeleopHooks, TeleopTick, TickRequest  # noqa: E402
from hardware_teleop.startup import run_startup_homing  # noqa: E402
from s4_pipeline.config import load_project_config  # noqa: E402
from s4_robot.control_mapping import ACTION_SLICES  # noqa: E402
from teleoperation.common import (  # noqa: E402
    detect_lan_ip,
    load_task_control_profiles,
    smooth_command,
)
from teleoperation.mapping import BimanualTeleopMapper  # noqa: E402
from teleoperation.protocol import LatestFrameStore  # noqa: E402
from teleoperation.server import QuestWebServer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the real S4 robot with Quest 3 and Pink IK (no Isaac)."
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=PROJECT_ROOT / "hardware_teleop/config/quest_hardware.yaml",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cert", type=Path, default=PROJECT_ROOT / ".local/teleoperation/cert.pem")
    parser.add_argument("--key", type=Path, default=PROJECT_ROOT / ".local/teleoperation/key.pem")
    parser.add_argument("--insecure-http", action="store_true")
    parser.add_argument("--report-period-s", type=float, default=0.5)
    parser.add_argument("--max-runtime-s", type=float, default=0.0)
    parser.add_argument("--input-debug", action="store_true")
    parser.add_argument(
        "--ik-backend",
        choices=("pink",),
        default=None,
        help="Pure hardware runtime currently supports only Pink.",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Read state and solve IK, but create no arm or hand command publishers.",
    )
    parser.add_argument(
        "--arm-output",
        action="store_true",
        help="Explicitly confirm that the supported real-robot command test may be armed.",
    )
    parser.add_argument(
        "--enabled-arms",
        choices=("left", "right", "both"),
        default="both",
        help="Limit which arm can move during staged commissioning.",
    )
    hands = parser.add_mutually_exclusive_group()
    hands.add_argument(
        "--enable-hands",
        action="store_true",
        help="Enable hand commands. quest_hardware.yaml hands.enabled also turns them on; --disable-hands overrides.",
    )
    hands.add_argument("--disable-hands", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-homing", action="store_true")
    parser.add_argument(
        "--max-arm-step-rad",
        type=float,
        default=None,
        help="Optional stricter per-cycle arm step for staged hardware tests.",
    )
    parser.add_argument(
        "--record-state-jsonl",
        type=Path,
        default=None,
        help="Record q14, FK, targets and commands for offline Pink replay.",
    )
    parser.add_argument("--overwrite-state-log", action="store_true")
    return parser


def _arm_enabled(mode: str, side: str) -> bool:
    return mode == "both" or mode == side


def _max_active_proximal_tracking_error(
    actual_q14: np.ndarray,
    command_q14: np.ndarray,
    *,
    left_active: bool,
    right_active: bool,
) -> float:
    """Return max shoulder/elbow command-feedback error for active arms."""
    actual = np.asarray(actual_q14, dtype=np.float64)
    command = np.asarray(command_q14, dtype=np.float64)
    errors: list[float] = []
    if left_active:
        errors.append(float(np.max(np.abs(command[:4] - actual[:4]))))
    if right_active:
        errors.append(float(np.max(np.abs(command[7:11] - actual[7:11]))))
    return max(errors, default=0.0)


def _validate_no_isaac_imports() -> None:
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "isaaclab"
        or name.startswith("isaaclab.")
        or name == "isaacsim"
        or name.startswith("isaacsim.")
        or name == "omni"
        or name.startswith("omni.")
    )
    if forbidden:
        raise RuntimeError(
            "pure Pink hardware runtime imported forbidden heavy modules: "
            + ", ".join(forbidden[:10])
        )


def _close_failed_initialization(
    bridge: HardwareRobotBridge,
    *,
    command_output_enabled: bool,
) -> None:
    """Release the ROS node without emitting an arm command packet."""
    del command_output_enabled
    bridge.close()


def run_pink_hardware_teleop(
    config: HardwareTeleopConfig,
    args: argparse.Namespace,
    hooks: TeleopHooks | None = None,
) -> None:
    _validate_no_isaac_imports()
    from hardware_teleop.env_check import validate_runtime_imports

    validate_runtime_imports(
        robot_profile=os.environ.get("S4_HW_TELEOP_RUNTIME_RESOLVED") == "system"
    )
    pink_cfg = replace(
        config.teleop.controller.pink,
        position_cost=config.hardware.commissioning_position_cost,
        max_joint_velocity_rad_s=config.ik.max_joint_velocity_rad_s,
        orientation_cost=(
            config.hardware.commissioning_orientation_cost
            if config.hardware.commissioning_orientation_enabled
            else 0.0
        ),
        elbow_avoidance=replace(
            config.teleop.controller.pink.elbow_avoidance,
            enabled=config.hardware.commissioning_elbow_avoidance_enabled,
            min_lateral_distance_base_m=(
                config.hardware.commissioning_elbow_min_lateral_distance_base_m
            ),
        ),
    )
    teleop_cfg = replace(
        config.teleop,
        network=replace(
            config.teleop.network,
            stale_timeout_s=config.hardware.input_stale_timeout_s,
        ),
        safety=replace(
            config.teleop.safety,
            workspace_min=np.asarray(config.hardware.commissioning_workspace_min),
            workspace_max=np.asarray(config.hardware.commissioning_workspace_max),
            max_translation_speed_m_s=config.hardware.max_tcp_translation_speed_m_s,
            max_rotation_speed_rad_s=config.hardware.max_tcp_rotation_speed_rad_s,
        ),
        mapping=replace(
            config.teleop.mapping,
            position_scale=config.hardware.commissioning_position_scale,
            orientation_enabled=config.hardware.commissioning_orientation_enabled,
            max_clutch_translation_m=config.hardware.commissioning_max_clutch_translation_m,
            controller_filter_time_constant_s=config.hardware.commissioning_input_filter_tau_s,
            invert_translation=config.hardware.commissioning_invert_translation,
            invert_orientation=config.hardware.commissioning_invert_orientation,
            translation_sign=config.hardware.commissioning_translation_sign,
        ),
        controller=replace(config.teleop.controller, pink=pink_cfg),
    )
    config = replace(config, teleop=teleop_cfg)
    configured_max_arm_step = min(
        teleop_cfg.smoothing.arm_max_joint_step_rad,
        config.hardware.max_joint_step_rad,
    )
    max_arm_step = (
        configured_max_arm_step
        if args.max_arm_step_rad is None
        else float(args.max_arm_step_rad)
    )
    if not 0.0 < max_arm_step <= configured_max_arm_step:
        raise ValueError(
            "--max-arm-step-rad must be positive and no larger than the configured "
            f"hardware limit {configured_max_arm_step:.6f} rad"
        )
    project = load_project_config()
    profiles, home_poses = load_task_control_profiles(project.dataset.task_id)
    mapper = BimanualTeleopMapper(teleop_cfg, **profiles, require_calibration=True)

    if not args.shadow and not args.arm_output:
        raise RuntimeError(
            "real-robot command output is disarmed by default; complete the online doctor "
            "and shadow test, support the robot, then restart with --arm-output"
        )

    cert = None if args.insecure_http else args.cert.resolve()
    key = None if args.insecure_http else args.key.resolve()
    if cert is not None and (not cert.is_file() or not key.is_file()):
        raise FileNotFoundError(
            f"Quest WebXR HTTPS certificate missing: cert={cert} key={key}. "
            "Run: bash run.sh teleop-cert"
        )

    host = args.host or teleop_cfg.network.host
    port = int(args.port or teleop_cfg.network.port)
    store = LatestFrameStore()
    server = QuestWebServer(store, host, port, cert, key, PROJECT_ROOT / "teleoperation/webxr")

    if not args.shadow and config.startup.require_sdk_arm_replay:
        sdk_pid, sdk_executable = find_verified_arm_replay_sdk_process(
            approved_sha256=config.startup.approved_sdk_sha256,
        )
        print(
            f"[HW-PINK] verified SDK arm-only replay: pid={sdk_pid} executable={sdk_executable}",
            flush=True,
        )
    try:
        bridge = HardwareRobotBridge(
            config.hardware,
            config.hands,
            gravity_cfg=config.gravity,
            project_root=config.project_root,
            check_arm_command_publishers=config.startup.check_arm_command_publishers,
            command_output_enabled=not args.shadow,
        )
    except RosImportError as exc:
        raise RuntimeError(str(exc)) from exc

    control_dt = 1.0 / float(config.hardware.control_rate_hz)
    hands_enabled = (config.hands.enabled or args.enable_hands) and not args.disable_hands
    print(
        f"[HW-PINK] runtime=hardware_no_isaac control_rate_hz={config.hardware.control_rate_hz:.1f} "
        f"state={config.hardware.state_source} shadow={args.shadow} enabled_arms={args.enabled_arms} "
        f"max_arm_step={max_arm_step:.4f}rad "
        f"ik_joint_speed={config.ik.max_joint_velocity_rad_s:.2f}rad/s "
        f"clutch_cap={config.hardware.commissioning_max_clutch_translation_m:.2f}m "
        f"tcp_speed={config.hardware.max_tcp_translation_speed_m_s:.2f}m/s "
        f"rot_speed={config.hardware.max_tcp_rotation_speed_rad_s:.2f}rad/s "
        f"input_filter_tau={config.hardware.commissioning_input_filter_tau_s:.3f}s "
        f"elbow_avoidance={int(pink_cfg.elbow_avoidance.enabled)} "
        f"elbow_min_y={pink_cfg.elbow_avoidance.min_lateral_distance_base_m:.2f}m "
        f"orientation={int(config.hardware.commissioning_orientation_enabled)} "
        f"orientation_cost={pink_cfg.orientation_cost:.2f} "
        f"position_cost={pink_cfg.position_cost:.2f} "
        f"joint_limit_cost={config.ik.joint_limit_avoidance_cost:.4f} "
        f"wrist_py_cost={config.ik.wrist_pitch_yaw_posture_cost:.4f} "
        f"wrist_py_speed={config.ik.wrist_pitch_yaw_max_velocity_rad_s:.2f}rad/s "
        f"invert_t={int(config.hardware.commissioning_invert_translation)} "
        f"invert_r={int(config.hardware.commissioning_invert_orientation)} "
        f"t_sign=({config.hardware.commissioning_translation_sign[0]:+.0f},"
        f"{config.hardware.commissioning_translation_sign[1]:+.0f},"
        f"{config.hardware.commissioning_translation_sign[2]:+.0f}) "
        f"arm_topic={config.hardware.arm_command_topic} "
        f"mode_ctrl={config.hardware.arm_command_mode_ctrl} "
        f"hands={int(hands_enabled)} "
        f"hands_topic={config.hardware.hands_cmd_topic}",
        flush=True,
    )

    if config.hardware.require_initial_state:
        print(
            f"[HW-PINK] waiting for {config.hardware.state_source} "
            f"(timeout={config.hardware.initial_state_timeout_s:.1f}s)...",
            flush=True,
        )
        try:
            bridge.wait_for_initial_state(config.hardware.initial_state_timeout_s)
        except BaseException:
            _close_failed_initialization(
                bridge,
                command_output_enabled=not args.shadow,
            )
            raise

    try:
        if args.shadow or args.skip_homing:
            command_action = bridge.read_bimanual_state()
            reason = "shadow" if args.shadow else "explicit"
            print(f"[HW-PINK] startup homing skipped ({reason})", flush=True)
        else:
            command_action = run_startup_homing(
                startup_cfg=config.startup,
                control_dt=control_dt,
                read_state=bridge.read_bimanual_state,
                publish_step=lambda action: bridge.publish_arm_command(action, allow_motion=True),
                spin_once=lambda: bridge.spin_once(timeout_sec=0.0),
                home_poses=home_poses,
                profiles=profiles,
            )

        bridge.update_hand_state_from_rad(
            command_action[ACTION_SLICES.left_hand],
            command_action[ACTION_SLICES.right_hand],
        )
        ik_backend = create_pure_hardware_ik_backend(config)
        initial_q14 = bimanual_to_arm_q14(command_action)
        ik_backend.set_posture_reference(initial_q14)
        left_tcp, right_tcp = ik_backend.forward(initial_q14)
        if not np.isfinite(left_tcp.position).all() or not np.isfinite(right_tcp.position).all():
            raise RuntimeError("Pink FK produced a non-finite initial TCP pose")
    except BaseException:
        _close_failed_initialization(
            bridge,
            command_output_enabled=not args.shadow,
        )
        raise
    print(f"[HW-PINK][IK] details={ik_backend.diagnostics()}", flush=True)
    print(
        f"[HW-PINK][FK] initial_tcp_L={tuple(np.round(left_tcp.position, 4))} "
        f"initial_tcp_R={tuple(np.round(right_tcp.position, 4))}",
        flush=True,
    )
    recorder = None
    try:
        if args.record_state_jsonl is not None:
            recorder = PinkStateRecorder(
                args.record_state_jsonl,
                overwrite=args.overwrite_state_log,
            )
            print(f"[HW-PINK] recording state replay: {recorder.path}", flush=True)
        server.start()
    except BaseException:
        if recorder is not None:
            recorder.close()
        server.close()
        _close_failed_initialization(
            bridge,
            command_output_enabled=not args.shadow,
        )
        raise
    scheme = "http" if args.insecure_http else "https"
    print("\n[HW-PINK] Meta Quest controller server ready", flush=True)
    print(f"[HW-PINK] Quest URL: {scheme}://{detect_lan_ip()}:{port}", flush=True)
    if args.shadow:
        print("[HW-PINK][SHADOW] no arm or hand command publisher was created", flush=True)

    start_time = time.monotonic()
    next_deadline = start_time
    last_mapping_time = start_time
    last_report = start_time
    report_start_time = start_time
    report_steps = 0
    fault = TeleopFaultLatch()
    try:
        while True:
            now = time.monotonic()
            if args.max_runtime_s > 0.0 and now - start_time >= args.max_runtime_s:
                break
            bridge.spin_once(timeout_sec=0.0)
            if not bridge.state_ready:
                time.sleep(0.01)
                continue

            mapping_dt = float(np.clip(now - last_mapping_time, control_dt, 0.05))
            last_mapping_time = now
            actual_action = bridge.read_bimanual_state()
            actual_q14 = bimanual_to_arm_q14(actual_action)
            command_q14 = bimanual_to_arm_q14(command_action)
            left_tcp, right_tcp = ik_backend.forward(actual_q14)
            # Clutch must re-anchor to the held command pose, not the measured TCP.
            # Without this, gravity sag between clutches becomes the new target and
            # each Grip press ratchets the arm downward.
            hold_left_tcp, hold_right_tcp = ik_backend.forward(command_q14)
            frame = store.snapshot()
            request = TickRequest()
            if hooks is not None:
                request = hooks.begin_tick(frame, now)
                if request.stop:
                    break
                if request.hud:
                    server.send_status({"type": "hud", **request.hud})
            mapped = mapper.update(frame, hold_left_tcp, hold_right_tcp, mapping_dt, now)
            state_feed_stale = bridge.is_state_feed_stale(config.hardware.max_state_age_s)
            arm_command_graph_conflict = bridge.is_arm_command_graph_conflicted()
            state_feed_ok = not state_feed_stale and not arm_command_graph_conflict

            if state_feed_stale:
                if fault.trip(
                    f"robot state stale for {bridge.last_state_age_s:.3f}s"
                ):
                    print(
                        "[HW-PINK][SAFETY] robot state feed stale; arm output stopped",
                        flush=True,
                    )
            elif arm_command_graph_conflict:
                if fault.trip("external arm-replay publisher detected"):
                    bridge.relinquish_without_arm_hold(fault.reason)
                    print(
                        "[HW-PINK][SAFETY] external arm-replay source detected; "
                        "arm output stopped",
                        flush=True,
                    )
                    raise RuntimeError(
                        "external arm-replay publisher detected; restart only after it is stopped"
                    )

            if fault.clear_if_released(
                left_clutch=mapped.left.clutch,
                right_clutch=mapped.right.clutch,
                state_feed_ok=state_feed_ok,
            ):
                ik_backend.set_posture_reference(command_q14)
                print("[HW-PINK][SAFETY] fault cleared after both grips were released", flush=True)

            left_active = (
                request.allow_teleop
                and mapped.left.clutch
                and _arm_enabled(args.enabled_arms, "left")
                and not mapped.stale
                and state_feed_ok
                and not fault.active
            )
            right_active = (
                request.allow_teleop
                and mapped.right.clutch
                and _arm_enabled(args.enabled_arms, "right")
                and not mapped.stale
                and state_feed_ok
                and not fault.active
            )

            proximal_tracking_error = _max_active_proximal_tracking_error(
                actual_q14,
                command_q14,
                left_active=left_active,
                right_active=right_active,
            )
            if proximal_tracking_error > config.ik.max_proximal_tracking_error_rad:
                if fault.trip(
                    "proximal command-feedback error "
                    f"{proximal_tracking_error:.3f} rad exceeds "
                    f"{config.ik.max_proximal_tracking_error_rad:.3f} rad"
                ):
                    print(
                        f"[HW-PINK][SAFETY] {fault.reason}; release both grips "
                        "and wait for the arm to catch up",
                        flush=True,
                    )
                left_active = False
                right_active = False

            if (mapped.left.clutch_rising and _arm_enabled(args.enabled_arms, "left")) or (
                mapped.right.clutch_rising and _arm_enabled(args.enabled_arms, "right")
            ):
                ik_backend.set_posture_reference(command_q14)

            arm_target_q14: np.ndarray | None = None
            if left_active or right_active:
                try:
                    # Integrate IK from the last command, not measured joints.
                    # Seeding from measured + step-limiting from measured kept the
                    # published target one tick (~0.02 rad) ahead of the arm; if
                    # the motors did not instantly close that gap, the upper arm
                    # never accumulated visible travel.
                    ik_seed_q14 = actual_q14.copy()
                    if left_active:
                        ik_seed_q14[:7] = command_q14[:7]
                    if right_active:
                        ik_seed_q14[7:] = command_q14[7:]
                    seed_left_tcp, seed_right_tcp = ik_backend.forward(ik_seed_q14)
                    arm_target_q14 = ik_backend.compute(
                        ik_seed_q14,
                        mapping_dt,
                        mapped.left.target if left_active else seed_left_tcp,
                        mapped.right.target if right_active else seed_right_tcp,
                    )
                    if not np.isfinite(arm_target_q14).all():
                        raise RuntimeError("Pink produced a non-finite arm command")
                except Exception as exc:
                    if fault.trip(f"{type(exc).__name__}: {exc}"):
                        print(
                            f"[HW-PINK][SAFETY] controller fault latched: {fault.reason}; "
                            "release both grips to recover",
                            flush=True,
                        )
                    left_active = False
                    right_active = False

            desired = np.asarray(actual_action, dtype=np.float32).copy()
            if arm_target_q14 is not None:
                solved = apply_arm_q14(desired, arm_target_q14)
                if left_active:
                    desired[ACTION_SLICES.left_arm] = solved[ACTION_SLICES.left_arm]
                if right_active:
                    desired[ACTION_SLICES.right_arm] = solved[ACTION_SLICES.right_arm]
            desired[ACTION_SLICES.left_hand] = mapped.left.hand6
            desired[ACTION_SLICES.right_hand] = mapped.right.hand6

            if left_active:
                command_action[ACTION_SLICES.left_arm] = smooth_command(
                    command_action[ACTION_SLICES.left_arm],
                    desired[ACTION_SLICES.left_arm],
                    teleop_cfg.smoothing.arm_command_alpha,
                    max_arm_step,
                )
            if right_active:
                command_action[ACTION_SLICES.right_arm] = smooth_command(
                    command_action[ACTION_SLICES.right_arm],
                    desired[ACTION_SLICES.right_arm],
                    teleop_cfg.smoothing.arm_command_alpha,
                    max_arm_step,
                )
            command_action[ACTION_SLICES.left_hand] = smooth_command(
                command_action[ACTION_SLICES.left_hand],
                desired[ACTION_SLICES.left_hand],
                teleop_cfg.smoothing.hand_command_alpha,
                teleop_cfg.smoothing.hand_max_joint_step_rad,
            )
            command_action[ACTION_SLICES.right_hand] = smooth_command(
                command_action[ACTION_SLICES.right_hand],
                desired[ACTION_SLICES.right_hand],
                teleop_cfg.smoothing.hand_command_alpha,
                teleop_cfg.smoothing.hand_max_joint_step_rad,
            )
            if (
                request.command_override is not None
                and state_feed_ok
                and not fault.active
            ):
                command_action = np.asarray(request.command_override, dtype=np.float32).copy()

            if not np.isfinite(command_action).all():
                if fault.trip("non-finite 26D command"):
                    print("[HW-PINK][SAFETY] non-finite command fault latched", flush=True)
                command_action = np.asarray(actual_action, dtype=np.float32).copy()
                left_active = False
                right_active = False

            if recorder is not None:
                recorder.write(
                    monotonic_s=now,
                    q14=actual_q14,
                    left_tcp=left_tcp,
                    right_tcp=right_tcp,
                    left_target=mapped.left.target,
                    right_target=mapped.right.target,
                    solved_q14=arm_target_q14,
                    commanded_q14=bimanual_to_arm_q14(command_action),
                    fault=fault.reason,
                )

            # Always publish the last command, including unclutched hold. Publishing
            # the measured pose instead lets gravity sag become the new target.
            allow_arm_motion = not args.shadow
            command_transport_ok = (
                not state_feed_stale
                and not arm_command_graph_conflict
            )
            bridge.update_hand_state_from_rad(
                command_action[ACTION_SLICES.left_hand],
                command_action[ACTION_SLICES.right_hand],
            )
            if not args.shadow and bridge.output_relinquished:
                print(
                    "[HW-PINK][SAFETY] arm-replay output relinquished; leaving control loop",
                    flush=True,
                )
                break
            published_arm: dict[str, float] = {}
            if not args.shadow and command_transport_ok and request.publish_arms:
                published_arm = bridge.publish_arm_command(
                    command_action,
                    allow_motion=allow_arm_motion,
                    hold_commanded=False,
                )
                if bridge.output_relinquished:
                    print(
                        "[HW-PINK][SAFETY] arm-replay output relinquished; leaving control loop",
                        flush=True,
                    )
                    break
            left_trigger = command_hand_trigger(
                tracking_valid=mapped.left.tracking_valid,
                stale=mapped.stale,
                fault_active=fault.active,
                trigger=mapper.states["left"].trigger_filtered,
            )
            right_trigger = command_hand_trigger(
                tracking_valid=mapped.right.tracking_valid,
                stale=mapped.stale,
                fault_active=fault.active,
                trigger=mapper.states["right"].trigger_filtered,
            )
            if request.hand_triggers is not None:
                left_trigger, right_trigger = request.hand_triggers
            if not args.shadow and hands_enabled:
                bridge.publish_hands(left_trigger, right_trigger)

            if hooks is not None:
                hooks.end_tick(
                    TeleopTick(
                        monotonic_s=now,
                        timestamp_ns=time.monotonic_ns(),
                        frame=frame,
                        actual_action=np.asarray(actual_action, dtype=np.float32).copy(),
                        command_action=np.asarray(command_action, dtype=np.float32).copy(),
                        published_arm=dict(published_arm),
                        left_trigger=float(left_trigger),
                        right_trigger=float(right_trigger),
                        left_active=bool(left_active),
                        right_active=bool(right_active),
                        fault_active=bool(fault.active),
                        output_relinquished=bool(bridge.output_relinquished),
                        state_feed_stale=bool(state_feed_stale),
                        state_age_s=float(bridge.last_state_age_s),
                        command_transport_ok=bool(command_transport_ok),
                        command_published=bool(published_arm),
                        input_stale=bool(mapped.stale),
                        fault_reason=str(fault.reason),
                        mapping_dt=float(mapping_dt),
                    )
                )

            report_steps += 1
            if now - last_report >= max(args.report_period_s, 0.1):
                report_elapsed = max(now - report_start_time, 1.0e-6)
                loop_hz = report_steps / report_elapsed
                left_error = float(np.linalg.norm(mapped.left.target.position - left_tcp.position))
                right_error = float(np.linalg.norm(mapped.right.target.position - right_tcp.position))
                left_sample = None if frame is None else frame.left
                right_sample = None if frame is None else frame.right
                diag = bridge.diagnostics()
                print(
                    f"[HW-PINK] loop={loop_hz:.1f}Hz target={config.hardware.control_rate_hz:.1f}Hz "
                    f"shadow={args.shadow} stale={mapped.stale} state_feed_stale={state_feed_stale} "
                    f"arm_graph_conflict={arm_command_graph_conflict} "
                    f"active(L/R)={int(left_active)}/{int(right_active)} "
                    f"grip(L/R)={mapper.clutch_value(left_sample):.2f}/{mapper.clutch_value(right_sample):.2f} "
                    f"trig(L/R)={left_trigger:.2f}/{right_trigger:.2f} "
                    f"trig_raw(L/R)="
                    f"{0.0 if left_sample is None else float(left_sample.trigger):.2f}/"
                    f"{0.0 if right_sample is None else float(right_sample.trigger):.2f} "
                    f"hands={int(hands_enabled)} "
                    f"track(L/R)={int(mapped.left.tracking_valid)}/{int(mapped.right.tracking_valid)} "
                    f"calibrated={int(mapped.calibrated)} "
                    f"calib_yaw={mapped.calibration_yaw_rad:+.3f}rad "
                    f"boundary_safe={int(mapped.boundary_safe)} "
                    f"boundary_distance="
                    f"{'n/a' if mapped.boundary_distance_m is None else f'{mapped.boundary_distance_m:.2f}m'} "
                    f"fault={fault.reason!r} "
                    f"proximal_track_err={proximal_tracking_error:.3f}rad "
                    f"tcp_err(L/R)={left_error:.3f}/{right_error:.3f}m "
                    f"state_age={bridge.last_state_age_s:.3f}s "
                    f"grav_sign={diag['gravity_sign']:+.1f} "
                    f"grav_tau_max={float(diag['gravity_tau_abs_max']):.2f}Nm",
                    flush=True,
                )
                if args.input_debug:
                    cmd_q14 = bimanual_to_arm_q14(command_action)
                    proximal = np.r_[0:4, 7:11]
                    session_id = "none" if frame is None else frame.session_id
                    reference_space = "none" if frame is None else frame.reference_space
                    calibration_id = 0 if frame is None else frame.calibration_id
                    print(
                        f"[HW-PINK][DEBUG] session={session_id} "
                        f"reference={reference_space} calibration_id={calibration_id} "
                        f"q14={np.round(actual_q14, 4).tolist()} "
                        f"cmd_q14={np.round(cmd_q14, 4).tolist()} "
                        f"proximal_cmd_err={float(np.max(np.abs(cmd_q14[proximal] - actual_q14[proximal]))):.4f}rad "
                        f"target_L={np.round(mapped.left.target.position, 4).tolist()} "
                        f"target_R={np.round(mapped.right.target.position, 4).tolist()} "
                        f"xr_dL={np.round(mapped.left.xr_delta, 3).tolist()} "
                        f"base_dL={np.round(mapped.left.base_delta, 3).tolist()} "
                        f"xr_dR={np.round(mapped.right.xr_delta, 3).tolist()} "
                        f"base_dR={np.round(mapped.right.base_delta, 3).tolist()} "
                        f"ik={ik_backend.diagnostics()} bridge={diag}",
                        flush=True,
                    )
                last_report = now
                report_start_time = now
                report_steps = 0

            next_deadline += control_dt
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.monotonic()
    finally:
        if recorder is not None:
            recorder.close()
        server.close()
        if not args.shadow:
            try:
                if bridge.is_arm_command_graph_conflicted(check_period_s=0.0):
                    bridge.relinquish_without_arm_hold(
                        "external arm-replay publisher detected during shutdown"
                    )
                else:
                    bridge.hold_current_and_relinquish("teleop process shutdown")
            except Exception:
                pass
        bridge.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_hardware_teleop_config(args.hardware_config)
        if args.ik_backend is not None:
            config = replace(config, ik=replace(config.ik, backend=str(args.ik_backend)))
        run_pink_hardware_teleop(config, args)
    except KeyboardInterrupt:
        print("\n[HW-PINK] interrupted; arm-replay output relinquished", flush=True)
        return 130
    except BaseException:
        print("[FATAL] Pink hardware Quest teleoperation failed:", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
