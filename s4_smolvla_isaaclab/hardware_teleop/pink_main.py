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
    HardwareIkConfig,
    HardwareTeleopConfig,
    load_hardware_teleop_config,
)
from hardware_teleop.hand_mapping import trigger_from_hand6  # noqa: E402
from hardware_teleop.ik import create_pure_hardware_ik_backend  # noqa: E402
from hardware_teleop.joint_mapping import (  # noqa: E402
    apply_arm_q14,
    bimanual_to_arm_q14,
)
from hardware_teleop.ros import HardwareRobotBridge, RosImportError  # noqa: E402
from hardware_teleop.replay import PinkStateRecorder  # noqa: E402
from hardware_teleop.safety import (  # noqa: E402
    TeleopFaultLatch,
    find_verified_mode5_sdk_process,
)
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
        help="Read state and solve IK, but create no lowcmd or hand command publishers.",
    )
    parser.add_argument(
        "--enabled-arms",
        choices=("left", "right", "both"),
        default="both",
        help="Limit which arm can move during staged commissioning.",
    )
    parser.add_argument("--disable-hands", action="store_true")
    parser.add_argument("--skip-homing", action="store_true")
    parser.add_argument(
        "--allow-existing-lowcmd-publishers",
        action="store_true",
        help=(
            "Allow more than one lowcmd graph publisher after manually identifying every "
            "source; valid standing-policy packets are still required."
        ),
    )
    parser.add_argument(
        "--allow-no-policy-lowcmd",
        action="store_true",
        help=(
            "DANGEROUS: bypass the standing-policy leg-cache gate. Use only on a supported "
            "robot fixture where disabling all leg motors is known to be safe."
        ),
    )
    parser.add_argument(
        "--allow-unverified-sdk-mode5-merge",
        action="store_true",
        help=(
            "DANGEROUS: bypass verification that the running sn_loco_server supports "
            "mode_ctrl=5 arm + standing-policy leg merging."
        ),
    )
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
    """Release the ROS node without emitting a mode_ctrl=5 packet."""
    del command_output_enabled
    bridge.close()


def run_pink_hardware_teleop(
    config: HardwareTeleopConfig,
    args: argparse.Namespace,
) -> None:
    _validate_no_isaac_imports()
    from hardware_teleop.env_check import validate_runtime_imports

    validate_runtime_imports(
        robot_profile=os.environ.get("S4_HW_TELEOP_RUNTIME_RESOLVED") == "system"
    )
    teleop_cfg = replace(
        config.teleop,
        network=replace(
            config.teleop.network,
            stale_timeout_s=config.hardware.input_stale_timeout_s,
        ),
        safety=replace(
            config.teleop.safety,
            max_translation_speed_m_s=config.hardware.max_tcp_translation_speed_m_s,
            max_rotation_speed_rad_s=config.hardware.max_tcp_rotation_speed_rad_s,
        ),
    )
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
    mapper = BimanualTeleopMapper(teleop_cfg, **profiles)

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

    if (
        not args.shadow
        and config.startup.require_sdk_mode5_merge
        and not args.allow_unverified_sdk_mode5_merge
    ):
        sdk_pid, sdk_executable = find_verified_mode5_sdk_process()
        print(
            f"[HW-PINK] verified SDK mode5 merge: pid={sdk_pid} executable={sdk_executable}",
            flush=True,
        )
    elif not args.shadow and args.allow_unverified_sdk_mode5_merge:
        print(
            "[HW-PINK][DANGER] running SDK mode_ctrl=5 merge verification was bypassed",
            flush=True,
        )

    try:
        bridge = HardwareRobotBridge(
            config.hardware,
            config.hands,
            gravity_cfg=config.gravity,
            project_root=config.project_root,
            check_lowcmd_publishers=(
                config.startup.check_lowcmd_publishers
                and not args.allow_existing_lowcmd_publishers
            ),
            command_output_enabled=not args.shadow,
        )
    except RosImportError as exc:
        raise RuntimeError(str(exc)) from exc

    control_dt = 1.0 / float(config.hardware.control_rate_hz)
    print(
        f"[HW-PINK] runtime=hardware_no_isaac control_rate_hz={config.hardware.control_rate_hz:.1f} "
        f"state={config.hardware.state_source} shadow={args.shadow} enabled_arms={args.enabled_arms} "
        f"allow_existing_lowcmd_publishers={args.allow_existing_lowcmd_publishers}",
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

    require_policy = config.startup.require_policy_lowcmd and not args.allow_no_policy_lowcmd
    if not args.shadow and require_policy:
        print(
            "[HW-PINK] waiting for fresh standing-policy lowcmd packets before enabling "
            "mode_ctrl=5 output...",
            flush=True,
        )
        try:
            bridge.wait_for_policy_lowcmd(
                config.startup.policy_initial_timeout_s,
                config.startup.policy_min_valid_frames,
                config.startup.max_policy_age_s,
            )
        except BaseException:
            _close_failed_initialization(
                bridge,
                command_output_enabled=False,
            )
            raise
    elif not args.shadow and args.allow_no_policy_lowcmd:
        print(
            "[HW-PINK][DANGER] standing-policy lowcmd gate was explicitly bypassed; "
            "mode_ctrl=5 placeholder legs may be sent without an SDK leg cache",
            flush=True,
        )

    staged_single_arm = args.enabled_arms != "both"
    try:
        if args.shadow or args.skip_homing or staged_single_arm:
            command_action = bridge.read_bimanual_state()
            reason = (
                "shadow"
                if args.shadow
                else "explicit"
                if args.skip_homing
                else "single-arm staging"
            )
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
        print("[HW-PINK][SHADOW] no lowcmd or hand command publisher was created", flush=True)

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
            left_tcp, right_tcp = ik_backend.forward(actual_q14)
            mapped = mapper.update(store.snapshot(), left_tcp, right_tcp, mapping_dt, now)
            state_feed_stale = bridge.is_state_feed_stale(config.hardware.max_state_age_s)
            policy_feed_stale = require_policy and bridge.is_policy_feed_stale(
                config.startup.max_policy_age_s
            )
            lowcmd_graph_conflict = bridge.is_lowcmd_graph_conflicted()
            state_feed_ok = (
                not state_feed_stale
                and not policy_feed_stale
                and not lowcmd_graph_conflict
            )

            if state_feed_stale:
                if fault.trip(
                    f"robot state stale for {bridge.last_state_age_s:.3f}s"
                ):
                    print(
                        "[HW-PINK][SAFETY] robot state feed stale; lowcmd output stopped",
                        flush=True,
                    )
            elif policy_feed_stale:
                if fault.trip(
                    f"standing policy stale for {bridge.last_policy_age_s:.3f}s"
                ):
                    print(
                        "[HW-PINK][SAFETY] standing-policy feed stale; lowcmd output stopped",
                        flush=True,
                    )
            elif lowcmd_graph_conflict:
                if fault.trip("multiple external lowcmd publishers detected"):
                    print(
                        "[HW-PINK][SAFETY] multiple external lowcmd sources detected; "
                        "lowcmd output stopped",
                        flush=True,
                    )

            if fault.clear_if_released(
                left_clutch=mapped.left.clutch,
                right_clutch=mapped.right.clutch,
                state_feed_ok=state_feed_ok,
            ):
                ik_backend.set_posture_reference(actual_q14)
                print("[HW-PINK][SAFETY] fault cleared after both grips were released", flush=True)

            left_active = (
                mapped.left.clutch
                and _arm_enabled(args.enabled_arms, "left")
                and not mapped.stale
                and state_feed_ok
                and not fault.active
            )
            right_active = (
                mapped.right.clutch
                and _arm_enabled(args.enabled_arms, "right")
                and not mapped.stale
                and state_feed_ok
                and not fault.active
            )

            if (mapped.left.clutch_rising and _arm_enabled(args.enabled_arms, "left")) or (
                mapped.right.clutch_rising and _arm_enabled(args.enabled_arms, "right")
            ):
                ik_backend.set_posture_reference(actual_q14)

            arm_target_q14: np.ndarray | None = None
            if left_active or right_active:
                try:
                    arm_target_q14 = ik_backend.compute(
                        actual_q14,
                        mapping_dt,
                        mapped.left.target,
                        mapped.right.target,
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

            command_action[ACTION_SLICES.left_arm] = smooth_command(
                actual_action[ACTION_SLICES.left_arm],
                desired[ACTION_SLICES.left_arm],
                teleop_cfg.smoothing.arm_command_alpha,
                max_arm_step,
            )
            command_action[ACTION_SLICES.right_arm] = smooth_command(
                actual_action[ACTION_SLICES.right_arm],
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

            allow_arm_motion = (left_active or right_active) and not args.shadow
            command_transport_ok = (
                not state_feed_stale
                and not policy_feed_stale
                and not lowcmd_graph_conflict
            )
            bridge.update_hand_state_from_rad(
                command_action[ACTION_SLICES.left_hand],
                command_action[ACTION_SLICES.right_hand],
            )
            if not args.shadow and command_transport_ok:
                bridge.publish_arm_command(
                    command_action,
                    allow_motion=allow_arm_motion,
                    hold_commanded=False,
                )
                if not args.disable_hands and not mapped.stale:
                    left_trigger = trigger_from_hand6(
                        profiles["left_open"], profiles["left_close"], mapped.left.hand6
                    )
                    right_trigger = trigger_from_hand6(
                        profiles["right_open"], profiles["right_close"], mapped.right.hand6
                    )
                    bridge.publish_hands(left_trigger, right_trigger)

            report_steps += 1
            if now - last_report >= max(args.report_period_s, 0.1):
                report_elapsed = max(now - report_start_time, 1.0e-6)
                loop_hz = report_steps / report_elapsed
                left_error = float(np.linalg.norm(mapped.left.target.position - left_tcp.position))
                right_error = float(np.linalg.norm(mapped.right.target.position - right_tcp.position))
                print(
                    f"[HW-PINK] loop={loop_hz:.1f}Hz target={config.hardware.control_rate_hz:.1f}Hz "
                    f"shadow={args.shadow} stale={mapped.stale} state_feed_stale={state_feed_stale} "
                    f"policy_feed_stale={policy_feed_stale} "
                    f"lowcmd_graph_conflict={lowcmd_graph_conflict} "
                    f"active(L/R)={int(left_active)}/{int(right_active)} fault={fault.reason!r} "
                    f"tcp_err(L/R)={left_error:.3f}/{right_error:.3f}m "
                    f"state_age={bridge.last_state_age_s:.3f}s",
                    flush=True,
                )
                if args.input_debug:
                    print(
                        f"[HW-PINK][DEBUG] q14={np.round(actual_q14, 4).tolist()} "
                        f"target_L={np.round(mapped.left.target.position, 4).tolist()} "
                        f"target_R={np.round(mapped.right.target.position, 4).tolist()} "
                        f"ik={ik_backend.diagnostics()} bridge={bridge.diagnostics()}",
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
        if (
            not args.shadow
            and not bridge.is_state_feed_stale(config.hardware.max_state_age_s)
            and (
                not require_policy
                or not bridge.is_policy_feed_stale(config.startup.max_policy_age_s)
            )
            and not bridge.is_lowcmd_graph_conflicted(check_period_s=0.0)
        ):
            try:
                bridge.hold_current_arms()
            except Exception:
                pass
        bridge.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_hardware_teleop_config(args.hardware_config)
        if args.ik_backend is not None:
            config = replace(config, ik=HardwareIkConfig(backend=str(args.ik_backend)))
        run_pink_hardware_teleop(config, args)
    except KeyboardInterrupt:
        print("\n[HW-PINK] interrupted; holding current arm state", flush=True)
        return 130
    except BaseException:
        print("[FATAL] Pink hardware Quest teleoperation failed:", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
