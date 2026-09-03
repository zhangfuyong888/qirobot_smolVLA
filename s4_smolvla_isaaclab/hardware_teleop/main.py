#!/usr/bin/env python
"""Quest hardware teleoperation with headless Isaac IK and direct lowcmd output."""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Control the real S4 robot with Meta Quest 3 controllers.")
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
parser.add_argument(
    "--ik-backend",
    choices=("rmpflow", "pinocchio"),
    default=None,
    help="Override hardware config ik.backend.",
)
parser.add_argument("--input-debug", action="store_true")
parser.add_argument("--max-runtime-s", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from hardware_teleop.config_loader import HardwareTeleopConfig, load_hardware_teleop_config
from hardware_teleop.hand_mapping import trigger_from_hand6
from hardware_teleop.ik import create_hardware_ik_backend
from hardware_teleop.ros import HardwareRobotBridge, RosImportError
from hardware_teleop.scene.minimal import build_minimal_robot_scene
from hardware_teleop.startup import run_startup_homing
from hardware_teleop.safety import find_verified_mode5_sdk_process
from s4_pipeline.config import load_project_config
from s4_robot.arm_control import DEFAULT_TCP_OFFSET_WRIST, smooth_command
from s4_robot.control_mapping import ACTION_SLICES, extract_bimanual_state, make_full_joint_target
from s4_robot.simulation import create_simulation_context
from teleoperation.isaaclab_teleop import (
    detect_lan_ip,
    load_task_control_profiles,
)
from teleoperation.mapping import BimanualTeleopMapper, TcpPose, matrix_to_quat_wxyz, quat_wxyz_to_matrix
from teleoperation.protocol import LatestFrameStore
from teleoperation.server import QuestWebServer


def _body_id(robot, name: str) -> int:
    ids, _ = robot.find_bodies(f"^{name}$")
    if len(ids) != 1:
        raise RuntimeError(f"Expected one robot body {name!r}, got ids={ids}")
    return int(ids[0])


def _pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def current_tcp_pose_base(robot, base_body_id: int, wrist_body_id: int) -> TcpPose:
    base = robot.data.body_pose_w[0, base_body_id].detach().cpu().numpy()
    wrist = robot.data.body_pose_w[0, wrist_body_id].detach().cpu().numpy()
    world_base = _pose_matrix(base[:3], base[3:7])
    world_wrist = _pose_matrix(wrist[:3], wrist[3:7])
    wrist_tcp = np.eye(4, dtype=np.float64)
    wrist_tcp[:3, 3] = np.asarray(DEFAULT_TCP_OFFSET_WRIST, dtype=np.float64)
    base_tcp = np.linalg.inv(world_base) @ world_wrist @ wrist_tcp
    return TcpPose(base_tcp[:3, 3].copy(), matrix_to_quat_wxyz(base_tcp[:3, :3]))


def sync_articulation_from_bimanual(robot, sim, action_26d: np.ndarray) -> None:
    actual_full = robot.data.joint_pos[0].detach().cpu().numpy()
    full_target = make_full_joint_target(action_26d, robot.joint_names, actual_full, include_mimic=True)
    joint_tensor = torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1)
    zero_vel = torch.zeros_like(joint_tensor)
    robot.write_joint_state_to_sim(joint_tensor, zero_vel)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(dt=float(sim.get_physics_dt()))


def run_hardware_teleop(config: HardwareTeleopConfig, args: argparse.Namespace) -> None:
    teleop_cfg = config.teleop
    project = load_project_config()
    profiles, home_poses = load_task_control_profiles(project.dataset.task_id)
    mapper = BimanualTeleopMapper(teleop_cfg, **profiles)

    cert = None if args.insecure_http else args.cert.resolve()
    key = None if args.insecure_http else args.key.resolve()
    if cert is not None and (not cert.is_file() or not key.is_file()):
        raise FileNotFoundError(
            f"Quest WebXR HTTPS certificate missing: cert={cert} key={key}. Run: bash run.sh teleop-cert"
        )

    host = args.host or teleop_cfg.network.host
    port = int(args.port or teleop_cfg.network.port)
    store = LatestFrameStore()
    server = QuestWebServer(store, host, port, cert, key, PROJECT_ROOT / "teleoperation/webxr")

    if config.startup.require_sdk_mode5_merge:
        sdk_pid, sdk_executable = find_verified_mode5_sdk_process(
            approved_sha256=config.startup.approved_sdk_sha256,
        )
        print(
            f"[HW-TELEOP] verified SDK mode5 merge: pid={sdk_pid} executable={sdk_executable}",
            flush=True,
        )

    try:
        bridge = HardwareRobotBridge(
            config.hardware,
            config.hands,
            gravity_cfg=config.gravity,
            startup_cfg=config.startup,
            project_root=config.project_root,
            check_lowcmd_publishers=config.startup.check_lowcmd_publishers,
        )
    except RosImportError as exc:
        raise RuntimeError(str(exc)) from exc

    sim = create_simulation_context(args.device)
    scene = build_minimal_robot_scene(config.scene)
    sim.reset()
    robot = scene["robot"]
    dt = float(sim.get_physics_dt())
    control_dt = 1.0 / float(config.hardware.control_rate_hz)

    base_id = _body_id(robot, "base_link")
    left_wrist_id = _body_id(robot, "left_wrist_yaw_link")
    right_wrist_id = _body_id(robot, "right_wrist_yaw_link")

    print(
        f"[HW-TELEOP] runtime=hardware ik={config.ik.backend} control_rate_hz={config.hardware.control_rate_hz:.1f} "
        f"state={config.hardware.state_source} lowcmd={config.hardware.lowcmd_topic} "
        f"hands={config.hardware.hands_cmd_topic}",
        flush=True,
    )
    print(
        "[HW-TELEOP] Shared lowcmd mode: keep exactly one standing-policy source and "
        "stop old teleop, MoveIt and replay controllers.",
        flush=True,
    )
    if config.startup.check_lowcmd_publishers:
        print("[HW-TELEOP] lowcmd publisher conflict check enabled.", flush=True)

    if config.hardware.require_initial_state:
        print(
            f"[HW-TELEOP] waiting for {config.hardware.state_source} "
            f"(timeout={config.hardware.initial_state_timeout_s:.1f}s)...",
            flush=True,
        )
        bridge.wait_for_initial_state(config.hardware.initial_state_timeout_s)
    if config.startup.require_policy_lowcmd:
        bridge.wait_for_policy_lowcmd(
            config.startup.policy_initial_timeout_s,
            config.startup.policy_min_valid_frames,
            config.startup.max_policy_age_s,
            config.startup.policy_stable_duration_s,
        )

    command_action = run_startup_homing(
        startup_cfg=config.startup,
        control_dt=control_dt,
        read_state=bridge.read_bimanual_state,
        publish_step=lambda action: bridge.publish_arm_command(action, allow_motion=True),
        spin_once=lambda: bridge.spin_once(timeout_sec=0.0),
        home_poses=home_poses,
        profiles=profiles,
    )
    sync_articulation_from_bimanual(robot, sim, command_action)
    bridge.update_hand_state_from_rad(
        command_action[ACTION_SLICES.left_hand],
        command_action[ACTION_SLICES.right_hand],
    )

    ik_backend = create_hardware_ik_backend(config, robot, sim.device, base_id)
    ik_backend.set_posture_reference(robot.data.joint_pos[0].detach().cpu().numpy())
    print(f"[HW-TELEOP][IK] backend={ik_backend.name} details={ik_backend.diagnostics()}", flush=True)

    server.start()
    scheme = "http" if args.insecure_http else "https"
    lan_ip = detect_lan_ip()
    print("\n[HW-TELEOP] Meta Quest controller server ready", flush=True)
    print(f"[HW-TELEOP] Quest URL: {scheme}://{lan_ip}:{port}", flush=True)
    print("[HW-TELEOP] Grip = arm clutch; Trigger = hand closure; stale input freezes arms.", flush=True)
    print("[HW-TELEOP] Waiting for Quest controller input...", flush=True)

    start_time = time.monotonic()
    next_deadline = start_time
    last_mapping_time = start_time
    last_report = 0.0
    report_start_time = start_time
    report_steps = 0
    reported_input_session: str | None = None

    try:
        while simulation_app.is_running():
            now = time.monotonic()
            if args.max_runtime_s > 0.0 and now - start_time >= args.max_runtime_s:
                break

            bridge.spin_once(timeout_sec=0.0)
            if not bridge.state_ready:
                time.sleep(0.01)
                continue

            mapping_dt = float(np.clip(now - last_mapping_time, control_dt, 0.10))
            last_mapping_time = now

            actual_action = bridge.read_bimanual_state()
            sync_articulation_from_bimanual(robot, sim, actual_action)
            current_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()

            left_tcp = current_tcp_pose_base(robot, base_id, left_wrist_id)
            right_tcp = current_tcp_pose_base(robot, base_id, right_wrist_id)
            mapped = mapper.update(store.snapshot(), left_tcp, right_tcp, mapping_dt, now)
            if mapped.left.clutch_rising or mapped.right.clutch_rising:
                ik_backend.set_posture_reference(current_joint_pos)

            controller_dt = mapping_dt if config.ik.backend == "rmpflow" else control_dt
            actual_before = extract_bimanual_state(current_joint_pos, robot.joint_names)
            arm_targets = None
            if mapped.left.clutch or mapped.right.clutch:
                arm_targets = ik_backend.compute(
                    current_joint_pos,
                    controller_dt,
                    mapped.left.target,
                    mapped.right.target,
                )

            desired = command_action.copy()
            if mapped.left.clutch and arm_targets is not None:
                desired[ACTION_SLICES.left_arm] = arm_targets[:7]
            else:
                desired[ACTION_SLICES.left_arm] = actual_before[ACTION_SLICES.left_arm]
            if mapped.right.clutch and arm_targets is not None:
                desired[ACTION_SLICES.right_arm] = arm_targets[7:14]
            else:
                desired[ACTION_SLICES.right_arm] = actual_before[ACTION_SLICES.right_arm]
            desired[ACTION_SLICES.left_hand] = mapped.left.hand6
            desired[ACTION_SLICES.right_hand] = mapped.right.hand6

            allow_arm_motion = (mapped.left.clutch or mapped.right.clutch) and not mapped.stale
            if config.hardware.stale_command_hold and mapped.stale:
                allow_arm_motion = False
            state_feed_stale = bridge.is_state_feed_stale(config.hardware.max_state_age_s)
            policy_feed_stale = (
                config.startup.require_policy_lowcmd
                and bridge.is_policy_feed_stale(config.startup.max_policy_age_s)
            )
            lowcmd_graph_conflict = bridge.is_lowcmd_graph_conflicted()
            if state_feed_stale:
                allow_arm_motion = False
            if policy_feed_stale:
                allow_arm_motion = False
            if lowcmd_graph_conflict:
                allow_arm_motion = False

            if mapped.left.clutch and allow_arm_motion:
                command_action[ACTION_SLICES.left_arm] = smooth_command(
                    command_action[ACTION_SLICES.left_arm],
                    desired[ACTION_SLICES.left_arm],
                    teleop_cfg.smoothing.arm_command_alpha,
                    teleop_cfg.smoothing.arm_max_joint_step_rad,
                )
            else:
                command_action[ACTION_SLICES.left_arm] = desired[ACTION_SLICES.left_arm]
            if mapped.right.clutch and allow_arm_motion:
                command_action[ACTION_SLICES.right_arm] = smooth_command(
                    command_action[ACTION_SLICES.right_arm],
                    desired[ACTION_SLICES.right_arm],
                    teleop_cfg.smoothing.arm_command_alpha,
                    teleop_cfg.smoothing.arm_max_joint_step_rad,
                )
            else:
                command_action[ACTION_SLICES.right_arm] = desired[ACTION_SLICES.right_arm]

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
                raise RuntimeError("hardware teleop produced a non-finite 26D command")

            bridge.update_hand_state_from_rad(
                command_action[ACTION_SLICES.left_hand],
                command_action[ACTION_SLICES.right_hand],
            )
            if not state_feed_stale and not policy_feed_stale and not lowcmd_graph_conflict:
                bridge.publish_arm_command(
                    command_action,
                    allow_motion=allow_arm_motion,
                    hold_commanded=False,
                )

            if not mapped.stale:
                left_trigger = trigger_from_hand6(profiles["left_open"], profiles["left_close"], mapped.left.hand6)
                right_trigger = trigger_from_hand6(profiles["right_open"], profiles["right_close"], mapped.right.hand6)
                bridge.publish_hands(left_trigger, right_trigger)

            report_steps += 1
            if now - last_report >= max(args.report_period_s, 0.1):
                report_elapsed = max(now - report_start_time, 1.0e-6)
                loop_hz = report_steps / report_elapsed
                stats = store.stats()
                frame = store.snapshot()
                if frame is not None and frame.session_id != reported_input_session:
                    print(
                        f"[HW-TELEOP][INPUT] session={frame.session_id} reference={frame.reference_space}",
                        flush=True,
                    )
                    reported_input_session = frame.session_id
                print(
                    f"[HW-TELEOP] loop={loop_hz:.1f}Hz target={config.hardware.control_rate_hz:.1f}Hz "
                    f"stale={mapped.stale} state_feed_stale={state_feed_stale} "
                    f"policy_feed_stale={policy_feed_stale} "
                    f"lowcmd_graph_conflict={lowcmd_graph_conflict} "
                    f"clutch(L/R)={mapped.left.clutch}/{mapped.right.clutch} "
                    f"js_age={bridge.last_state_age_s:.3f}s bridge={bridge.diagnostics()}",
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
        server.close()
        if (
            not bridge.is_state_feed_stale(config.hardware.max_state_age_s)
            and (
                not config.startup.require_policy_lowcmd
                or not bridge.is_policy_feed_stale(config.startup.max_policy_age_s)
            )
            and not bridge.is_lowcmd_graph_conflicted(check_period_s=0.0)
        ):
            try:
                bridge.release_to_policy("legacy teleop process shutdown")
            except Exception:
                pass
        bridge.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        hw_config = load_hardware_teleop_config(args_cli.hardware_config)
        if args_cli.ik_backend is not None:
            from dataclasses import replace

            from hardware_teleop.config_loader import HardwareIkConfig

            hw_config = replace(hw_config, ik=HardwareIkConfig(backend=str(args_cli.ik_backend)))
        run_hardware_teleop(hw_config, args_cli)
    except BaseException:
        print("[FATAL] hardware Quest teleoperation failed:", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None
