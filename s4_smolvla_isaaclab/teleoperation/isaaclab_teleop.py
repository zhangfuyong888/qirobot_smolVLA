#!/usr/bin/env python
"""Run Meta Quest 3 controller-only bimanual teleoperation in IsaacLab."""

from __future__ import annotations

import argparse
import importlib
import math
import socket
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Control S4 dual-arm TCPs with Meta Quest 3 controllers.")
parser.add_argument("--teleop-config", type=Path, default=PROJECT_ROOT / "configs/teleoperation/meta_quest3.yaml")
parser.add_argument("--host", default=None, help="HTTPS/WSS bind host; defaults to YAML network.host.")
parser.add_argument("--port", type=int, default=None, help="HTTPS/WSS port; defaults to YAML network.port.")
parser.add_argument("--cert", type=Path, default=PROJECT_ROOT / ".local/teleoperation/cert.pem")
parser.add_argument("--key", type=Path, default=PROJECT_ROOT / ".local/teleoperation/key.pem")
parser.add_argument(
    "--insecure-http",
    action="store_true",
    help="Desktop protocol testing only. Quest WebXR requires HTTPS.",
)
parser.add_argument("--report-period-s", type=float, default=0.5)
parser.add_argument(
    "--controller-backend",
    choices=("rmpflow", "pink", "pinocchio"),
    default=None,
    help="Arm controller for this teleop process; defaults to YAML controller.backend.",
)
parser.add_argument(
    "--input-debug",
    action="store_true",
    help="Print raw WebXR buttons, axes, poses, mapping deltas, and command tracking.",
)
parser.add_argument("--max-runtime-s", type=float, default=0.0, help="Exit automatically after N seconds; 0 runs until closed.")
parser.add_argument(
    "--synthetic-input",
    action="store_true",
    help="Test only: inject a deterministic clutch-and-forward controller motion.",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Pace the 120 Hz simulation to wall time. Defaults to YAML simulation.real_time.",
)
parser.add_argument(
    "--render-every-n-steps",
    type=int,
    default=None,
    help="Render the Isaac viewport every N physics steps (default: YAML simulation.render_every_n_steps).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from teleoperation.config import load_teleop_config
from teleoperation.runtime import TeleopRuntimeMode, resolve_runtime_mode

if resolve_runtime_mode(load_teleop_config(args_cli.teleop_config)) == TeleopRuntimeMode.HARDWARE:
    from teleoperation.hardware_teleop import run_hardware_teleop

    run_hardware_teleop(load_teleop_config(args_cli.teleop_config), args_cli)
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

from s4_pipeline.config import load_project_config
from s4_robot.arm_control import DEFAULT_TCP_OFFSET_WRIST, smooth_command
from s4_robot.control_mapping import (
    ACTION_SLICES,
    BIMANUAL_ACTION_DIM,
    extract_bimanual_state,
    make_full_joint_target,
)
from s4_robot.s4_robot_cfg import ALL_DRIVE_JOINTS
from s4_robot.simulation import SceneBuildCfg, create_simulation_context, reset_camera, reset_viewport
from tasks import get_task_spec
from tasks.loading import load_yaml
from teleoperation.config import TeleopConfig, load_teleop_config
from teleoperation.controllers import create_arm_controller
from teleoperation.mapping import BimanualTeleopMapper, TcpPose, quat_wxyz_to_matrix
from teleoperation.protocol import ControllerFrame, ControllerSample, LatestFrameStore
from teleoperation.server import QuestWebServer


DEFAULT_HANDS = {
    "left_open": [0.9, 0.0, 0.05, 0.05, 0.05, 0.05],
    "left_close": [1.0, 0.22, 0.85, 0.85, 0.85, 0.85],
    "right_open": [0.9, 0.0, 0.05, 0.05, 0.05, 0.05],
    "right_close": [1.0, 0.42, 0.85, 0.85, 0.85, 0.85],
}


def detect_lan_ip() -> str:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        if sock is not None:
            sock.close()


def resolve_scene_builder(task_id: str):
    path = get_task_spec(task_id).scene_builder
    module_name, function_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def load_task_control_profiles(task_id: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = dict(DEFAULT_HANDS)
    home_poses: dict[str, np.ndarray] = {}
    task_spec = get_task_spec(task_id)
    if task_spec.scripted_config is not None and task_spec.scripted_config.is_file():
        scripted = load_yaml(task_spec.scripted_config)
        configured = scripted.get("hands", {})
        for key in values:
            if key in configured:
                values[key] = configured[key]
        configured_home = scripted.get("home_poses", {})
        for key in ("left_arm", "right_arm"):
            if key in configured_home:
                home_poses[key] = np.asarray(configured_home[key], dtype=np.float32)
        source = task_spec.scripted_config
    else:
        source = "teleoperation fallback"
    profiles = {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}
    for key, value in profiles.items():
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} hand profile from {source}: {value}")
    for key, value in home_poses.items():
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} home pose from {source}: {value}")
    print(f"[TELEOP] hand profiles source={source}", flush=True)
    return profiles, home_poses


def make_scene_cfg(config: TeleopConfig) -> SceneBuildCfg:
    project = load_project_config()
    return SceneBuildCfg(
        table_top_z=project.scene.table_top_z,
        joint_stiffness=config.simulation.joint_stiffness,
        joint_damping=config.simulation.joint_damping,
        joint_effort_limit=config.simulation.joint_effort_limit,
        robot_base_z=0.98,
        scene_usd=project.scene.scene_usd,
        table_usd=project.scene.table_usd,
        spawn_rgb_cameras=config.simulation.spawn_rgb_cameras,
    )


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
    from teleoperation.mapping import matrix_to_quat_wxyz

    return TcpPose(base_tcp[:3, 3].copy(), matrix_to_quat_wxyz(base_tcp[:3, :3]))


def update_scene_buffers(scene: dict[str, object], dt: float) -> None:
    scene["robot"].update(dt=dt)
    drawer = scene.get("drawer")
    if drawer is not None:
        drawer.update(dt=dt)
    for obj in scene.get("dynamic_objects", []):
        obj.update(dt=dt)


def apply_gravity_compensation(robot, joint_ids: list[int], config: TeleopConfig) -> None:
    if not joint_ids:
        return
    if config.simulation.gravity_compensation:
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        efforts = gravity[:, joint_ids] * config.simulation.gravity_comp_scale
    else:
        efforts = torch.zeros(1, len(joint_ids), dtype=torch.float32, device=robot.device)
    robot.set_joint_effort_target(efforts, joint_ids=joint_ids)


def _rotation_error_angle(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> float:
    relative = quat_wxyz_to_matrix(target_wxyz) @ quat_wxyz_to_matrix(current_wxyz).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def run_simulation_teleop(config: TeleopConfig, args_cli: argparse.Namespace) -> None:
    project = load_project_config()
    task_spec = get_task_spec(project.dataset.task_id)
    if task_spec.data.control_mode != "bimanual":
        print(
            f"[TELEOP][WARN] active task {task_spec.task_id!r} has data control_mode={task_spec.data.control_mode!r}; "
            "teleoperation still controls the physical 26D bimanual robot but does not change that task's dataset contract.",
            flush=True,
        )
    profiles, home_poses = load_task_control_profiles(project.dataset.task_id)
    mapper = BimanualTeleopMapper(config, **profiles)

    cert = None if args_cli.insecure_http else args_cli.cert.resolve()
    key = None if args_cli.insecure_http else args_cli.key.resolve()
    if cert is not None and (not cert.is_file() or not key.is_file()):
        raise FileNotFoundError(
            f"Quest WebXR HTTPS certificate missing: cert={cert} key={key}. "
            "Run: bash run.sh teleop-cert"
        )
    host = args_cli.host or config.network.host
    port = int(args_cli.port or config.network.port)
    store = LatestFrameStore()
    server = QuestWebServer(store, host, port, cert, key, PROJECT_ROOT / "teleoperation/webxr")

    scene_cfg = make_scene_cfg(config)
    sim = create_simulation_context(args_cli.device)
    scene_builder = resolve_scene_builder(project.dataset.task_id)
    print(f"[BOOT] teleop scene builder: {scene_builder.__module__}:{scene_builder.__name__}", flush=True)
    scene = scene_builder(scene_cfg)
    sim.reset()
    if scene.get("camera") is not None:
        reset_camera(scene["camera"], sim, scene_cfg)
    else:
        reset_viewport(sim, scene_cfg)

    render_every_n_steps = (
        int(args_cli.render_every_n_steps)
        if args_cli.render_every_n_steps is not None
        else config.simulation.render_every_n_steps
    )
    if render_every_n_steps < 1:
        raise ValueError("--render-every-n-steps must be at least 1")
    print(
        f"[TELEOP] runtime=simulation spawn_rgb_cameras={config.simulation.spawn_rgb_cameras} "
        f"viewport_render_every={render_every_n_steps} physics_steps",
        flush=True,
    )

    robot = scene["robot"]
    dt = float(sim.get_physics_dt())
    base_id = _body_id(robot, "base_link")
    left_wrist_id = _body_id(robot, "left_wrist_yaw_link")
    right_wrist_id = _body_id(robot, "right_wrist_yaw_link")
    gravity_ids = [robot.joint_names.index(name) for name in ALL_DRIVE_JOINTS if name in robot.joint_names]

    robot.update(dt=dt)
    actual_full = robot.data.joint_pos[0].detach().cpu().numpy().copy()
    command_action = extract_bimanual_state(actual_full, robot.joint_names)
    if "left_arm" in home_poses:
        command_action[ACTION_SLICES.left_arm] = home_poses["left_arm"]
    if "right_arm" in home_poses:
        command_action[ACTION_SLICES.right_arm] = home_poses["right_arm"]
    command_action[ACTION_SLICES.left_hand] = profiles["left_open"]
    command_action[ACTION_SLICES.right_hand] = profiles["right_open"]
    full_target = make_full_joint_target(command_action, robot.joint_names, actual_full, include_mimic=True)
    initial_state = torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1)
    robot.write_joint_state_to_sim(initial_state, torch.zeros_like(initial_state))
    robot.reset()
    robot.update(dt=dt)
    settle_steps = int(math.ceil(max(config.simulation.reset_settle_s, 0.0) / max(dt, 1.0e-6)))
    print(f"[TELEOP] settling scene for {settle_steps} steps ({config.simulation.reset_settle_s:.1f}s sim time)", flush=True)
    for settle_step in range(settle_steps):
        robot.set_joint_position_target(torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1))
        apply_gravity_compensation(robot, gravity_ids, config)
        robot.write_data_to_sim()
        render = not args_cli.headless and settle_step % render_every_n_steps == 0
        sim.step(render=render)
        update_scene_buffers(scene, dt)

    controller_backend = args_cli.controller_backend or config.controller.backend
    arm_controller = create_arm_controller(controller_backend, config, robot, sim.device, base_id)
    controller_joint_state = robot.data.joint_pos[0].detach().cpu().numpy()
    arm_controller.set_posture_reference(controller_joint_state)
    if controller_backend == "pink":
        pink_left_tcp, pink_right_tcp = arm_controller.forward(controller_joint_state)
        isaac_left_tcp = current_tcp_pose_base(robot, base_id, left_wrist_id)
        isaac_right_tcp = current_tcp_pose_base(robot, base_id, right_wrist_id)
        left_position_error = float(np.linalg.norm(pink_left_tcp.position - isaac_left_tcp.position))
        right_position_error = float(np.linalg.norm(pink_right_tcp.position - isaac_right_tcp.position))
        left_rotation_error = _rotation_error_angle(pink_left_tcp.quat_wxyz, isaac_left_tcp.quat_wxyz)
        right_rotation_error = _rotation_error_angle(pink_right_tcp.quat_wxyz, isaac_right_tcp.quat_wxyz)
        print(
            "[TELEOP][PINK][FK-PARITY] "
            f"position_error_m(L/R)={left_position_error:.6f}/{right_position_error:.6f} "
            f"rotation_error_rad(L/R)={left_rotation_error:.6f}/{right_rotation_error:.6f}",
            flush=True,
        )
        if max(left_position_error, right_position_error) > 2.0e-3 or max(
            left_rotation_error, right_rotation_error
        ) > 2.0e-2:
            raise RuntimeError("Pink and Isaac TCP forward kinematics disagree")
    print(f"[TELEOP][CTRL] backend={controller_backend} details={arm_controller.diagnostics()}", flush=True)

    server.start()
    scheme = "http" if args_cli.insecure_http else "https"
    lan_ip = detect_lan_ip()
    print("\n[TELEOP] Meta Quest controller server ready", flush=True)
    print(f"[TELEOP] Quest URL: {scheme}://{lan_ip}:{port}", flush=True)
    print("[TELEOP] Grip/Squeeze = independent arm clutch; Trigger = hand closure 0..1", flush=True)
    print("[TELEOP] Releasing Grip holds the arm at its current pose. Stale tracking freezes both arms.", flush=True)
    if args_cli.insecure_http:
        print("[TELEOP][WARN] insecure HTTP is for desktop tests; Quest immersive WebXR will not start.", flush=True)

    real_time = config.simulation.real_time if args_cli.real_time is None else bool(args_cli.real_time)
    start_time = time.monotonic()
    initial_arm_command = command_action[np.r_[0:7, 13:20]].copy()
    initial_left_tcp_quat = current_tcp_pose_base(robot, base_id, left_wrist_id).quat_wxyz.copy()
    max_synthetic_arm_delta = 0.0
    max_synthetic_tcp_rotation = 0.0
    next_deadline = start_time
    last_mapping_time = start_time
    last_report = 0.0
    report_start_time = start_time
    report_steps = 0
    simulation_step = 0
    reported_input_session: str | None = None
    try:
        while simulation_app.is_running():
            now = time.monotonic()
            mapping_dt = float(np.clip(now - last_mapping_time, dt, 0.05))
            last_mapping_time = now
            if args_cli.max_runtime_s > 0.0 and now - start_time >= args_cli.max_runtime_s:
                break
            if args_cli.synthetic_input:
                elapsed = now - start_time
                forward = min(max(elapsed - 0.15, 0.0) * 0.035, 0.05)
                rotation = min(max(elapsed - 0.30, 0.0) * 0.30, 0.45)
                trigger = min(max(elapsed - 0.25, 0.0) * 0.5, 1.0)
                synthetic = ControllerSample(
                    valid=True,
                    position=(0.0, 1.2, -forward),
                    orientation_xyzw=(0.0, math.sin(rotation * 0.5), 0.0, math.cos(rotation * 0.5)),
                    trigger=trigger,
                    squeeze=0.0 if elapsed < 0.15 else 1.0,
                )
                store.publish(
                    ControllerFrame(
                        session_id="synthetic-smoke",
                        sequence=int(elapsed / max(dt, 1.0e-6)),
                        client_time_ms=elapsed * 1000.0,
                        reference_space="local-floor",
                        left=synthetic,
                        right=synthetic,
                        received_monotonic=now,
                    )
                )
            current_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
            left_tcp = current_tcp_pose_base(robot, base_id, left_wrist_id)
            right_tcp = current_tcp_pose_base(robot, base_id, right_wrist_id)
            # Controller filtering and Cartesian speed limits are wall-time
            # operations. Using fixed physics dt makes an interactive target
            # artificially slow whenever rendered Isaac Sim runs below 120 Hz.
            mapped = mapper.update(store.snapshot(), left_tcp, right_tcp, mapping_dt, now)
            if mapped.left.clutch_rising or mapped.right.clutch_rising:
                arm_controller.set_posture_reference(current_joint_pos)

            # RMPflow is an interactive wall-time controller. When rendered
            # Isaac Sim runs below 120 Hz, advancing it with fixed physics dt
            # makes the robot visibly lag the controller. The Pinocchio
            # compatibility backend retains its original physics-dt behavior.
            controller_dt = mapping_dt if controller_backend in {"rmpflow", "pink"} else dt
            actual_action_before_command = extract_bimanual_state(current_joint_pos, robot.joint_names)
            ik_step_left = 0.0
            ik_step_right = 0.0
            arm_targets = None
            if mapped.left.clutch or mapped.right.clutch:
                arm_targets = arm_controller.compute(
                    current_joint_pos,
                    controller_dt,
                    mapped.left.target,
                    mapped.right.target,
                )
                ik_step_left = float(
                    np.max(np.abs(arm_targets[:7] - actual_action_before_command[ACTION_SLICES.left_arm]))
                )
                ik_step_right = float(
                    np.max(np.abs(arm_targets[7:14] - actual_action_before_command[ACTION_SLICES.right_arm]))
                )
            desired = command_action.copy()
            if mapped.left.clutch:
                desired[ACTION_SLICES.left_arm] = arm_targets[:7]
            else:
                desired[ACTION_SLICES.left_arm] = actual_action_before_command[ACTION_SLICES.left_arm]
            if mapped.right.clutch:
                desired[ACTION_SLICES.right_arm] = arm_targets[7:14]
            else:
                desired[ACTION_SLICES.right_arm] = actual_action_before_command[ACTION_SLICES.right_arm]
            desired[ACTION_SLICES.left_hand] = mapped.left.hand6
            desired[ACTION_SLICES.right_hand] = mapped.right.hand6
            if mapped.left.clutch:
                command_action[ACTION_SLICES.left_arm] = smooth_command(
                    command_action[ACTION_SLICES.left_arm],
                    desired[ACTION_SLICES.left_arm],
                    config.smoothing.arm_command_alpha,
                    config.smoothing.arm_max_joint_step_rad,
                )
            else:
                command_action[ACTION_SLICES.left_arm] = desired[ACTION_SLICES.left_arm]
            if mapped.right.clutch:
                command_action[ACTION_SLICES.right_arm] = smooth_command(
                    command_action[ACTION_SLICES.right_arm],
                    desired[ACTION_SLICES.right_arm],
                    config.smoothing.arm_command_alpha,
                    config.smoothing.arm_max_joint_step_rad,
                )
            else:
                command_action[ACTION_SLICES.right_arm] = desired[ACTION_SLICES.right_arm]
            command_action[ACTION_SLICES.left_hand] = smooth_command(
                command_action[ACTION_SLICES.left_hand],
                desired[ACTION_SLICES.left_hand],
                config.smoothing.hand_command_alpha,
                config.smoothing.hand_max_joint_step_rad,
            )
            command_action[ACTION_SLICES.right_hand] = smooth_command(
                command_action[ACTION_SLICES.right_hand],
                desired[ACTION_SLICES.right_hand],
                config.smoothing.hand_command_alpha,
                config.smoothing.hand_max_joint_step_rad,
            )
            max_synthetic_arm_delta = max(
                max_synthetic_arm_delta,
                float(np.max(np.abs(command_action[np.r_[0:7, 13:20]] - initial_arm_command))),
            )
            max_synthetic_tcp_rotation = max(
                max_synthetic_tcp_rotation,
                _rotation_error_angle(left_tcp.quat_wxyz, initial_left_tcp_quat),
            )
            if command_action.shape != (BIMANUAL_ACTION_DIM,) or not np.isfinite(command_action).all():
                raise RuntimeError("teleoperation produced an invalid 26D command")
            full_target = make_full_joint_target(command_action, robot.joint_names, full_target, include_mimic=True)
            robot.set_joint_position_target(
                torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1)
            )
            apply_gravity_compensation(robot, gravity_ids, config)
            robot.write_data_to_sim()
            render = not args_cli.headless and simulation_step % render_every_n_steps == 0
            sim.step(render=render)
            simulation_step += 1
            update_scene_buffers(scene, dt)
            report_steps += 1

            if now - last_report >= max(args_cli.report_period_s, 0.1):
                report_elapsed = max(now - report_start_time, 1.0e-6)
                loop_hz = report_steps / report_elapsed
                real_time_factor = loop_hz * dt
                stats = store.stats()
                frame = store.snapshot()
                if frame is not None and frame.session_id != reported_input_session:
                    print(
                        f"[TELEOP][INPUT] session={frame.session_id} reference={frame.reference_space} "
                        f"profiles(L/R)={list(frame.left.profiles)}/{list(frame.right.profiles)} "
                        f"buttons(L/R)={len(frame.left.buttons)}/{len(frame.right.buttons)}",
                        flush=True,
                    )
                    reported_input_session = frame.session_id
                actual_action = extract_bimanual_state(current_joint_pos, robot.joint_names)
                arm_track_left = float(
                    np.max(np.abs(command_action[ACTION_SLICES.left_arm] - actual_action[ACTION_SLICES.left_arm]))
                )
                arm_track_right = float(
                    np.max(np.abs(command_action[ACTION_SLICES.right_arm] - actual_action[ACTION_SLICES.right_arm]))
                )
                hand_track_left = float(
                    np.max(np.abs(command_action[ACTION_SLICES.left_hand] - actual_action[ACTION_SLICES.left_hand]))
                )
                hand_track_right = float(
                    np.max(np.abs(command_action[ACTION_SLICES.right_hand] - actual_action[ACTION_SLICES.right_hand]))
                )
                clutch_input_left = mapper.clutch_value(None if frame is None else frame.left)
                clutch_input_right = mapper.clutch_value(None if frame is None else frame.right)
                raw_left_trigger = 0.0 if frame is None else frame.left.trigger
                raw_right_trigger = 0.0 if frame is None else frame.right.trigger
                left_tcp_error = float(np.linalg.norm(mapped.left.target.position - left_tcp.position))
                right_tcp_error = float(np.linalg.norm(mapped.right.target.position - right_tcp.position))
                age = "none" if not math.isfinite(mapped.frame_age_s) else f"{mapped.frame_age_s * 1000.0:.0f}ms"
                print(
                    f"[TELEOP] clients={stats['clients']} frame_age={age} stale={mapped.stale} "
                    f"loop={loop_hz:.1f}Hz rtf={real_time_factor:.2f} "
                    f"grip_input(L/R)={clutch_input_left:.2f}/{clutch_input_right:.2f} "
                    f"clutch(L/R)={int(mapped.left.clutch)}/{int(mapped.right.clutch)} "
                    f"trigger_raw(L/R)={raw_left_trigger:.2f}/{raw_right_trigger:.2f} "
                    f"trigger_cmd(L/R)={mapper.states['left'].trigger_filtered:.2f}/{mapper.states['right'].trigger_filtered:.2f} "
                    f"target_L=({mapped.left.target.position[0]:+.3f},{mapped.left.target.position[1]:+.3f},{mapped.left.target.position[2]:+.3f}) "
                    f"target_R=({mapped.right.target.position[0]:+.3f},{mapped.right.target.position[1]:+.3f},{mapped.right.target.position[2]:+.3f}) "
                    f"tcp_err(L/R)={left_tcp_error:.3f}/{right_tcp_error:.3f}m "
                    f"track_arm(L/R)={arm_track_left:.3f}/{arm_track_right:.3f}rad "
                    f"track_hand(L/R)={hand_track_left:.3f}/{hand_track_right:.3f}rad",
                    flush=True,
                )
                if args_cli.input_debug and frame is not None:
                    for side_name, sample, tcp, mapped_side in (
                        ("L", frame.left, left_tcp, mapped.left),
                        ("R", frame.right, right_tcp, mapped.right),
                    ):
                        state = mapper.states["left" if side_name == "L" else "right"]
                        if not mapped_side.tracking_valid:
                            clutch_reason = "tracking_invalid"
                        elif state.requires_release:
                            clutch_reason = "release_required"
                        elif mapped_side.clutch:
                            clutch_reason = "engaged"
                        else:
                            clutch_reason = "below_engage"
                        if state.controller_reference_position is None:
                            delta_xr = np.zeros(3, dtype=np.float64)
                        else:
                            delta_xr = np.asarray(sample.position) - state.controller_reference_position
                        delta_base = config.mapping.controller_to_base_rotation @ delta_xr
                        buttons = ",".join(f"{value:.2f}" for value in sample.buttons)
                        axes = ",".join(f"{value:+.2f}" for value in sample.axes)
                        print(
                            f"[TELEOP][RAW][{side_name}] valid={int(sample.valid)} "
                            f"pos_xr=({sample.position[0]:+.3f},{sample.position[1]:+.3f},{sample.position[2]:+.3f}) "
                            f"buttons=[{buttons}] axes=[{axes}] squeeze_event={sample.squeeze:.2f} "
                            f"trigger={sample.trigger:.2f} profiles={list(sample.profiles)}",
                            flush=True,
                        )
                        print(
                            f"[TELEOP][MAP][{side_name}] clutch={int(mapped_side.clutch)} "
                            f"reason={clutch_reason} requires_release={int(state.requires_release)} "
                            f"delta_xr=({delta_xr[0]:+.3f},{delta_xr[1]:+.3f},{delta_xr[2]:+.3f}) "
                            f"delta_base=({delta_base[0]:+.3f},{delta_base[1]:+.3f},{delta_base[2]:+.3f}) "
                            f"tcp_current=({tcp.position[0]:+.3f},{tcp.position[1]:+.3f},{tcp.position[2]:+.3f}) "
                            f"tcp_target=({mapped_side.target.position[0]:+.3f},{mapped_side.target.position[1]:+.3f},{mapped_side.target.position[2]:+.3f})",
                            flush=True,
                        )
                    left_rot_error = _rotation_error_angle(mapped.left.target.quat_wxyz, left_tcp.quat_wxyz)
                    right_rot_error = _rotation_error_angle(mapped.right.target.quat_wxyz, right_tcp.quat_wxyz)
                    command_motion_left = float(
                        np.max(np.abs(command_action[ACTION_SLICES.left_arm] - initial_arm_command[:7]))
                    )
                    command_motion_right = float(
                        np.max(np.abs(command_action[ACTION_SLICES.right_arm] - initial_arm_command[7:14]))
                    )
                    print(
                        f"[TELEOP][CTRL] backend={controller_backend} "
                        f"tcp_pos_err(L/R)={left_tcp_error:.4f}/{right_tcp_error:.4f}m "
                        f"tcp_rot_err(L/R)={left_rot_error:.3f}/{right_rot_error:.3f}rad "
                        f"solver_q_step(L/R)={ik_step_left:.4f}/{ik_step_right:.4f}rad "
                        f"command_motion_from_start(L/R)={command_motion_left:.4f}/{command_motion_right:.4f}rad "
                        f"actual_track(L/R)={arm_track_left:.4f}/{arm_track_right:.4f}rad",
                        flush=True,
                    )
                last_report = now
                report_start_time = now
                report_steps = 0

            if real_time:
                next_deadline += dt
                remaining = next_deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                elif remaining < -0.25:
                    next_deadline = time.monotonic()
    finally:
        server.close()
    if args_cli.synthetic_input:
        if max_synthetic_arm_delta < 1.0e-3:
            raise RuntimeError("synthetic teleoperation smoke did not change arm commands")
        if max_synthetic_tcp_rotation < 1.0e-2:
            raise RuntimeError("synthetic teleoperation smoke did not rotate the measured TCP")
        print(
            f"[TELEOP][SMOKE] arm_command_max_delta={max_synthetic_arm_delta:.4f}rad "
            f"measured_tcp_rotation={max_synthetic_tcp_rotation:.4f}rad",
            flush=True,
        )


if __name__ == "__main__":
    try:
        teleop_config = load_teleop_config(args_cli.teleop_config)
        run_simulation_teleop(teleop_config, args_cli)
    except BaseException:
        print("[FATAL] Quest teleoperation failed:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
