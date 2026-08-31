#!/usr/bin/env python
"""Thin Isaac Lab entry for S4 scene debug and right-arm reach control."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run S4 Isaac Lab grasping debug utilities.")
parser.add_argument("--table-top-z", type=float, default=None, help="World Z height for task objects.")
parser.add_argument("--scene-usd", type=Path, default=None, help="Local background scene USD.")
parser.add_argument("--table-usd", type=Path, default=None, help="Local visual table USD.")
parser.add_argument("--table-visual-z", type=float, default=0.0, help="World Z translation for the visual table USD.")
parser.add_argument("--table-scale", type=float, default=1.0, help="Uniform scale for the visual table USD.")
parser.add_argument(
    "--clean-table-clutter",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep the visual table but deactivate known PackingTable clutter prims.",
)
parser.add_argument("--robot-base-z", type=float, default=0.98, help="World Z for fixed robot base_link.")
parser.add_argument("--task-x", type=float, default=0.50, help="World X for block centers.")
parser.add_argument("--task-y", type=float, default=-0.05, help="World Y center for table and task objects.")
parser.add_argument("--block-y-offset", type=float, default=0.20, help="Half spacing between red and blue blocks.")
parser.add_argument("--plate-x", type=float, default=0.50, help="World X for plate center.")
parser.add_argument("--camera-eye", type=float, nargs=3, default=[0.10, 0.0, 1.80], metavar=("X", "Y", "Z"))
parser.add_argument("--camera-target", type=float, nargs=3, default=[0.68, 0.0, 1.02], metavar=("X", "Y", "Z"))
parser.add_argument("--camera-rpy-deg", type=float, nargs=3, default=[0.0, -23.0, -90.0], metavar=("R", "P", "Y"))
parser.add_argument("--camera-convention", choices=["opengl", "ros", "world"], default="opengl")
parser.add_argument(
    "--camera-look-at",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use --camera-eye -> --camera-target look-at for /World/DebugFrontCamera. Default uses explicit --camera-rpy-deg.",
)
parser.add_argument("--camera-width", type=int, default=680)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument("--left-wrist-camera-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--right-wrist-camera-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--left-wrist-camera-quat-wxyz", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
parser.add_argument("--right-wrist-camera-quat-wxyz", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
parser.add_argument("--left-wrist-camera-rpy-deg", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
parser.add_argument("--right-wrist-camera-rpy-deg", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
parser.add_argument("--wrist-camera-convention", choices=["opengl", "ros", "world"], default=None)
parser.add_argument("--continuous", action="store_true", help="Run forever for debug.")
parser.add_argument("--keyboard-jog", action="store_true", help="Enable live keyboard joint jogging.")
parser.add_argument("--jog-step", type=float, default=0.03, help="Joint increment for keyboard jogging, in radians.")
parser.add_argument("--control-file", type=Path, default=Path("/tmp/s4_joint_command.json"))
parser.add_argument("--arm-control-file", type=Path, default=Path("/tmp/s4_arm_control.json"))
parser.add_argument("--print-layout", action="store_true")
parser.add_argument("--joint-stiffness", type=float, default=600.0)
parser.add_argument("--joint-damping", type=float, default=80.0)
parser.add_argument("--joint-effort-limit", type=float, default=300.0)
parser.add_argument("--target-alpha", type=float, default=0.32)
parser.add_argument("--max-joint-step", type=float, default=0.050)
parser.add_argument("--hand-max-joint-step", type=float, default=0.015)
parser.add_argument("--reach-max-cart-step", type=float, default=0.020)
parser.add_argument("--reach-max-joint-delta", type=float, default=0.050)
parser.add_argument("--reach-damping", type=float, default=0.16)
parser.add_argument("--reach-posture-gain", type=float, default=0.30)
parser.add_argument(
    "--tcp-posture-gain",
    type=float,
    default=0.05,
    help="Null-space posture gain for base_link TCP IK. Biases both arms toward DEFAULT_POSE while preserving TCP tasks.",
)
parser.add_argument("--tcp-ik-damping", type=float, default=0.08, help="DLS damping for base_link TCP IK.")
parser.add_argument("--tcp-max-joint-delta", type=float, default=0.050, help="Per-step joint delta limit for base_link TCP IK.")
parser.add_argument("--reach-max-error", type=float, default=0.85)
parser.add_argument(
    "--reach-jacobian-body-shift",
    type=int,
    default=None,
    help="Body row offset from right_wrist_yaw_link for PhysX Jacobian. Default keeps IsaacLab fixed-base convention.",
)
parser.add_argument("--reach-jacobian-sign", type=float, default=1.0, help="Set -1 to flip the raw PhysX Jacobian sign for diagnostics.")
parser.add_argument(
    "--reach-adaptive-direction-sign",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Auto-flip reach direction if measured TCP motion repeatedly moves away from the target. Disabled by default.",
)
parser.add_argument(
    "--reach-min-tcp-below-block",
    type=float,
    default=0.04,
    help="Hold reach control if TCP falls this far below the target block center.",
)
parser.add_argument("--unstable-arm-threshold", type=float, default=3.2)
parser.add_argument("--unstable-arm-velocity-threshold", type=float, default=18.0)
parser.add_argument("--reset-settle-steps", type=int, default=0, help="Minimum physics steps to settle the robot after reset before syncing hold targets.")
parser.add_argument("--reset-settle-s", type=float, default=2.0, help="Minimum simulated seconds to settle after scene load/reset before starting a task.")
parser.add_argument(
    "--gravity-compensation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply PhysX joint-space gravity compensation as feed-forward effort.",
)
parser.add_argument("--gravity-comp-scale", type=float, default=1.0, help="Scale for gravity compensation feed-forward effort.")
parser.add_argument("--show-tcp-frames", action="store_true", help="Visualize current hand TCP and target block TCP frames.")
parser.add_argument("--show-drawer-handle-frame", action="store_true", help="Visualize the top drawer handle frame.")
parser.add_argument("--show-wrist-camera-frustums", action="store_true", help="Visualize left/right wrist camera view frustums.")
parser.add_argument("--wrist-camera-frustum-depth", type=float, default=0.45, help="Depth in meters for wrist camera frustum visualization.")
parser.add_argument("--wrist-camera-frustum-line-width", type=float, default=5.0, help="Screen-space line width in pixels for wrist camera frustum visualization.")
parser.add_argument(
    "--wrist-camera-frustum-scale",
    type=float,
    default=0.30,
    help="Uniform debug-frustum geometry scale. Default 0.30 means 70%% smaller than the configured base size.",
)
parser.add_argument(
    "--live-usd-transforms",
    action="store_true",
    help="Debug only: run CPU PhysX without Fabric so Stage transform gizmos follow current articulation poses.",
)
parser.add_argument(
    "--drawer-handle-frame-prim",
    type=str,
    default="/World/DrawerTask/DrawerCabinet/drawer_handle_frame",
    help="USD prim path for drawer_handle_frame. Falls back to searching by leaf name.",
)
parser.add_argument("--print-tcp-pose", action="store_true", help="Print left/right TCP poses in world/base_link frame.")
parser.add_argument("--tcp-print-period", type=float, default=0.5, help="Seconds between --print-tcp-pose log lines.")
parser.add_argument("--drawer-open", action="store_true", help="Preview drawer opening by driving the drawer USD joint.")
parser.add_argument("--drawer-joint-filter", type=str, default="drawer_top_joint", help="Substring used to select drawer joint prims.")
parser.add_argument("--drawer-target", type=float, default=0.35, help="Drawer joint drive target position for --drawer-open.")
parser.add_argument("--drawer-drive-stiffness", type=float, default=800.0)
parser.add_argument("--drawer-drive-damping", type=float, default=120.0)
parser.add_argument("--drawer-drive-max-force", type=float, default=800.0)
parser.add_argument(
    "--drawer-coast-diagnostic",
    action="store_true",
    help="Run an isolated top-drawer coast test with the can moved away, then exit.",
)
parser.add_argument("--drawer-coast-start", type=float, default=0.18, help="Initial drawer position for coast test.")
parser.add_argument("--drawer-coast-velocity", type=float, default=-0.15, help="Initial drawer velocity for coast test.")
parser.add_argument("--drawer-coast-steps", type=int, default=600, help="Physics steps for coast test.")
parser.add_argument("--record-output", type=Path, default=None, help="Write HDF5 episodes while running scripted grasp.")
parser.add_argument(
    "--failure-log",
    type=Path,
    default=None,
    help="JSONL path for one durable record per failed collection attempt. Defaults beside --record-output.",
)
parser.add_argument(
    "--failure-summary",
    type=Path,
    default=None,
    help="JSON summary path for failure counts by phase/type/reason. Defaults beside --record-output.",
)
parser.add_argument(
    "--max-failed-attempts",
    type=int,
    default=None,
    help="Abort immediately when failed attempts exceed this budget. Zero enforces a failure-free run.",
)
parser.add_argument("--record-episodes", type=int, default=1, help="Number of scripted grasp episodes to record.")
parser.add_argument(
    "--resume",
    action="store_true",
    help="Append to --record-output and continue until --record-episodes total successes.",
)
parser.add_argument(
    "--record-every-n",
    type=int,
    default=6,
    help="Record every N simulation steps. Default 6 gives 20 Hz data from the 120 Hz physics loop.",
)
parser.add_argument(
    "--drawer-scripted-config",
    type=Path,
    default=None,
    help="Drawer scripted YAML. Defaults to configs/tasks/drawer_insert_close.scripted.yaml.",
)
parser.add_argument(
    "--record-episode-timeout-s",
    type=float,
    default=300.0,
    help="Discard the active recorded episode and reset if it exceeds this wall-clock timeout.",
)
parser.add_argument(
    "--success-check",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Only write recorded episodes when the target cylinder finishes inside the plate region.",
)
parser.add_argument(
    "--success-xy-tolerance",
    type=float,
    default=None,
    help="Maximum final XY distance from cylinder center to plate center. Default is plate_radius - cylinder_radius.",
)
parser.add_argument(
    "--success-z-min-above-plate",
    type=float,
    default=-0.02,
    help="Minimum final cylinder center height relative to plate center for success.",
)
parser.add_argument(
    "--success-z-max-above-plate",
    type=float,
    default=0.20,
    help="Maximum final cylinder center height relative to plate center for success.",
)
parser.add_argument("--auto-grasp", action="store_true", help="Automatically start grasp-block when recording starts.")
parser.add_argument("--auto-grasp-block", choices=["red", "blue"], default="blue")
parser.add_argument(
    "--randomize-blue-xy",
    type=float,
    default=0.0,
    help="Uniform per-episode randomization range for blue cylinder x/y position in meters.",
)
parser.add_argument(
    "--random-seed",
    type=int,
    default=None,
    help="Override the task YAML randomization seed. Default: use YAML seed (normally 42).",
)
parser.add_argument(
    "--can-xy-randomization",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Randomize grasp-can XY each episode. Default: follow scripted.yaml "
        "randomization.can_xy.enabled."
    ),
)
parser.add_argument(
    "--distractor-cans",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Spawn the three cabinet-top YCB distractors during recording. Default: follow "
        "scripted.yaml randomization.distractor_cans.enabled."
    ),
)
parser.add_argument(
    "--verbose-status",
    action="store_true",
    help="Print high-frequency TCP/Jacobian/status diagnostics. Default keeps logs concise.",
)
parser.add_argument(
    "--color-logs",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Color important collection events. Default enables colors only on an interactive terminal.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_failed_attempts is not None and args_cli.max_failed_attempts < 0:
    parser.error("--max-failed-attempts must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul

from s4_robot.arm_control import (
    KeyboardJog,
    RightArmReachController,
    CLOSE_RIGHT_HAND,
    DEFAULT_TCP_OFFSET_WRIST,
    OPEN_RIGHT_HAND,
    quat_rotate_wxyz,
    read_control_action,
    smooth_command,
    write_default_control_file,
)
from s4_robot.control_mapping import ACTION_SLICES, action_to_joint_targets, extract_bimanual_state, format_action_layout
from s4_robot.s4_robot_cfg import (
    ALL_DRIVE_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    get_default_joint_positions,
    get_joint_limits,
)
from s4_robot.simulation import (
    BLOCK_CYLINDER_RADIUS,
    LEFT_WRIST_CAMERA_LOCAL_POS,
    LEFT_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
    PLATE_RADIUS,
    RIGHT_WRIST_CAMERA_LOCAL_POS,
    RIGHT_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
    SceneBuildCfg,
    TASK_OBJECT_KEYS,
    TaskLayout,
    WRIST_CAMERA_OFFSET_CONVENTION,
    build_scene as build_default_scene,
    create_simulation_context,
    format_layout,
    reset_camera,
    reset_scene,
    write_object_pose,
)
from s4_pipeline.config import load_project_config
from s4_pipeline.drawer_distractors import (
    DEFAULT_DISTRACTOR_RANGES,
    DISTRACTOR_OBJECT_NAMES,
    GRASP_CAN_NOMINAL_POSITION,
    GRASP_CAN_SCALE,
    apply_distractor_spawn_env,
    asset_contract as distractor_asset_contract,
    can_xy_enabled_from_scripted,
    distractor_cans_enabled_from_scripted,
)
from s4_pipeline.failure_reporting import CollectionFailureReporter
from s4_pipeline.language_phases import load_language_phase_contract
from s4_pipeline.paths import DATASET_CONFIG_PATH
from s4_pipeline.randomization import StratifiedGrid2D, sample_separated_xy, sample_xyz_range
from s4_pipeline.retry_policy import decide_drawer_retry
from data.dataset_writer import EpisodeBuffer, Hdf5DemoWriter
from tasks import get_task_spec
from tasks.loading import import_symbol, load_yaml
from tasks.progress_dashboard import DashboardSnapshot, format_compact, format_dashboard


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = DATASET_CONFIG_PATH
JOINT_LIMITS = get_joint_limits()
OPEN_LEFT_HAND = OPEN_RIGHT_HAND.copy()
CLOSE_LEFT_HAND = CLOSE_RIGHT_HAND.copy()


_LOG_COLORS = {
    "cyan": "\033[36;1m",
    "blue": "\033[34;1m",
    "yellow": "\033[33;1m",
    "red": "\033[31;1m",
    "green": "\033[32;1m",
    "magenta": "\033[35;1m",
    "gray": "\033[90m",
}
_LOG_RESET = "\033[0m"
_DASHBOARD_ACTIVE = False


def _color_logs_enabled() -> bool:
    if args_cli.color_logs is not None:
        return bool(args_cli.color_logs)
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def log_collection_event(tag: str, message: str, color: str = "cyan") -> None:
    global _DASHBOARD_ACTIVE
    if _DASHBOARD_ACTIVE:
        print("\033[2J\033[H", end="")
        _DASHBOARD_ACTIVE = False
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{timestamp}] [{tag}]"
    if _color_logs_enabled():
        time_prefix = f"{_LOG_COLORS['gray']}[{timestamp}]{_LOG_RESET}"
        event = f"{_LOG_COLORS[color]}[{tag}] {message}{_LOG_RESET}"
        print(f"{time_prefix} {event}", flush=True)
        return
    print(f"{prefix} {message}", flush=True)


def render_collection_dashboard(
    snapshot: DashboardSnapshot,
    config: dict[str, object],
) -> None:
    """Refresh an interactive panel or emit one compact redirected-log line."""
    global _DASHBOARD_ACTIVE
    interactive = (
        bool(config.get("enabled", True))
        and sys.stdout.isatty()
        and os.environ.get("TERM", "") != "dumb"
    )
    if interactive:
        panel = format_dashboard(
            snapshot,
            width=int(config.get("width", 78)),
            bar_width=int(config.get("bar_width", 24)),
            color=_color_logs_enabled(),
        )
        print("\033[2J\033[H" + panel, end="", flush=True)
        _DASHBOARD_ACTIVE = True
        return
    print(
        "[PROGRESS] " + format_compact(snapshot, bar_width=int(config.get("bar_width", 24))),
        flush=True,
    )


def load_table_top_z() -> float:
    if args_cli.table_top_z is not None:
        return float(args_cli.table_top_z)
    return float(load_project_config(CONFIG_PATH).scene.table_top_z)


def make_scene_cfg() -> SceneBuildCfg:
    project_cfg = load_project_config(CONFIG_PATH)
    scene_usd = args_cli.scene_usd or project_cfg.scene.scene_usd
    table_usd = args_cli.table_usd if args_cli.table_usd is not None else project_cfg.scene.table_usd
    layout = TaskLayout(
        table_center_x=float(args_cli.task_x),
        table_center_y=float(args_cli.task_y),
        block_x=float(args_cli.task_x),
        block_y_offset=float(args_cli.block_y_offset),
        plate_x=float(args_cli.plate_x),
    )
    return SceneBuildCfg(
        table_top_z=load_table_top_z(),
        joint_stiffness=float(args_cli.joint_stiffness),
        joint_damping=float(args_cli.joint_damping),
        joint_effort_limit=float(args_cli.joint_effort_limit),
        scene_usd=scene_usd,
        table_usd=table_usd,
        robot_base_z=float(args_cli.robot_base_z),
        table_visual_z=float(args_cli.table_visual_z),
        table_scale=float(args_cli.table_scale),
        clean_table_clutter=bool(args_cli.clean_table_clutter),
        layout=layout,
        camera_eye=tuple(float(x) for x in args_cli.camera_eye),
        camera_target=tuple(float(x) for x in args_cli.camera_target),
        camera_rpy_deg=None if args_cli.camera_look_at else tuple(float(x) for x in args_cli.camera_rpy_deg),
        camera_convention=str(args_cli.camera_convention),
        camera_width=max(int(args_cli.camera_width), 1),
        camera_height=max(int(args_cli.camera_height), 1),
        left_wrist_camera_pos=tuple(float(x) for x in (args_cli.left_wrist_camera_pos or LEFT_WRIST_CAMERA_LOCAL_POS)),
        right_wrist_camera_pos=tuple(float(x) for x in (args_cli.right_wrist_camera_pos or RIGHT_WRIST_CAMERA_LOCAL_POS)),
        left_wrist_camera_quat_wxyz=tuple(float(x) for x in (args_cli.left_wrist_camera_quat_wxyz or LEFT_WRIST_CAMERA_LOCAL_QUAT_WXYZ)),
        right_wrist_camera_quat_wxyz=tuple(float(x) for x in (args_cli.right_wrist_camera_quat_wxyz or RIGHT_WRIST_CAMERA_LOCAL_QUAT_WXYZ)),
        left_wrist_camera_rpy_deg=None if args_cli.left_wrist_camera_rpy_deg is None else tuple(float(x) for x in args_cli.left_wrist_camera_rpy_deg),
        right_wrist_camera_rpy_deg=None if args_cli.right_wrist_camera_rpy_deg is None else tuple(float(x) for x in args_cli.right_wrist_camera_rpy_deg),
        wrist_camera_convention=str(args_cli.wrist_camera_convention or WRIST_CAMERA_OFFSET_CONVENTION),
    )


def _fmt_tuple(values, precision: int = 4) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(round(float(x), precision) for x in values)


def resolve_scene_builder():
    project_cfg = load_project_config(CONFIG_PATH)
    spec = get_task_spec(project_cfg.dataset.task_id)
    module_name, func_name = spec.scene_builder.split(":", 1)
    if module_name == "s4_robot.simulation" and func_name == "build_scene":
        return build_default_scene
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_active_drawer_scripted_cfg() -> dict[str, object] | None:
    """Load the active drawer scripted YAML when the current task is drawer_insert_close."""
    project_cfg = load_project_config(CONFIG_PATH)
    if project_cfg.dataset.task_id != "drawer_insert_close":
        return None
    task_spec = get_task_spec(project_cfg.dataset.task_id)
    scripted_path = args_cli.drawer_scripted_config or task_spec.scripted_config
    if scripted_path is None:
        return None
    return load_yaml(Path(scripted_path).resolve())


def resolve_record_can_xy_enabled(scripted_cfg: dict[str, object] | None) -> bool:
    if args_cli.can_xy_randomization is not None:
        return bool(args_cli.can_xy_randomization)
    return can_xy_enabled_from_scripted(scripted_cfg)


def resolve_record_distractor_cans_enabled(scripted_cfg: dict[str, object] | None) -> bool:
    if args_cli.distractor_cans is not None:
        return bool(args_cli.distractor_cans)
    return distractor_cans_enabled_from_scripted(scripted_cfg)


def set_named_joint_targets(
    full_target: np.ndarray,
    robot: Articulation,
    joint_names: list[str],
    values: np.ndarray,
) -> None:
    for name, value in zip(joint_names, values, strict=True):
        if name in robot.joint_names:
            safe_value = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
            if name in JOINT_LIMITS:
                limit = JOINT_LIMITS[name]
                safe_value = float(np.clip(safe_value, limit["lower"], limit["upper"]))
            full_target[robot.joint_names.index(name)] = safe_value


def parse_arm_target(payload: dict, key: str) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None
    target = np.asarray(value, dtype=np.float32)
    if target.shape != (7,):
        print(f"[WARN] ignored invalid {key} target: expected 7 values")
        return None
    return target


def control_action_from_full_target(full_target: np.ndarray, robot: Articulation) -> np.ndarray:
    return extract_bimanual_state(full_target, robot.joint_names)


def control_action_from_sim(robot: Articulation) -> np.ndarray:
    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
    return extract_bimanual_state(joint_pos, robot.joint_names)


def write_action_to_full_target(full_target: np.ndarray, robot: Articulation, action: np.ndarray) -> None:
    targets = action_to_joint_targets(action, include_mimic=True)
    for name, value in targets.items():
        if name in robot.joint_names:
            full_target[robot.joint_names.index(name)] = float(value)


def right_hand_command_error(commanded_action: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(commanded_action[ACTION_SLICES.right_hand] - np.asarray(target, dtype=np.float32))))


def pose7_from_rigid_object(obj) -> np.ndarray:
    pos = obj.data.root_pos_w[0].detach().cpu().numpy()
    quat = obj.data.root_quat_w[0].detach().cpu().numpy()
    return np.concatenate([pos, quat]).astype(np.float32)


def final_cylinder_in_plate(scene: dict[str, object], block: str) -> tuple[bool, dict[str, float]]:
    block_obj = scene["blue"] if block == "blue" else scene["red"]
    block_pos = block_obj.data.root_pos_w[0].detach().cpu().numpy()
    plate_pos = scene["plate"].data.root_pos_w[0].detach().cpu().numpy()
    xy_dist = float(np.linalg.norm(block_pos[:2] - plate_pos[:2]))
    z_above_plate = float(block_pos[2] - plate_pos[2])
    xy_tolerance = (
        float(args_cli.success_xy_tolerance)
        if args_cli.success_xy_tolerance is not None
        else max(float(PLATE_RADIUS - BLOCK_CYLINDER_RADIUS), 0.01)
    )
    z_min = float(args_cli.success_z_min_above_plate)
    z_max = float(args_cli.success_z_max_above_plate)
    ok = bool(xy_dist <= xy_tolerance and z_min <= z_above_plate <= z_max)
    return ok, {
        "xy_dist": xy_dist,
        "xy_tolerance": xy_tolerance,
        "z_above_plate": z_above_plate,
        "z_min": z_min,
        "z_max": z_max,
        "block_x": float(block_pos[0]),
        "block_y": float(block_pos[1]),
        "block_z": float(block_pos[2]),
        "plate_x": float(plate_pos[0]),
        "plate_y": float(plate_pos[1]),
        "plate_z": float(plate_pos[2]),
    }


def camera_rgb_uint8(camera) -> np.ndarray:
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).astype(np.uint8)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return rgb


def update_all_cameras(scene: dict[str, object], camera, dt: float) -> None:
    camera.update(dt=dt)
    for wrist_camera in scene.get("wrist_cameras", {}).values():
        wrist_camera.update(dt=dt)


def append_record_frame(
    episode: EpisodeBuffer,
    scene: dict[str, object],
    robot: Articulation,
    camera,
    action: np.ndarray,
    reach_controller: RightArmReachController | None,
    tcp_offset_wrist: np.ndarray,
) -> None:
    episode.actions.append(np.asarray(action, dtype=np.float32).copy())
    episode.full_joint_pos.append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy())
    episode.active_joint_pos.append(control_action_from_sim(robot).astype(np.float32).copy())
    episode.chest_front_rgb.append(camera_rgb_uint8(camera))
    wrist_cameras = scene.get("wrist_cameras", {})
    if "left_wrist" in wrist_cameras:
        episode.left_wrist_rgb.append(camera_rgb_uint8(wrist_cameras["left_wrist"]))
    if "right_wrist" in wrist_cameras:
        episode.right_wrist_rgb.append(camera_rgb_uint8(wrist_cameras["right_wrist"]))
    if reach_controller is not None:
        right_tcp = estimate_right_hand_tcp_pose(robot, reach_controller, tcp_offset_wrist)
        if right_tcp is not None:
            episode.right_eef_pose.append(np.concatenate([right_tcp[0], right_tcp[1]]).astype(np.float32))
    episode.red_block_pose.append(pose7_from_rigid_object(scene["red"]))
    episode.blue_block_pose.append(pose7_from_rigid_object(scene["blue"]))
    episode.plate_pose.append(pose7_from_rigid_object(scene["plate"]))


def append_bimanual_record_frame(
    episode: EpisodeBuffer,
    scene: dict[str, object],
    robot: Articulation,
    camera,
    action: np.ndarray,
    task_description: str,
    language_phase_id: str,
    expert_phase_name: str,
) -> None:
    episode.actions.append(np.asarray(action, dtype=np.float32).copy())
    episode.full_joint_pos.append(robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy())
    episode.active_joint_pos.append(control_action_from_sim(robot).astype(np.float32).copy())
    episode.task_descriptions.append(str(task_description))
    episode.language_phase_ids.append(str(language_phase_id))
    episode.expert_phase_names.append(str(expert_phase_name))
    episode.chest_front_rgb.append(camera_rgb_uint8(camera))
    wrist_cameras = scene.get("wrist_cameras", {})
    if "left_wrist" in wrist_cameras:
        episode.left_wrist_rgb.append(camera_rgb_uint8(wrist_cameras["left_wrist"]))
    if "right_wrist" in wrist_cameras:
        episode.right_wrist_rgb.append(camera_rgb_uint8(wrist_cameras["right_wrist"]))
    left_tcp = estimate_left_hand_tcp_pose_from_robot(robot)
    if left_tcp is not None:
        episode.left_eef_pose.append(np.concatenate([left_tcp[0], left_tcp[1]]).astype(np.float32))
    right_tcp = estimate_right_hand_tcp_pose_from_robot(robot)
    if right_tcp is not None:
        episode.right_eef_pose.append(np.concatenate([right_tcp[0], right_tcp[1]]).astype(np.float32))
    can_obj = scene.get("named_objects", {}).get("can")
    if can_obj is not None:
        episode.drawer_task_object_pose.append(pose7_from_rigid_object(can_obj))


def default_grasp_payload(block: str) -> dict[str, object]:
    return {
        "mode": "grasp-block",
        "block": block,
        "base_offset": [-0.06, -0.05],
        "approach_z": 0.10,
        "grasp_z": 0.01,
        "lift_z": 0.15,
        "place_approach_z": 0.18,
        "place_z": 0.10,
        "place_offset": [0.0, -0.05],
        "tcp_offset_wrist": [0.0, 0.0, -0.10],
        "offset_frame": "world",
        "grasp_pose": "current",
        "grasp_rpy": [0.0, 0.1, -0.20],
        "place_rpy": [0.40, 0.0, 0.0],
        "tolerance": 0.05,
        "approach_steps": 120,
        "lower_steps": 120,
        "close_steps": 70,
        "pre_close_hold_steps": 120,
        "hand_complete_tolerance": 0.015,
        "lift_steps": 60,
        "place_steps": 150,
        "pre_release_hold_steps": 120,
        "release_steps": 50,
    }


def settle_scene_to_target(scene: dict[str, object], camera, full_target: np.ndarray, sim, steps: int) -> np.ndarray:
    """Settle physics under a target, then return the settled robot joint state."""
    robot: Articulation = scene["robot"]
    steps = max(int(steps), 0)
    target_tensor = torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1)
    for _ in range(steps):
        robot.set_joint_position_target(target_tensor)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt=sim.get_physics_dt())
        for key in TASK_OBJECT_KEYS:
            scene[key].update(dt=sim.get_physics_dt())
        update_all_cameras(scene, camera, dt=sim.get_physics_dt())
    settled = robot.data.joint_pos[0].detach().cpu().numpy().copy()
    return settled


def sample_blue_xy_offset(rng: np.random.Generator, randomize_xy: float) -> np.ndarray:
    """Sample a world-frame x/y offset for blue-object data collection."""
    span = max(float(randomize_xy), 0.0)
    if span <= 0.0:
        return np.zeros(2, dtype=np.float32)
    return rng.uniform(-span, span, size=2).astype(np.float32)


def apply_blue_xy_offset(
    scene: dict[str, object],
    cfg: SceneBuildCfg,
    sim,
    offset_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Move only the blue cylinder by a world-frame x/y offset."""
    offset_xy = np.asarray(offset_xy, dtype=np.float32)
    if offset_xy.shape != (2,):
        raise ValueError(f"offset_xy must have shape (2,), got {offset_xy.shape}")
    block_pos = cfg.layout.blue_block_pos(cfg.table_top_z).copy()
    block_pos[:2] += offset_xy

    write_object_pose(scene["blue"], block_pos, sim.device)
    scene["blue"].update(dt=sim.get_physics_dt())
    return {"blue": block_pos}


def control_action_bias_from_target(full_target: np.ndarray, robot: Articulation) -> np.ndarray:
    """Return actuator target minus actual joint state in 26D action order."""
    return np.zeros_like(control_action_from_sim(robot))


def resolve_existing_joint_ids(robot: Articulation, joint_names: list[str]) -> list[int]:
    return [robot.joint_names.index(name) for name in joint_names if name in robot.joint_names]


def apply_gravity_compensation(
    robot: Articulation,
    joint_ids: list[int],
    scale: float,
    enabled: bool,
) -> tuple[float, float]:
    """Apply joint-space gravity compensation and return max/mean absolute effort."""
    if not joint_ids:
        return 0.0, 0.0
    if not enabled:
        zeros = torch.zeros(1, len(joint_ids), dtype=torch.float32, device=robot.device)
        robot.set_joint_effort_target(zeros, joint_ids=joint_ids)
        return 0.0, 0.0
    gravity = robot.root_physx_view.get_gravity_compensation_forces()
    if gravity.shape[1] <= max(joint_ids):
        zeros = torch.zeros(1, len(joint_ids), dtype=torch.float32, device=robot.device)
        robot.set_joint_effort_target(zeros, joint_ids=joint_ids)
        return 0.0, 0.0
    efforts = gravity[:, joint_ids] * float(scale)
    robot.set_joint_effort_target(efforts, joint_ids=joint_ids)
    abs_efforts = torch.abs(efforts)
    return float(torch.max(abs_efforts)), float(torch.mean(abs_efforts))


def reset_unstable_arm_state(
    robot: Articulation,
    joint_names: list[str],
    full_target: np.ndarray,
    threshold_rad: float,
    velocity_threshold_rad_s: float,
) -> bool:
    joint_ids = [robot.joint_names.index(name) for name in joint_names if name in robot.joint_names]
    if not joint_ids:
        return False
    q = robot.data.joint_pos.clone()
    qd = robot.data.joint_vel.clone()
    arm_q = q[0, joint_ids]
    arm_qd = qd[0, joint_ids]
    finite = torch.isfinite(arm_q).all() and torch.isfinite(arm_qd).all()
    within_position = float(torch.max(torch.abs(arm_q))) <= threshold_rad
    within_velocity = float(torch.max(torch.abs(arm_qd))) <= velocity_threshold_rad_s
    if finite and within_position and within_velocity:
        return False
    target = torch.tensor(full_target[joint_ids], dtype=torch.float32, device=robot.device)
    q[0, joint_ids] = target
    qd[0, joint_ids] = 0.0
    robot.write_joint_state_to_sim(q, qd)
    robot.reset()
    return True


class TcpFrameVisualizer:
    """Visualize estimated TCP, target, and task frames."""

    def __init__(self, device: str):
        right_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/RightHandTCP")
        right_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        left_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/LeftHandTCP")
        left_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        target_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/TargetBlockTCP")
        target_cfg.markers["frame"].scale = (0.11, 0.11, 0.11)
        handle_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/DrawerHandleTop")
        handle_cfg.markers["frame"].scale = (0.09, 0.09, 0.09)
        self.right_marker = VisualizationMarkers(right_cfg)
        self.left_marker = VisualizationMarkers(left_cfg)
        self.target_marker = VisualizationMarkers(target_cfg)
        self.handle_marker = VisualizationMarkers(handle_cfg)
        self.right_marker.set_visibility(False)
        self.left_marker.set_visibility(False)
        self.target_marker.set_visibility(False)
        self.handle_marker.set_visibility(False)
        self.device = device
        self.identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def _visualize_marker(
        self,
        marker: VisualizationMarkers,
        pose: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        if pose is None:
            marker.set_visibility(False)
            return
        pos, quat = pose
        marker.set_visibility(True)
        marker.visualize(
            translations=torch.tensor(pos, dtype=torch.float32, device=self.device).view(1, 3),
            orientations=torch.tensor(quat, dtype=torch.float32, device=self.device).view(1, 4),
        )

    def visualize(
        self,
        hand_tcp_pose: tuple[np.ndarray, np.ndarray] | None,
        target_tcp_pos: np.ndarray | None,
        target_tcp_quat: np.ndarray | None = None,
    ) -> None:
        self._visualize_marker(self.right_marker, hand_tcp_pose)
        if target_tcp_pos is not None:
            pos = torch.tensor(target_tcp_pos, dtype=torch.float32, device=self.device).view(1, 3)
            if target_tcp_quat is None:
                quat = self.identity_quat
            else:
                quat = torch.tensor(target_tcp_quat, dtype=torch.float32, device=self.device).view(1, 4)
            self.target_marker.set_visibility(True)
            self.target_marker.visualize(translations=pos, orientations=quat)
        else:
            self.target_marker.set_visibility(False)

    def visualize_task_frames(
        self,
        left_tcp_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_tcp_pose: tuple[np.ndarray, np.ndarray] | None = None,
        drawer_handle_pose: tuple[np.ndarray, np.ndarray] | None = None,
        target_tcp_pos: np.ndarray | None = None,
        target_tcp_quat: np.ndarray | None = None,
    ) -> None:
        self._visualize_marker(self.left_marker, left_tcp_pose)
        self.visualize(right_tcp_pose, target_tcp_pos, target_tcp_quat)
        self._visualize_marker(self.handle_marker, drawer_handle_pose)


def quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class WristCameraFrustumVisualizer:
    """Draw camera-local frustums that inherit each wrist camera's Fabric pose."""

    def __init__(self, scene: dict[str, object], depth: float, line_width: float, visual_scale: float):
        from pxr import Gf, UsdGeom

        self.visual_scale = max(float(visual_scale), 0.01)
        self.base_depth = max(float(depth), 0.02)
        self.depth = self.base_depth * self.visual_scale
        self.line_width = max(float(line_width), 1.0)
        base_line_radius = max(0.003, min(0.03, self.line_width * 0.0015))
        base_point_radius = max(0.025, base_line_radius * 2.5)
        self.line_radius = base_line_radius * self.visual_scale
        self.point_radius = base_point_radius * self.visual_scale
        self.colors = {
            "left_wrist": (0.1, 0.8, 1.0, 1.0),
            "right_wrist": (1.0, 0.55, 0.1, 1.0),
        }
        import omni.usd

        self.Gf = Gf
        self.UsdGeom = UsdGeom
        self.stage = omni.usd.get_context().get_stage()
        self.legacy_root_path = "/World/Visuals/WristCameraFrustums"
        self.camera_geometry: dict[str, dict[str, object]] = {}
        # Remove old world-space attempts and stale camera-local geometry.
        stale_paths = ["/World/Visuals/WristCameraFrustumPoints", self.legacy_root_path]
        for camera in scene.get("wrist_cameras", {}).values():
            stale_paths.append(f"{camera.cfg.prim_path}/DebugFrustum")
        for stale_path in stale_paths:
            if self.stage.GetPrimAtPath(stale_path).IsValid():
                self.stage.RemovePrim(stale_path)
        self.last_line_count = 0
        self.last_point_count = 0
        self._reported_ready = False
        self._create_geometry(scene)
        self.update(scene)

    def _create_display_prim(self, kind: str, path: str, color: tuple[float, float, float, float]):
        if kind == "line":
            geom = self.UsdGeom.Cylinder.Define(self.stage, path)
            geom.CreateRadiusAttr(1.0)
            geom.CreateHeightAttr(1.0)
            geom.CreateAxisAttr("Z")
        else:
            geom = self.UsdGeom.Sphere.Define(self.stage, path)
            geom.CreateRadiusAttr(1.0)
        geom.CreateDisplayColorAttr([self.Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        xform = self.UsdGeom.Xformable(geom.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(precision=self.UsdGeom.XformOp.PrecisionDouble)
        orient_op = xform.AddOrientOp(precision=self.UsdGeom.XformOp.PrecisionDouble)
        scale_op = xform.AddScaleOp(precision=self.UsdGeom.XformOp.PrecisionDouble)
        return translate_op, orient_op, scale_op

    def _create_geometry(self, scene: dict[str, object]) -> None:
        wrist_cameras = scene.get("wrist_cameras", {})
        for camera_name in ("left_wrist", "right_wrist"):
            camera = wrist_cameras.get(camera_name)
            if camera is None:
                continue
            root_path = f"{camera.cfg.prim_path}/DebugFrustum"
            self.UsdGeom.Xform.Define(self.stage, root_path)
            line_ops = []
            point_ops = []
            color = self.colors.get(camera_name, (1.0, 1.0, 1.0, 1.0))
            for i in range(8):
                line_ops.append(
                    self._create_display_prim("line", f"{root_path}/line_{i:02d}", color)
                )
            for i in range(5):
                point_ops.append(
                    self._create_display_prim("point", f"{root_path}/point_{i:02d}", color)
                )
            self.camera_geometry[camera_name] = {
                "root_path": root_path,
                "line_ops": line_ops,
                "point_ops": point_ops,
            }

    @staticmethod
    def _quat_from_z_to_direction(direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        target = direction / norm
        source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot > 0.999999:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if dot < -0.999999:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        axis = np.cross(source, target)
        s = math.sqrt((1.0 + dot) * 2.0)
        return np.array([0.5 * s, axis[0] / s, axis[1] / s, axis[2] / s], dtype=np.float64)

    def _set_xform(
        self,
        ops,
        translation: np.ndarray,
        quat_wxyz: np.ndarray,
        scale: tuple[float, float, float],
    ) -> None:
        translate_op, orient_op, scale_op = ops
        translate_op.Set(self.Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
        orient_op.Set(
            self.Gf.Quatd(
                float(quat_wxyz[0]),
                self.Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            )
        )
        scale_op.Set(self.Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))

    def _update_usd_geometry(
        self,
        line_ops,
        point_ops,
        starts: list[tuple[float, float, float]],
        ends: list[tuple[float, float, float]],
        points: list[tuple[float, float, float]],
    ) -> None:
        zero_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        hidden_scale = (1e-6, 1e-6, 1e-6)
        for i, ops in enumerate(line_ops):
            if i < len(starts):
                start = np.asarray(starts[i], dtype=np.float64)
                end = np.asarray(ends[i], dtype=np.float64)
                delta = end - start
                length = max(float(np.linalg.norm(delta)), 1e-6)
                midpoint = 0.5 * (start + end)
                quat = self._quat_from_z_to_direction(delta)
                self._set_xform(ops, midpoint, quat, (self.line_radius, self.line_radius, length))
            else:
                self._set_xform(ops, np.zeros(3, dtype=np.float64), zero_quat, hidden_scale)
        for i, ops in enumerate(point_ops):
            if i < len(points):
                point = np.asarray(points[i], dtype=np.float64)
                self._set_xform(ops, point, zero_quat, (self.point_radius, self.point_radius, self.point_radius))
            else:
                self._set_xform(ops, np.zeros(3, dtype=np.float64), zero_quat, hidden_scale)

    def _camera_intrinsic_from_usd(self, camera) -> tuple[float, float, float, float, int, int]:
        height = int(camera.cfg.height)
        width = int(camera.cfg.width)
        focal_length = 18.0
        horizontal_aperture = 20.955
        vertical_aperture = horizontal_aperture * float(height) / max(float(width), 1.0)
        prim = self.stage.GetPrimAtPath(str(camera.cfg.prim_path))
        if prim.IsValid():
            usd_camera = self.UsdGeom.Camera(prim)
            focal_attr = usd_camera.GetFocalLengthAttr()
            h_ap_attr = usd_camera.GetHorizontalApertureAttr()
            v_ap_attr = usd_camera.GetVerticalApertureAttr()
            if focal_attr and focal_attr.HasValue():
                focal_length = float(focal_attr.Get())
            if h_ap_attr and h_ap_attr.HasValue():
                horizontal_aperture = max(float(h_ap_attr.Get()), 1e-6)
            if v_ap_attr and v_ap_attr.HasValue():
                vertical_aperture = max(float(v_ap_attr.Get()), 1e-6)
        fx = focal_length / horizontal_aperture * float(width)
        fy = focal_length / vertical_aperture * float(height)
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        return fx, fy, cx, cy, height, width

    def _camera_local_frustum_geometry(
        self,
        camera,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        # These coordinates are authored directly under the Camera prim. The
        # renderer applies the camera's current Fabric transform to both the image
        # and this geometry, so no world-pose synchronization is required.
        fx, fy, cx, cy, height, width = self._camera_intrinsic_from_usd(camera)
        z = self.depth
        pixel_corners = np.array(
            [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ],
            dtype=np.float64,
        )
        x_cam = (pixel_corners[:, 0] - float(cx)) / max(float(fx), 1e-6) * z
        y_cam = (pixel_corners[:, 1] - float(cy)) / max(float(fy), 1e-6) * z
        # USD/OpenGL camera frame: -Z forward, +X image right, +Y image up.
        # Pixel v grows down, therefore y_cam is negated.
        corners_cam = np.stack([x_cam, -y_cam, np.full(4, -z, dtype=np.float64)], axis=1)
        origin = np.zeros(3, dtype=np.float64)
        starts = []
        ends = []
        for corner in corners_cam:
            starts.append(origin)
            ends.append(corner)
        for i in range(4):
            starts.append(corners_cam[i])
            ends.append(corners_cam[(i + 1) % 4])
        points = [origin] + [corner for corner in corners_cam]
        return (
            [tuple(float(v) for v in p) for p in starts],
            [tuple(float(v) for v in p) for p in ends],
            [tuple(float(v) for v in p) for p in points],
        )

    def update(self, scene: dict[str, object]) -> None:
        wrist_cameras = scene.get("wrist_cameras", {})
        line_count = 0
        point_count = 0
        for name in ("left_wrist", "right_wrist"):
            camera = wrist_cameras.get(name)
            geometry = self.camera_geometry.get(name)
            if camera is None or geometry is None:
                continue
            try:
                starts, ends, points = self._camera_local_frustum_geometry(camera)
            except Exception as exc:
                print(f"[WARN] failed to update {name} camera frustum: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._update_usd_geometry(
                geometry["line_ops"],
                geometry["point_ops"],
                starts,
                ends,
                points,
            )
            line_count += len(starts)
            point_count += len(points)
        self.last_line_count = line_count
        self.last_point_count = point_count
        if self.last_line_count > 0 and not self._reported_ready:
            print(
                f"[VIS] wrist camera frustums active: lines={self.last_line_count} "
                f"points={self.last_point_count} scale={self.visual_scale:.2f} "
                f"base_depth={self.base_depth:.3f}m effective_depth={self.depth:.3f}m "
                "attachment=camera_local optical_axis=local_-Z "
                f"left_prim={self.camera_geometry.get('left_wrist', {}).get('root_path')} "
                f"right_prim={self.camera_geometry.get('right_wrist', {}).get('root_path')}",
                flush=True,
            )
            self._reported_ready = True


def estimate_right_hand_tcp_pose(
    robot: Articulation,
    reach_controller: RightArmReachController | None,
    tcp_offset_wrist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if reach_controller is None:
        return None
    wrist_pose_w = robot.data.body_pose_w[:, reach_controller.right_wrist_id]
    tcp_offset = torch.tensor(tcp_offset_wrist, dtype=torch.float32, device=robot.device).view(1, 3)
    tcp_offset_w = reach_controller.rotate_wrist_vector_to_world(tcp_offset)
    tcp_pos = wrist_pose_w[0, 0:3] + tcp_offset_w[0]
    tcp_quat = wrist_pose_w[0, 3:7]
    return tcp_pos.detach().cpu().numpy(), tcp_quat.detach().cpu().numpy()


def estimate_right_hand_tcp_pose_from_robot(
    robot: Articulation,
    tcp_offset_wrist: np.ndarray = DEFAULT_TCP_OFFSET_WRIST,
) -> tuple[np.ndarray, np.ndarray] | None:
    return estimate_hand_tcp_pose_from_robot(robot, "right_wrist_yaw_link", tcp_offset_wrist)


def estimate_left_hand_tcp_pose_from_robot(
    robot: Articulation,
    tcp_offset_wrist: np.ndarray = DEFAULT_TCP_OFFSET_WRIST,
) -> tuple[np.ndarray, np.ndarray] | None:
    return estimate_hand_tcp_pose_from_robot(robot, "left_wrist_yaw_link", tcp_offset_wrist)


def estimate_hand_tcp_pose_from_robot(
    robot: Articulation,
    wrist_body_name: str,
    tcp_offset_wrist: np.ndarray = DEFAULT_TCP_OFFSET_WRIST,
) -> tuple[np.ndarray, np.ndarray] | None:
    body_ids, _ = robot.find_bodies(f"^{wrist_body_name}$")
    if not body_ids:
        print(f"[WARN] cannot estimate TCP pose; body not found: {wrist_body_name}")
        return None
    wrist_pose_w = robot.data.body_pose_w[:, body_ids[0]]
    tcp_offset = torch.tensor(tcp_offset_wrist, dtype=torch.float32, device=robot.device).view(1, 3)
    tcp_offset_w = quat_rotate_wxyz(wrist_pose_w[:, 3:7], tcp_offset)
    tcp_pos = wrist_pose_w[0, 0:3] + tcp_offset_w[0]
    tcp_quat = wrist_pose_w[0, 3:7]
    return tcp_pos.detach().cpu().numpy(), tcp_quat.detach().cpu().numpy()


def estimate_body_pose_from_robot(robot: Articulation, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    body_ids, _ = robot.find_bodies(f"^{body_name}$")
    if not body_ids:
        return None
    pose_w = robot.data.body_pose_w[:, body_ids[0]]
    return pose_w[0, 0:3].detach().cpu().numpy(), pose_w[0, 3:7].detach().cpu().numpy()


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_wxyz_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(out)
    return out if norm < 1.0e-8 else out / norm


def quat_to_matrix_wxyz_np(quat: np.ndarray) -> np.ndarray:
    from s4_robot.pink_bimanual_ik import quat_wxyz_to_matrix

    return quat_wxyz_to_matrix(np.asarray(quat, dtype=np.float64))


def pose_world_to_base(
    pose_w: tuple[np.ndarray, np.ndarray],
    base_pose_w: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if base_pose_w is None:
        return None
    pos_w, quat_w = pose_w
    base_pos_w, base_quat_w = base_pose_w
    rot_bw = quat_to_matrix_wxyz_np(base_quat_w).T
    pos_b = rot_bw @ (np.asarray(pos_w, dtype=np.float64) - np.asarray(base_pos_w, dtype=np.float64))
    quat_b = quat_mul_wxyz_np(quat_conjugate_wxyz(base_quat_w), quat_w)
    return pos_b.astype(np.float32), quat_b.astype(np.float32)


def get_usd_prim_world_pose(prim_path: str, fallback_leaf_name: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as exc:
        print(f"[WARN] could not import USD helpers for frame pose: {exc}")
        return None

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() and fallback_leaf_name:
        for candidate in stage.Traverse():
            if candidate.GetName() == fallback_leaf_name:
                prim = candidate
                break
    if not prim.IsValid():
        print(f"[WARN] frame prim not found: {prim_path}")
        return None
    xformable = UsdGeom.Xformable(prim)
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imag = rotation.GetImaginary()
    pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
    quat = np.array([rotation.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-6:
        quat = quat / quat_norm
    return pos, quat


def get_drawer_handle_top_pose() -> tuple[np.ndarray, np.ndarray] | None:
    return get_usd_prim_world_pose(
        str(args_cli.drawer_handle_frame_prim),
        fallback_leaf_name=Path(str(args_cli.drawer_handle_frame_prim)).name,
    )


def format_tcp_pose_line(
    prefix: str,
    pose: tuple[np.ndarray, np.ndarray] | None,
    base_pose_w: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    if pose is None:
        return f"{prefix} unavailable"
    pos, quat = pose
    quat_t = torch.tensor(quat, dtype=torch.float32).view(1, 4)
    roll, pitch, yaw = euler_xyz_from_quat(quat_t)
    rpy_deg = torch.rad2deg(torch.stack((roll[0], pitch[0], yaw[0]))).numpy()
    text = (
        f"{prefix} pos_w=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
        f"quat_wxyz=({quat[0]:+.5f},{quat[1]:+.5f},{quat[2]:+.5f},{quat[3]:+.5f}) "
        f"rpy_deg=({rpy_deg[0]:+.2f},{rpy_deg[1]:+.2f},{rpy_deg[2]:+.2f})"
    )
    pose_b = pose_world_to_base(pose, base_pose_w)
    if pose_b is not None:
        pos_b, quat_b = pose_b
        quat_b_t = torch.tensor(quat_b, dtype=torch.float32).view(1, 4)
        roll_b, pitch_b, yaw_b = euler_xyz_from_quat(quat_b_t)
        rpy_b_deg = torch.rad2deg(torch.stack((roll_b[0], pitch_b[0], yaw_b[0]))).numpy()
        text += (
            f" pos_base=({pos_b[0]:+.4f},{pos_b[1]:+.4f},{pos_b[2]:+.4f}) "
            f"quat_base_wxyz=({quat_b[0]:+.5f},{quat_b[1]:+.5f},{quat_b[2]:+.5f},{quat_b[3]:+.5f}) "
            f"rpy_base_deg=({rpy_b_deg[0]:+.2f},{rpy_b_deg[1]:+.2f},{rpy_b_deg[2]:+.2f})"
        )
    return text


def right_tcp_position(
    robot: Articulation,
    reach_controller: RightArmReachController,
    tcp_offset_wrist: np.ndarray,
) -> np.ndarray:
    pose = estimate_right_hand_tcp_pose(robot, reach_controller, tcp_offset_wrist)
    if pose is None:
        return np.zeros(3, dtype=np.float32)
    return pose[0]


def compose_grasp_quat(
    current_tcp_quat: np.ndarray,
    rpy: np.ndarray,
    mode: str,
    device: str,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode != "current":
        raise ValueError(f"Unsupported grasp pose mode: {mode}")
    return compose_local_rpy_quat(current_tcp_quat, rpy, device)


def compose_local_rpy_quat(
    base_quat: np.ndarray,
    rpy: np.ndarray,
    device: str,
) -> np.ndarray:
    current = torch.tensor(base_quat, dtype=torch.float32, device=device).view(1, 4)
    current = current / torch.linalg.norm(current, dim=1, keepdim=True).clamp_min(1e-6)
    rpy_t = torch.tensor(rpy, dtype=torch.float32, device=device).view(3)
    delta = quat_from_euler_xyz(rpy_t[0:1], rpy_t[1:2], rpy_t[2:3])
    target = quat_mul(current, delta)
    target = target / torch.linalg.norm(target, dim=1, keepdim=True).clamp_min(1e-6)
    return target[0].detach().cpu().numpy().astype(np.float32)


def print_right_arm_diagnostics(
    robot: Articulation,
    reach_controller: RightArmReachController,
    tcp_offset_wrist: np.ndarray,
    full_command_target: np.ndarray,
    sim,
    eps: float,
    hold_steps: int,
    drive_steps: int,
) -> None:
    eps = max(float(eps), 1e-4)
    hold_steps = max(int(hold_steps), 0)
    drive_steps = max(int(drive_steps), 0)
    joint_ids = [robot.joint_names.index(name) for name in RIGHT_ARM_JOINTS]
    q0 = robot.data.joint_pos.clone()
    qd0 = robot.data.joint_vel.clone()
    target_tensor = torch.tensor(full_command_target, dtype=torch.float32, device=robot.device).view(1, -1)
    tcp0 = right_tcp_position(robot, reach_controller, tcp_offset_wrist)
    print("[DIAG] right arm chain diagnostic begin")
    print(f"[DIAG] {reach_controller.resolution_summary()}")
    print(
        f"[DIAG] tcp0=({tcp0[0]:.5f},{tcp0[1]:.5f},{tcp0[2]:.5f}) "
        f"eps={eps:.5f} hold_steps={hold_steps} drive_steps={drive_steps}"
    )
    print(
        "[DIAG] right arm q="
        + ",".join(f"{name}:{float(q0[0, jid]):.4f}" for name, jid in zip(RIGHT_ARM_JOINTS, joint_ids, strict=True))
    )
    print(
        "[DIAG] right arm q_target="
        + ",".join(
            f"{name}:{float(full_command_target[jid]):.4f}" for name, jid in zip(RIGHT_ARM_JOINTS, joint_ids, strict=True)
        )
    )
    print(
        "[DIAG] right arm gains="
        + ",".join(
            f"{name}:kp={float(robot.data.joint_stiffness[0, jid]):.1f}/kd={float(robot.data.joint_damping[0, jid]):.1f}"
            for name, jid in zip(RIGHT_ARM_JOINTS, joint_ids, strict=True)
        )
    )
    print(
        "[DIAG] right arm effort_limits="
        + ",".join(
            f"{name}:{float(robot.data.joint_effort_limits[0, jid]):.1f}"
            for name, jid in zip(RIGHT_ARM_JOINTS, joint_ids, strict=True)
        )
    )

    if hold_steps > 0:
        hold_start = tcp0.copy()
        for _ in range(hold_steps):
            robot.set_joint_position_target(target_tensor)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(dt=sim.get_physics_dt())
        hold_end = right_tcp_position(robot, reach_controller, tcp_offset_wrist)
        hold_delta = hold_end - hold_start
        print(
            f"[DIAG] hold_drift steps={hold_steps} "
            f"delta=({hold_delta[0]:.5f},{hold_delta[1]:.5f},{hold_delta[2]:.5f}) "
            f"tcp_end=({hold_end[0]:.5f},{hold_end[1]:.5f},{hold_end[2]:.5f})"
        )

    robot.write_joint_state_to_sim(q0, qd0)
    robot.set_joint_position_target(target_tensor)
    robot.write_data_to_sim()
    robot.update(dt=sim.get_physics_dt())
    tcp0 = right_tcp_position(robot, reach_controller, tcp_offset_wrist)

    jacobians = robot.root_physx_view.get_jacobians()
    candidate_rows = []
    for row in [reach_controller.right_wrist_id - 1, reach_controller.right_wrist_id]:
        if 0 <= row < jacobians.shape[1] and row not in candidate_rows:
            candidate_rows.append(row)
    tcp_offset_t = torch.tensor(tcp_offset_wrist, dtype=torch.float32, device=robot.device).view(1, 3)
    tcp_offset_w = reach_controller.rotate_wrist_vector_to_world(tcp_offset_t)

    jac_cols_by_row = {}
    for row in candidate_rows:
        jac = jacobians[:, row, :, joint_ids]
        linear = jac[:, 0:3, :]
        angular = jac[:, 3:6, :]
        offset_cols = tcp_offset_w.unsqueeze(-1).expand_as(angular)
        tcp_jac = linear + torch.cross(angular, offset_cols, dim=1)
        jac_cols_by_row[row] = tcp_jac[0].detach().cpu().numpy()

    for local_i, (name, jid) in enumerate(zip(RIGHT_ARM_JOINTS, joint_ids, strict=True)):
        q_plus = q0.clone()
        q_minus = q0.clone()
        q_plus[0, jid] += eps
        q_minus[0, jid] -= eps
        robot.write_joint_state_to_sim(q_plus, torch.zeros_like(q_plus))
        robot.update(dt=sim.get_physics_dt())
        tcp_plus = right_tcp_position(robot, reach_controller, tcp_offset_wrist)
        robot.write_joint_state_to_sim(q_minus, torch.zeros_like(q_minus))
        robot.update(dt=sim.get_physics_dt())
        tcp_minus = right_tcp_position(robot, reach_controller, tcp_offset_wrist)
        fd_col = (tcp_plus - tcp_minus) / (2.0 * eps)
        row_parts = []
        for row, cols in jac_cols_by_row.items():
            col = cols[:, local_i]
            denom = max(float(np.linalg.norm(fd_col) * np.linalg.norm(col)), 1e-8)
            cos = float(np.dot(fd_col, col) / denom)
            row_parts.append(f"row{row}=({col[0]:+.4f},{col[1]:+.4f},{col[2]:+.4f}) cos={cos:+.3f}")
        print(
            f"[DIAG] fd {name}[id={jid}] "
            f"fd=({fd_col[0]:+.4f},{fd_col[1]:+.4f},{fd_col[2]:+.4f}) "
            + " ".join(row_parts)
        )

    if drive_steps > 0:
        print(f"[DIAG] positive joint-target response begin drive_steps={drive_steps}")
        for name, jid in zip(RIGHT_ARM_JOINTS, joint_ids, strict=True):
            hold_target = target_tensor.clone()
            robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
            robot.set_joint_position_target(hold_target)
            robot.write_data_to_sim()
            robot.reset()
            robot.update(dt=sim.get_physics_dt())
            for _ in range(4):
                robot.set_joint_position_target(hold_target)
                robot.write_data_to_sim()
                sim.step(render=True)
                robot.update(dt=sim.get_physics_dt())
            q_start = robot.data.joint_pos.clone()
            tcp_start = right_tcp_position(robot, reach_controller, tcp_offset_wrist)
            drive_target = hold_target.clone()
            drive_target[0, jid] = drive_target[0, jid] + eps
            for _ in range(drive_steps):
                robot.set_joint_position_target(drive_target)
                robot.write_data_to_sim()
                sim.step(render=True)
                robot.update(dt=sim.get_physics_dt())
            q_end = float(robot.data.joint_pos[0, jid])
            right_q_delta = (robot.data.joint_pos[0, joint_ids] - q_start[0, joint_ids]).detach().cpu().numpy()
            q_delta = q_end - float(q_start[0, jid])
            tcp_delta = right_tcp_position(robot, reach_controller, tcp_offset_wrist) - tcp_start
            print(
                f"[DIAG] drive+ {name}[id={jid}] target_delta=+{eps:.5f} q_delta={q_delta:+.5f} "
                f"tcp_delta=({tcp_delta[0]:+.5f},{tcp_delta[1]:+.5f},{tcp_delta[2]:+.5f}) "
                f"all_right_dq=({','.join(f'{x:+.5f}' for x in right_q_delta)})"
            )
        print("[DIAG] positive joint-target response end")

    robot.write_joint_state_to_sim(q0, qd0)
    robot.set_joint_position_target(target_tensor)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(dt=sim.get_physics_dt())
    print("[DIAG] right arm chain diagnostic end")


def make_right_reach_controller(robot: Articulation, device: str) -> RightArmReachController:
    return RightArmReachController(
        robot,
        device,
        max_cart_step=float(args_cli.reach_max_cart_step),
        max_joint_delta=float(args_cli.reach_max_joint_delta),
        damping=float(args_cli.reach_damping),
        posture_gain=float(args_cli.reach_posture_gain),
        max_reach_error=float(args_cli.reach_max_error),
        jacobian_body_shift=args_cli.reach_jacobian_body_shift,
        jacobian_sign=float(args_cli.reach_jacobian_sign),
        adaptive_direction_sign=bool(args_cli.reach_adaptive_direction_sign),
        min_tcp_below_block=float(args_cli.reach_min_tcp_below_block),
    )


def reset_robot_only(scene: dict[str, object], sim) -> np.ndarray:
    robot: Articulation = scene["robot"]
    default_drive = get_default_joint_positions()
    init_pos = torch.zeros(1, robot.num_joints, device=sim.device)
    for drive_i, joint_name in enumerate(ALL_DRIVE_JOINTS):
        if joint_name in robot.joint_names:
            init_pos[0, robot.joint_names.index(joint_name)] = float(default_drive[drive_i])
    robot.write_joint_state_to_sim(init_pos, torch.zeros_like(init_pos))
    robot.reset()
    return init_pos[0].detach().cpu().numpy()


def _drawer_joint_type_name(prim) -> str:
    try:
        type_name = prim.GetTypeName()
    except Exception:
        return ""
    return str(type_name)


def list_drawer_joint_prims(root_path: str = "/World/DrawerTask/DrawerCabinet") -> list[tuple[str, str]]:
    try:
        import omni.usd
    except Exception as exc:
        print(f"[WARN] could not import USD helpers for drawer joints: {exc}")
        return []

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"[WARN] drawer root not found for joint scan: {root_path}")
        return []
    joints: list[tuple[str, str]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path):
            continue
        type_name = _drawer_joint_type_name(prim)
        if "Joint" in type_name or path.endswith("_joint"):
            joints.append((path, type_name))
    return joints


def configure_drawer_drive() -> list[str]:
    try:
        import omni.usd
        from pxr import UsdPhysics
    except Exception as exc:
        print(f"[WARN] could not import USD physics helpers for drawer drive: {exc}")
        return []

    stage = omni.usd.get_context().get_stage()
    joints = list_drawer_joint_prims()
    if joints:
        print("[DRAWER] detected joints:")
        for path, type_name in joints:
            print(f"[DRAWER]   {path} type={type_name}")

    selected: list[str] = []
    joint_filter = str(args_cli.drawer_joint_filter)
    for path, type_name in joints:
        if joint_filter and joint_filter not in path:
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        drive_kind = "linear" if "Prismatic" in type_name else "angular"
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_kind)
        drive.CreateTargetPositionAttr().Set(float(args_cli.drawer_target))
        drive.CreateStiffnessAttr().Set(float(args_cli.drawer_drive_stiffness))
        drive.CreateDampingAttr().Set(float(args_cli.drawer_drive_damping))
        drive.CreateMaxForceAttr().Set(float(args_cli.drawer_drive_max_force))
        selected.append(path)
        print(
            f"[DRAWER] drive {path} kind={drive_kind} target={float(args_cli.drawer_target):.3f} "
            f"kp={float(args_cli.drawer_drive_stiffness):.1f} kd={float(args_cli.drawer_drive_damping):.1f}",
            flush=True,
        )
    if not selected:
        print(f"[WARN] no drawer joints matched filter: {joint_filter!r}")
    return selected


def run_static_task_scene(scene: dict[str, object], cfg: SceneBuildCfg, sim) -> None:
    robot: Articulation = scene["robot"]
    drawer: Articulation | None = scene.get("drawer")
    camera = scene["camera"]
    sim_dt = sim.get_physics_dt()
    reset_settle_steps = max(
        int(args_cli.reset_settle_steps),
        int(math.ceil(max(float(args_cli.reset_settle_s), 0.0) / max(float(sim_dt), 1.0e-6))),
    )

    task_spec = get_task_spec(str(scene.get("task_id")))
    drawer_scripted_config = None
    drawer_randomization_cfg: dict[str, object] = {}
    drawer_task_cfg: dict[str, object] = {}
    drawer_dashboard_cfg: dict[str, object] = {}
    drawer_rng = None
    can_grid_sampler: StratifiedGrid2D | None = None
    current_grid_sample = None
    grasp_retry_count = 0
    skipped_grid_cells: list[dict[str, int]] = []
    episode_context: dict[str, object] = {}
    phase_state_history: list[dict[str, object]] = []
    closed_handle_pose_w = get_drawer_handle_top_pose() if scene.get("task_id") == "drawer_insert_close" else None
    if scene.get("task_id") == "drawer_insert_close":
        if task_spec.scripted_config is None:
            raise ValueError(f"Task has no scripted_config: {task_spec.task_id}")
        drawer_scripted_config = Path(args_cli.drawer_scripted_config or task_spec.scripted_config).resolve()
        scripted_cfg = load_yaml(drawer_scripted_config)
        drawer_language_contract = load_language_phase_contract(scripted_cfg)
        drawer_randomization_cfg = scripted_cfg.get("randomization", {})
        drawer_task_cfg = dict(scripted_cfg.get("drawer", {}) or {})
        logging_cfg = scripted_cfg.get("logging", {})
        if isinstance(logging_cfg, dict):
            dashboard_cfg = logging_cfg.get("progress_dashboard", {})
            if isinstance(dashboard_cfg, dict):
                drawer_dashboard_cfg = dashboard_cfg
        drawer_seed = int(
            args_cli.random_seed
            if args_cli.random_seed is not None
            else drawer_randomization_cfg.get("seed", 42)
        )
        drawer_randomization_cfg = dict(drawer_randomization_cfg)
        drawer_randomization_cfg["effective_seed"] = drawer_seed
        can_cfg = dict(drawer_randomization_cfg.get("can_xy", {}) or {})
        distractor_cfg = dict(drawer_randomization_cfg.get("distractor_cans", {}) or {})
        can_cfg["enabled"] = resolve_record_can_xy_enabled(scripted_cfg)
        distractor_cfg["enabled"] = resolve_record_distractor_cans_enabled(scripted_cfg)
        drawer_randomization_cfg["can_xy"] = can_cfg
        drawer_randomization_cfg["distractor_cans"] = distractor_cfg
        print(
            f"[RECORD] can_xy_randomization={bool(can_cfg['enabled'])} "
            f"distractor_cans={bool(distractor_cfg['enabled'])} "
            f"drawer_initial_open=fixed:{float(drawer_task_cfg.get('initial_open_m', 0.0)):.3f}m",
            flush=True,
        )
        drawer_rng = np.random.default_rng(drawer_seed)
        if can_cfg.get("enabled", False) and str(can_cfg.get("sampling", "uniform")) == "stratified_grid":
            grid_cells = can_cfg.get("grid_cells", [5, 5])
            if not isinstance(grid_cells, (list, tuple)) or len(grid_cells) != 2:
                raise ValueError("randomization.can_xy.grid_cells must contain [x_cells, y_cells]")
            can_grid_sampler = StratifiedGrid2D(
                drawer_rng,
                x_range=tuple(float(value) for value in can_cfg.get("x_range", [-0.05, 0.05])),
                y_range=tuple(float(value) for value in can_cfg.get("y_range", [-0.05, 0.05])),
                cells_x=int(grid_cells[0]),
                cells_y=int(grid_cells[1]),
            )

    def sample_drawer_episode(*, grid_sample_override=None) -> dict[str, object]:
        nonlocal current_grid_sample
        if drawer_rng is None:
            return {}
        can_cfg = drawer_randomization_cfg.get("can_xy", {})
        x_range = can_cfg.get("x_range", [0.0, 0.0]) if can_cfg.get("enabled", False) else [0.0, 0.0]
        y_range = can_cfg.get("y_range", [0.0, 0.0]) if can_cfg.get("enabled", False) else [0.0, 0.0]
        if can_grid_sampler is not None:
            grid_sample = grid_sample_override or can_grid_sampler.sample()
            current_grid_sample = grid_sample
            can_xy_offset = grid_sample.xy.tolist()
            grid_metadata: dict[str, object] = {
                "can_grid_cell": [grid_sample.cell_x, grid_sample.cell_y],
                "can_grid_cycle": grid_sample.cycle,
                "can_grid_index_in_cycle": grid_sample.index_in_cycle,
                "can_grid_point_attempt": int(grasp_retry_count + 1),
                "can_grasp_retry_count": int(grasp_retry_count),
            }
        else:
            can_xy_offset = [
                float(drawer_rng.uniform(float(x_range[0]), float(x_range[1]))),
                float(drawer_rng.uniform(float(y_range[0]), float(y_range[1]))),
            ]
            grid_metadata = {}
        lift_cfg = drawer_randomization_cfg.get("right_can_lift", {})
        lift_ranges = lift_cfg.get("offset_ranges", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        lift_offset = (
            sample_xyz_range(drawer_rng, lift_ranges).tolist()
            if lift_cfg.get("enabled", False)
            else [0.0, 0.0, 0.0]
        )
        distractor_positions: dict[str, list[float]] = {}
        distractor_cfg = drawer_randomization_cfg.get("distractor_cans", {})
        distractor_names = DISTRACTOR_OBJECT_NAMES
        named_objects = scene.get("named_objects", {})
        if distractor_cfg.get("enabled", False) and all(name in named_objects for name in distractor_names):
            main_xy = np.asarray(scene.get("can_initial_position", GRASP_CAN_NOMINAL_POSITION)[:2], dtype=np.float32) + np.asarray(
                can_xy_offset, dtype=np.float32
            )
            sampled_xy = sample_separated_xy(
                drawer_rng,
                ranges=distractor_cfg.get(
                    "ranges",
                    DEFAULT_DISTRACTOR_RANGES,
                ),
                forbidden_xy=[main_xy.tolist()],
                min_center_distance=float(distractor_cfg.get("min_center_distance_m", 0.16)),
            )
            # Randomly permute object-to-region assignment as well as sampling
            # continuously within each region. All three styles stay present.
            region_order = drawer_rng.permutation(len(distractor_names))
            distractor_positions = {
                name: sampled_xy[int(region_order[index])].tolist()
                for index, name in enumerate(distractor_names)
            }
        return {
            "can_xy_offset": can_xy_offset,
            **grid_metadata,
            "distractor_can_xy": distractor_positions,
            "right_can_lift_offset": lift_offset,
        }

    def collection_state() -> dict[str, object]:
        return {
            "version": 2,
            "rng_state": None if drawer_rng is None else drawer_rng.bit_generator.state,
            "grid_state": None if can_grid_sampler is None else can_grid_sampler.state_dict(),
            "episode_context": dict(episode_context),
            "grasp_retry_count": int(grasp_retry_count),
            "skipped_grid_cells": list(skipped_grid_cells),
        }

    def restore_collection_state(state: dict[str, object]) -> bool:
        nonlocal episode_context, current_grid_sample, grasp_retry_count, skipped_grid_cells
        if drawer_rng is None or can_grid_sampler is None:
            return False
        rng_state = state.get("rng_state")
        grid_state = state.get("grid_state")
        if not isinstance(rng_state, dict) or not isinstance(grid_state, dict):
            return False
        drawer_rng.bit_generator.state = rng_state
        can_grid_sampler.load_state_dict(grid_state)
        episode_context = dict(state.get("episode_context", {}) or {})
        grasp_retry_count = int(
            state.get("grasp_retry_count", max(int(state.get("grid_point_attempt", 1)) - 1, 0))
        )
        skipped_grid_cells = list(state.get("skipped_grid_cells", []) or [])
        if episode_context and "can_grid_cell" in episode_context:
            cell = episode_context["can_grid_cell"]
            from s4_pipeline.randomization import StratifiedGridSample

            current_grid_sample = StratifiedGridSample(
                xy=np.asarray(episode_context["can_xy_offset"], dtype=np.float32),
                cell_x=int(cell[0]),
                cell_y=int(cell[1]),
                cycle=int(episode_context.get("can_grid_cycle", 0)),
                index_in_cycle=int(episode_context.get("can_grid_index_in_cycle", 0)),
            )
        return True

    def select_after_failed_attempt(phase_name: str, reason: str) -> None:
        """Apply phase-aware retry policy without advancing the stratified cell."""
        nonlocal episode_context, current_grid_sample, grasp_retry_count
        can_cfg = drawer_randomization_cfg.get("can_xy", {})
        max_retries = max(int(can_cfg.get("max_grasp_retries_same_position", 3)), 0)
        decision = decide_drawer_retry(
            phase_name,
            grasp_retry_count=grasp_retry_count,
            max_grasp_retries_same_position=max_retries,
        )
        if decision.retry_same_position:
            grasp_retry_count = decision.next_grasp_retry_count
            episode_context["can_grid_point_attempt"] = int(grasp_retry_count + 1)
            episode_context["can_grasp_retry_count"] = int(grasp_retry_count)
            log_collection_event(
                "GRASP-RETRY",
                f"same_position retry={grasp_retry_count}/{max_retries} "
                f"cell={episode_context.get('can_grid_cell')} "
                f"xy={episode_context.get('can_xy_offset')} after={reason}",
                "yellow",
            )
            return

        if decision.exhausted_grasp_position:
            log_collection_event(
                "GRASP-POSITION-EXHAUSTED",
                f"same_position retries={max_retries}; replacing the precise point "
                f"inside cell={episode_context.get('can_grid_cell')}",
                "yellow",
            )
        else:
            log_collection_event(
                "NON-GRASP-DISCARD",
                f"phase={phase_name}; replacing the precise point inside "
                f"cell={episode_context.get('can_grid_cell')} after={reason}",
                "yellow",
            )
        grasp_retry_count = 0
        if can_grid_sampler is None or current_grid_sample is None:
            episode_context = sample_drawer_episode()
            return
        next_sample = can_grid_sampler.resample_cell(current_grid_sample)
        episode_context = sample_drawer_episode(grid_sample_override=next_sample)
        log_collection_event(
            "GRID-RESAMPLE",
            f"cell=({next_sample.cell_x},{next_sample.cell_y}) new_xy={next_sample.xy.tolist()}",
            "yellow",
        )

    drawer_top_joint_id = None
    if drawer is not None:
        drawer_joint_name = str(drawer_task_cfg.get("joint_name", "drawer_top_joint"))
        drawer_joint_ids, _ = drawer.find_joints(f"^{drawer_joint_name}$")
        if len(drawer_joint_ids) != 1:
            raise RuntimeError(f"Expected one {drawer_joint_name}, found ids={drawer_joint_ids}")
        drawer_top_joint_id = int(drawer_joint_ids[0])
        drawer_limits = drawer.data.soft_joint_pos_limits[0, drawer_top_joint_id].detach().cpu().numpy()
        drawer_stiffness = float(drawer.data.joint_stiffness[0, drawer_top_joint_id].item())
        drawer_damping = float(drawer.data.joint_damping[0, drawer_top_joint_id].item())
        drawer_static_friction = float(
            drawer.data.joint_friction_coeff[0, drawer_top_joint_id].item()
        )
        drawer_dynamic_friction = float(
            drawer.data.joint_dynamic_friction_coeff[0, drawer_top_joint_id].item()
        )
        drawer_viscous_friction = float(
            drawer.data.joint_viscous_friction_coeff[0, drawer_top_joint_id].item()
        )
        print(
            f"[DRAWER] {drawer_joint_name} passive joint limits="
            f"[{float(drawer_limits[0]):.3f},{float(drawer_limits[1]):.3f}]m "
            f"stiffness={drawer_stiffness:.3f} damping={drawer_damping:.3f} "
            f"friction={drawer_static_friction:.3f}/{drawer_dynamic_friction:.3f}/"
            f"{drawer_viscous_friction:.3f}",
            flush=True,
        )

    def reset_drawer(context: dict[str, object]) -> None:
        if drawer is None or drawer_top_joint_id is None:
            return
        amount = float(drawer_task_cfg.get("initial_open_m", 0.0))
        sign = float(drawer_task_cfg.get("joint_position_sign", 1.0))
        drawer.reset()
        joint_pos = drawer.data.default_joint_pos.clone()
        joint_vel = drawer.data.default_joint_vel.clone()
        joint_pos[:, drawer_top_joint_id] = sign * amount
        joint_vel.zero_()
        drawer.write_joint_state_to_sim(joint_pos, joint_vel)

    def current_drawer_open_m() -> float | None:
        if drawer is None or drawer_top_joint_id is None:
            return None
        sign = float(drawer_task_cfg.get("joint_position_sign", 1.0))
        return sign * float(drawer.data.joint_pos[0, drawer_top_joint_id].item())

    def evaluate_drawer_task_success() -> tuple[bool, dict[str, object]]:
        """Validate the final physical task state before persisting an episode."""
        success_cfg = scripted_cfg.get("success", {})
        drawer_open = current_drawer_open_m()
        bounds_cfg = success_cfg.get("can_world_bounds", {}) or {}
        axes = ("x", "y", "z")
        can_bounds: dict[str, tuple[float, float]] = {}
        for axis in axes:
            values = bounds_cfg.get(axis)
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"success.can_world_bounds.{axis} requires [min_m, max_m]")
            lower, upper = (float(value) for value in values)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"success.can_world_bounds.{axis} requires finite min_m < max_m")
            can_bounds[axis] = (lower, upper)

        can_obj = scene.get("named_objects", {}).get("can")
        can_world_position = [float("nan"), float("nan"), float("nan")]
        if can_obj is not None:
            can_pose_tensor = can_obj.data.root_pose_w[0]
            can_world_position = [float(value) for value in can_pose_tensor[0:3].detach().cpu().tolist()]
        can_in_drawer = bool(
            all(
                np.isfinite(value) and can_bounds[axis][0] <= value <= can_bounds[axis][1]
                for axis, value in zip(axes, can_world_position)
            )
        )
        details: dict[str, object] = {
            "accepted": can_in_drawer,
            "drawer_open_m": None if drawer_open is None else float(drawer_open),
            "can_world_position_m": can_world_position,
            "can_in_drawer": can_in_drawer,
            "can_world_bounds_m": {axis: list(can_bounds[axis]) for axis in axes},
        }
        return can_in_drawer, details

    def settle_static_target(full_target: np.ndarray) -> np.ndarray:
        target_tensor = torch.tensor(full_target, dtype=torch.float32, device=robot.device).view(1, -1)
        for _ in range(reset_settle_steps):
            robot.set_joint_position_target(target_tensor)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(dt=sim_dt)
            if drawer is not None:
                drawer.update(dt=sim_dt)
            for obj in scene.get("dynamic_objects", []):
                obj.update(dt=sim_dt)
            update_all_cameras(scene, camera, dt=sim_dt)
        return robot.data.joint_pos[0].detach().cpu().numpy().copy()

    def reset_static_objects(context: dict[str, object]) -> None:
        can_offset = np.asarray(context.get("can_xy_offset", [0.0, 0.0]), dtype=np.float32)
        distractor_xy = context.get("distractor_can_xy", {})
        if not isinstance(distractor_xy, dict):
            distractor_xy = {}
        named_objects = scene.get("named_objects", {})
        for obj, pos, quat in scene.get("object_initial_poses", []):
            object_pos = np.asarray(pos, dtype=np.float32).copy()
            if obj is named_objects.get("can"):
                object_pos[:2] += can_offset
            else:
                for name in DISTRACTOR_OBJECT_NAMES:
                    if obj is named_objects.get(name) and name in distractor_xy:
                        object_pos[:2] = np.asarray(distractor_xy[name], dtype=np.float32)
                        break
            write_object_pose(obj, object_pos, sim.device, quat)
            obj.update(dt=sim_dt)

    def reset_static_attempt(*, sample_new_context: bool = False) -> np.ndarray:
        nonlocal episode_context
        if sample_new_context or not episode_context:
            episode_context = sample_drawer_episode()
        sim.reset()
        next_target = reset_robot_only(scene, sim)
        if scene.get("task_id") == "drawer_insert_close":
            hand_cfg = scripted_cfg.get("hands", {})
            reset_action = control_action_from_full_target(next_target, robot)
            reset_action[ACTION_SLICES.left_hand] = np.asarray(
                hand_cfg.get("left_open", [0.9, 0.0, 0.05, 0.05, 0.05, 0.05]),
                dtype=np.float32,
            )
            reset_action[ACTION_SLICES.right_hand] = np.asarray(
                hand_cfg.get("right_open", [0.9, 0.0, 0.05, 0.05, 0.05, 0.05]),
                dtype=np.float32,
            )
            write_action_to_full_target(next_target, robot, reset_action)
            # Reset both the measured state and the drive target to the same
            # configured open-hand pose. Previously the state was written at
            # DEFAULT_POSE (0.3-rad bent fingers) and only the target was made
            # open, which could occasionally leave one finger resting above
            # the 0-rad hard stop for an entire readiness phase.
            reset_joint_state = torch.tensor(
                next_target, dtype=torch.float32, device=sim.device
            ).view(1, -1)
            robot.write_joint_state_to_sim(
                reset_joint_state,
                torch.zeros_like(reset_joint_state),
            )
            robot.reset()
        reset_drawer(episode_context)
        reset_static_objects(episode_context)
        robot.set_joint_position_target(torch.tensor(next_target, device=sim.device).view(1, -1))
        robot.write_data_to_sim()
        reset_camera(camera, sim, cfg)
        settled = settle_static_target(next_target)
        next_target[:] = settled
        return next_target

    target = reset_static_attempt(sample_new_context=True)

    if args_cli.drawer_coast_diagnostic:
        if drawer is None or drawer_top_joint_id is None:
            raise RuntimeError("--drawer-coast-diagnostic requires the drawer task articulation")
        # Remove task contacts from the experiment. This isolates the authored
        # drawer joint, cabinet collision geometry, and PhysX articulation.
        for obj, pos, quat in scene.get("object_initial_poses", []):
            isolated_pos = np.asarray(pos, dtype=np.float32).copy()
            isolated_pos[2] += 3.0
            write_object_pose(obj, isolated_pos, sim.device, quat)
            obj.update(dt=sim_dt)
        joint_pos = drawer.data.joint_pos.clone()
        joint_vel = drawer.data.joint_vel.clone()
        joint_pos[:, drawer_top_joint_id] = float(args_cli.drawer_coast_start)
        joint_vel.zero_()
        joint_vel[:, drawer_top_joint_id] = float(args_cli.drawer_coast_velocity)
        drawer.write_joint_state_to_sim(joint_pos, joint_vel)
        print(
            f"[DRAWER-DIAG] isolated coast start q={float(args_cli.drawer_coast_start):+.6f}m "
            f"qd={float(args_cli.drawer_coast_velocity):+.6f}m/s steps={int(args_cli.drawer_coast_steps)}",
            flush=True,
        )
        for step in range(max(int(args_cli.drawer_coast_steps), 1)):
            robot.set_joint_position_target(torch.tensor(target, device=sim.device).view(1, -1))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt=sim_dt)
            drawer.update(dt=sim_dt)
            for obj in scene.get("dynamic_objects", []):
                obj.update(dt=sim_dt)
            if step % 12 == 0 or step == int(args_cli.drawer_coast_steps) - 1:
                q = float(drawer.data.joint_pos[0, drawer_top_joint_id].item())
                qd = float(drawer.data.joint_vel[0, drawer_top_joint_id].item())
                print(f"[DRAWER-DIAG] step={step:04d} q={q:+.6f}m qd={qd:+.6f}m/s", flush=True)
        q = float(drawer.data.joint_pos[0, drawer_top_joint_id].item())
        qd = float(drawer.data.joint_vel[0, drawer_top_joint_id].item())
        print(f"[DRAWER-DIAG] final q={q:+.6f}m qd={qd:+.6f}m/s", flush=True)
        # Kit can hang while tearing down camera/render resources after this
        # short diagnostic, leaving a multi-GB GPU process behind. No state
        # needs to be persisted here, so terminate after flushing diagnostics.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    print(scene.get("layout_text", "[SCENE] static task scene ready."))
    print(
        "Wrist cameras: "
        f"left_pos={_fmt_tuple(cfg.left_wrist_camera_pos, 4)} "
        f"left_quat_wxyz={_fmt_tuple(cfg.left_wrist_camera_quat_wxyz, 4)} "
        f"left_rpy_override_deg={_fmt_tuple(cfg.left_wrist_camera_rpy_deg, 2)} "
        f"right_pos={_fmt_tuple(cfg.right_wrist_camera_pos, 4)} "
        f"right_quat_wxyz={_fmt_tuple(cfg.right_wrist_camera_quat_wxyz, 4)} "
        f"right_rpy_override_deg={_fmt_tuple(cfg.right_wrist_camera_rpy_deg, 2)} "
        f"convention={cfg.wrist_camera_convention} optical_axis={'local_+Z' if cfg.wrist_camera_convention == 'ros' else 'local_-Z'}"
    )
    drawer_drive_paths = configure_drawer_drive() if args_cli.drawer_open else []
    if drawer_drive_paths:
        print("[SCENE] drawer drive preview: physics is stepped so the driven joint can move.")
    elif args_cli.show_wrist_camera_frustums and not args_cli.headless:
        print("[SCENE] static preview mode: physics/render is stepped so wrist camera frustums can refresh.")
    else:
        print("[SCENE] static preview mode: GUI keeps rendering; physics is not stepped unless headless.")
    tcp_visualizer = None
    if (args_cli.show_tcp_frames or args_cli.show_drawer_handle_frame) and not args_cli.headless:
        tcp_visualizer = TcpFrameVisualizer(sim.device)
        print(
            "Debug frames: /World/Visuals/LeftHandTCP, /World/Visuals/RightHandTCP, "
            "/World/Visuals/DrawerHandleTop"
        )
    wrist_frustum_visualizer = None
    if args_cli.show_wrist_camera_frustums and not args_cli.headless:
        wrist_frustum_visualizer = scene.get("wrist_frustum_visualizer")
        if wrist_frustum_visualizer is None:
            raise RuntimeError("wrist frustum geometry was not created before SimulationContext.reset()")
        print(
            "Wrist camera frustums: camera-local USD geometry "
            "(LeftWristCamera/DebugFrustum=cyan, RightWristCamera/DebugFrustum=orange)"
        )
    action = control_action_from_full_target(target, robot)
    keyboard_jog = None
    if args_cli.keyboard_jog and not args_cli.headless:
        candidate = KeyboardJog(action, jog_step=float(args_cli.jog_step))
        if candidate.start():
            keyboard_jog = candidate
            print("Keyboard jog: '['/']' select joint, 'u' increase, 'j' decrease, 'r' reset, 'p' print selected.")
    args_cli.arm_control_file.parent.mkdir(parents=True, exist_ok=True)
    args_cli.arm_control_file.write_text(json.dumps({"mode": "idle"}, indent=2), encoding="utf-8")
    last_arm_control_mtime = args_cli.arm_control_file.stat().st_mtime_ns
    print(f"Arm control file: {args_cli.arm_control_file}")
    last_tcp_print = 0.0
    pink_tcp_controller = None
    drawer_controller = None
    scripted_drawer_enabled = bool(
        scene.get("task_id") == "drawer_insert_close" and (args_cli.record_output is not None or args_cli.auto_grasp)
    )
    writer = None
    failure_reporter = None
    recording_episode = None
    recorded_episodes = 0
    record_attempt = 1
    failed_attempts_total = 0
    record_complete = False
    record_step = 0
    record_wall_start = None
    collection_wall_start = None
    current_scripted_task = str(scene.get("task_description", "Open the drawer, place the object inside, and close the drawer."))
    current_scripted_phase = "idle"
    last_logged_phase_index = -1
    max_record_episodes = max(int(args_cli.record_episodes), 1)
    record_every_n = max(int(args_cli.record_every_n), 1)
    arm_control_active = False
    tcp_pose_active = False
    tcp_pose_goal_left = None
    tcp_pose_goal_right = None
    last_arm_debug = 0.0
    gravity_comp_joint_ids = resolve_existing_joint_ids(robot, list(ALL_DRIVE_JOINTS))
    last_gravity_comp_stats = (0.0, 0.0)
    print(
        f"[SCENE] drawer preview gravity_compensation={bool(args_cli.gravity_compensation)} "
        f"scale={float(args_cli.gravity_comp_scale):.2f} joints={len(gravity_comp_joint_ids)}",
        flush=True,
    )
    if args_cli.record_output is not None:
        failure_log_path = args_cli.failure_log or args_cli.record_output.with_name(
            f"{args_cli.record_output.stem}_failures.jsonl"
        )
        failure_summary_path = args_cli.failure_summary or args_cli.record_output.with_name(
            f"{args_cli.record_output.stem}_failure_summary.json"
        )
        recording_env_args = {
                "task": str(scene.get("task_id", "drawer_insert_close")),
                "source": "scripted_yaml_bimanual_tcp_ik",
                "sim_dt": float(sim_dt),
                "record_every_n": int(record_every_n),
                "record_episode_timeout_s": float(max(float(args_cli.record_episode_timeout_s), 1.0)),
                "reset_settle_s": float(max(float(args_cli.reset_settle_s), 0.0)),
                "reset_settle_steps": int(reset_settle_steps),
                "scripted_config": str(drawer_scripted_config) if drawer_scripted_config is not None else None,
                "ik_runtime": {
                    "solver": "PinkBimanualTcpController",
                    "posture_gain": float(args_cli.tcp_posture_gain),
                    "damping": float(args_cli.tcp_ik_damping),
                    "max_joint_delta": float(args_cli.tcp_max_joint_delta),
                    "tcp_offset_wrist_m": [float(value) for value in DEFAULT_TCP_OFFSET_WRIST],
                },
                "gravity_compensation": {
                    "enabled": bool(args_cli.gravity_compensation),
                    "scale": float(args_cli.gravity_comp_scale),
                },
                "randomization": drawer_randomization_cfg,
                "success_filter": scripted_cfg.get("success", {}),
                "distractor_cans_enabled": bool(
                    (drawer_randomization_cfg.get("distractor_cans") or {}).get("enabled", False)
                )
                and all(name in scene.get("named_objects", {}) for name in DISTRACTOR_OBJECT_NAMES),
                "distractor_assets": (
                    distractor_asset_contract()
                    if bool((drawer_randomization_cfg.get("distractor_cans") or {}).get("enabled", False))
                    and all(name in scene.get("named_objects", {}) for name in DISTRACTOR_OBJECT_NAMES)
                    else []
                ),
                "grasp_can_nominal_position": list(GRASP_CAN_NOMINAL_POSITION),
                "grasp_can_scale": list(GRASP_CAN_SCALE),
                "record_fps": float(1.0 / (sim_dt * record_every_n)),
                "camera": {
                    "eye": list(cfg.camera_eye),
                    "target": list(cfg.camera_target),
                    "rpy_deg": None if cfg.camera_rpy_deg is None else list(cfg.camera_rpy_deg),
                    "convention": str(cfg.camera_convention),
                    "width": int(cfg.camera_width),
                    "height": int(cfg.camera_height),
                },
                "data_contract": {
                    "state_dim": 26,
                    "action_dim": 26,
                    "state_order": "left_arm_7,left_hand_6,right_arm_7,right_hand_6",
                },
                "language_contract": {
                    "version": drawer_language_contract.version,
                    "phases": drawer_language_contract.as_portable_records(),
                },
            }
        writer = Hdf5DemoWriter(
            args_cli.record_output,
            env_args=recording_env_args,
            resume=bool(args_cli.resume),
            overwrite=False,
        )
        # Open/validate the HDF5 first. This prevents a mistyped non-resume
        # command from truncating the existing failure log before HDF5 refuses
        # to overwrite its existing data.
        failure_reporter = CollectionFailureReporter(
            failure_log_path,
            failure_summary_path,
            resume=bool(args_cli.resume),
        )
        failed_attempts_total = len(failure_reporter.events)
        log_collection_event(
            "FAILURE-LOG",
            f"events={failure_log_path.resolve()} | summary={failure_summary_path.resolve()}",
            "cyan",
        )
        recorded_episodes = writer.episode_count
        if (
            args_cli.max_failed_attempts is not None
            and failed_attempts_total > int(args_cli.max_failed_attempts)
        ):
            failure_reporter.finalize(
                completed=False,
                accepted_episodes=recorded_episodes,
                target_episodes=max_record_episodes,
                skipped_grid_cells=skipped_grid_cells,
                hdf5_path=args_cli.record_output,
            )
            writer.close()
            writer = None
            raise RuntimeError(
                f"Cannot resume strict collection: existing failed_attempts={failed_attempts_total} "
                f"exceeds --max-failed-attempts={int(args_cli.max_failed_attempts)}"
            )
        if recorded_episodes > max_record_episodes:
            raise ValueError(
                f"Resume target {max_record_episodes} is below existing episode count {recorded_episodes}"
            )
        if args_cli.resume and recorded_episodes:
            saved_state = writer.read_collection_state()
            restored = bool(saved_state and restore_collection_state(saved_state))
            if restored and int(saved_state.get("completed_episodes", -1)) != recorded_episodes:
                restored = False
            if not restored:
                # Legacy recordings have no collection_state. Rebuild the RNG
                # and grid traversal from their accepted episode count. The old
                # retry behavior did not consume random samples on failure.
                drawer_rng.bit_generator.state = np.random.default_rng(drawer_seed).bit_generator.state
                if can_grid_sampler is not None:
                    can_grid_sampler.load_state_dict({"order": [], "cursor": 0, "cycle": -1})
                grasp_retry_count = 0
                skipped_grid_cells = []
                episode_context = {}
                for _ in range(recorded_episodes):
                    sample_drawer_episode()
                episode_context = sample_drawer_episode()
                log_collection_event(
                    "RESUME",
                    f"replayed legacy sampler to {recorded_episodes} accepted episode(s)",
                    "yellow",
                )
            else:
                log_collection_event(
                    "RESUME",
                    f"restored exact sampler state at {recorded_episodes} accepted episode(s)",
                    "cyan",
                )
            target = reset_static_attempt()
            action = control_action_from_full_target(target, robot)
        if recorded_episodes >= max_record_episodes:
            log_collection_event(
                "COMPLETE",
                f"file already contains target {recorded_episodes}/{max_record_episodes} episodes",
                "green",
            )
            failure_reporter.finalize(
                completed=True,
                accepted_episodes=recorded_episodes,
                target_episodes=max_record_episodes,
                skipped_grid_cells=skipped_grid_cells,
                hdf5_path=args_cli.record_output,
            )
            writer.close()
            return
        writer.write_collection_state({**collection_state(), "completed_episodes": recorded_episodes})
        log_collection_event(
            "COLLECT",
            f"output={args_cli.record_output} | existing={recorded_episodes} | "
            f"target_total_successes={max_record_episodes} | "
            f"fps={1.0 / (sim_dt * record_every_n):.1f} | "
            f"timeout={max(float(args_cli.record_episode_timeout_s), 1.0):.1f}s",
            "cyan",
        )

    def current_drawer_anchors_base() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
        if base_pose_w is None:
            raise RuntimeError("Robot base_link pose is unavailable for drawer task anchors")
        can_obj = scene.get("named_objects", {}).get("can")
        if can_obj is None:
            raise RuntimeError("Drawer task scene did not expose named_objects.can")
        can_pose_tensor = can_obj.data.root_pose_w[0]
        can_pose_w = (
            can_pose_tensor[0:3].detach().cpu().numpy(),
            can_pose_tensor[3:7].detach().cpu().numpy(),
        )
        can_pose_b = pose_world_to_base(can_pose_w, base_pose_w)
        closed_handle_pose_b = pose_world_to_base(closed_handle_pose_w, base_pose_w) if closed_handle_pose_w else None
        if can_pose_b is None or closed_handle_pose_b is None:
            raise RuntimeError("Can or closed drawer-handle pose is unavailable")
        opening_axis = np.asarray(drawer_task_cfg.get("opening_axis_base", [-1.0, 0.0, 0.0]), dtype=np.float32)
        axis_norm = float(np.linalg.norm(opening_axis))
        if axis_norm < 1.0e-6:
            raise ValueError("drawer.opening_axis_base must be non-zero")
        opening_axis /= axis_norm
        initial_open = float(drawer_task_cfg.get("initial_open_m", 0.0))
        target_open = float(drawer_task_cfg.get("target_open_m", 0.06))
        initial_handle_pose_b = (
            closed_handle_pose_b[0] + opening_axis * initial_open,
            closed_handle_pose_b[1].copy(),
        )
        open_handle_pose_b = (
            closed_handle_pose_b[0] + opening_axis * target_open,
            closed_handle_pose_b[1].copy(),
        )
        return {
            "can": can_pose_b,
            "drawer_handle_closed": closed_handle_pose_b,
            "drawer_handle_initial": initial_handle_pose_b,
            "drawer_handle_open": open_handle_pose_b,
        }

    def make_recording_episode() -> EpisodeBuffer:
        return EpisodeBuffer(
            metadata={
                "randomization": dict(episode_context),
                "scripted_config": str(drawer_scripted_config) if drawer_scripted_config is not None else None,
                "language_contract_version": drawer_language_contract.version,
            }
        )

    def new_drawer_controller(initial_action: np.ndarray):
        nonlocal pink_tcp_controller
        if pink_tcp_controller is None:
            from s4_robot.pink_bimanual_ik import PinkBimanualTcpController

            pink_tcp_controller = PinkBimanualTcpController(
                robot,
                sim.device,
                posture_gain=args_cli.tcp_posture_gain,
                damping=args_cli.tcp_ik_damping,
                max_joint_delta=args_cli.tcp_max_joint_delta,
            )
            log_collection_event("IK", "Pinocchio DLS ready | target_frame=base_link", "cyan")
        anchors = current_drawer_anchors_base()
        controller_class = import_symbol(task_spec.scripted_controller)
        return controller_class(
            pink_tcp_controller,
            config_path=drawer_scripted_config,
            initial_action=initial_action,
            anchors=anchors,
            target_offsets={
                "right_can_lift": np.asarray(
                    episode_context.get("right_can_lift_offset", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                )
            },
        )

    def log_attempt_start(controller) -> None:
        phase_state_history.clear()
        log_collection_event(
            "EPISODE",
            f"EP{recorded_episodes + 1:03d}/{max_record_episodes:03d} "
            f"TRY{record_attempt:02d} | phases={len(controller.phases)}",
            "cyan",
        )

    def capture_current_phase_state(controller) -> None:
        """Keep detailed phase snapshots for failure files without flashing terminal output."""
        phase = controller.current_phase
        can_obj = scene.get("named_objects", {}).get("can")
        can_pos = None if can_obj is None else can_obj.data.root_pos_w[0].detach().cpu().numpy()
        right_pose_w = estimate_right_hand_tcp_pose_from_robot(robot)
        base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
        right_pose_b = pose_world_to_base(right_pose_w, base_pose_w) if right_pose_w is not None else None
        right_target = None if phase.right is None else phase.right.pos
        right_error = (
            float("nan")
            if right_pose_b is None or right_target is None
            else float(np.linalg.norm(np.asarray(right_target) - np.asarray(right_pose_b[0])))
        )
        start_pos = controller._task_object_start_position_world
        can_displacement = (
            float("nan")
            if can_pos is None or start_pos is None
            else float(np.linalg.norm(np.asarray(can_pos) - np.asarray(start_pos)))
        )
        actual_action = control_action_from_sim(robot)
        fingertip_positions: dict[str, list[float]] = {}
        for body_name in (
            "rh_thumb_distal",
            "rh_index_distal",
            "rh_middle_distal",
            "rh_ring_distal",
            "rh_pinky_distal",
        ):
            try:
                body_id = robot.body_names.index(body_name)
                fingertip_positions[body_name] = [
                    float(value)
                    for value in robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
                ]
            except (ValueError, AttributeError, IndexError):
                continue
        fingertip_centroid = (
            None
            if not fingertip_positions
            else np.mean(np.asarray(list(fingertip_positions.values()), dtype=np.float64), axis=0)
        )
        phase_snapshot = {
            "phase": str(phase.name),
            "can_world_m": None if can_pos is None else [float(value) for value in can_pos],
            "can_shift_m": None if not np.isfinite(can_displacement) else float(can_displacement),
            "right_tcp_world_m": None if right_pose_w is None else [float(value) for value in right_pose_w[0]],
            "right_tcp_base_m": None if right_pose_b is None else [float(value) for value in right_pose_b[0]],
            "right_tcp_error_m": None if not np.isfinite(right_error) else float(right_error),
            "left_tcp_world_m": (
                None
                if (left_pose_w := estimate_left_hand_tcp_pose_from_robot(robot)) is None
                else [float(value) for value in left_pose_w[0]]
            ),
            "can_minus_right_tcp_world_m": (
                None
                if can_pos is None or right_pose_w is None
                else [float(value) for value in np.asarray(can_pos) - np.asarray(right_pose_w[0])]
            ),
            "right_fingertip_centroid_world_m": (
                None if fingertip_centroid is None else [float(value) for value in fingertip_centroid]
            ),
            "can_minus_right_fingertip_centroid_world_m": (
                None
                if can_pos is None or fingertip_centroid is None
                else [float(value) for value in np.asarray(can_pos) - fingertip_centroid]
            ),
            "right_fingertip_link_positions_world_m": fingertip_positions,
            "drawer_open_m": current_drawer_open_m(),
            "left_hand_command_rad": [float(value) for value in action[ACTION_SLICES.left_hand]],
            "left_hand_actual_rad": [float(value) for value in actual_action[ACTION_SLICES.left_hand]],
            "right_hand_command_rad": [float(value) for value in action[ACTION_SLICES.right_hand]],
            "right_hand_actual_rad": [float(value) for value in actual_action[ACTION_SLICES.right_hand]],
        }
        phase_state_history.append(phase_snapshot)

    def record_attempt_failure(failure_type: str, reason: str, controller, wall_elapsed_s: float) -> None:
        """Persist enough state to diagnose exactly where and where-in-space an attempt failed."""
        if failure_reporter is None:
            return

        def vector(values) -> list[float] | None:
            if values is None:
                return None
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            return [float(value) for value in np.asarray(values).reshape(-1)]

        phase = None
        phase_index = None
        phase_total = None
        phase_step = None
        if controller is not None:
            phase_total = len(controller.phases)
            phase_index = int(controller.phase_index) + 1
            phase = controller.current_phase
            phase_step = int(controller.phase_steps)

        can_obj = scene.get("named_objects", {}).get("can")
        can_position_w = None
        can_linear_velocity_w = None
        if can_obj is not None:
            can_position_w = vector(can_obj.data.root_pos_w[0])
            can_linear_velocity_w = vector(can_obj.data.root_lin_vel_w[0])
        can_initial = np.asarray(scene.get("can_initial_position", GRASP_CAN_NOMINAL_POSITION), dtype=np.float64)
        can_offset = np.asarray(episode_context.get("can_xy_offset", [0.0, 0.0]), dtype=np.float64)
        can_spawn_w = can_initial.copy()
        can_spawn_w[:2] += can_offset
        base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
        left_pose_w = estimate_left_hand_tcp_pose_from_robot(robot)
        right_pose_w = estimate_right_hand_tcp_pose_from_robot(robot)
        left_pose_b = pose_world_to_base(left_pose_w, base_pose_w) if left_pose_w is not None else None
        right_pose_b = pose_world_to_base(right_pose_w, base_pose_w) if right_pose_w is not None else None
        tcp_errors = {"left_pos": None, "left_rot": None, "right_pos": None, "right_rot": None}
        if controller is not None:
            try:
                tcp_errors = controller.tcp_error_metrics(left_pose_b, right_pose_b)
            except Exception:
                # Failure reporting must not hide the original controller error.
                pass
        actual_action = control_action_from_sim(robot)
        left_hand_tracking_error = float(
            np.max(np.abs(action[ACTION_SLICES.left_hand] - actual_action[ACTION_SLICES.left_hand]))
        )
        right_arm_tracking_error = float(
            np.max(np.abs(action[ACTION_SLICES.right_arm] - actual_action[ACTION_SLICES.right_arm]))
        )
        right_hand_tracking_error = float(
            np.max(np.abs(action[ACTION_SLICES.right_hand] - actual_action[ACTION_SLICES.right_hand]))
        )
        can_trace = None
        if recording_episode is not None and recording_episode.drawer_task_object_pose:
            poses = np.asarray(recording_episode.drawer_task_object_pose, dtype=np.float32)
            below_support = np.flatnonzero(poses[:, 2] < 1.10)
            first_below_index = int(below_support[0]) if below_support.size else None
            displacement_vectors = poses[:, :3] - poses[0, :3]
            displacement_norms = np.linalg.norm(displacement_vectors, axis=1)

            def first_displacement(threshold_m: float) -> tuple[int | None, str | None, str | None]:
                indices = np.flatnonzero(displacement_norms >= threshold_m)
                frame_index = int(indices[0]) if indices.size else None
                if frame_index is None or frame_index >= len(recording_episode.task_descriptions):
                    return frame_index, None, None
                task_text = str(recording_episode.task_descriptions[frame_index])
                phase_name = (
                    str(recording_episode.expert_phase_names[frame_index])
                    if frame_index < len(recording_episode.expert_phase_names)
                    else next(
                        (candidate.name for candidate in controller.phases if candidate.task == task_text),
                        None,
                    )
                )
                return frame_index, phase_name, task_text

            first_5mm_frame, first_5mm_phase, first_5mm_task = first_displacement(0.005)
            first_10mm_frame, first_10mm_phase, first_10mm_task = first_displacement(0.010)
            can_trace = {
                "first_position_world_m": vector(poses[0, :3]),
                "last_position_world_m": vector(poses[-1, :3]),
                "last_displacement_from_first_m": vector(displacement_vectors[-1]),
                "min_world_z_m": float(np.min(poses[:, 2])),
                "max_world_z_m": float(np.max(poses[:, 2])),
                "max_displacement_from_first_m": float(np.max(displacement_norms)),
                "max_xy_displacement_from_first_m": float(
                    np.max(np.linalg.norm(displacement_vectors[:, :2], axis=1))
                ),
                "first_displacement_5mm_frame": first_5mm_frame,
                "first_displacement_5mm_phase": first_5mm_phase,
                "first_displacement_5mm_task": first_5mm_task,
                "first_displacement_10mm_frame": first_10mm_frame,
                "first_displacement_10mm_phase": first_10mm_phase,
                "first_displacement_10mm_task": first_10mm_task,
                "first_below_support_frame": first_below_index,
                "first_below_support_phase": (
                    None
                    if first_below_index is None
                    or first_below_index >= len(recording_episode.expert_phase_names)
                    else str(recording_episode.expert_phase_names[first_below_index])
                ),
            }
        sampled_can_shift = (
            float("nan")
            if can_trace is None
            else float(can_trace["max_displacement_from_first_m"])
        )
        settled_can_start = (
            None if controller is None else controller._task_object_start_position_world
        )
        current_can_array = None if can_position_w is None else np.asarray(can_position_w, dtype=np.float32)
        settled_can_displacement = (
            None
            if settled_can_start is None or current_can_array is None
            else current_can_array - np.asarray(settled_can_start, dtype=np.float32)
        )
        settled_can_shift = (
            float("nan")
            if settled_can_displacement is None
            else float(np.linalg.norm(settled_can_displacement))
        )
        finite_can_shifts = [value for value in (sampled_can_shift, settled_can_shift) if np.isfinite(value)]
        can_shift = max(finite_can_shifts, default=float("nan"))
        phase_name = "controller_exception" if phase is None else str(phase.name)
        if phase_name in {"right_pregrasp_can", "right_grasp_can"} and can_shift > 0.010:
            diagnostic_cause = "can_pushed_before_grasp_closure"
        elif (
            phase_name == "right_lift_can"
            and can_trace is not None
            and float(can_trace["max_world_z_m"]) < 1.20
        ):
            diagnostic_cause = "grasp_not_secured_or_slipped_during_lift"
        elif phase_name == "initial_open_hands" and right_hand_tracking_error > 0.030:
            diagnostic_cause = "initial_right_hand_not_fully_open"
        elif phase_name == "initial_open_hands" and left_hand_tracking_error > 0.030:
            diagnostic_cause = "initial_left_hand_not_fully_open"
        elif phase_name in (
            "left_open_hand",
            "left_clear_handle_after_release",
            "left_joint_transition_after_release",
        ) and left_hand_tracking_error > 0.030:
            diagnostic_cause = "left_hand_release_blocked_by_drawer_handle"
        elif right_arm_tracking_error > 0.10:
            diagnostic_cause = "right_arm_command_tracking_error"
        elif tcp_errors.get("right_pos") is not None and float(tcp_errors["right_pos"]) > 0.010:
            diagnostic_cause = "right_tcp_target_not_reached"
        else:
            diagnostic_cause = "phase_gate_or_final_state_failure"
        right_target_base = None if phase is None or phase.right is None else vector(phase.right.pos)
        right_target_delta = (
            None
            if phase is None or phase.right is None or right_pose_b is None
            else vector(np.asarray(phase.right.pos) - np.asarray(right_pose_b[0]))
        )
        event = {
            "schema_version": 1,
            "failure_type": str(failure_type),
            "reason": str(reason),
            "task": str(scene.get("task_id", "unknown")),
            "accepted_episodes_before_attempt": int(recorded_episodes),
            "target_episodes": int(max_record_episodes),
            "attempt_for_episode": int(record_attempt),
            "phase_index": phase_index,
            "phase_total": phase_total,
            "phase_name": phase_name,
            "phase_step": phase_step,
            "wall_elapsed_s": float(wall_elapsed_s),
            "sim_elapsed_s": float(record_step * sim_dt),
            "simulation_realtime_factor": float(
                (record_step * sim_dt) / max(float(wall_elapsed_s), 1.0e-6)
            ),
            "recorded_frames": 0 if recording_episode is None else int(len(recording_episode)),
            "can_grid_cell": episode_context.get("can_grid_cell"),
            "can_grid_cycle": episode_context.get("can_grid_cycle"),
            "can_grid_index_in_cycle": episode_context.get("can_grid_index_in_cycle"),
            "can_grid_point_attempt": int(episode_context.get("can_grid_point_attempt", grasp_retry_count + 1)),
            "can_grasp_retry_count": int(episode_context.get("can_grasp_retry_count", grasp_retry_count)),
            "can_xy_offset_m": vector(can_offset),
            "can_spawn_position_world_m": vector(can_spawn_w),
            "can_position_world_m": can_position_w,
            "can_linear_velocity_world_m_s": can_linear_velocity_w,
            "can_settled_start_position_world_m": vector(settled_can_start),
            "can_displacement_from_settled_start_m": vector(settled_can_displacement),
            "can_displacement_norm_from_settled_start_m": (
                None if not np.isfinite(settled_can_shift) else settled_can_shift
            ),
            "drawer_open_m": current_drawer_open_m(),
            "left_tcp_position_world_m": None if left_pose_w is None else vector(left_pose_w[0]),
            "left_tcp_position_base_m": None if left_pose_b is None else vector(left_pose_b[0]),
            "right_tcp_position_world_m": None if right_pose_w is None else vector(right_pose_w[0]),
            "right_tcp_position_base_m": None if right_pose_b is None else vector(right_pose_b[0]),
            "right_tcp_target_base_m": right_target_base,
            "right_tcp_target_delta_m": right_target_delta,
            "tcp_error": {
                key: None if value is None else float(value)
                for key, value in tcp_errors.items()
            },
            "right_hand_command_rad": vector(action[ACTION_SLICES.right_hand]),
            "right_hand_actual_rad": vector(actual_action[ACTION_SLICES.right_hand]),
            "left_hand_command_rad": vector(action[ACTION_SLICES.left_hand]),
            "left_hand_actual_rad": vector(actual_action[ACTION_SLICES.left_hand]),
            "left_hand_command_actual_max_error_rad": left_hand_tracking_error,
            "right_arm_command_actual_max_error_rad": right_arm_tracking_error,
            "right_hand_command_actual_max_error_rad": right_hand_tracking_error,
            "gravity_compensation": {
                "enabled": bool(args_cli.gravity_compensation),
                "scale": float(args_cli.gravity_comp_scale),
                "last_max_abs_effort": float(last_gravity_comp_stats[0]),
                "last_mean_abs_effort": float(last_gravity_comp_stats[1]),
            },
            "diagnostic_cause": diagnostic_cause,
            "phase_state_history": list(phase_state_history),
            "can_trace": can_trace,
        }
        failure_reporter.record(event)
        failure_reporter.finalize(
            completed=False,
            accepted_episodes=recorded_episodes,
            target_episodes=max_record_episodes,
            skipped_grid_cells=skipped_grid_cells,
            hdf5_path=args_cli.record_output,
        )
        log_collection_event(
            "FAILURE-RECORDED",
            f"phase={event['phase_name']} | reason={reason} | "
            f"diagnostic={diagnostic_cause} | grid={event['can_grid_cell']} | "
            f"can_world={event['can_position_world_m']} | can_shift={can_shift:.4f}m | "
            f"right_arm_track={right_arm_tracking_error:.4f}rad | "
            f"gravity_effort_max={float(last_gravity_comp_stats[0]):.2f}",
            "yellow",
        )

    def enforce_failure_budget() -> None:
        budget = args_cli.max_failed_attempts
        if budget is not None and failed_attempts_total > int(budget):
            raise RuntimeError(
                f"Collection aborted: failed_attempts={failed_attempts_total} exceeded "
                f"--max-failed-attempts={int(budget)}. Partial HDF5 and logs are preserved."
            )

    if scripted_drawer_enabled:
        drawer_controller = new_drawer_controller(action)
        recording_episode = make_recording_episode() if writer is not None else None
        record_wall_start = time.monotonic()
        collection_wall_start = record_wall_start
        arm_control_active = True
        log_collection_event(
            "TASK",
            f"drawer_insert_close | config={drawer_controller.config_path}",
            "cyan",
        )
        if writer is not None:
            log_attempt_start(drawer_controller)
            capture_current_phase_state(drawer_controller)
            last_logged_phase_index = drawer_controller.phase_index
    try:
        while simulation_app.is_running():
            try:
                arm_control_mtime = args_cli.arm_control_file.stat().st_mtime_ns
            except FileNotFoundError:
                arm_control_mtime = None
            if arm_control_mtime != last_arm_control_mtime:
                last_arm_control_mtime = arm_control_mtime
                if arm_control_mtime is not None:
                    try:
                        payload = json.loads(args_cli.arm_control_file.read_text(encoding="utf-8"))
                    except Exception as exc:
                        print(f"[WARN] ignoring invalid arm control file: {exc}")
                        payload = {}
                    mode = payload.get("mode")
                    if mode == "test-right-arm":
                        right_arm = parse_arm_target(payload, "right_arm")
                        if right_arm is not None:
                            set_named_joint_targets(target, robot, RIGHT_ARM_JOINTS, right_arm)
                            action = control_action_from_full_target(target, robot)
                            arm_control_active = True
                            tcp_pose_active = False
                            print(f"[ARM] drawer preview right-arm target: {[round(float(x), 3) for x in right_arm]}", flush=True)
                    elif mode == "test-left-arm":
                        left_arm = parse_arm_target(payload, "left_arm")
                        if left_arm is not None:
                            set_named_joint_targets(target, robot, LEFT_ARM_JOINTS, left_arm)
                            action = control_action_from_full_target(target, robot)
                            arm_control_active = True
                            tcp_pose_active = False
                            print(f"[ARM] drawer preview left-arm target: {[round(float(x), 3) for x in left_arm]}", flush=True)
                    elif mode == "test-bimanual-arm":
                        left_arm = parse_arm_target(payload, "left_arm")
                        right_arm = parse_arm_target(payload, "right_arm")
                        if left_arm is not None:
                            set_named_joint_targets(target, robot, LEFT_ARM_JOINTS, left_arm)
                        if right_arm is not None:
                            set_named_joint_targets(target, robot, RIGHT_ARM_JOINTS, right_arm)
                        if left_arm is not None or right_arm is not None:
                            action = control_action_from_full_target(target, robot)
                            arm_control_active = True
                            tcp_pose_active = False
                            print("[ARM] drawer preview bimanual joint target updated", flush=True)
                    elif mode == "hand" and payload.get("hand") in {"open", "close"}:
                        side = payload.get("side", "right")
                        if side not in {"left", "right", "both"}:
                            side = "right"
                        action = control_action_from_full_target(target, robot)
                        hand_values_left = CLOSE_LEFT_HAND if payload["hand"] == "close" else OPEN_LEFT_HAND
                        hand_values_right = CLOSE_RIGHT_HAND if payload["hand"] == "close" else OPEN_RIGHT_HAND
                        if side in {"left", "both"}:
                            action[ACTION_SLICES.left_hand] = hand_values_left
                        if side in {"right", "both"}:
                            action[ACTION_SLICES.right_hand] = hand_values_right
                        write_action_to_full_target(target, robot, action)
                        arm_control_active = True
                        tcp_pose_active = False
                        print(f"[ARM] drawer preview hand {payload['hand']} side={side}", flush=True)
                    elif mode == "tcp-pose":
                        if payload.get("frame") != "base_link":
                            print(f"[WARN] tcp-pose only supports frame=base_link, got {payload.get('frame')!r}")
                            continue
                        try:
                            if pink_tcp_controller is None:
                                from s4_robot.pink_bimanual_ik import PinkBimanualTcpController

                                pink_tcp_controller = PinkBimanualTcpController(
                                    robot,
                                    sim.device,
                                    posture_gain=args_cli.tcp_posture_gain,
                                    damping=args_cli.tcp_ik_damping,
                                    max_joint_delta=args_cli.tcp_max_joint_delta,
                                )
                                print("[ARM] Pinocchio DLS bimanual TCP controller ready (target frame: base_link)")
                            tcp_pose_goal_left = payload.get("left")
                            tcp_pose_goal_right = payload.get("right")
                            tcp_pose_active = True
                            arm_control_active = True
                            print(
                                "[ARM] Pinocchio DLS tcp-pose goal accepted; solving continuously "
                                f"left={tcp_pose_goal_left is not None} right={tcp_pose_goal_right is not None} "
                                f"left_goal={tcp_pose_goal_left} right_goal={tcp_pose_goal_right}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(f"[WARN] Pinocchio DLS tcp-pose setup failed: {exc}", flush=True)
                            traceback.print_exc()
                    elif mode == "reset-scene":
                        target = reset_static_attempt()
                        action = control_action_from_full_target(target, robot)
                        pink_tcp_controller = None
                        drawer_controller = None
                        if scripted_drawer_enabled:
                            drawer_controller = new_drawer_controller(action)
                            recording_episode = make_recording_episode() if writer is not None else None
                            record_wall_start = time.monotonic()
                            record_step = 0
                            last_logged_phase_index = -1
                            if writer is not None:
                                log_attempt_start(drawer_controller)
                                capture_current_phase_state(drawer_controller)
                                last_logged_phase_index = drawer_controller.phase_index
                        arm_control_active = False
                        tcp_pose_active = False
                        tcp_pose_goal_left = None
                        tcp_pose_goal_right = None
                        print("[SCENE] drawer preview robot reset.", flush=True)
                    elif mode == "idle":
                        arm_control_active = False
                        tcp_pose_active = False
                        print("[ARM] drawer preview idle", flush=True)
            if keyboard_jog is not None:
                action = keyboard_jog.update(action)
                write_action_to_full_target(target, robot, action)
            scripted_done = False
            if drawer_controller is not None and not record_complete:
                try:
                    current_q = robot.data.joint_pos[0].detach().cpu().numpy()
                    base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
                    left_pose_w = estimate_left_hand_tcp_pose_from_robot(robot)
                    right_pose_w = estimate_right_hand_tcp_pose_from_robot(robot)
                    left_pose_b = pose_world_to_base(left_pose_w, base_pose_w) if left_pose_w is not None else None
                    right_pose_b = pose_world_to_base(right_pose_w, base_pose_w) if right_pose_w is not None else None
                    actual_action = control_action_from_sim(robot)
                    can_obj = scene.get("named_objects", {}).get("can")
                    can_position_w = (
                        None
                        if can_obj is None
                        else can_obj.data.root_pos_w[0].detach().cpu().numpy()
                    )
                    can_linear_velocity_w = (
                        None
                        if can_obj is None
                        else can_obj.data.root_lin_vel_w[0].detach().cpu().numpy()
                    )
                    desired_action, current_scripted_phase, current_scripted_task, scripted_done = drawer_controller.step(
                        current_q,
                        max(sim_dt, 1.0 / 120.0),
                        left_pose_b,
                        right_pose_b,
                        current_drawer_open_m(),
                        commanded_action=action,
                        actual_action=actual_action,
                        task_object_position_world=can_position_w,
                        task_object_linear_velocity_world=can_linear_velocity_w,
                    )
                    if drawer_controller.phase_index != last_logged_phase_index and not drawer_controller.failed:
                        capture_current_phase_state(drawer_controller)
                        last_logged_phase_index = drawer_controller.phase_index
                    active_phase = drawer_controller.current_phase
                    phase_alpha = float(
                        active_phase.target_alpha
                        if active_phase.target_alpha is not None
                        else args_cli.target_alpha
                    )
                    phase_max_joint_step = float(
                        active_phase.max_joint_step
                        if active_phase.max_joint_step is not None
                        else args_cli.max_joint_step
                    )
                    next_action = smooth_command(
                        action,
                        desired_action,
                        alpha=phase_alpha,
                        max_joint_step=phase_max_joint_step,
                    )
                    for action_slice in (ACTION_SLICES.left_hand, ACTION_SLICES.right_hand):
                        hand_delta = np.clip(
                            next_action[action_slice] - action[action_slice],
                            -float(args_cli.hand_max_joint_step),
                            float(args_cli.hand_max_joint_step),
                        )
                        next_action[action_slice] = action[action_slice] + hand_delta
                    action = next_action
                    write_action_to_full_target(target, robot, action)
                    arm_control_active = True
                    tcp_pose_active = False
                except Exception as exc:
                    print(f"[WARN] drawer scripted controller failed: {exc}", flush=True)
                    traceback.print_exc()
                    if writer is not None:
                        record_attempt_failure(
                            "controller_exception",
                            f"{type(exc).__name__}: {exc}",
                            drawer_controller,
                            time.monotonic() - record_wall_start if record_wall_start is not None else 0.0,
                        )
                        raise RuntimeError("Dataset collection stopped after a controller exception") from exc
                    drawer_controller = None
            if tcp_pose_active and pink_tcp_controller is not None:
                try:
                    current_q = robot.data.joint_pos[0].detach().cpu().numpy()
                    arm_targets = pink_tcp_controller.compute(
                        current_q,
                        max(sim.get_physics_dt(), 1.0 / 60.0),
                        tcp_pose_goal_left,
                        tcp_pose_goal_right,
                    )
                    left_arm = arm_targets[: len(LEFT_ARM_JOINTS)]
                    right_arm = arm_targets[len(LEFT_ARM_JOINTS) :]
                    set_named_joint_targets(target, robot, LEFT_ARM_JOINTS, left_arm)
                    set_named_joint_targets(target, robot, RIGHT_ARM_JOINTS, right_arm)
                    action = control_action_from_full_target(target, robot)
                except Exception as exc:
                    print(f"[WARN] Pinocchio DLS continuous solve failed: {exc}", flush=True)
                    traceback.print_exc()
                    tcp_pose_active = False
            target_tensor = torch.tensor(target, device=sim.device).view(1, -1)
            if (
                args_cli.headless
                or drawer_drive_paths
                or keyboard_jog is not None
                or arm_control_active
                or wrist_frustum_visualizer is not None
            ):
                robot.set_joint_position_target(target_tensor)
                last_gravity_comp_stats = apply_gravity_compensation(
                    robot,
                    gravity_comp_joint_ids,
                    scale=float(args_cli.gravity_comp_scale),
                    enabled=bool(args_cli.gravity_compensation),
                )
                robot.write_data_to_sim()
                sim.step(render=True)
                robot.update(dt=sim.get_physics_dt())
                if drawer is not None:
                    drawer.update(dt=sim.get_physics_dt())
                for obj in scene.get("dynamic_objects", []):
                    obj.update(dt=sim.get_physics_dt())
                update_all_cameras(scene, camera, dt=sim.get_physics_dt())
            else:
                robot.set_joint_position_target(target_tensor)
                last_gravity_comp_stats = apply_gravity_compensation(
                    robot,
                    gravity_comp_joint_ids,
                    scale=float(args_cli.gravity_comp_scale),
                    enabled=bool(args_cli.gravity_compensation),
                )
                robot.write_data_to_sim()
                simulation_app.update()
            if recording_episode is not None and drawer_controller is not None:
                wall_elapsed = time.monotonic() - record_wall_start if record_wall_start is not None else 0.0
                record_timeout_s = max(float(args_cli.record_episode_timeout_s), 1.0)
                if wall_elapsed >= record_timeout_s:
                    failed_attempts_total += 1
                    record_attempt_failure("timeout", "episode_timeout", drawer_controller, wall_elapsed)
                    enforce_failure_budget()
                    log_collection_event(
                        "TIMEOUT",
                        f"episode={recorded_episodes + 1:03d}/{max_record_episodes:03d} attempt={record_attempt:02d} | "
                        f"elapsed={wall_elapsed:.1f}/{record_timeout_s:.1f}s | frames={len(recording_episode)} | discarded",
                        "red",
                    )
                    select_after_failed_attempt(drawer_controller.current_phase.name, "timeout")
                    writer.write_collection_state({**collection_state(), "completed_episodes": recorded_episodes})
                    target = reset_static_attempt()
                    action = control_action_from_full_target(target, robot)
                    pink_tcp_controller = None
                    drawer_controller = new_drawer_controller(action)
                    recording_episode = make_recording_episode()
                    record_wall_start = time.monotonic()
                    record_step = 0
                    record_attempt += 1
                    last_logged_phase_index = -1
                    log_collection_event("RETRY", "scene reset; trying the selected grid point", "yellow")
                    log_attempt_start(drawer_controller)
                    capture_current_phase_state(drawer_controller)
                    last_logged_phase_index = drawer_controller.phase_index
                    continue
                if record_step % record_every_n == 0:
                    append_bimanual_record_frame(
                        recording_episode,
                        scene,
                        robot,
                        camera,
                        action,
                        drawer_controller.current_language_task,
                        drawer_controller.current_language_phase_id,
                        current_scripted_phase,
                    )
                record_step += 1
                if scripted_done:
                    if drawer_controller.failed:
                        failed_attempts_total += 1
                        record_attempt_failure(
                            "controller_failed",
                            drawer_controller.failure_reason or "controller_failed",
                            drawer_controller,
                            wall_elapsed,
                        )
                        enforce_failure_budget()
                        log_collection_event(
                            "DISCARD",
                            f"episode={recorded_episodes + 1:03d}/{max_record_episodes:03d} attempt={record_attempt:02d} | "
                            f"frames={len(recording_episode)} | {drawer_controller.failure_reason}",
                            "red",
                        )
                        select_after_failed_attempt(
                            drawer_controller.current_phase.name,
                            drawer_controller.failure_reason or "controller_failed",
                        )
                        writer.write_collection_state({**collection_state(), "completed_episodes": recorded_episodes})
                        target = reset_static_attempt()
                        action = control_action_from_full_target(target, robot)
                        pink_tcp_controller = None
                        drawer_controller = new_drawer_controller(action)
                        recording_episode = make_recording_episode()
                        record_wall_start = time.monotonic()
                        record_step = 0
                        record_attempt += 1
                        last_logged_phase_index = -1
                        log_collection_event("RETRY", "scene reset; trying the selected grid point", "yellow")
                        log_attempt_start(drawer_controller)
                        capture_current_phase_state(drawer_controller)
                        last_logged_phase_index = drawer_controller.phase_index
                        continue
                    task_success, success_details = evaluate_drawer_task_success()
                    drawer_open_text = success_details["drawer_open_m"]
                    drawer_open_value = float("nan") if drawer_open_text is None else float(drawer_open_text)
                    can_world_position = success_details["can_world_position_m"]
                    log_collection_event(
                        "VERIFY",
                        f"accepted={task_success} | drawer={drawer_open_value:+.4f}m (telemetry only) | "
                        f"can_xyz=({float(can_world_position[0]):+.3f},{float(can_world_position[1]):+.3f},"
                        f"{float(can_world_position[2]):+.3f}) | "
                        f"can_in_drawer={success_details['can_in_drawer']} | "
                        f"bounds={success_details['can_world_bounds_m']}",
                        "green" if task_success else "red",
                    )
                    if not task_success:
                        failed_attempts_total += 1
                        failed_checks = []
                        if not success_details["can_in_drawer"]:
                            failed_checks.append("can_not_in_drawer")
                        reason = ",".join(failed_checks) or "final_state_invalid"
                        record_attempt_failure("final_state_invalid", reason, drawer_controller, wall_elapsed)
                        enforce_failure_budget()
                        log_collection_event(
                            "DISCARD",
                            f"episode={recorded_episodes + 1:03d}/{max_record_episodes:03d} attempt={record_attempt:02d} | "
                            f"reason={reason} | frames={len(recording_episode)}",
                            "red",
                        )
                        select_after_failed_attempt(drawer_controller.current_phase.name, reason)
                        writer.write_collection_state({**collection_state(), "completed_episodes": recorded_episodes})
                        target = reset_static_attempt()
                        action = control_action_from_full_target(target, robot)
                        pink_tcp_controller = None
                        drawer_controller = new_drawer_controller(action)
                        recording_episode = make_recording_episode()
                        record_wall_start = time.monotonic()
                        record_step = 0
                        record_attempt += 1
                        last_logged_phase_index = -1
                        log_collection_event("RETRY", "scene reset; trying the selected grid point", "yellow")
                        log_attempt_start(drawer_controller)
                        capture_current_phase_state(drawer_controller)
                        last_logged_phase_index = drawer_controller.phase_index
                        continue
                    recording_episode.metadata["final_success"] = success_details
                    grasp_retry_count = 0
                    episode_context = sample_drawer_episode()
                    next_collection_state = {
                        **collection_state(),
                        "completed_episodes": recorded_episodes + 1,
                    }
                    demo_name = (
                        writer.write_episode(
                            recording_episode,
                            collection_state=next_collection_state,
                        )
                        if writer is not None
                        else "demo"
                    )
                    sim_seconds = record_step * sim_dt
                    wall_seconds = time.monotonic() - record_wall_start if record_wall_start is not None else float("nan")
                    log_collection_event(
                        "ACCEPT",
                        f"{demo_name} | success={recorded_episodes + 1:03d}/{max_record_episodes:03d} | "
                        f"frames={len(recording_episode)} | sim={sim_seconds:.2f}s | wall={wall_seconds:.2f}s",
                        "green",
                    )
                    recorded_episodes += 1
                    if writer is not None and recorded_episodes >= max_record_episodes:
                        record_complete = True
                        break
                    target = reset_static_attempt()
                    action = control_action_from_full_target(target, robot)
                    pink_tcp_controller = None
                    drawer_controller = new_drawer_controller(action)
                    recording_episode = make_recording_episode()
                    record_wall_start = time.monotonic()
                    record_step = 0
                    record_attempt = 1
                    last_logged_phase_index = -1
                    log_attempt_start(drawer_controller)
                    capture_current_phase_state(drawer_controller)
                    last_logged_phase_index = drawer_controller.phase_index
            if tcp_visualizer is not None:
                tcp_visualizer.visualize_task_frames(
                    left_tcp_pose=estimate_left_hand_tcp_pose_from_robot(robot),
                    right_tcp_pose=estimate_right_hand_tcp_pose_from_robot(robot),
                    drawer_handle_pose=get_drawer_handle_top_pose() if args_cli.show_drawer_handle_frame else None,
                )
            if wrist_frustum_visualizer is not None:
                wrist_frustum_visualizer.update(scene)
            if args_cli.print_tcp_pose:
                now = time.monotonic()
                if now - last_tcp_print >= max(float(args_cli.tcp_print_period), 0.05):
                    base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
                    print(format_tcp_pose_line("[TCP] left_hand", estimate_left_hand_tcp_pose_from_robot(robot), base_pose_w))
                    print(format_tcp_pose_line("[TCP] right_hand", estimate_right_hand_tcp_pose_from_robot(robot), base_pose_w))
                    print(format_tcp_pose_line("[TCP] drawer_handle_frame", get_drawer_handle_top_pose(), base_pose_w))
                    last_tcp_print = now
            if arm_control_active:
                now = time.monotonic()
                refresh_period = (
                    max(float(drawer_dashboard_cfg.get("refresh_seconds", 0.5)), 0.05)
                    if drawer_controller is not None
                    else 1.0
                )
                if now - last_arm_debug >= refresh_period:
                    if drawer_controller is not None:
                        base_pose_w = estimate_body_pose_from_robot(robot, "base_link")
                        left_pose_w = estimate_left_hand_tcp_pose_from_robot(robot)
                        right_pose_w = estimate_right_hand_tcp_pose_from_robot(robot)
                        left_pose_b = pose_world_to_base(left_pose_w, base_pose_w) if left_pose_w is not None else None
                        right_pose_b = pose_world_to_base(right_pose_w, base_pose_w) if right_pose_w is not None else None
                        wall_elapsed = time.monotonic() - record_wall_start if record_wall_start is not None else 0.0
                        record_timeout_s = max(float(args_cli.record_episode_timeout_s), 1.0)
                        errors = drawer_controller.tcp_error_metrics(left_pose_b, right_pose_b)
                        phase = drawer_controller.current_phase
                        dashboard_drawer_open = current_drawer_open_m()
                        render_collection_dashboard(
                            DashboardSnapshot(
                                episode=recorded_episodes + 1,
                                episode_total=max_record_episodes,
                                attempt=record_attempt,
                                clock_time=datetime.now().astimezone().strftime("%H:%M:%S"),
                                success_count=recorded_episodes,
                                failure_count=failed_attempts_total,
                                phase_index=drawer_controller.phase_index + 1,
                                phase_total=len(drawer_controller.phases),
                                phase_name=phase.name,
                                step=drawer_controller.phase_steps,
                                step_total=phase.max_steps,
                                elapsed_s=wall_elapsed,
                                timeout_s=record_timeout_s,
                                episode_sim_s=record_step * sim_dt,
                                collection_elapsed_s=(
                                    now - collection_wall_start
                                    if collection_wall_start is not None
                                    else 0.0
                                ),
                                frames=len(recording_episode) if recording_episode is not None else 0,
                                left_pos=errors["left_pos"],
                                left_rot=errors["left_rot"],
                                right_pos=errors["right_pos"],
                                right_rot=errors["right_rot"],
                                left_pos_limit=phase.tolerance,
                                left_rot_limit=phase.orientation_tolerance,
                                right_pos_limit=phase.tolerance,
                                right_rot_limit=phase.orientation_tolerance,
                                left_tcp_gate=(
                                    phase.require_left_tcp_reached and phase.left is not None
                                ),
                                right_tcp_gate=(
                                    phase.require_right_tcp_reached and phase.right is not None
                                ),
                                drawer_open_m=(
                                    float(dashboard_drawer_open)
                                    if dashboard_drawer_open is not None
                                    else float("nan")
                                ),
                                drawer_open_min_m=(
                                    float(phase.drawer_open_min)
                                    if phase.drawer_open_min is not None
                                    else float("nan")
                                ),
                                drawer_open_limit_m=(
                                    float(phase.drawer_open_max)
                                    if phase.drawer_open_max is not None
                                    else float("nan")
                                ),
                            ),
                            drawer_dashboard_cfg,
                        )
                    else:
                        q = robot.data.joint_pos[0].detach().cpu().numpy()
                        right_err = 0.0
                        left_err = 0.0
                        for name in RIGHT_ARM_JOINTS:
                            if name in robot.joint_names:
                                idx = robot.joint_names.index(name)
                                right_err = max(right_err, abs(float(target[idx] - q[idx])))
                        for name in LEFT_ARM_JOINTS:
                            if name in robot.joint_names:
                                idx = robot.joint_names.index(name)
                                left_err = max(left_err, abs(float(target[idx] - q[idx])))
                        print(
                            f"[ARMDBG] tcp_pose={tcp_pose_active} "
                            f"q_track(L/R)={left_err:.3f}/{right_err:.3f}rad "
                            f"gravity=max:{last_gravity_comp_stats[0]:.2f}/mean:{last_gravity_comp_stats[1]:.2f}",
                            flush=True,
                        )
                    last_arm_debug = now
    finally:
        if keyboard_jog is not None:
            keyboard_jog.stop()
        if failure_reporter is not None:
            failure_reporter.finalize(
                completed=record_complete,
                accepted_episodes=recorded_episodes,
                target_episodes=max_record_episodes,
                skipped_grid_cells=skipped_grid_cells,
                hdf5_path=args_cli.record_output,
            )
        if writer is not None:
            writer.close()
        if record_complete and args_cli.record_output is not None:
            log_collection_event(
                "COMPLETE",
                f"accepted={recorded_episodes}/{max_record_episodes} | "
                f"failed_attempts={failed_attempts_total} | skipped_grid_cells={len(skipped_grid_cells)} | "
                f"output={args_cli.record_output}",
                "green",
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def run_debug(scene: dict[str, object], cfg: SceneBuildCfg, sim) -> None:
    if scene.get("task_id") not in (None, "right_blue_cylinder_plate"):
        run_static_task_scene(scene, cfg, sim)
        return

    robot: Articulation = scene["robot"]
    camera = scene["camera"]
    sim_dt = sim.get_physics_dt()
    reset_settle_steps = max(
        int(args_cli.reset_settle_steps),
        int(math.ceil(max(float(args_cli.reset_settle_s), 0.0) / max(float(sim_dt), 1.0e-6))),
    )
    effective_random_seed = int(args_cli.random_seed if args_cli.random_seed is not None else 42)
    rng = np.random.default_rng(effective_random_seed)
    randomize_blue_xy = max(float(args_cli.randomize_blue_xy), 0.0)
    default_target = reset_scene(scene, cfg, sim)
    blue_offset_xy = sample_blue_xy_offset(rng, randomize_blue_xy)
    blue_randomized = apply_blue_xy_offset(scene, cfg, sim, blue_offset_xy)
    if randomize_blue_xy > 0.0:
        blue_pos = blue_randomized["blue"]
        print(
            f"[RANDOMIZE] blue xy offset=({blue_offset_xy[0]:+.4f},{blue_offset_xy[1]:+.4f}) "
            f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f})"
        )
    settle_scene_to_target(scene, camera, default_target, sim, reset_settle_steps)
    full_command_target = default_target.copy()
    robot.set_joint_position_target(torch.tensor(full_command_target, device=sim.device).view(1, -1))
    robot.write_data_to_sim()
    action = control_action_from_sim(robot)
    commanded_action = action.copy()
    hold_action = action.copy()
    action_target_bias = control_action_bias_from_target(full_command_target, robot)

    reach_controller = None
    arm_mode = "idle"
    reach_block = None
    reach_offset = np.array([0.0, 0.0, 0.14], dtype=np.float32)
    reach_offset_frame = "world"
    reach_tcp_offset = DEFAULT_TCP_OFFSET_WRIST.copy()
    hand_target = OPEN_RIGHT_HAND.copy()
    test_right_arm = None
    reach_q_target = None
    reach_q_current = None
    target_tcp_pos = None
    target_tcp_quat = None
    grasp_plan = None
    grasp_phase = None
    grasp_phase_steps = 0
    gravity_comp_joint_ids = resolve_existing_joint_ids(robot, list(ALL_DRIVE_JOINTS))
    last_gravity_comp_stats = (0.0, 0.0)
    tcp_visualizer = None
    writer = None
    recording_episode = None
    recorded_episodes = 0
    record_complete = False
    record_step = 0
    record_wall_start = None
    auto_grasp_pending = bool(args_cli.auto_grasp or args_cli.record_output is not None)
    max_record_episodes = max(int(args_cli.record_episodes), 1)
    record_every_n = max(int(args_cli.record_every_n), 1)
    if args_cli.record_output is not None:
        writer = Hdf5DemoWriter(
            args_cli.record_output,
            env_args={
                "task": "s4_right_blue_cylinder_plate_scripted",
                "source": "scripted_ik",
                "sim_dt": float(sim.get_physics_dt()),
                "record_every_n": int(record_every_n),
                "record_episode_timeout_s": float(max(float(args_cli.record_episode_timeout_s), 1.0)),
                "reset_settle_s": float(max(float(args_cli.reset_settle_s), 0.0)),
                "reset_settle_steps": int(reset_settle_steps),
                "record_fps": float(1.0 / (sim.get_physics_dt() * record_every_n)),
                "randomization": {
                    "blue_xy_range_m": float(randomize_blue_xy),
                    "random_seed": effective_random_seed,
                    "distribution": "uniform",
                },
                "success_filter": {
                    "enabled": bool(args_cli.success_check),
                    "target_block": str(args_cli.auto_grasp_block),
                    "xy_tolerance": (
                        None if args_cli.success_xy_tolerance is None else float(args_cli.success_xy_tolerance)
                    ),
                    "default_xy_tolerance": float(max(PLATE_RADIUS - BLOCK_CYLINDER_RADIUS, 0.01)),
                    "z_min_above_plate": float(args_cli.success_z_min_above_plate),
                    "z_max_above_plate": float(args_cli.success_z_max_above_plate),
                },
                "camera": {
                    "eye": list(cfg.camera_eye),
                    "target": list(cfg.camera_target),
                    "rpy_deg": None if cfg.camera_rpy_deg is None else list(cfg.camera_rpy_deg),
                    "convention": str(cfg.camera_convention),
                    "width": int(cfg.camera_width),
                    "height": int(cfg.camera_height),
                },
                "layout": {
                    "task_x": float(cfg.layout.block_x),
                    "task_y": float(cfg.layout.table_center_y),
                    "block_y_offset": float(cfg.layout.block_y_offset),
                    "plate_x": float(cfg.layout.plate_x),
                },
            },
        )
    if (args_cli.show_tcp_frames or args_cli.show_drawer_handle_frame) and not args_cli.headless:
        tcp_visualizer = TcpFrameVisualizer(sim.device)
        reach_controller = make_right_reach_controller(robot, sim.device)
        if args_cli.verbose_status:
            print(f"[ARM] reach resolution: {reach_controller.resolution_summary()}")
    wrist_frustum_visualizer = None
    if args_cli.show_wrist_camera_frustums and not args_cli.headless:
        wrist_frustum_visualizer = scene.get("wrist_frustum_visualizer")
        if wrist_frustum_visualizer is None:
            raise RuntimeError("wrist frustum geometry was not created before SimulationContext.reset()")

    keyboard_jog = None
    if args_cli.keyboard_jog and not args_cli.headless:
        candidate = KeyboardJog(action, jog_step=float(args_cli.jog_step))
        if candidate.start():
            keyboard_jog = candidate

    write_default_control_file(args_cli.control_file, overwrite=True)
    args_cli.arm_control_file.parent.mkdir(parents=True, exist_ok=True)
    args_cli.arm_control_file.write_text(json.dumps({"mode": "idle"}, indent=2), encoding="utf-8")
    print("\nS4 debug running.")
    print(format_layout(cfg))
    print(
        "Recording camera: /World/DebugFrontCamera "
        f"mode={'look_at' if cfg.camera_rpy_deg is None else 'rpy'} "
        f"eye=({cfg.camera_eye[0]:.3f},{cfg.camera_eye[1]:.3f},{cfg.camera_eye[2]:.3f}) "
        f"target=({cfg.camera_target[0]:.3f},{cfg.camera_target[1]:.3f},{cfg.camera_target[2]:.3f}) "
        f"rpy_deg={cfg.camera_rpy_deg} convention={cfg.camera_convention} "
        f"size={cfg.camera_width}x{cfg.camera_height} sensor_render=True ui_headless={bool(args_cli.headless)}"
    )
    print(
        "Wrist cameras: "
        f"left_pos={_fmt_tuple(cfg.left_wrist_camera_pos, 4)} "
        f"left_quat_wxyz={_fmt_tuple(cfg.left_wrist_camera_quat_wxyz, 4)} "
        f"left_rpy_override_deg={_fmt_tuple(cfg.left_wrist_camera_rpy_deg, 2)} "
        f"right_pos={_fmt_tuple(cfg.right_wrist_camera_pos, 4)} "
        f"right_quat_wxyz={_fmt_tuple(cfg.right_wrist_camera_quat_wxyz, 4)} "
        f"right_rpy_override_deg={_fmt_tuple(cfg.right_wrist_camera_rpy_deg, 2)} "
        f"convention={cfg.wrist_camera_convention} optical_axis={'local_+Z' if cfg.wrist_camera_convention == 'ros' else 'local_-Z'}"
    )
    if tcp_visualizer is not None:
        print(
            "Debug frames: /World/Visuals/LeftHandTCP, /World/Visuals/RightHandTCP, "
            "/World/Visuals/TargetBlockTCP, /World/Visuals/DrawerHandleTop"
        )
    if wrist_frustum_visualizer is not None:
        print(
            "Wrist camera frustums: camera-local USD geometry "
            "(LeftWristCamera/DebugFrustum=cyan, RightWristCamera/DebugFrustum=orange)"
        )
    if keyboard_jog is not None:
        print("Keyboard jog: '['/']' select joint, 'u' increase, 'j' decrease, 'r' reset, 'p' print selected.")
    print("Reset command: bash run.sh control reset-scene")
    if writer is not None:
        print(
            f"HDF5 recording: {args_cli.record_output} episodes={max_record_episodes} "
            f"every_n={record_every_n} timeout={max(float(args_cli.record_episode_timeout_s), 1.0):.1f}s"
        )
        success_xy_tolerance = (
            float(args_cli.success_xy_tolerance)
            if args_cli.success_xy_tolerance is not None
            else max(float(PLATE_RADIUS - BLOCK_CYLINDER_RADIUS), 0.01)
        )
        print(
            "Success filter: "
            f"enabled={bool(args_cli.success_check)} block={args_cli.auto_grasp_block} "
            f"xy_dist<={success_xy_tolerance:.3f}m "
            f"z_above_plate=[{float(args_cli.success_z_min_above_plate):.3f},"
            f"{float(args_cli.success_z_max_above_plate):.3f}]m"
        )
    if args_cli.verbose_status:
        print(f"Joint control file: {args_cli.control_file}")
        print(f"Arm control file: {args_cli.arm_control_file}")
        print("Arm controller: idle at startup; reach command is inactive until a control command is written.")
        print(
            f"Robot dynamics: fixed_base={robot.is_fixed_base} "
            f"joint_stiffness={float(args_cli.joint_stiffness):.1f} "
            f"joint_damping={float(args_cli.joint_damping):.1f} "
            f"joint_effort_limit={float(args_cli.joint_effort_limit):.1f} "
            f"gravity_compensation={bool(args_cli.gravity_compensation)} "
            f"gravity_comp_scale={float(args_cli.gravity_comp_scale):.2f}"
        )
        print("Reach controller: IsaacLab DifferentialIKController, root-frame TCP target, PhysX geometric Jacobian.")
        print(
            "Reach Jacobian params: "
            f"body_shift={args_cli.reach_jacobian_body_shift} "
            f"sign={float(args_cli.reach_jacobian_sign):.1f} "
            f"adaptive_direction_sign={bool(args_cli.reach_adaptive_direction_sign)} "
            f"min_tcp_below_block={float(args_cli.reach_min_tcp_below_block):.3f}m"
        )
        print(f"Hand smoothing: max_joint_step={float(args_cli.hand_max_joint_step):.4f} rad/step")
    print()

    try:
        last_report = time.monotonic()
        last_tcp_report = 0.0
        last_unstable_report = 0.0
        last_grasp_wait_report = 0.0
        last_control_mtime = None
        last_arm_control_mtime = None
        while simulation_app.is_running():
            if auto_grasp_pending:
                args_cli.arm_control_file.write_text(
                    json.dumps(default_grasp_payload(args_cli.auto_grasp_block), indent=2),
                    encoding="utf-8",
                )
                auto_grasp_pending = False
            try:
                control_mtime = args_cli.control_file.stat().st_mtime_ns
            except OSError:
                control_mtime = None
            if control_mtime != last_control_mtime:
                action = read_control_action(args_cli.control_file, action)
                last_control_mtime = control_mtime
            try:
                arm_control_mtime = args_cli.arm_control_file.stat().st_mtime_ns
            except OSError:
                arm_control_mtime = None
            if arm_control_mtime != last_arm_control_mtime:
                last_arm_control_mtime = arm_control_mtime
                if arm_control_mtime is None:
                    arm_mode = "idle"
                    reach_block = None
                    target_tcp_pos = None
                    target_tcp_quat = None
                else:
                    try:
                        payload = json.loads(args_cli.arm_control_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        print(f"[WARN] ignoring invalid arm control file: {exc}")
                    else:
                        mode = payload.get("mode", "idle")
                        if mode in {"reach-block", "grasp-block"} and payload.get("block") in {"red", "blue"}:
                            arm_mode = mode
                            reach_block = payload["block"]
                            test_right_arm = None
                            action = control_action_from_sim(robot)
                            commanded_action = action.copy()
                            hold_action = action.copy()
                            action_target_bias = control_action_bias_from_target(full_command_target, robot)
                            target_tcp_quat = None
                            if mode == "grasp-block":
                                if writer is not None:
                                    recording_episode = EpisodeBuffer()
                                    record_step = 0
                                base_offset = np.asarray(payload.get("base_offset", [0.0, 0.0]), dtype=np.float32)
                                if base_offset.shape != (2,):
                                    base_offset = np.array([0.0, 0.0], dtype=np.float32)
                                grasp_plan = {
                                    "base_offset": base_offset,
                                    "approach_z": float(payload.get("approach_z", 0.10)),
                                    "grasp_z": float(payload.get("grasp_z", 0.01)),
                                    "lift_z": float(payload.get("lift_z", 0.15)),
                                    "place_approach_z": float(payload.get("place_approach_z", 0.18)),
                                    "place_z": float(payload.get("place_z", 0.10)),
                                    "place_offset": np.asarray(payload.get("place_offset", [0.0, -0.05]), dtype=np.float32),
                                    "grasp_pose": payload.get("grasp_pose", "current"),
                                    "grasp_rpy": np.asarray(payload.get("grasp_rpy", [0.0, 0.1, -0.20]), dtype=np.float32),
                                    "place_rpy": np.asarray(payload.get("place_rpy", [0.40, 0.0, 0.0]), dtype=np.float32),
                                    "tolerance": max(float(payload.get("tolerance", 0.05)), 0.01),
                                    "approach_steps": max(int(payload.get("approach_steps", 360)), 1),
                                    "lower_steps": max(int(payload.get("lower_steps", 240)), 1),
                                    "close_steps": max(int(payload.get("close_steps", 160)), 1),
                                    "pre_close_hold_steps": max(int(payload.get("pre_close_hold_steps", 120)), 0),
                                    "hand_complete_tolerance": max(float(payload.get("hand_complete_tolerance", 0.015)), 0.001),
                                    "lift_steps": max(int(payload.get("lift_steps", 120)), 1),
                                    "place_steps": max(int(payload.get("place_steps", 360)), 1),
                                    "pre_release_hold_steps": max(int(payload.get("pre_release_hold_steps", 120)), 0),
                                    "release_steps": max(int(payload.get("release_steps", 120)), 1),
                                }
                                if grasp_plan["place_offset"].shape != (2,):
                                    grasp_plan["place_offset"] = np.array([0.0, -0.05], dtype=np.float32)
                                grasp_phase = "approach"
                                grasp_phase_steps = 0
                                reach_offset = np.array(
                                    [base_offset[0], base_offset[1], grasp_plan["approach_z"]],
                                    dtype=np.float32,
                                )
                            else:
                                grasp_plan = None
                                grasp_phase = None
                                grasp_phase_steps = 0
                                reach_offset = np.asarray(payload.get("offset", [0.0, 0.0, 0.20]), dtype=np.float32)
                                if reach_offset.shape != (3,):
                                    reach_offset = np.array([0.0, 0.0, 0.20], dtype=np.float32)
                            reach_offset_frame = payload.get("offset_frame", "world")
                            if reach_offset_frame not in {"world", "wrist"}:
                                reach_offset_frame = "world"
                            reach_tcp_offset = np.asarray(
                                payload.get("tcp_offset_wrist", DEFAULT_TCP_OFFSET_WRIST),
                                dtype=np.float32,
                            )
                            if reach_tcp_offset.shape != (3,):
                                reach_tcp_offset = DEFAULT_TCP_OFFSET_WRIST.copy()
                            hand_target = CLOSE_RIGHT_HAND.copy() if payload.get("hand") == "close" else OPEN_RIGHT_HAND.copy()
                            if mode == "grasp-block":
                                hand_target = OPEN_RIGHT_HAND.copy()
                                if writer is not None:
                                    record_wall_start = time.monotonic()
                            if reach_controller is None:
                                reach_controller = make_right_reach_controller(robot, sim.device)
                                print(f"[ARM] reach resolution: {reach_controller.resolution_summary()}")
                            else:
                                reach_controller.reset_diagnostics()
                            block_pos = scene[reach_block].data.root_pos_w[0].detach().cpu().numpy()
                            if mode == "grasp-block":
                                grasp_plan["anchor_block_pos_w"] = block_pos.copy()
                                grasp_plan["anchor_plate_pos_w"] = scene["plate"].data.root_pos_w[0].detach().cpu().numpy().copy()
                            if reach_offset_frame == "world":
                                preview_offset_w = reach_offset
                            else:
                                offset_t = torch.tensor(reach_offset, dtype=torch.float32, device=sim.device).view(1, 3)
                                preview_offset_w = reach_controller.rotate_wrist_vector_to_world(offset_t)[0].detach().cpu().numpy()
                            preview_target_tcp = block_pos + preview_offset_w
                            current_tcp_pose = estimate_right_hand_tcp_pose(robot, reach_controller, reach_tcp_offset)
                            current_tcp = current_tcp_pose[0] if current_tcp_pose is not None else np.zeros(3, dtype=np.float32)
                            current_tcp_quat = (
                                current_tcp_pose[1] if current_tcp_pose is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                            )
                            if mode == "grasp-block":
                                grasp_pose_mode = grasp_plan.get("grasp_pose", "current")
                                grasp_rpy = grasp_plan.get("grasp_rpy", np.zeros(3, dtype=np.float32))
                                if np.asarray(grasp_rpy).shape != (3,):
                                    grasp_rpy = np.zeros(3, dtype=np.float32)
                                target_tcp_quat = compose_grasp_quat(
                                    current_tcp_quat,
                                    np.asarray(grasp_rpy, dtype=np.float32),
                                    grasp_pose_mode,
                                    sim.device,
                                )
                                grasp_plan["target_tcp_quat_w"] = None if target_tcp_quat is None else target_tcp_quat.copy()
                                grasp_plan["carry_tcp_quat_w"] = None
                                grasp_plan["place_tcp_quat_w"] = None
                                grasp_plan["active_target_tcp_quat_w"] = (
                                    None if target_tcp_quat is None else target_tcp_quat.copy()
                                )
                            preview_delta = preview_target_tcp - current_tcp
                            if mode == "grasp-block":
                                print(
                                    f"[ARM] grasp {reach_block}: approach_z={grasp_plan['approach_z']:.3f} "
                                    f"grasp_z={grasp_plan['grasp_z']:.3f} lift_z={grasp_plan['lift_z']:.3f} "
                                    f"tol={grasp_plan['tolerance']:.3f} offset_frame={reach_offset_frame} "
                                    f"grasp_pose={grasp_plan['grasp_pose']} "
                                    f"place_offset=({grasp_plan['place_offset'][0]:.3f},{grasp_plan['place_offset'][1]:.3f}) "
                                    f"grasp_rpy=({grasp_plan['grasp_rpy'][0]:.3f},{grasp_plan['grasp_rpy'][1]:.3f},{grasp_plan['grasp_rpy'][2]:.3f}) "
                                    f"place_rpy=({grasp_plan['place_rpy'][0]:.3f},{grasp_plan['place_rpy'][1]:.3f},{grasp_plan['place_rpy'][2]:.3f}) "
                                    f"tcp_offset=({reach_tcp_offset[0]:.3f}, {reach_tcp_offset[1]:.3f}, {reach_tcp_offset[2]:.3f})"
                                )
                                print(
                                    f"[ARM] grasp anchor locked in fixed world/base frame: "
                                    f"({block_pos[0]:.3f},{block_pos[1]:.3f},{block_pos[2]:.3f})"
                                )
                                plate_anchor = grasp_plan["anchor_plate_pos_w"]
                                print(
                                    f"[ARM] plate anchor locked in fixed world/base frame: "
                                    f"({plate_anchor[0]:.3f},{plate_anchor[1]:.3f},{plate_anchor[2]:.3f})"
                                )
                                if target_tcp_quat is not None:
                                    print(
                                        f"[ARM] grasp approach TCP quat locked: "
                                        f"({target_tcp_quat[0]:.4f},{target_tcp_quat[1]:.4f},"
                                        f"{target_tcp_quat[2]:.4f},{target_tcp_quat[3]:.4f})"
                                    )
                            else:
                                print(
                                    f"[ARM] reach {reach_block} cartesian-step + "
                                    f"({reach_offset[0]:.3f}, {reach_offset[1]:.3f}, {reach_offset[2]:.3f}) m "
                                    f"offset_frame={reach_offset_frame} "
                                    f"tcp_offset=({reach_tcp_offset[0]:.3f}, {reach_tcp_offset[1]:.3f}, {reach_tcp_offset[2]:.3f}) "
                                    f"hand={payload.get('hand', 'open')}"
                                )
                            print(
                                f"[ARM] preview block=({block_pos[0]:.3f},{block_pos[1]:.3f},{block_pos[2]:.3f}) "
                                f"target_tcp=({preview_target_tcp[0]:.3f},{preview_target_tcp[1]:.3f},{preview_target_tcp[2]:.3f}) "
                                f"tcp=({current_tcp[0]:.3f},{current_tcp[1]:.3f},{current_tcp[2]:.3f}) "
                                f"target_minus_tcp=({preview_delta[0]:+.3f},{preview_delta[1]:+.3f},{preview_delta[2]:+.3f})"
                            )
                            if preview_delta[2] < -0.005:
                                print(
                                    "[WARN] reach target is below the current TCP in world Z; "
                                    "the first motion will intentionally include a downward component. "
                                    "Use a larger --z-offset for an above-block approach."
                                )
                        elif mode == "hand" and payload.get("hand") in {"open", "close"}:
                            arm_mode = "hold"
                            reach_block = None
                            test_right_arm = None
                            target_tcp_pos = None
                            target_tcp_quat = None
                            grasp_plan = None
                            grasp_phase = None
                            grasp_phase_steps = 0
                            action = control_action_from_sim(robot)
                            commanded_action = action.copy()
                            hold_action = action.copy()
                            action_target_bias = control_action_bias_from_target(full_command_target, robot)
                            hand_target = CLOSE_RIGHT_HAND.copy() if payload["hand"] == "close" else OPEN_RIGHT_HAND.copy()
                            print(f"[ARM] hand {payload['hand']} while holding current right-arm state")
                        elif mode == "test-right-arm":
                            right_arm = np.asarray(payload.get("right_arm", []), dtype=np.float32)
                            if right_arm.shape == (7,):
                                arm_mode = "test-right-arm"
                                reach_block = None
                                test_right_arm = right_arm
                                target_tcp_pos = None
                                target_tcp_quat = None
                                print(f"[ARM] direct right-arm target: {[round(float(x), 3) for x in right_arm]}")
                            else:
                                arm_mode = "idle"
                                reach_block = None
                                test_right_arm = None
                                target_tcp_pos = None
                                target_tcp_quat = None
                                print("[ARM] idle: invalid test-right-arm target")
                        elif mode == "reset-scene":
                            default_target = reset_scene(scene, cfg, sim)
                            blue_offset_xy = sample_blue_xy_offset(rng, randomize_blue_xy)
                            blue_randomized = apply_blue_xy_offset(scene, cfg, sim, blue_offset_xy)
                            if randomize_blue_xy > 0.0:
                                blue_pos = blue_randomized["blue"]
                                print(
                                    f"[RANDOMIZE] blue xy offset=({blue_offset_xy[0]:+.4f},{blue_offset_xy[1]:+.4f}) "
                                    f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f})"
                                )
                            reset_camera(camera, sim, cfg)
                            settle_scene_to_target(
                                scene,
                                camera,
                                default_target,
                                sim,
                                reset_settle_steps,
                            )
                            full_command_target = default_target.copy()
                            action = control_action_from_sim(robot)
                            commanded_action = action.copy()
                            hold_action = action.copy()
                            action_target_bias = control_action_bias_from_target(full_command_target, robot)
                            arm_mode = "idle"
                            reach_block = None
                            test_right_arm = None
                            target_tcp_pos = None
                            target_tcp_quat = None
                            grasp_plan = None
                            grasp_phase = None
                            grasp_phase_steps = 0
                            reach_q_target = None
                            reach_q_current = None
                            hand_target = OPEN_RIGHT_HAND.copy()
                            reach_offset_frame = "world"
                            if reach_controller is not None:
                                reach_controller.reset_diagnostics()
                            robot.set_joint_position_target(torch.tensor(full_command_target, device=sim.device).view(1, -1))
                            robot.write_data_to_sim()
                            print(f"[SCENE] reset robot, task objects, camera, and control state after {reset_settle_steps} settle steps.")
                        elif mode == "diagnose-right-arm":
                            arm_mode = "idle"
                            reach_block = None
                            test_right_arm = None
                            target_tcp_pos = None
                            target_tcp_quat = None
                            grasp_plan = None
                            grasp_phase = None
                            grasp_phase_steps = 0
                            if reach_controller is None:
                                reach_controller = make_right_reach_controller(robot, sim.device)
                                print(f"[ARM] reach resolution: {reach_controller.resolution_summary()}")
                            diag_eps = float(payload.get("eps", 0.01))
                            diag_hold_steps = int(payload.get("hold_steps", 60))
                            diag_drive_steps = int(payload.get("drive_steps", 30))
                            print_right_arm_diagnostics(
                                robot,
                                reach_controller,
                                reach_tcp_offset,
                                full_command_target,
                                sim,
                                eps=diag_eps,
                                hold_steps=diag_hold_steps,
                                drive_steps=diag_drive_steps,
                            )
                            action = control_action_from_sim(robot)
                            commanded_action = action.copy()
                            hold_action = action.copy()
                            action_target_bias = control_action_bias_from_target(full_command_target, robot)
                            args_cli.arm_control_file.write_text(json.dumps({"mode": "idle"}, indent=2), encoding="utf-8")
                            print("[ARM] idle after diagnostics")
                        else:
                            arm_mode = "idle"
                            reach_block = None
                            test_right_arm = None
                            target_tcp_pos = None
                            target_tcp_quat = None
                            grasp_plan = None
                            grasp_phase = None
                            grasp_phase_steps = 0
                            print("[ARM] idle")
            if keyboard_jog is not None:
                action = keyboard_jog.update(action)

            desired_action = hold_action.copy() if arm_mode in {"idle", "hold"} else commanded_action.copy()
            target_pos = None
            reach_debug = None
            if arm_mode == "grasp-block" and grasp_plan is not None and reach_block is not None:
                grasp_phase_steps += 1
                block_anchor = grasp_plan.get("anchor_block_pos_w")
                if block_anchor is None:
                    block_anchor = scene[reach_block].data.root_pos_w[0].detach().cpu().numpy()
                plate_anchor = grasp_plan.get("anchor_plate_pos_w")
                if plate_anchor is None:
                    plate_anchor = scene["plate"].data.root_pos_w[0].detach().cpu().numpy()
                base_offset = grasp_plan["base_offset"]
                place_offset = grasp_plan["place_offset"]
                block_phase_offsets = {
                    "approach": np.array([base_offset[0], base_offset[1], grasp_plan["approach_z"]], dtype=np.float32),
                    "lower": np.array([base_offset[0], base_offset[1], grasp_plan["grasp_z"]], dtype=np.float32),
                    "pre_close_hold": np.array([base_offset[0], base_offset[1], grasp_plan["grasp_z"]], dtype=np.float32),
                    "close": np.array([base_offset[0], base_offset[1], grasp_plan["grasp_z"]], dtype=np.float32),
                    "lift": np.array([base_offset[0], base_offset[1], grasp_plan["lift_z"]], dtype=np.float32),
                }
                plate_phase_offsets = {
                    "move_to_plate": np.array(
                        [place_offset[0], place_offset[1], grasp_plan["place_approach_z"]],
                        dtype=np.float32,
                    ),
                    "place_lower": np.array(
                        [place_offset[0], place_offset[1], grasp_plan["place_z"]],
                        dtype=np.float32,
                    ),
                    "pre_release_hold": np.array(
                        [place_offset[0], place_offset[1], grasp_plan["place_z"]],
                        dtype=np.float32,
                    ),
                    "release": np.array(
                        [place_offset[0], place_offset[1], grasp_plan["place_z"]],
                        dtype=np.float32,
                    ),
                }
                if grasp_phase in plate_phase_offsets:
                    phase_anchor = plate_anchor
                    phase_offset = plate_phase_offsets[grasp_phase]
                else:
                    phase_anchor = block_anchor
                    phase_offset = block_phase_offsets.get(grasp_phase, block_phase_offsets["approach"])
                if reach_controller is not None:
                    if reach_offset_frame == "world":
                        phase_offset_w = phase_offset
                    else:
                        offset_t = torch.tensor(phase_offset, dtype=torch.float32, device=sim.device).view(1, 3)
                        phase_offset_w = reach_controller.rotate_wrist_vector_to_world(offset_t)[0].detach().cpu().numpy()
                    current_tcp_pose = estimate_right_hand_tcp_pose(robot, reach_controller, reach_tcp_offset)
                    current_tcp = current_tcp_pose[0] if current_tcp_pose is not None else np.zeros(3, dtype=np.float32)
                    phase_dist = float(np.linalg.norm(phase_anchor + phase_offset_w - current_tcp))
                else:
                    phase_dist = float("inf")
                old_phase = grasp_phase
                current_tcp_pose = None
                current_tcp_quat = None
                if reach_controller is not None:
                    current_tcp_pose = estimate_right_hand_tcp_pose(robot, reach_controller, reach_tcp_offset)
                    if current_tcp_pose is not None:
                        current_tcp_quat = current_tcp_pose[1]
                if grasp_phase == "approach" and (
                    phase_dist <= grasp_plan["tolerance"] or grasp_phase_steps >= grasp_plan["approach_steps"]
                ):
                    grasp_phase = "lower"
                    grasp_phase_steps = 0
                elif grasp_phase == "lower":
                    if phase_dist <= grasp_plan["tolerance"]:
                        grasp_phase = "pre_close_hold"
                        grasp_phase_steps = 0
                        print(
                            "[ARM] pre-close hold: keeping hand open at grasp pose "
                            f"for {grasp_plan['pre_close_hold_steps']} sim steps before closing."
                        )
                    elif grasp_phase_steps >= grasp_plan["lower_steps"]:
                        now = time.monotonic()
                        if now - last_grasp_wait_report > 1.0:
                            print(
                                "[ARM] waiting at lower target before closing: "
                                f"dist={phase_dist:.3f}m tol={grasp_plan['tolerance']:.3f}m. "
                                "The hand will not close high; tune --grasp-z/--tolerance if this stays stuck."
                            )
                            last_grasp_wait_report = now
                        grasp_phase_steps = grasp_plan["lower_steps"]
                elif grasp_phase == "pre_close_hold" and grasp_phase_steps >= grasp_plan["pre_close_hold_steps"]:
                    grasp_phase = "close"
                    grasp_phase_steps = 0
                elif grasp_phase == "close":
                    hand_err = right_hand_command_error(commanded_action, CLOSE_RIGHT_HAND)
                    if grasp_phase_steps >= grasp_plan["close_steps"] and hand_err <= grasp_plan["hand_complete_tolerance"]:
                        grasp_phase = "lift"
                        grasp_phase_steps = 0
                        if current_tcp_quat is not None:
                            grasp_plan["carry_tcp_quat_w"] = current_tcp_quat.copy()
                            place_rpy = grasp_plan.get("place_rpy", np.zeros(3, dtype=np.float32))
                            if np.asarray(place_rpy).shape != (3,):
                                place_rpy = np.zeros(3, dtype=np.float32)
                            grasp_plan["place_tcp_quat_w"] = compose_local_rpy_quat(
                                current_tcp_quat,
                                np.asarray(place_rpy, dtype=np.float32),
                                sim.device,
                            )
                            if args_cli.verbose_status:
                                print(
                                    "[ARM] carry/place TCP quat locked from actual grasp pose: "
                                    f"({current_tcp_quat[0]:.4f},{current_tcp_quat[1]:.4f},"
                                    f"{current_tcp_quat[2]:.4f},{current_tcp_quat[3]:.4f})"
                                )
                                place_quat = grasp_plan["place_tcp_quat_w"]
                                print(
                                    "[ARM] place TCP quat with local x-roll offset: "
                                    f"({place_quat[0]:.4f},{place_quat[1]:.4f},"
                                    f"{place_quat[2]:.4f},{place_quat[3]:.4f})"
                                )
                    elif grasp_phase_steps >= grasp_plan["close_steps"]:
                        now = time.monotonic()
                        if now - last_grasp_wait_report > 1.0:
                            print(
                                "[ARM] waiting for hand close before lifting: "
                                f"cmd_err={hand_err:.3f} tol={grasp_plan['hand_complete_tolerance']:.3f}"
                            )
                            last_grasp_wait_report = now
                        grasp_phase_steps = grasp_plan["close_steps"]
                elif grasp_phase == "lift" and (
                    phase_dist <= grasp_plan["tolerance"] or grasp_phase_steps >= grasp_plan["lift_steps"]
                ):
                    grasp_phase = "move_to_plate"
                    grasp_phase_steps = 0
                elif grasp_phase == "move_to_plate" and (
                    phase_dist <= grasp_plan["tolerance"] or grasp_phase_steps >= grasp_plan["place_steps"]
                ):
                    grasp_phase = "place_lower"
                    grasp_phase_steps = 0
                elif grasp_phase == "place_lower":
                    if phase_dist <= grasp_plan["tolerance"]:
                        grasp_phase = "pre_release_hold"
                        grasp_phase_steps = 0
                        print(
                            "[ARM] pre-release hold: keeping hand closed at place pose "
                            f"for {grasp_plan['pre_release_hold_steps']} sim steps before opening."
                        )
                    elif grasp_phase_steps >= grasp_plan["place_steps"]:
                        now = time.monotonic()
                        if now - last_grasp_wait_report > 1.0:
                            print(
                                "[ARM] waiting at plate release target before opening: "
                                f"dist={phase_dist:.3f}m tol={grasp_plan['tolerance']:.3f}m. "
                                "The hand will not open high; tune --place-z/--tolerance if this stays stuck."
                            )
                            last_grasp_wait_report = now
                        grasp_phase_steps = grasp_plan["place_steps"]
                elif grasp_phase == "pre_release_hold" and grasp_phase_steps >= grasp_plan["pre_release_hold_steps"]:
                    grasp_phase = "release"
                    grasp_phase_steps = 0
                elif grasp_phase == "release":
                    hand_err = right_hand_command_error(commanded_action, OPEN_RIGHT_HAND)
                    if grasp_phase_steps >= grasp_plan["release_steps"] and hand_err <= grasp_plan["hand_complete_tolerance"]:
                        commanded_action[ACTION_SLICES.right_hand] = OPEN_RIGHT_HAND.copy()
                        desired_action[ACTION_SLICES.right_hand] = OPEN_RIGHT_HAND.copy()
                        hand_target = OPEN_RIGHT_HAND.copy()
                        grasp_phase = "done"
                        grasp_phase_steps = 0
                    elif grasp_phase_steps >= grasp_plan["release_steps"]:
                        now = time.monotonic()
                        if now - last_grasp_wait_report > 1.0:
                            print(
                                "[ARM] waiting for hand open before ending: "
                                f"cmd_err={hand_err:.3f} tol={grasp_plan['hand_complete_tolerance']:.3f}"
                            )
                            last_grasp_wait_report = now
                        grasp_phase_steps = grasp_plan["release_steps"]
                if grasp_phase != old_phase:
                    print(f"[ARM] grasp phase {old_phase} -> {grasp_phase} dist={phase_dist:.3f}m")
                if grasp_phase == "lift":
                    carry_quat = grasp_plan.get("carry_tcp_quat_w")
                    grasp_plan["active_target_tcp_quat_w"] = carry_quat if carry_quat is not None else grasp_plan.get("target_tcp_quat_w")
                elif grasp_phase in {"move_to_plate", "place_lower", "pre_release_hold", "release", "done"}:
                    place_quat = grasp_plan.get("place_tcp_quat_w")
                    carry_quat = grasp_plan.get("carry_tcp_quat_w")
                    grasp_plan["active_target_tcp_quat_w"] = (
                        place_quat if place_quat is not None else carry_quat if carry_quat is not None else grasp_plan.get("target_tcp_quat_w")
                    )
                else:
                    grasp_plan["active_target_tcp_quat_w"] = grasp_plan.get("target_tcp_quat_w")
                if grasp_phase in plate_phase_offsets:
                    target_block_pos_w = plate_anchor
                    reach_offset = plate_phase_offsets[grasp_phase]
                else:
                    target_block_pos_w = block_anchor
                    reach_offset = block_phase_offsets.get(grasp_phase, block_phase_offsets["approach"])
                grasp_plan["active_anchor_pos_w"] = target_block_pos_w.copy()
                hand_target = (
                    CLOSE_RIGHT_HAND.copy()
                    if grasp_phase in {"close", "lift", "move_to_plate", "place_lower", "pre_release_hold"}
                    else OPEN_RIGHT_HAND.copy()
                )
                if grasp_phase == "done":
                    arm_mode = "hold"
                    hold_action = commanded_action.copy()
                    commanded_action = hold_action.copy()
                    desired_action = hold_action.copy()
                    hand_target = OPEN_RIGHT_HAND.copy()
                    print("[ARM] grasp-place sequence done; released cylinder and holding final pose.")
            if (
                arm_mode in {"reach-block", "grasp-block"}
                and reach_controller is not None
                and reach_block is not None
            ):
                target_block_pos_w = None
                target_pose_quat_w = None
                if arm_mode == "grasp-block" and grasp_plan is not None:
                    target_block_pos_w = grasp_plan.get("active_anchor_pos_w", grasp_plan.get("anchor_block_pos_w"))
                    target_pose_quat_w = grasp_plan.get("active_target_tcp_quat_w", grasp_plan.get("target_tcp_quat_w"))
                desired_action, reach_debug, reach_q_target, reach_q_current = reach_controller.update_action(
                    desired_action,
                    scene[reach_block],
                    reach_offset,
                    reach_tcp_offset,
                    hand_target,
                    offset_frame=(
                        reach_offset_frame
                    ),
                    target_block_pos_w=target_block_pos_w,
                    target_tcp_quat_w=target_pose_quat_w,
                )
                target_pos = reach_debug.target_tcp_pos
                target_tcp_pos = reach_debug.target_tcp_pos
                target_tcp_quat = reach_debug.target_tcp_quat
                if reach_debug.held_for_safety:
                    now = time.monotonic()
                    if now - last_unstable_report > 1.0:
                        print(f"[WARN] reach held for safety: {reach_debug.safety_reason}")
                        last_unstable_report = now
            elif arm_mode == "test-right-arm" and test_right_arm is not None:
                desired_action[ACTION_SLICES.right_arm] = test_right_arm
                desired_action[ACTION_SLICES.right_hand] = hand_target
            if hand_target is not None and arm_mode != "idle":
                desired_action[ACTION_SLICES.right_hand] = hand_target

            next_commanded_action = smooth_command(
                commanded_action,
                desired_action,
                alpha=float(args_cli.target_alpha),
                max_joint_step=float(args_cli.max_joint_step),
            )
            hand_delta = np.clip(
                next_commanded_action[ACTION_SLICES.right_hand] - commanded_action[ACTION_SLICES.right_hand],
                -float(args_cli.hand_max_joint_step),
                float(args_cli.hand_max_joint_step),
            )
            next_commanded_action[ACTION_SLICES.right_hand] = commanded_action[ACTION_SLICES.right_hand] + hand_delta
            commanded_action = next_commanded_action
            if arm_mode == "idle":
                full_command_target = default_target.copy()
            else:
                full_command_target = default_target.copy()
                right_arm_drive_target = commanded_action[ACTION_SLICES.right_arm]
                set_named_joint_targets(
                    full_command_target,
                    robot,
                    RIGHT_ARM_JOINTS,
                    right_arm_drive_target,
                )
                set_named_joint_targets(
                    full_command_target,
                    robot,
                    RIGHT_HAND_JOINTS,
                    commanded_action[ACTION_SLICES.right_hand],
                )
            if reset_unstable_arm_state(
                robot,
                RIGHT_ARM_JOINTS,
                full_command_target,
                threshold_rad=float(args_cli.unstable_arm_threshold),
                velocity_threshold_rad_s=float(args_cli.unstable_arm_velocity_threshold),
            ):
                now = time.monotonic()
                if now - last_unstable_report > 1.0:
                    print("[WARN] right arm joint state became unstable; reset right-arm state to clamped target.")
                    last_unstable_report = now
            robot.set_joint_position_target(torch.tensor(full_command_target, device=sim.device).view(1, -1))
            last_gravity_comp_stats = apply_gravity_compensation(
                robot,
                gravity_comp_joint_ids,
                scale=float(args_cli.gravity_comp_scale),
                enabled=bool(args_cli.gravity_compensation),
            )
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(dt=sim_dt)
            for key in TASK_OBJECT_KEYS:
                scene[key].update(dt=sim_dt)
            update_all_cameras(scene, camera, dt=sim_dt)
            if recording_episode is not None:
                wall_elapsed = time.monotonic() - record_wall_start if record_wall_start is not None else 0.0
                record_timeout_s = max(float(args_cli.record_episode_timeout_s), 1.0)
                if wall_elapsed >= record_timeout_s:
                    print(
                        f"[RECORD][TIMEOUT] discarded episode attempt for index {recorded_episodes}: "
                        f"wall_seconds={wall_elapsed:.1f}s timeout={record_timeout_s:.1f}s "
                        f"frames={len(recording_episode)} sim_steps={record_step}. Resetting and retrying."
                    )
                    recording_episode = None
                    record_wall_start = None
                    record_step = 0
                    default_target = reset_scene(scene, cfg, sim)
                    blue_offset_xy = sample_blue_xy_offset(rng, randomize_blue_xy)
                    blue_randomized = apply_blue_xy_offset(scene, cfg, sim, blue_offset_xy)
                    if randomize_blue_xy > 0.0:
                        blue_pos = blue_randomized["blue"]
                        print(
                            f"[RANDOMIZE] blue xy offset=({blue_offset_xy[0]:+.4f},{blue_offset_xy[1]:+.4f}) "
                            f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f})"
                        )
                    reset_camera(camera, sim, cfg)
                    settle_scene_to_target(scene, camera, default_target, sim, reset_settle_steps)
                    full_command_target = default_target.copy()
                    action = control_action_from_sim(robot)
                    commanded_action = action.copy()
                    hold_action = action.copy()
                    action_target_bias = control_action_bias_from_target(full_command_target, robot)
                    arm_mode = "idle"
                    reach_block = None
                    target_tcp_pos = None
                    target_tcp_quat = None
                    grasp_plan = None
                    grasp_phase = None
                    grasp_phase_steps = 0
                    hand_target = OPEN_RIGHT_HAND.copy()
                    auto_grasp_pending = True
                    continue
                if record_step % record_every_n == 0:
                    append_record_frame(
                        recording_episode,
                        scene,
                        robot,
                        camera,
                        commanded_action,
                        reach_controller,
                        reach_tcp_offset,
                    )
                record_step += 1
                if arm_mode == "hold" and grasp_phase == "done":
                    success_ok, success_stats = final_cylinder_in_plate(scene, args_cli.auto_grasp_block)
                    if bool(args_cli.success_check) and not success_ok:
                        wall_seconds = time.monotonic() - record_wall_start if record_wall_start is not None else float("nan")
                        print(
                            f"[RECORD][FAIL] discarded episode attempt for index {recorded_episodes}: "
                            f"{args_cli.auto_grasp_block}_xy_dist={success_stats['xy_dist']:.3f}m "
                            f"tol={success_stats['xy_tolerance']:.3f}m "
                            f"z_above_plate={success_stats['z_above_plate']:.3f}m "
                            f"allowed=[{success_stats['z_min']:.3f},{success_stats['z_max']:.3f}]m "
                            f"frames={len(recording_episode)} sim_steps={record_step} "
                            f"wall_seconds={wall_seconds:.2f}s. Resetting and retrying."
                        )
                        recording_episode = None
                        record_wall_start = None
                        record_step = 0
                        default_target = reset_scene(scene, cfg, sim)
                        blue_offset_xy = sample_blue_xy_offset(rng, randomize_blue_xy)
                        blue_randomized = apply_blue_xy_offset(scene, cfg, sim, blue_offset_xy)
                        if randomize_blue_xy > 0.0:
                            blue_pos = blue_randomized["blue"]
                            print(
                                f"[RANDOMIZE] blue xy offset=({blue_offset_xy[0]:+.4f},{blue_offset_xy[1]:+.4f}) "
                                f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f})"
                            )
                        reset_camera(camera, sim, cfg)
                        settle_scene_to_target(scene, camera, default_target, sim, reset_settle_steps)
                        full_command_target = default_target.copy()
                        action = control_action_from_sim(robot)
                        commanded_action = action.copy()
                        hold_action = action.copy()
                        action_target_bias = control_action_bias_from_target(full_command_target, robot)
                        arm_mode = "idle"
                        reach_block = None
                        target_tcp_pos = None
                        target_tcp_quat = None
                        grasp_plan = None
                        grasp_phase = None
                        grasp_phase_steps = 0
                        hand_target = OPEN_RIGHT_HAND.copy()
                        auto_grasp_pending = True
                        continue
                    demo_name = writer.write_episode(recording_episode) if writer is not None else "demo"
                    sim_seconds = record_step * sim_dt
                    wall_seconds = time.monotonic() - record_wall_start if record_wall_start is not None else float("nan")
                    realtime_factor = sim_seconds / wall_seconds if wall_seconds > 1e-6 else float("nan")
                    print(
                        f"[RECORD] wrote {demo_name}: {len(recording_episode)} frames "
                        f"sim_steps={record_step} sim_seconds={sim_seconds:.2f}s "
                        f"wall_seconds={wall_seconds:.2f}s realtime_factor={realtime_factor:.2f}x "
                        f"{args_cli.auto_grasp_block}_xy_dist={success_stats['xy_dist']:.3f}m "
                        f"z_above_plate={success_stats['z_above_plate']:.3f}m"
                    )
                    recording_episode = None
                    record_wall_start = None
                    recorded_episodes += 1
                    if writer is not None and recorded_episodes >= max_record_episodes:
                        record_complete = True
                        break
                    default_target = reset_scene(scene, cfg, sim)
                    blue_offset_xy = sample_blue_xy_offset(rng, randomize_blue_xy)
                    blue_randomized = apply_blue_xy_offset(scene, cfg, sim, blue_offset_xy)
                    if randomize_blue_xy > 0.0:
                        blue_pos = blue_randomized["blue"]
                        print(
                            f"[RANDOMIZE] blue xy offset=({blue_offset_xy[0]:+.4f},{blue_offset_xy[1]:+.4f}) "
                            f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f})"
                        )
                    reset_camera(camera, sim, cfg)
                    settle_scene_to_target(scene, camera, default_target, sim, reset_settle_steps)
                    full_command_target = default_target.copy()
                    action = control_action_from_sim(robot)
                    commanded_action = action.copy()
                    hold_action = action.copy()
                    arm_mode = "idle"
                    reach_block = None
                    grasp_plan = None
                    grasp_phase = None
                    grasp_phase_steps = 0
                    hand_target = OPEN_RIGHT_HAND.copy()
                    auto_grasp_pending = True
            if tcp_visualizer is not None:
                hand_tcp_pose = estimate_right_hand_tcp_pose(robot, reach_controller, reach_tcp_offset)
                if hand_tcp_pose is None:
                    hand_tcp_pose = estimate_right_hand_tcp_pose_from_robot(robot, reach_tcp_offset)
                if target_tcp_pos is None and reach_block is not None:
                    block_pos = scene[reach_block].data.root_pos_w[0].detach().cpu().numpy()
                    target_tcp_pos = block_pos + reach_offset
                tcp_visualizer.visualize_task_frames(
                    left_tcp_pose=estimate_left_hand_tcp_pose_from_robot(robot, reach_tcp_offset),
                    right_tcp_pose=hand_tcp_pose,
                    drawer_handle_pose=get_drawer_handle_top_pose() if args_cli.show_drawer_handle_frame else None,
                    target_tcp_pos=target_tcp_pos,
                    target_tcp_quat=target_tcp_quat,
                )
            if wrist_frustum_visualizer is not None:
                wrist_frustum_visualizer.update(scene)

            now = time.monotonic()
            if args_cli.print_tcp_pose and now - last_tcp_report >= max(float(args_cli.tcp_print_period), 0.05):
                pose = estimate_right_hand_tcp_pose(robot, reach_controller, reach_tcp_offset)
                if pose is None:
                    pose = estimate_right_hand_tcp_pose_from_robot(robot, reach_tcp_offset)
                print(format_tcp_pose_line("[TCP] right_hand", pose, estimate_body_pose_from_robot(robot, "base_link")))
                last_tcp_report = now
            if args_cli.verbose_status and now - last_report > 2.0:
                red_pos = scene["red"].data.root_pos_w[0].detach().cpu().numpy()
                blue_pos = scene["blue"].data.root_pos_w[0].detach().cpu().numpy()
                plate_pos = scene["plate"].data.root_pos_w[0].detach().cpu().numpy()
                status = (
                    f"red=({red_pos[0]:.3f},{red_pos[1]:.3f},{red_pos[2]:.3f}) "
                    f"blue=({blue_pos[0]:.3f},{blue_pos[1]:.3f},{blue_pos[2]:.3f}) "
                    f"plate=({plate_pos[0]:.3f},{plate_pos[1]:.3f},{plate_pos[2]:.3f})"
                )
                if arm_mode == "grasp-block" and grasp_phase is not None:
                    status += f" grasp_phase={grasp_phase}[{grasp_phase_steps}]"
                if reach_debug is not None:
                    status += (
                        f" block=({reach_debug.block_pos[0]:.3f},{reach_debug.block_pos[1]:.3f},{reach_debug.block_pos[2]:.3f}) "
                        f"offset_frame={reach_debug.offset_frame} "
                        f"offset_w=({reach_debug.offset_world[0]:.3f},{reach_debug.offset_world[1]:.3f},{reach_debug.offset_world[2]:.3f}) "
                        f"target_tcp=({reach_debug.target_tcp_pos[0]:.3f},{reach_debug.target_tcp_pos[1]:.3f},{reach_debug.target_tcp_pos[2]:.3f}) "
                        f"tcp=({reach_debug.current_tcp_pos[0]:.3f},{reach_debug.current_tcp_pos[1]:.3f},{reach_debug.current_tcp_pos[2]:.3f}) "
                        f"tcp_err=({reach_debug.tcp_error[0]:.3f},{reach_debug.tcp_error[1]:.3f},{reach_debug.tcp_error[2]:.3f}) "
                        f"rot_err=({reach_debug.rot_error_axis_angle[0]:.3f},{reach_debug.rot_error_axis_angle[1]:.3f},{reach_debug.rot_error_axis_angle[2]:.3f}) "
                        f"step_w=({reach_debug.step_error_world[0]:.4f},{reach_debug.step_error_world[1]:.4f},{reach_debug.step_error_world[2]:.4f}) "
                        f"pred_w=({reach_debug.predicted_tcp_delta[0]:.4f},{reach_debug.predicted_tcp_delta[1]:.4f},{reach_debug.predicted_tcp_delta[2]:.4f}) "
                        f"actual_d=({reach_debug.actual_delta_world[0]:.4f},{reach_debug.actual_delta_world[1]:.4f},{reach_debug.actual_delta_world[2]:.4f}) "
                        f"dq=({','.join(f'{x:+.4f}' for x in reach_debug.joint_delta)}) "
                        f"jac_row={reach_debug.jacobian_body_row} "
                        f"jac_sign={reach_debug.jacobian_sign:.1f} "
                        f"dir_sign={reach_debug.direction_sign:.1f} "
                        f"progress={reach_debug.actual_progress:.5f} "
                        f"right_tcp_dist={reach_debug.tcp_dist:.3f}m"
                    )
                    if reach_debug.held_for_safety:
                        status += f" reach_held={reach_debug.safety_reason}"
                if reach_q_target is not None and reach_q_current is not None:
                    q_err = float(np.max(np.abs(reach_q_target - reach_q_current)))
                    q_lag = float(np.max(np.abs(reach_q_target - commanded_action[ACTION_SLICES.right_arm])))
                    cmd_delta = commanded_action[ACTION_SLICES.right_arm] - reach_q_current
                    drive_delta = (
                        commanded_action[ACTION_SLICES.right_arm]
                        + action_target_bias[ACTION_SLICES.right_arm]
                        - reach_q_current
                    )
                    status += (
                        f" right_arm_q_err={q_err:.3f} right_arm_cmd_lag={q_lag:.3f} "
                        f"cmd_delta=({','.join(f'{x:+.4f}' for x in cmd_delta)}) "
                        f"drive_delta=({','.join(f'{x:+.4f}' for x in drive_delta)})"
                    )
                if args_cli.gravity_compensation:
                    status += (
                        f" gravity_comp=max:{last_gravity_comp_stats[0]:.2f}"
                        f"/mean:{last_gravity_comp_stats[1]:.2f}"
                    )
                if arm_mode == "test-right-arm" and test_right_arm is not None:
                    q_lag = float(np.max(np.abs(test_right_arm - commanded_action[ACTION_SLICES.right_arm])))
                    status += f" direct_right_arm_cmd_lag={q_lag:.3f}"
                print(status)
                last_report = now
    finally:
        if keyboard_jog is not None:
            keyboard_jog.stop()
        if writer is not None:
            writer.close()
        if record_complete and args_cli.record_output is not None:
            print(f"[RECORD] complete: wrote {recorded_episodes} episode(s) to {args_cli.record_output}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def main() -> None:
    # Match collection and rollout to the same scripted YAML switches. Preview /
    # teleop stay distractor-free unless recording (or an explicit CLI override)
    # enables them.
    scripted_cfg = load_active_drawer_scripted_cfg()
    distractor_enabled = (
        resolve_record_distractor_cans_enabled(scripted_cfg)
        if args_cli.record_output is not None
        else bool(args_cli.distractor_cans) if args_cli.distractor_cans is not None else False
    )
    apply_distractor_spawn_env(distractor_enabled)
    cfg = make_scene_cfg()
    if args_cli.print_layout:
        print(format_action_layout())
        project_cfg = load_project_config(CONFIG_PATH)
        if project_cfg.dataset.task_id == "right_blue_cylinder_plate":
            print(format_layout(cfg))
        else:
            print(f"Task layout will be printed by scene builder: {project_cfg.dataset.task_id}")

    print("[BOOT] creating SimulationContext...", flush=True)
    sim_device = args_cli.device
    use_fabric = True
    if args_cli.live_usd_transforms:
        sim_device = "cpu"
        use_fabric = False
        print(
            "[DEBUG] --live-usd-transforms: CPU PhysX + Fabric disabled; "
            "Stage transforms will follow current articulation poses.",
            flush=True,
        )
    if args_cli.drawer_open and str(sim_device).startswith("cuda"):
        sim_device = "cpu"
        print(
            "[DRAWER] --drawer-open uses CPU PhysX because direct GPU API forbids runtime articulation drive targets.",
            flush=True,
        )
    sim = create_simulation_context(sim_device, use_fabric=use_fabric)
    print("[BOOT] building scene...", flush=True)
    scene_builder = resolve_scene_builder()
    print(f"[BOOT] scene builder: {scene_builder.__module__}:{scene_builder.__name__}", flush=True)
    scene = scene_builder(cfg)
    if args_cli.show_wrist_camera_frustums and not args_cli.headless:
        # Fabric population happens during the first reset/play. Create camera-local
        # debug children beforehand so they inherit the live camera hierarchy.
        print("[BOOT] creating camera-local wrist frustums before Fabric population...", flush=True)
        scene["wrist_frustum_visualizer"] = WristCameraFrustumVisualizer(
            scene,
            depth=float(args_cli.wrist_camera_frustum_depth),
            line_width=float(args_cli.wrist_camera_frustum_line_width),
            visual_scale=float(args_cli.wrist_camera_frustum_scale),
        )
    print("[BOOT] resetting simulation...", flush=True)
    sim.reset()
    print("[BOOT] resetting camera...", flush=True)
    reset_camera(scene["camera"], sim, cfg)
    print("[BOOT] entering run loop...", flush=True)
    run_debug(scene, cfg, sim)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[FATAL] record_dataset.py failed before normal shutdown:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
