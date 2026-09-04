"""Configuration loading and validation for the isolated teleoperation subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class NetworkConfig:
    host: str
    port: int
    stale_timeout_s: float


@dataclass(frozen=True)
class ClutchConfig:
    engage_threshold: float
    release_threshold: float
    button_indices: tuple[int, ...]


@dataclass(frozen=True)
class MappingConfig:
    controller_to_base_rotation: np.ndarray
    position_scale: float
    orientation_enabled: bool
    max_clutch_translation_m: float
    clutch: ClutchConfig
    controller_filter_time_constant_s: float = 0.0
    invert_translation: bool = False
    invert_orientation: bool = False
    # Per-axis sign applied after the rotation basis. (-1,-1,+1) keeps
    # robot Z (up) while flipping the horizontal plane.
    translation_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SafetyConfig:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_translation_speed_m_s: float
    max_rotation_speed_rad_s: float


@dataclass(frozen=True)
class SmoothingConfig:
    trigger_time_constant_s: float
    trigger_deadband: float
    arm_command_alpha: float
    arm_max_joint_step_rad: float
    hand_command_alpha: float
    hand_max_joint_step_rad: float


@dataclass(frozen=True)
class IkConfig:
    posture_gain: float
    damping: float
    max_joint_delta_rad: float
    orientation_weight: float


@dataclass(frozen=True)
class RmpFlowArmConfig:
    descriptor_file: Path
    policy_config_file: Path
    frame_name: str


@dataclass(frozen=True)
class RmpFlowConfig:
    name: str
    urdf_file: Path
    evaluations_per_frame: float
    update_every_n_steps: int
    ignore_robot_state_updates: bool
    left: RmpFlowArmConfig
    right: RmpFlowArmConfig


@dataclass(frozen=True)
class PinkElbowAvoidanceConfig:
    enabled: bool
    left_frame_name: str
    right_frame_name: str
    min_lateral_distance_base_m: float
    gain: float


@dataclass(frozen=True)
class PinkConfig:
    urdf_file: Path
    solver: str
    left_frame_name: str
    right_frame_name: str
    tcp_offset_wrist: np.ndarray
    position_cost: float
    orientation_cost: float
    posture_cost: float
    task_gain: float
    lm_damping: float
    damping: float
    max_joint_velocity_rad_s: float
    elbow_avoidance: PinkElbowAvoidanceConfig


@dataclass(frozen=True)
class ControllerConfig:
    backend: str
    rmpflow: RmpFlowConfig
    pink: PinkConfig


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    hardware: dict[str, Any]


@dataclass(frozen=True)
class SimulationConfig:
    reset_settle_s: float
    real_time: bool
    render_every_n_steps: int
    spawn_rgb_cameras: bool
    gravity_compensation: bool
    gravity_comp_scale: float
    joint_stiffness: float
    joint_damping: float
    joint_effort_limit: float


@dataclass(frozen=True)
class TeleopConfig:
    network: NetworkConfig
    mapping: MappingConfig
    safety: SafetyConfig
    smoothing: SmoothingConfig
    ik: IkConfig
    controller: ControllerConfig
    runtime: RuntimeConfig
    simulation: SimulationConfig
    raw: dict[str, Any]
    source: Path


def _vec(mapping: dict[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(mapping[key], dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {length} finite values")
    return value


def _project_path(source: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_root = source.parents[2]
    return (project_root / path).resolve()


def _rmpflow_arm(source: Path, mapping: dict[str, Any], side: str) -> RmpFlowArmConfig:
    arm = mapping[side]
    return RmpFlowArmConfig(
        descriptor_file=_project_path(source, str(arm["descriptor_file"])),
        policy_config_file=_project_path(source, str(arm["policy_config_file"])),
        frame_name=str(arm["frame_name"]),
    )


def load_teleop_config(path: Path) -> TeleopConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping: {source}")
    network = raw["network"]
    mapping = raw["mapping"]
    clutch = mapping["clutch"]
    safety = raw["safety"]
    smoothing = raw["smoothing"]
    ik = raw["ik"]
    controller = raw.get("controller", {})
    rmpflow = controller.get("rmpflow", {})
    pink = controller.get("pink", {})
    simulation = raw["simulation"]
    runtime = raw.get("runtime", {})

    basis = np.asarray(mapping["controller_to_base_rotation"], dtype=np.float64)
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ValueError("mapping.controller_to_base_rotation must be a finite 3x3 matrix")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-5) or not np.isclose(
        np.linalg.det(basis), 1.0, atol=1.0e-5
    ):
        raise ValueError("mapping.controller_to_base_rotation must be a proper rotation matrix")
    engage = float(clutch["engage_threshold"])
    release = float(clutch["release_threshold"])
    if not 0.0 <= release < engage <= 1.0:
        raise ValueError("clutch thresholds require 0 <= release < engage <= 1")
    button_indices = tuple(int(value) for value in clutch.get("button_indices", [1]))
    if not button_indices or any(value < 0 or value >= 16 for value in button_indices):
        raise ValueError("mapping.clutch.button_indices must contain indices in [0, 15]")
    max_clutch_translation_m = float(mapping.get("max_clutch_translation_m", float("inf")))
    if max_clutch_translation_m <= 0.0:
        raise ValueError("mapping.max_clutch_translation_m must be positive")
    controller_filter_time_constant_s = float(
        mapping.get("controller_filter_time_constant_s", 0.0)
    )
    if controller_filter_time_constant_s < 0.0:
        raise ValueError("mapping.controller_filter_time_constant_s must be non-negative")
    translation_sign = np.asarray(mapping.get("translation_sign", [1.0, 1.0, 1.0]), dtype=np.float64)
    if translation_sign.shape != (3,) or not np.isfinite(translation_sign).all():
        raise ValueError("mapping.translation_sign must contain 3 finite values")
    if not np.all(np.isin(translation_sign, (-1.0, 1.0))):
        raise ValueError("mapping.translation_sign entries must be +1 or -1")
    workspace_min = _vec(safety, "workspace_min_base_m", 3)
    workspace_max = _vec(safety, "workspace_max_base_m", 3)
    if np.any(workspace_min >= workspace_max):
        raise ValueError("workspace_min_base_m must be below workspace_max_base_m")
    backend = str(controller.get("backend", "pinocchio")).lower()
    if backend not in {"pinocchio", "pink", "rmpflow"}:
        raise ValueError("controller.backend must be 'pinocchio', 'pink' or 'rmpflow'")
    if not rmpflow:
        raise ValueError("controller.rmpflow configuration is required")
    evaluations_per_frame = float(rmpflow.get("evaluations_per_frame", 4.0))
    if evaluations_per_frame <= 0.0:
        raise ValueError("controller.rmpflow.evaluations_per_frame must be positive")
    update_every_n_steps = int(rmpflow.get("update_every_n_steps", 1))
    if update_every_n_steps < 1:
        raise ValueError("controller.rmpflow.update_every_n_steps must be at least 1")
    rmpflow_config = RmpFlowConfig(
        name=str(rmpflow.get("name", "rmp_flow")),
        urdf_file=_project_path(source, str(rmpflow["urdf_file"])),
        evaluations_per_frame=evaluations_per_frame,
        update_every_n_steps=update_every_n_steps,
        ignore_robot_state_updates=bool(rmpflow.get("ignore_robot_state_updates", False)),
        left=_rmpflow_arm(source, rmpflow, "left"),
        right=_rmpflow_arm(source, rmpflow, "right"),
    )
    required_files = (
        rmpflow_config.urdf_file,
        rmpflow_config.left.descriptor_file,
        rmpflow_config.left.policy_config_file,
        rmpflow_config.right.descriptor_file,
        rmpflow_config.right.policy_config_file,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("RMPflow configuration files are missing: " + ", ".join(missing))
    if not pink:
        raise ValueError("controller.pink configuration is required")
    pink_urdf = _project_path(source, str(pink.get("urdf_file", rmpflow["urdf_file"])))
    if not pink_urdf.is_file():
        raise FileNotFoundError(f"Pink URDF is missing: {pink_urdf}")
    tcp_offset = _vec(pink, "tcp_offset_wrist_m", 3)
    solver = str(pink.get("solver", "daqp"))
    if not solver:
        raise ValueError("controller.pink.solver must not be empty")
    elbow_avoidance = pink.get("elbow_avoidance", {})
    if not isinstance(elbow_avoidance, dict):
        raise TypeError("controller.pink.elbow_avoidance must be a mapping")
    elbow_avoidance_config = PinkElbowAvoidanceConfig(
        enabled=bool(elbow_avoidance.get("enabled", False)),
        left_frame_name=str(elbow_avoidance.get("left_frame_name", "left_elbow_link")),
        right_frame_name=str(elbow_avoidance.get("right_frame_name", "right_elbow_link")),
        min_lateral_distance_base_m=float(
            elbow_avoidance.get("min_lateral_distance_base_m", 0.28)
        ),
        gain=float(elbow_avoidance.get("gain", 1.0)),
    )
    if not elbow_avoidance_config.left_frame_name or not elbow_avoidance_config.right_frame_name:
        raise ValueError("controller.pink.elbow_avoidance frame names must not be empty")
    if not 0.0 < elbow_avoidance_config.min_lateral_distance_base_m < 1.0:
        raise ValueError(
            "controller.pink.elbow_avoidance.min_lateral_distance_base_m "
            "must be in (0, 1)"
        )
    if elbow_avoidance_config.gain <= 0.0:
        raise ValueError("controller.pink.elbow_avoidance.gain must be positive")
    pink_config = PinkConfig(
        urdf_file=pink_urdf,
        solver=solver,
        left_frame_name=str(pink.get("left_frame_name", "left_wrist_yaw_link")),
        right_frame_name=str(pink.get("right_frame_name", "right_wrist_yaw_link")),
        tcp_offset_wrist=tcp_offset,
        position_cost=float(pink.get("position_cost", 1.0)),
        orientation_cost=float(pink.get("orientation_cost", 0.65)),
        posture_cost=float(pink.get("posture_cost", 1.0e-3)),
        task_gain=float(pink.get("task_gain", 0.5)),
        lm_damping=float(pink.get("lm_damping", 0.1)),
        damping=float(pink.get("damping", 1.0e-6)),
        max_joint_velocity_rad_s=float(pink.get("max_joint_velocity_rad_s", 4.8)),
        elbow_avoidance=elbow_avoidance_config,
    )
    positive_pink_values = {
        "position_cost": pink_config.position_cost,
        "posture_cost": pink_config.posture_cost,
        "task_gain": pink_config.task_gain,
        "max_joint_velocity_rad_s": pink_config.max_joint_velocity_rad_s,
    }
    invalid_pink = [name for name, value in positive_pink_values.items() if value <= 0.0]
    if invalid_pink:
        raise ValueError("controller.pink values must be positive: " + ", ".join(invalid_pink))
    if not 0.0 <= pink_config.orientation_cost:
        raise ValueError("controller.pink.orientation_cost must be non-negative")
    if not 0.0 < pink_config.task_gain <= 1.0:
        raise ValueError("controller.pink.task_gain must be in (0, 1]")
    render_every_n_steps = int(simulation.get("render_every_n_steps", 1))
    if render_every_n_steps < 1:
        raise ValueError("simulation.render_every_n_steps must be at least 1")
    spawn_rgb_cameras = bool(simulation.get("spawn_rgb_cameras", True))
    runtime_mode = str(runtime.get("mode", "simulation")).lower()
    if runtime_mode not in {"simulation", "hardware"}:
        raise ValueError("runtime.mode must be 'simulation' or 'hardware'")
    hardware_cfg = runtime.get("hardware", {})
    if not isinstance(hardware_cfg, dict):
        raise TypeError("runtime.hardware must be a mapping")

    return TeleopConfig(
        network=NetworkConfig(
            host=str(network.get("host", "0.0.0.0")),
            port=int(network.get("port", 8443)),
            stale_timeout_s=float(network.get("stale_timeout_s", 0.25)),
        ),
        mapping=MappingConfig(
            controller_to_base_rotation=basis,
            position_scale=float(mapping.get("position_scale", 1.0)),
            orientation_enabled=bool(mapping.get("orientation_enabled", True)),
            max_clutch_translation_m=max_clutch_translation_m,
            clutch=ClutchConfig(engage, release, button_indices),
            controller_filter_time_constant_s=controller_filter_time_constant_s,
            invert_translation=bool(mapping.get("invert_translation", False)),
            invert_orientation=bool(mapping.get("invert_orientation", False)),
            translation_sign=(
                float(translation_sign[0]),
                float(translation_sign[1]),
                float(translation_sign[2]),
            ),
        ),
        safety=SafetyConfig(
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            max_translation_speed_m_s=float(safety["max_translation_speed_m_s"]),
            max_rotation_speed_rad_s=float(safety["max_rotation_speed_rad_s"]),
        ),
        smoothing=SmoothingConfig(
            trigger_time_constant_s=float(smoothing["trigger_time_constant_s"]),
            trigger_deadband=float(smoothing["trigger_deadband"]),
            arm_command_alpha=float(smoothing["arm_command_alpha"]),
            arm_max_joint_step_rad=float(smoothing["arm_max_joint_step_rad"]),
            hand_command_alpha=float(smoothing["hand_command_alpha"]),
            hand_max_joint_step_rad=float(smoothing["hand_max_joint_step_rad"]),
        ),
        ik=IkConfig(
            posture_gain=float(ik["posture_gain"]),
            damping=float(ik["damping"]),
            max_joint_delta_rad=float(ik["max_joint_delta_rad"]),
            orientation_weight=float(ik["orientation_weight"]),
        ),
        controller=ControllerConfig(backend=backend, rmpflow=rmpflow_config, pink=pink_config),
        runtime=RuntimeConfig(mode=runtime_mode, hardware=dict(hardware_cfg)),
        simulation=SimulationConfig(
            reset_settle_s=float(simulation["reset_settle_s"]),
            real_time=bool(simulation["real_time"]),
            render_every_n_steps=render_every_n_steps,
            spawn_rgb_cameras=spawn_rgb_cameras,
            gravity_compensation=bool(simulation["gravity_compensation"]),
            gravity_comp_scale=float(simulation["gravity_comp_scale"]),
            joint_stiffness=float(simulation["joint_stiffness"]),
            joint_damping=float(simulation["joint_damping"]),
            joint_effort_limit=float(simulation["joint_effort_limit"]),
        ),
        raw=raw,
        source=source,
    )
