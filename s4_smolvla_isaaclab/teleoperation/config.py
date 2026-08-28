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
    clutch: ClutchConfig


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
class ControllerConfig:
    backend: str
    rmpflow: RmpFlowConfig


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
    workspace_min = _vec(safety, "workspace_min_base_m", 3)
    workspace_max = _vec(safety, "workspace_max_base_m", 3)
    if np.any(workspace_min >= workspace_max):
        raise ValueError("workspace_min_base_m must be below workspace_max_base_m")
    backend = str(controller.get("backend", "pinocchio")).lower()
    if backend not in {"pinocchio", "rmpflow"}:
        raise ValueError("controller.backend must be 'pinocchio' or 'rmpflow'")
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
            clutch=ClutchConfig(engage, release, button_indices),
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
        controller=ControllerConfig(backend=backend, rmpflow=rmpflow_config),
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
