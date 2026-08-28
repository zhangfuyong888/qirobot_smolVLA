"""Hardware teleoperation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from teleoperation.config import TeleopConfig, load_teleop_config


@dataclass(frozen=True)
class HardwareRosConfig:
    control_rate_hz: float
    state_source: str
    lowstate_topic: str
    joint_states_topic: str
    lowcmd_topic: str
    hands_cmd_topic: str
    body_dof: int
    arm_kp: float
    arm_kd: float
    reversed_joint_names: tuple[str, ...]
    max_joint_step_rad: float
    require_initial_state: bool
    initial_state_timeout_s: float
    stale_command_hold: bool
    max_state_age_s: float


@dataclass(frozen=True)
class HardwareIkConfig:
    backend: str


@dataclass(frozen=True)
class HardwareHandsConfig:
    left_open_uint16: tuple[int, ...]
    left_close_uint16: tuple[int, ...]
    right_open_uint16: tuple[int, ...]
    right_close_uint16: tuple[int, ...]
    duration_ms: int
    left_hand_id: int
    right_hand_id: int
    left_hand_array_index: int
    right_hand_array_index: int
    hands_mode: int
    hands_mode_ctrl: int
    hand_mode: int


@dataclass(frozen=True)
class HardwareSceneConfig:
    robot_base_z: float
    joint_stiffness: float
    joint_damping: float
    joint_effort_limit: float


@dataclass(frozen=True)
class HardwareStartupConfig:
    move_to_home: bool
    duration_s: float
    max_joint_step_rad: float
    position_tolerance_rad: float
    check_lowcmd_publishers: bool


@dataclass(frozen=True)
class HardwareGravityCompConfig:
    enabled: bool
    urdf_path: str
    scale: float
    sign: float
    tau_limit: float
    ramp_time_s: float
    gravity_vector: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class HardwareTeleopConfig:
    teleop: TeleopConfig
    hardware: HardwareRosConfig
    ik: HardwareIkConfig
    hands: HardwareHandsConfig
    scene: HardwareSceneConfig
    startup: HardwareStartupConfig
    gravity: HardwareGravityCompConfig
    source: Path
    project_root: Path


def _uint16_list(raw: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or len(raw) != 6:
        raise ValueError(f"{field} must be a list of six integers")
    values = tuple(int(item) for item in raw)
    if any(value < 0 or value > 255 for value in values):
        raise ValueError(f"{field} values must be in 0..255")
    return values


def _resolve_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if (parent / "configs" / "teleoperation" / "meta_quest3.yaml").is_file():
            return parent
    raise FileNotFoundError(
        f"could not locate project root from hardware config {config_path}; "
        "expected configs/teleoperation/meta_quest3.yaml in an ancestor directory"
    )


def load_hardware_teleop_config(path: Path) -> HardwareTeleopConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"hardware teleop config must be a mapping: {path}")

    teleop_rel = raw.get("teleop_config")
    if not teleop_rel:
        raise ValueError("hardware teleop config must include teleop_config")

    hardware = raw.get("hardware", {})
    ik = raw.get("ik", {})
    hands = raw.get("hands", {})
    scene = raw.get("scene", {})
    startup = raw.get("startup", {})
    gravity = raw.get("gravity_compensation", {})

    backend = str(ik.get("backend", "rmpflow")).lower()
    if backend not in {"rmpflow", "pinocchio"}:
        raise ValueError("ik.backend must be 'rmpflow' or 'pinocchio'")

    project_root = _resolve_project_root(path)
    teleop_path = (project_root / str(teleop_rel)).resolve()
    teleop = load_teleop_config(teleop_path)

    reversed_names = hardware.get("reversed_joint_names", [])
    if not isinstance(reversed_names, list):
        raise TypeError("hardware.reversed_joint_names must be a list")

    control_rate_hz = float(hardware.get("control_rate_hz", 30.0))
    if control_rate_hz <= 0.0:
        raise ValueError("hardware.control_rate_hz must be positive")

    state_source = str(hardware.get("state_source", "lowstate")).lower()
    if state_source not in {"lowstate", "joint_states"}:
        raise ValueError("hardware.state_source must be 'lowstate' or 'joint_states'")

    gravity_source = str(gravity.get("source", "current")).lower()
    if gravity_source not in {"current", "target"}:
        raise ValueError("gravity_compensation.source must be 'current' or 'target'")

    gravity_vector_raw = gravity.get("gravity_vector", [0.0, 0.0, -9.81])
    if not isinstance(gravity_vector_raw, list) or len(gravity_vector_raw) != 3:
        raise ValueError("gravity_compensation.gravity_vector must be a list of three floats")
    gravity_vector = tuple(float(value) for value in gravity_vector_raw)

    return HardwareTeleopConfig(
        teleop=teleop,
        hardware=HardwareRosConfig(
            control_rate_hz=control_rate_hz,
            state_source=state_source,
            lowstate_topic=str(hardware.get("lowstate_topic", "lowstate")),
            joint_states_topic=str(hardware.get("joint_states_topic", "/joint_states")),
            lowcmd_topic=str(hardware.get("lowcmd_topic", "lowcmd")),
            hands_cmd_topic=str(hardware.get("hands_cmd_topic", "/handscmd")),
            body_dof=int(hardware.get("body_dof", 26)),
            arm_kp=float(hardware.get("arm_kp", 60.0)),
            arm_kd=float(hardware.get("arm_kd", 2.0)),
            reversed_joint_names=tuple(str(name) for name in reversed_names),
            max_joint_step_rad=float(hardware.get("max_joint_step_rad", 0.065)),
            require_initial_state=bool(hardware.get("require_initial_state", True)),
            initial_state_timeout_s=float(hardware.get("initial_state_timeout_s", 15.0)),
            stale_command_hold=bool(hardware.get("stale_command_hold", True)),
            max_state_age_s=float(hardware.get("max_state_age_s", 0.5)),
        ),
        ik=HardwareIkConfig(backend=backend),
        hands=HardwareHandsConfig(
            left_open_uint16=_uint16_list(hands.get("left_open_uint16"), field="hands.left_open_uint16"),
            left_close_uint16=_uint16_list(hands.get("left_close_uint16"), field="hands.left_close_uint16"),
            right_open_uint16=_uint16_list(hands.get("right_open_uint16"), field="hands.right_open_uint16"),
            right_close_uint16=_uint16_list(hands.get("right_close_uint16"), field="hands.right_close_uint16"),
            duration_ms=int(hands.get("duration_ms", 200)),
            left_hand_id=int(hands.get("left_hand_id", 0)),
            right_hand_id=int(hands.get("right_hand_id", 1)),
            left_hand_array_index=int(hands.get("left_hand_array_index", 0)),
            right_hand_array_index=int(hands.get("right_hand_array_index", 1)),
            hands_mode=int(hands.get("hands_mode", 1)),
            hands_mode_ctrl=int(hands.get("hands_mode_ctrl", 5)),
            hand_mode=int(hands.get("hand_mode", 1)),
        ),
        scene=HardwareSceneConfig(
            robot_base_z=float(scene.get("robot_base_z", 0.98)),
            joint_stiffness=float(scene.get("joint_stiffness", 600.0)),
            joint_damping=float(scene.get("joint_damping", 60.0)),
            joint_effort_limit=float(scene.get("joint_effort_limit", 400.0)),
        ),
        startup=HardwareStartupConfig(
            move_to_home=bool(startup.get("move_to_home", True)),
            duration_s=float(startup.get("duration_s", 4.0)),
            max_joint_step_rad=float(startup.get("max_joint_step_rad", 0.03)),
            position_tolerance_rad=float(startup.get("position_tolerance_rad", 0.02)),
            check_lowcmd_publishers=bool(startup.get("check_lowcmd_publishers", True)),
        ),
        gravity=HardwareGravityCompConfig(
            enabled=bool(gravity.get("enabled", True)),
            urdf_path=str(
                gravity.get("urdf_path", "assets/my_robot/urdf/s4_40dof_merged.urdf")
            ),
            scale=float(gravity.get("scale", 0.6)),
            sign=float(gravity.get("sign", 1.0)),
            tau_limit=float(gravity.get("tau_limit", 12.0)),
            ramp_time_s=float(gravity.get("ramp_time", 2.0)),
            gravity_vector=gravity_vector,
            source=gravity_source,
        ),
        source=path,
        project_root=project_root,
    )
