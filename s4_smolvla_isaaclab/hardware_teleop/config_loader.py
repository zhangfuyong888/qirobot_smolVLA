"""Hardware teleoperation configuration."""

from __future__ import annotations

import math
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
    max_state_joint_jump_rad: float
    input_stale_timeout_s: float
    max_tcp_translation_speed_m_s: float
    max_tcp_rotation_speed_rad_s: float
    commissioning_position_scale: float
    commissioning_orientation_enabled: bool
    commissioning_workspace_min: tuple[float, float, float]
    commissioning_workspace_max: tuple[float, float, float]
    commissioning_max_clutch_translation_m: float
    command_watchdog_timeout_s: float
    release_duration_s: float
    release_max_joint_step_rad: float
    release_tolerance_rad: float


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
    require_policy_lowcmd: bool
    policy_initial_timeout_s: float
    policy_min_valid_frames: int
    max_policy_age_s: float
    require_sdk_mode5_merge: bool
    policy_stable_duration_s: float = 3.0
    max_abs_roll_pitch_rad: float = 0.35
    max_abs_imu_gyro_rad_s: float = 0.5
    max_leg_velocity_rad_s: float = 1.0
    max_leg_tracking_error_rad: float = 0.35
    approved_sdk_sha256: tuple[str, ...] = ()


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


def _float3(raw: Any, *, field: str) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field} must be a list of three floats")
    values = tuple(float(item) for item in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} must contain finite values")
    return values  # type: ignore[return-value]


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

    backend = str(ik.get("backend", "pink")).lower()
    if backend not in {"pink", "rmpflow", "pinocchio"}:
        raise ValueError("ik.backend must be 'pink', 'rmpflow' or 'pinocchio'")

    project_root = _resolve_project_root(path)
    teleop_path = (project_root / str(teleop_rel)).resolve()
    teleop = load_teleop_config(teleop_path)

    reversed_names = hardware.get("reversed_joint_names", [])
    if not isinstance(reversed_names, list):
        raise TypeError("hardware.reversed_joint_names must be a list")

    control_rate_hz = float(hardware.get("control_rate_hz", 30.0))
    if control_rate_hz <= 0.0:
        raise ValueError("hardware.control_rate_hz must be positive")

    max_state_age_s = float(hardware.get("max_state_age_s", 0.5))
    max_state_joint_jump_rad = float(hardware.get("max_state_joint_jump_rad", 0.35))
    input_stale_timeout_s = float(hardware.get("input_stale_timeout_s", 0.25))
    max_tcp_translation_speed_m_s = float(
        hardware.get("max_tcp_translation_speed_m_s", 0.5)
    )
    max_tcp_rotation_speed_rad_s = float(
        hardware.get("max_tcp_rotation_speed_rad_s", 1.5)
    )
    commissioning_position_scale = float(
        hardware.get("commissioning_position_scale", 0.5)
    )
    commissioning_workspace_min = _float3(
        hardware.get("commissioning_workspace_min_base_m", [0.10, -0.55, 0.00]),
        field="hardware.commissioning_workspace_min_base_m",
    )
    commissioning_workspace_max = _float3(
        hardware.get("commissioning_workspace_max_base_m", [0.70, 0.55, 0.60]),
        field="hardware.commissioning_workspace_max_base_m",
    )
    commissioning_max_clutch_translation_m = float(
        hardware.get("commissioning_max_clutch_translation_m", 0.05)
    )
    command_watchdog_timeout_s = float(
        hardware.get("command_watchdog_timeout_s", 0.15)
    )
    release_duration_s = float(hardware.get("release_duration_s", 5.0))
    release_max_joint_step_rad = float(
        hardware.get("release_max_joint_step_rad", 0.006)
    )
    release_tolerance_rad = float(hardware.get("release_tolerance_rad", 0.03))
    if max_state_age_s <= 0.0:
        raise ValueError("hardware.max_state_age_s must be positive")
    if max_state_joint_jump_rad <= 0.0:
        raise ValueError("hardware.max_state_joint_jump_rad must be positive")
    if input_stale_timeout_s <= 0.0:
        raise ValueError("hardware.input_stale_timeout_s must be positive")
    if max_tcp_translation_speed_m_s <= 0.0:
        raise ValueError("hardware.max_tcp_translation_speed_m_s must be positive")
    if max_tcp_rotation_speed_rad_s <= 0.0:
        raise ValueError("hardware.max_tcp_rotation_speed_rad_s must be positive")
    if commissioning_position_scale <= 0.0:
        raise ValueError("hardware.commissioning_position_scale must be positive")
    if any(
        lo >= hi
        for lo, hi in zip(commissioning_workspace_min, commissioning_workspace_max)
    ):
        raise ValueError("hardware commissioning workspace min must be below max")
    for field, value in (
        (
            "commissioning_max_clutch_translation_m",
            commissioning_max_clutch_translation_m,
        ),
        ("command_watchdog_timeout_s", command_watchdog_timeout_s),
        ("release_duration_s", release_duration_s),
        ("release_max_joint_step_rad", release_max_joint_step_rad),
        ("release_tolerance_rad", release_tolerance_rad),
    ):
        if value <= 0.0:
            raise ValueError(f"hardware.{field} must be positive")

    policy_initial_timeout_s = float(startup.get("policy_initial_timeout_s", 5.0))
    policy_min_valid_frames = int(startup.get("policy_min_valid_frames", 3))
    max_policy_age_s = float(startup.get("max_policy_age_s", 0.5))
    policy_stable_duration_s = float(startup.get("policy_stable_duration_s", 3.0))
    max_abs_roll_pitch_rad = float(startup.get("max_abs_roll_pitch_rad", 0.35))
    max_abs_imu_gyro_rad_s = float(startup.get("max_abs_imu_gyro_rad_s", 0.5))
    max_leg_velocity_rad_s = float(startup.get("max_leg_velocity_rad_s", 1.0))
    max_leg_tracking_error_rad = float(startup.get("max_leg_tracking_error_rad", 0.35))
    if policy_initial_timeout_s <= 0.0:
        raise ValueError("startup.policy_initial_timeout_s must be positive")
    if policy_min_valid_frames < 1:
        raise ValueError("startup.policy_min_valid_frames must be at least 1")
    if max_policy_age_s <= 0.0:
        raise ValueError("startup.max_policy_age_s must be positive")
    for field, value in (
        ("policy_stable_duration_s", policy_stable_duration_s),
        ("max_abs_roll_pitch_rad", max_abs_roll_pitch_rad),
        ("max_abs_imu_gyro_rad_s", max_abs_imu_gyro_rad_s),
        ("max_leg_velocity_rad_s", max_leg_velocity_rad_s),
        ("max_leg_tracking_error_rad", max_leg_tracking_error_rad),
    ):
        if value <= 0.0:
            raise ValueError(f"startup.{field} must be positive")
    approved_sdk_sha256 = tuple(
        str(value).strip().lower() for value in startup.get("approved_sdk_sha256", [])
    )
    if any(
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in approved_sdk_sha256
    ):
        raise ValueError("startup.approved_sdk_sha256 entries must be SHA256 hex digests")
    require_sdk_mode5_merge = bool(startup.get("require_sdk_mode5_merge", True))
    if require_sdk_mode5_merge and not approved_sdk_sha256:
        raise ValueError(
            "startup.approved_sdk_sha256 must pin at least one reviewed SDK binary"
        )

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
            max_joint_step_rad=float(hardware.get("max_joint_step_rad", 0.020)),
            require_initial_state=bool(hardware.get("require_initial_state", True)),
            initial_state_timeout_s=float(hardware.get("initial_state_timeout_s", 15.0)),
            stale_command_hold=bool(hardware.get("stale_command_hold", True)),
            max_state_age_s=max_state_age_s,
            max_state_joint_jump_rad=max_state_joint_jump_rad,
            input_stale_timeout_s=input_stale_timeout_s,
            max_tcp_translation_speed_m_s=max_tcp_translation_speed_m_s,
            max_tcp_rotation_speed_rad_s=max_tcp_rotation_speed_rad_s,
            commissioning_position_scale=commissioning_position_scale,
            commissioning_orientation_enabled=bool(
                hardware.get("commissioning_orientation_enabled", False)
            ),
            commissioning_workspace_min=commissioning_workspace_min,
            commissioning_workspace_max=commissioning_workspace_max,
            commissioning_max_clutch_translation_m=commissioning_max_clutch_translation_m,
            command_watchdog_timeout_s=command_watchdog_timeout_s,
            release_duration_s=release_duration_s,
            release_max_joint_step_rad=release_max_joint_step_rad,
            release_tolerance_rad=release_tolerance_rad,
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
            max_joint_step_rad=float(startup.get("max_joint_step_rad", 0.01)),
            position_tolerance_rad=float(startup.get("position_tolerance_rad", 0.02)),
            check_lowcmd_publishers=bool(startup.get("check_lowcmd_publishers", True)),
            require_policy_lowcmd=bool(startup.get("require_policy_lowcmd", True)),
            policy_initial_timeout_s=policy_initial_timeout_s,
            policy_min_valid_frames=policy_min_valid_frames,
            max_policy_age_s=max_policy_age_s,
            require_sdk_mode5_merge=require_sdk_mode5_merge,
            policy_stable_duration_s=policy_stable_duration_s,
            max_abs_roll_pitch_rad=max_abs_roll_pitch_rad,
            max_abs_imu_gyro_rad_s=max_abs_imu_gyro_rad_s,
            max_leg_velocity_rad_s=max_leg_velocity_rad_s,
            max_leg_tracking_error_rad=max_leg_tracking_error_rad,
            approved_sdk_sha256=approved_sdk_sha256,
        ),
        gravity=HardwareGravityCompConfig(
            enabled=bool(gravity.get("enabled", False)),
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
