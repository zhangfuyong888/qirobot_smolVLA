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
    arm_command_topic: str
    arm_command_mode_ctrl: int
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
    commissioning_orientation_cost: float
    commissioning_position_cost: float
    commissioning_elbow_avoidance_enabled: bool
    commissioning_elbow_min_lateral_distance_base_m: float
    commissioning_input_filter_tau_s: float
    commissioning_workspace_min: tuple[float, float, float]
    commissioning_workspace_max: tuple[float, float, float]
    commissioning_max_clutch_translation_m: float
    commissioning_invert_translation: bool
    commissioning_invert_orientation: bool
    commissioning_translation_sign: tuple[float, float, float]
    command_watchdog_timeout_s: float
    shutdown_hold_duration_s: float


@dataclass(frozen=True)
class HardwareIkConfig:
    backend: str
    max_joint_velocity_rad_s: float = 0.9
    joint_limit_avoidance_cost: float = 0.002
    joint_limit_activation_ratio: float = 0.8
    joint_limit_avoidance_gain: float = 0.2
    elbow_max_angle_rad: float = -0.08
    shoulder_posture_cost: float = 0.006
    elbow_posture_cost: float = 0.010
    wrist_pitch_yaw_posture_cost: float = 0.003
    shoulder_max_velocity_rad_s: float = 0.55
    elbow_max_velocity_rad_s: float = 0.65
    wrist_pitch_yaw_max_velocity_rad_s: float = 0.50
    shoulder_max_reference_deviation_rad: float = 0.85
    max_proximal_tracking_error_rad: float = 0.18


@dataclass(frozen=True)
class HardwareHandsConfig:
    enabled: bool
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
    check_arm_command_publishers: bool
    require_sdk_arm_replay: bool
    approved_sdk_sha256: tuple[str, ...] = ()
    home_left_arm: tuple[float, ...] = ()
    home_right_arm: tuple[float, ...] = ()


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


def _float7(raw: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != 7:
        raise ValueError(f"{field} must be a list of seven floats")
    values = tuple(float(item) for item in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} must contain finite values")
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

    backend = str(ik.get("backend", "pink")).lower()
    if backend not in {"pink", "rmpflow", "pinocchio"}:
        raise ValueError("ik.backend must be 'pink', 'rmpflow' or 'pinocchio'")

    ik_max_joint_velocity_rad_s = float(
        ik.get("max_joint_velocity_rad_s", 0.9)
    )
    joint_limit_avoidance_cost = float(
        ik.get("joint_limit_avoidance_cost", 0.002)
    )
    joint_limit_activation_ratio = float(
        ik.get("joint_limit_activation_ratio", 0.8)
    )
    joint_limit_avoidance_gain = float(
        ik.get("joint_limit_avoidance_gain", 0.2)
    )
    elbow_max_angle_rad = float(ik.get("elbow_max_angle_rad", -0.08))
    shoulder_posture_cost = float(ik.get("shoulder_posture_cost", 0.006))
    elbow_posture_cost = float(ik.get("elbow_posture_cost", 0.010))
    wrist_pitch_yaw_posture_cost = float(
        ik.get("wrist_pitch_yaw_posture_cost", 0.003)
    )
    shoulder_max_velocity_rad_s = float(
        ik.get("shoulder_max_velocity_rad_s", 0.55)
    )
    elbow_max_velocity_rad_s = float(
        ik.get("elbow_max_velocity_rad_s", 0.65)
    )
    wrist_pitch_yaw_max_velocity_rad_s = float(
        ik.get("wrist_pitch_yaw_max_velocity_rad_s", 0.50)
    )
    shoulder_max_reference_deviation_rad = float(
        ik.get("shoulder_max_reference_deviation_rad", 0.85)
    )
    max_proximal_tracking_error_rad = float(
        ik.get("max_proximal_tracking_error_rad", 0.18)
    )
    if ik_max_joint_velocity_rad_s <= 0.0:
        raise ValueError("ik.max_joint_velocity_rad_s must be positive")
    if joint_limit_avoidance_cost < 0.0:
        raise ValueError("ik.joint_limit_avoidance_cost must be non-negative")
    if not 0.0 < joint_limit_activation_ratio < 1.0:
        raise ValueError("ik.joint_limit_activation_ratio must be in (0, 1)")
    if not 0.0 < joint_limit_avoidance_gain <= 1.0:
        raise ValueError("ik.joint_limit_avoidance_gain must be in (0, 1]")
    if not -0.5 < elbow_max_angle_rad < 0.0:
        raise ValueError("ik.elbow_max_angle_rad must be in (-0.5, 0)")
    if (
        shoulder_posture_cost < 0.0
        or elbow_posture_cost < 0.0
        or wrist_pitch_yaw_posture_cost < 0.0
    ):
        raise ValueError("ik posture costs must be non-negative")
    for field, value in (
        ("shoulder_max_velocity_rad_s", shoulder_max_velocity_rad_s),
        ("elbow_max_velocity_rad_s", elbow_max_velocity_rad_s),
        (
            "wrist_pitch_yaw_max_velocity_rad_s",
            wrist_pitch_yaw_max_velocity_rad_s,
        ),
    ):
        if not 0.0 < value <= ik_max_joint_velocity_rad_s:
            raise ValueError(
                f"ik.{field} must be positive and no greater than "
                "ik.max_joint_velocity_rad_s"
            )
    if not 0.0 < shoulder_max_reference_deviation_rad < math.pi:
        raise ValueError(
            "ik.shoulder_max_reference_deviation_rad must be in (0, pi)"
        )
    if max_proximal_tracking_error_rad <= 0.0:
        raise ValueError("ik.max_proximal_tracking_error_rad must be positive")

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
    commissioning_orientation_cost = float(
        hardware.get("commissioning_orientation_cost", 0.15)
    )
    commissioning_position_cost = float(
        hardware.get("commissioning_position_cost", 1.0)
    )
    commissioning_input_filter_tau_s = float(
        hardware.get("commissioning_input_filter_tau_s", 0.0)
    )
    commissioning_elbow_min_lateral_distance_base_m = float(
        hardware.get("commissioning_elbow_min_lateral_distance_base_m", 0.28)
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
    commissioning_invert_translation = bool(
        hardware.get("commissioning_invert_translation", False)
    )
    commissioning_invert_orientation = bool(
        hardware.get("commissioning_invert_orientation", False)
    )
    commissioning_translation_sign = _float3(
        hardware.get("commissioning_translation_sign", [1.0, 1.0, 1.0]),
        field="hardware.commissioning_translation_sign",
    )
    if any(abs(value) != 1.0 for value in commissioning_translation_sign):
        raise ValueError("hardware.commissioning_translation_sign entries must be +1 or -1")
    command_watchdog_timeout_s = float(
        hardware.get("command_watchdog_timeout_s", 0.15)
    )
    shutdown_hold_duration_s = float(
        hardware.get("shutdown_hold_duration_s", 0.5)
    )
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
    if commissioning_orientation_cost < 0.0:
        raise ValueError("hardware.commissioning_orientation_cost must be non-negative")
    if commissioning_position_cost < 0.0:
        raise ValueError("hardware.commissioning_position_cost must be non-negative")
    if commissioning_input_filter_tau_s < 0.0:
        raise ValueError("hardware.commissioning_input_filter_tau_s must be non-negative")
    if not 0.0 < commissioning_elbow_min_lateral_distance_base_m < 1.0:
        raise ValueError(
            "hardware.commissioning_elbow_min_lateral_distance_base_m must be in (0, 1)"
        )
    if any(
        lo >= hi
        for lo, hi in zip(commissioning_workspace_min, commissioning_workspace_max)
    ):
        raise ValueError("hardware commissioning workspace min must be below max")
    home_left_arm = (
        _float7(startup.get("home_left_arm"), field="startup.home_left_arm")
        if "home_left_arm" in startup
        else ()
    )
    home_right_arm = (
        _float7(startup.get("home_right_arm"), field="startup.home_right_arm")
        if "home_right_arm" in startup
        else ()
    )
    for field, home in (
        ("startup.home_left_arm", home_left_arm),
        ("startup.home_right_arm", home_right_arm),
    ):
        if home and home[3] > elbow_max_angle_rad:
            raise ValueError(
                f"{field} elbow angle {home[3]:.3f} exceeds "
                f"ik.elbow_max_angle_rad {elbow_max_angle_rad:.3f}"
            )
    for field, value in (
        (
            "commissioning_max_clutch_translation_m",
            commissioning_max_clutch_translation_m,
        ),
        ("command_watchdog_timeout_s", command_watchdog_timeout_s),
        ("shutdown_hold_duration_s", shutdown_hold_duration_s),
    ):
        if value <= 0.0:
            raise ValueError(f"hardware.{field} must be positive")

    approved_sdk_sha256 = tuple(
        str(value).strip().lower() for value in startup.get("approved_sdk_sha256", [])
    )
    if any(
        len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in approved_sdk_sha256
    ):
        raise ValueError("startup.approved_sdk_sha256 entries must be SHA256 hex digests")
    require_sdk_arm_replay = bool(startup.get("require_sdk_arm_replay", True))
    check_arm_command_publishers = bool(
        startup.get("check_arm_command_publishers", True)
    )
    if not check_arm_command_publishers:
        raise ValueError("startup.check_arm_command_publishers must remain true")
    if not require_sdk_arm_replay:
        raise ValueError("startup.require_sdk_arm_replay must remain true")
    if require_sdk_arm_replay and not approved_sdk_sha256:
        raise ValueError(
            "startup.approved_sdk_sha256 must pin at least one reviewed SDK binary"
        )

    state_source = str(hardware.get("state_source", "lowstate")).lower()
    if state_source not in {"lowstate", "joint_states"}:
        raise ValueError("hardware.state_source must be 'lowstate' or 'joint_states'")

    arm_command_topic = str(hardware.get("arm_command_topic", "/lowcmd_replay"))
    arm_command_mode_ctrl = int(hardware.get("arm_command_mode_ctrl", 4))
    if arm_command_topic.rstrip("/") != "/lowcmd_replay":
        raise ValueError(
            "hardware.arm_command_topic must be '/lowcmd_replay' for SDK arm-only replay"
        )
    if arm_command_mode_ctrl != 4:
        raise ValueError(
            "hardware.arm_command_mode_ctrl must be 4 for SDK arm-only replay"
        )

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
            arm_command_topic=arm_command_topic,
            arm_command_mode_ctrl=arm_command_mode_ctrl,
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
            commissioning_orientation_cost=commissioning_orientation_cost,
            commissioning_position_cost=commissioning_position_cost,
            commissioning_elbow_avoidance_enabled=bool(
                hardware.get("commissioning_elbow_avoidance_enabled", False)
            ),
            commissioning_elbow_min_lateral_distance_base_m=(
                commissioning_elbow_min_lateral_distance_base_m
            ),
            commissioning_input_filter_tau_s=commissioning_input_filter_tau_s,
            commissioning_workspace_min=commissioning_workspace_min,
            commissioning_workspace_max=commissioning_workspace_max,
            commissioning_max_clutch_translation_m=commissioning_max_clutch_translation_m,
            commissioning_invert_translation=commissioning_invert_translation,
            commissioning_invert_orientation=commissioning_invert_orientation,
            commissioning_translation_sign=commissioning_translation_sign,
            command_watchdog_timeout_s=command_watchdog_timeout_s,
            shutdown_hold_duration_s=shutdown_hold_duration_s,
        ),
        ik=HardwareIkConfig(
            backend=backend,
            max_joint_velocity_rad_s=ik_max_joint_velocity_rad_s,
            joint_limit_avoidance_cost=joint_limit_avoidance_cost,
            joint_limit_activation_ratio=joint_limit_activation_ratio,
            joint_limit_avoidance_gain=joint_limit_avoidance_gain,
            elbow_max_angle_rad=elbow_max_angle_rad,
            shoulder_posture_cost=shoulder_posture_cost,
            elbow_posture_cost=elbow_posture_cost,
            wrist_pitch_yaw_posture_cost=wrist_pitch_yaw_posture_cost,
            shoulder_max_velocity_rad_s=shoulder_max_velocity_rad_s,
            elbow_max_velocity_rad_s=elbow_max_velocity_rad_s,
            wrist_pitch_yaw_max_velocity_rad_s=(
                wrist_pitch_yaw_max_velocity_rad_s
            ),
            shoulder_max_reference_deviation_rad=(
                shoulder_max_reference_deviation_rad
            ),
            max_proximal_tracking_error_rad=max_proximal_tracking_error_rad,
        ),
        hands=HardwareHandsConfig(
            enabled=bool(hands.get("enabled", False)),
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
            check_arm_command_publishers=check_arm_command_publishers,
            require_sdk_arm_replay=require_sdk_arm_replay,
            approved_sdk_sha256=approved_sdk_sha256,
            home_left_arm=home_left_arm,
            home_right_arm=home_right_arm,
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
