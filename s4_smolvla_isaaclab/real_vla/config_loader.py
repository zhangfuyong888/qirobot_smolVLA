"""Load real_vla YAML configs without touching simulation task configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from real_vla import SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION_CONFIG = Path(__file__).resolve().parent / "config" / "collection.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping")
    return payload


def _float7(values: Any, field: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError(f"{field} must contain 7 numbers")
    result = tuple(float(item) for item in values)
    if not all(map(lambda value: value == value, result)):
        raise ValueError(f"{field} contains NaN")
    return result


@dataclass(frozen=True)
class TaskConfig:
    text: str


@dataclass(frozen=True)
class RobotPolicyConfig:
    active_arm: str
    control_hz: float
    arm_dim: int
    gripper_dim: int
    action_semantics: str
    home_left_arm: tuple[float, ...]
    home_right_arm: tuple[float, ...]


@dataclass(frozen=True)
class CameraStreamConfig:
    name: str
    enabled: bool
    serial: str
    model: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class CamerasConfig:
    backend: str
    warmup_s: float
    writer_queue_frames: int
    head: CameraStreamConfig
    wrist_left: CameraStreamConfig
    wrist_right: CameraStreamConfig
    active_wrist: str

    def active_wrist_name(self, active_arm: str) -> str:
        if self.active_wrist == "auto":
            return f"wrist_{active_arm}"
        if self.active_wrist in {"wrist_left", "wrist_right", "left", "right"}:
            name = self.active_wrist
            if not name.startswith("wrist_"):
                name = f"wrist_{name}"
            return name
        raise ValueError(f"unsupported cameras.active_wrist: {self.active_wrist!r}")

    def enabled_streams(self, active_arm: str) -> tuple[CameraStreamConfig, ...]:
        wrist_name = self.active_wrist_name(active_arm)
        wrist = self.wrist_left if wrist_name == "wrist_left" else self.wrist_right
        streams = []
        if self.head.enabled:
            streams.append(self.head)
        if wrist.enabled:
            streams.append(wrist)
        return tuple(streams)


@dataclass(frozen=True)
class GripperConfig:
    mode: str
    open_threshold: float
    grasp_threshold: float


@dataclass(frozen=True)
class HomeConfig:
    gripper: str
    tolerance_rad: float
    stable_time_s: float
    duration_s: float
    max_joint_step_rad: float


@dataclass(frozen=True)
class ButtonsConfig:
    discard_hold_s: float
    a_index: int
    b_index: int
    x_index: int
    y_index: int
    press_threshold: float


@dataclass(frozen=True)
class StorageConfig:
    root: Path
    video_codec: str
    video_container: str
    writer_queue_frames: int
    min_free_disk_gb: float
    critical_free_disk_gb: float


@dataclass(frozen=True)
class QualityConfig:
    camera_gap_warning_ms: float
    camera_gap_invalid_ms: float
    robot_state_invalid_ms: float
    min_duration_s: float
    min_camera_frames: int


@dataclass(frozen=True)
class CollectionConfig:
    schema_version: str
    task: TaskConfig
    robot: RobotPolicyConfig
    cameras: CamerasConfig
    gripper: GripperConfig
    home: HomeConfig
    buttons: ButtonsConfig
    storage: StorageConfig
    quality: QualityConfig
    hardware_teleop_config: Path
    source_path: Path

    @property
    def active_arm(self) -> str:
        return self.robot.active_arm

    @property
    def active_wrist_name(self) -> str:
        return self.cameras.active_wrist_name(self.active_arm)


def _camera_stream(raw: dict[str, Any], name: str) -> CameraStreamConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"cameras.{name} must be a mapping")
    serial = str(raw.get("serial", "")).strip()
    if not serial:
        raise ValueError(f"cameras.{name}.serial is required")
    return CameraStreamConfig(
        name=name,
        enabled=bool(raw.get("enabled", True)),
        serial=serial,
        model=str(raw.get("model", "")),
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 480)),
        fps=int(raw.get("fps", 30)),
    )


def load_collection_config(path: Path | None = None) -> CollectionConfig:
    collection_path = (path or DEFAULT_COLLECTION_CONFIG).resolve()
    raw = _read_yaml(collection_path)
    schema_version = str(raw.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    config_dir = collection_path.parent
    robot_rel = str(raw.get("robot", {}).get("config", "robot.yaml")) if isinstance(raw.get("robot"), dict) else "robot.yaml"
    cameras_rel = str(raw.get("cameras", {}).get("config", "cameras.yaml")) if isinstance(raw.get("cameras"), dict) else "cameras.yaml"
    if isinstance(raw.get("robot"), dict) and "active_arm" in raw["robot"]:
        robot_raw = raw["robot"]
    else:
        robot_raw = _read_yaml(config_dir / robot_rel)
        robot_raw = {**robot_raw, **{k: v for k, v in dict(raw.get("robot") or {}).items() if k != "config"}}
    if "head" in (raw.get("cameras") or {}):
        cameras_raw = raw["cameras"]
    else:
        cameras_raw = _read_yaml(config_dir / cameras_rel)

    active_arm = str(robot_raw.get("active_arm", "right"))
    if active_arm not in {"left", "right"}:
        raise ValueError("robot.active_arm must be left or right")

    cameras = CamerasConfig(
        backend=str(cameras_raw.get("backend", "realsense")),
        warmup_s=float(cameras_raw.get("warmup_s", 1.5)),
        writer_queue_frames=int(cameras_raw.get("writer_queue_frames", 12)),
        head=_camera_stream(cameras_raw.get("head", {}), "head"),
        wrist_left=_camera_stream(cameras_raw.get("wrist_left", {}), "wrist_left"),
        wrist_right=_camera_stream(cameras_raw.get("wrist_right", {}), "wrist_right"),
        active_wrist=str(cameras_raw.get("active_wrist", "auto")),
    )
    storage_raw = raw.get("storage") or {}
    quality_raw = raw.get("quality") or {}
    gripper_raw = raw.get("gripper") or {}
    home_raw = raw.get("home") or {}
    buttons_raw = raw.get("buttons") or {}
    hardware_rel = Path(str(raw.get("hardware_teleop_config", "hardware_teleop/config/quest_hardware.yaml")))
    hardware_path = hardware_rel if hardware_rel.is_absolute() else (PROJECT_ROOT / hardware_rel)

    open_threshold = float(gripper_raw.get("open_threshold", 0.35))
    grasp_threshold = float(gripper_raw.get("grasp_threshold", 0.65))
    if not 0.0 <= open_threshold < grasp_threshold <= 1.0:
        raise ValueError("gripper thresholds must satisfy 0 <= open < grasp <= 1")

    return CollectionConfig(
        schema_version=schema_version,
        task=TaskConfig(text=str((raw.get("task") or {}).get("text", "")).strip()),
        robot=RobotPolicyConfig(
            active_arm=active_arm,
            control_hz=float(robot_raw.get("control_hz", 30)),
            arm_dim=int((robot_raw.get("state") or {}).get("arm_dim", 7)),
            gripper_dim=int((robot_raw.get("state") or {}).get("gripper_dim", 1)),
            action_semantics=str((robot_raw.get("action") or {}).get("semantics", "absolute_joint_target")),
            home_left_arm=_float7(robot_raw.get("home_left_arm"), "robot.home_left_arm"),
            home_right_arm=_float7(robot_raw.get("home_right_arm"), "robot.home_right_arm"),
        ),
        cameras=cameras,
        gripper=GripperConfig(
            mode=str(gripper_raw.get("mode", "binary")),
            open_threshold=open_threshold,
            grasp_threshold=grasp_threshold,
        ),
        home=HomeConfig(
            gripper=str(home_raw.get("gripper", "open")),
            tolerance_rad=float(home_raw.get("tolerance_rad", 0.03)),
            stable_time_s=float(home_raw.get("stable_time_s", 0.3)),
            duration_s=float(home_raw.get("duration_s", 6.0)),
            max_joint_step_rad=float(home_raw.get("max_joint_step_rad", 0.025)),
        ),
        buttons=ButtonsConfig(
            discard_hold_s=float(buttons_raw.get("discard_hold_s", 0.6)),
            a_index=int(buttons_raw.get("a_index", 4)),
            b_index=int(buttons_raw.get("b_index", 5)),
            x_index=int(buttons_raw.get("x_index", 4)),
            y_index=int(buttons_raw.get("y_index", 5)),
            press_threshold=float(buttons_raw.get("press_threshold", 0.5)),
        ),
        storage=StorageConfig(
            root=Path(str(storage_raw.get("root", "~/real_vla_data"))).expanduser(),
            video_codec=str(storage_raw.get("video_codec", "h264")),
            video_container=str(storage_raw.get("video_container", "mkv")),
            writer_queue_frames=int(storage_raw.get("writer_queue_frames", 12)),
            min_free_disk_gb=float(storage_raw.get("min_free_disk_gb", 10)),
            critical_free_disk_gb=float(storage_raw.get("critical_free_disk_gb", 2)),
        ),
        quality=QualityConfig(
            camera_gap_warning_ms=float(quality_raw.get("camera_gap_warning_ms", 80)),
            camera_gap_invalid_ms=float(quality_raw.get("camera_gap_invalid_ms", 100)),
            robot_state_invalid_ms=float(quality_raw.get("robot_state_invalid_ms", 100)),
            min_duration_s=float(quality_raw.get("min_duration_s", 0.5)),
            min_camera_frames=int(quality_raw.get("min_camera_frames", 5)),
        ),
        hardware_teleop_config=hardware_path.resolve(),
        source_path=collection_path,
    )
