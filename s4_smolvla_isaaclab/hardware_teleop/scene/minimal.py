"""Minimal Isaac scene for headless hardware IK (robot articulation only)."""

from __future__ import annotations

from hardware_teleop.config_loader import HardwareSceneConfig
from s4_robot.simulation import build_robot


def build_minimal_robot_scene(scene_cfg: HardwareSceneConfig) -> dict[str, object]:
    """Spawn only the S4 articulation used by RMPflow / Pinocchio FK."""
    robot = build_robot(
        "/World/Robot",
        scene_cfg.joint_stiffness,
        scene_cfg.joint_damping,
        scene_cfg.joint_effort_limit,
        scene_cfg.robot_base_z,
    )
    print("[HW-TELEOP][BOOT] minimal headless robot scene ready (no task assets)", flush=True)
    return {"robot": robot}
