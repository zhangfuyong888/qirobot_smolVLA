"""Isaac-independent helpers shared by simulation and hardware teleoperation."""

from __future__ import annotations

import socket

import numpy as np

from tasks import get_task_spec
from tasks.loading import load_yaml


DEFAULT_HANDS = {
    "left_open": [0.9, 0.0, 0.05, 0.05, 0.05, 0.05],
    "left_close": [1.0, 0.22, 0.85, 0.85, 0.85, 0.85],
    "right_open": [0.9, 0.0, 0.05, 0.05, 0.05, 0.05],
    "right_close": [1.0, 0.42, 0.85, 0.85, 0.85, 0.85],
}


def detect_lan_ip() -> str:
    """Return the preferred outbound IPv4 address without sending traffic."""
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


def load_task_control_profiles(
    task_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load hand profiles and optional arm home poses for a task."""
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


def smooth_command(
    previous: np.ndarray,
    desired: np.ndarray,
    alpha: float,
    max_joint_step: float,
) -> np.ndarray:
    """Low-pass and step-limit one finite joint command vector."""
    previous_np = np.asarray(previous, dtype=np.float32)
    desired_np = np.asarray(desired, dtype=np.float32)
    if previous_np.shape != desired_np.shape:
        raise ValueError(
            f"smooth_command shape mismatch: previous={previous_np.shape}, desired={desired_np.shape}"
        )
    if not np.isfinite(previous_np).all() or not np.isfinite(desired_np).all():
        raise ValueError("smooth_command requires finite previous and desired values")
    alpha_clipped = float(np.clip(alpha, 0.0, 1.0))
    max_step = max(float(max_joint_step), 1.0e-6)
    delta = np.clip(alpha_clipped * (desired_np - previous_np), -max_step, max_step)
    return previous_np + delta
