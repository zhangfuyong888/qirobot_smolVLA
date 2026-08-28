from pathlib import Path

import numpy as np
import yaml

from teleoperation.config import load_teleop_config


def test_meta_quest_config_has_proper_coordinate_basis() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_teleop_config(root / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    assert np.allclose(basis.T @ basis, np.eye(3))
    assert np.linalg.det(basis) == 1.0
    assert config.mapping.clutch.release_threshold < config.mapping.clutch.engage_threshold
    assert config.mapping.clutch.button_indices == (1,)
    assert config.network.stale_timeout_s == 1.0
    assert config.controller.backend == "rmpflow"
    assert config.controller.rmpflow.urdf_file.is_file()
    assert config.controller.rmpflow.left.frame_name == "left_wrist_yaw_link"
    assert config.controller.rmpflow.right.frame_name == "right_wrist_yaw_link"
    assert config.controller.rmpflow.update_every_n_steps == 2
    assert config.simulation.render_every_n_steps == 6
    assert config.simulation.spawn_rgb_cameras is False
    assert config.runtime.mode == "simulation"
    assert config.mapping.position_scale == 2.0
    assert config.safety.max_translation_speed_m_s == 1.6
    assert config.safety.max_rotation_speed_rad_s == 5.5
    assert config.smoothing.arm_max_joint_step_rad == 0.065


def test_rmpflow_descriptors_keep_independent_seven_joint_cspaces() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_teleop_config(root / "configs/teleoperation/meta_quest3.yaml")
    left = yaml.safe_load(config.controller.rmpflow.left.descriptor_file.read_text(encoding="utf-8"))
    right = yaml.safe_load(config.controller.rmpflow.right.descriptor_file.read_text(encoding="utf-8"))
    assert left["cspace"] == [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    assert right["cspace"] == [name.replace("left_", "right_", 1) for name in left["cspace"]]
    assert len(left["collision_spheres"]) == 5
    assert len(right["collision_spheres"]) == 5


def test_rmpflow_policy_has_only_same_side_body_collision_controllers() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_teleop_config(root / "configs/teleoperation/meta_quest3.yaml")
    for side, arm in (("left", config.controller.rmpflow.left), ("right", config.controller.rmpflow.right)):
        policy = yaml.safe_load(arm.policy_config_file.read_text(encoding="utf-8"))
        names = [entry["name"] for entry in policy["body_collision_controllers"]]
        assert names
        assert all(name.startswith(side + "_") for name in names)
