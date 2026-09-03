from __future__ import annotations

from pathlib import Path

import pytest

from hardware_teleop import config_loader
from hardware_teleop.config_loader import load_hardware_teleop_config


ROOT = Path(__file__).resolve().parents[1]


def test_hardware_config_loads() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    assert config.hardware.control_rate_hz == 30.0
    assert config.hardware.state_source == "lowstate"
    assert config.hardware.lowstate_topic == "lowstate"
    assert config.hardware.arm_command_topic == "/lowcmd_replay"
    assert config.hardware.arm_command_mode_ctrl == 4
    assert config.hardware.max_state_age_s == pytest.approx(0.2)
    assert config.hardware.max_state_joint_jump_rad == pytest.approx(0.35)
    assert config.hardware.input_stale_timeout_s == pytest.approx(0.12)
    assert config.hardware.max_tcp_translation_speed_m_s == pytest.approx(0.40)
    assert config.hardware.max_tcp_rotation_speed_rad_s == pytest.approx(0.80)
    assert config.hardware.max_joint_step_rad == pytest.approx(0.032)
    assert config.hardware.commissioning_position_scale == pytest.approx(2.0)
    assert config.hardware.commissioning_orientation_enabled is True
    assert config.hardware.commissioning_orientation_cost == pytest.approx(0.15)
    assert config.hardware.commissioning_position_cost == pytest.approx(0.90)
    assert config.hardware.commissioning_elbow_avoidance_enabled is True
    assert config.hardware.commissioning_elbow_min_lateral_distance_base_m == pytest.approx(0.25)
    assert config.hardware.commissioning_input_filter_tau_s == pytest.approx(0.06)
    assert config.hardware.commissioning_max_clutch_translation_m == pytest.approx(0.35)
    assert config.hardware.commissioning_invert_translation is False
    assert config.hardware.commissioning_invert_orientation is False
    assert config.hardware.commissioning_translation_sign == pytest.approx((-1.0, -1.0, 1.0))
    assert config.hardware.commissioning_workspace_min == pytest.approx((-0.25, -0.65, -0.30))
    assert config.hardware.commissioning_workspace_max == pytest.approx((0.90, 0.65, 0.75))
    assert config.hardware.command_watchdog_timeout_s == pytest.approx(0.10)
    assert config.hardware.shutdown_hold_duration_s == pytest.approx(0.5)
    assert config.hardware.arm_kd == pytest.approx(3.0)
    assert config.ik.backend == "pink"
    assert config.ik.max_joint_velocity_rad_s == pytest.approx(0.90)
    assert config.ik.joint_limit_avoidance_cost == pytest.approx(0.002)
    assert config.ik.joint_limit_activation_ratio == pytest.approx(0.80)
    assert config.ik.joint_limit_avoidance_gain == pytest.approx(0.20)
    assert config.ik.elbow_max_angle_rad == pytest.approx(-0.08)
    assert config.ik.shoulder_posture_cost == pytest.approx(0.006)
    assert config.ik.elbow_posture_cost == pytest.approx(0.010)
    assert config.ik.shoulder_max_velocity_rad_s == pytest.approx(0.55)
    assert config.ik.elbow_max_velocity_rad_s == pytest.approx(0.65)
    assert config.ik.shoulder_max_reference_deviation_rad == pytest.approx(0.85)
    assert config.ik.max_proximal_tracking_error_rad == pytest.approx(0.18)
    assert len(config.hands.left_open_uint16) == 6
    assert config.hands.enabled is True
    assert config.hands.left_close_uint16 == (150, 80, 120, 120, 120, 120)
    assert config.hands.right_close_uint16 == (150, 80, 120, 120, 120, 120)
    assert config.hands.duration_ms == 255
    assert config.teleop.mapping.position_scale == pytest.approx(2.0)
    assert config.startup.move_to_home is True
    assert config.startup.duration_s == pytest.approx(6.0)
    assert config.startup.home_left_arm == pytest.approx((-0.25, 0.45, -0.34, -0.52, -0.65, -0.33, 0.19))
    assert config.startup.home_right_arm == pytest.approx((-0.25, -0.45, -0.34, -0.52, 0.65, -0.33, 0.19))
    assert config.startup.check_arm_command_publishers is True
    assert len(config.startup.approved_sdk_sha256) == 1
    assert config.startup.require_sdk_arm_replay is True
    assert config.gravity.enabled is False
    assert config.gravity.scale == pytest.approx(0.4)
    assert config.gravity.sign == pytest.approx(1.0)
    assert config.hardware.reversed_joint_names == (
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_yaw_joint",
    )


def test_hardware_config_rejects_bad_backend(tmp_path: Path) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(encoding="utf-8")
    bad = source.replace("backend: pink", "backend: invalid")
    path = tmp_path / "quest_hardware_bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="ik.backend"):
        load_hardware_teleop_config(path)


def test_hardware_config_rejects_reverse_elbow_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(
        encoding="utf-8"
    )
    bad = source.replace(
        "home_left_arm: [-0.25, 0.45, -0.34, -0.52, -0.65, -0.33, 0.19]",
        "home_left_arm: [-0.25, 0.45, -0.34, 0.00, -0.65, -0.33, 0.19]",
    )
    path = tmp_path / "quest_hardware_reverse_elbow_home.yaml"
    path.write_text(bad, encoding="utf-8")
    monkeypatch.setattr(config_loader, "_resolve_project_root", lambda _path: ROOT)
    with pytest.raises(ValueError, match="home_left_arm elbow angle"):
        load_hardware_teleop_config(path)


def test_hardware_config_requires_reviewed_sdk_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(
        encoding="utf-8"
    )
    digest = "39e11f69338e6e3505af223f20c7ab47b48a87d5142bb6e7f251702dc106c09e"
    path = tmp_path / "quest_hardware_missing_sdk_hash.yaml"
    path.write_text(
        source.replace(f"  approved_sdk_sha256:\n    - {digest}\n", ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_loader, "_resolve_project_root", lambda _path: ROOT)
    with pytest.raises(ValueError, match="approved_sdk_sha256"):
        load_hardware_teleop_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("arm_command_topic: /lowcmd_replay", "arm_command_topic: /lowcmd", "arm_command_topic"),
        ("arm_command_mode_ctrl: 4", "arm_command_mode_ctrl: 5", "arm_command_mode_ctrl"),
    ],
)
def test_hardware_config_locks_arm_only_sdk_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    message: str,
) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "quest_hardware_bad_arm_contract.yaml"
    path.write_text(source.replace(old, new), encoding="utf-8")
    monkeypatch.setattr(config_loader, "_resolve_project_root", lambda _path: ROOT)
    with pytest.raises(ValueError, match=message):
        load_hardware_teleop_config(path)


@pytest.mark.parametrize(
    "setting",
    ("check_arm_command_publishers", "require_sdk_arm_replay"),
)
def test_hardware_config_rejects_disabled_arm_safety_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "quest_hardware_disabled_gate.yaml"
    path.write_text(
        source.replace(f"  {setting}: true", f"  {setting}: false"),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_loader, "_resolve_project_root", lambda _path: ROOT)
    with pytest.raises(ValueError, match=setting):
        load_hardware_teleop_config(path)
