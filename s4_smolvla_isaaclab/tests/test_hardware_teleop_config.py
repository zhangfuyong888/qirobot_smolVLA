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
    assert config.hardware.lowcmd_topic == "lowcmd"
    assert config.hardware.max_state_age_s == pytest.approx(0.2)
    assert config.hardware.max_state_joint_jump_rad == pytest.approx(0.35)
    assert config.hardware.input_stale_timeout_s == pytest.approx(0.12)
    assert config.hardware.max_tcp_translation_speed_m_s == pytest.approx(0.08)
    assert config.hardware.max_tcp_rotation_speed_rad_s == pytest.approx(0.3)
    assert config.hardware.max_joint_step_rad == pytest.approx(0.003)
    assert config.hardware.commissioning_position_scale == pytest.approx(0.5)
    assert config.hardware.commissioning_orientation_enabled is False
    assert config.hardware.commissioning_max_clutch_translation_m == pytest.approx(0.05)
    assert config.hardware.command_watchdog_timeout_s == pytest.approx(0.10)
    assert config.hardware.arm_kd == pytest.approx(3.0)
    assert config.ik.backend == "pink"
    assert len(config.hands.left_open_uint16) == 6
    assert config.teleop.mapping.position_scale == pytest.approx(2.0)
    assert config.startup.move_to_home is False
    assert config.startup.check_lowcmd_publishers is True
    assert config.startup.require_policy_lowcmd is True
    assert config.startup.policy_min_valid_frames == 90
    assert config.startup.max_policy_age_s == pytest.approx(0.2)
    assert config.startup.policy_stable_duration_s == pytest.approx(3.0)
    assert len(config.startup.approved_sdk_sha256) == 1
    assert config.startup.require_sdk_mode5_merge is True
    assert config.gravity.enabled is False
    assert config.gravity.scale == pytest.approx(0.6)


def test_hardware_config_rejects_bad_backend(tmp_path: Path) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(encoding="utf-8")
    bad = source.replace("backend: pink", "backend: invalid")
    path = tmp_path / "quest_hardware_bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="ik.backend"):
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
