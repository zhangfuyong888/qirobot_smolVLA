from __future__ import annotations

from pathlib import Path

import pytest

from hardware_teleop.config_loader import load_hardware_teleop_config


ROOT = Path(__file__).resolve().parents[1]


def test_hardware_config_loads() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    assert config.hardware.control_rate_hz == 30.0
    assert config.hardware.state_source == "lowstate"
    assert config.hardware.lowstate_topic == "lowstate"
    assert config.hardware.lowcmd_topic == "lowcmd"
    assert config.hardware.max_state_age_s == pytest.approx(0.5)
    assert config.ik.backend == "rmpflow"
    assert len(config.hands.left_open_uint16) == 6
    assert config.teleop.mapping.position_scale == pytest.approx(2.2)
    assert config.startup.move_to_home is True
    assert config.startup.check_lowcmd_publishers is True
    assert config.gravity.enabled is True
    assert config.gravity.scale == pytest.approx(0.6)


def test_hardware_config_rejects_bad_backend(tmp_path: Path) -> None:
    source = (ROOT / "hardware_teleop/config/quest_hardware.yaml").read_text(encoding="utf-8")
    bad = source.replace("backend: rmpflow", "backend: invalid")
    path = tmp_path / "quest_hardware_bad.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="ik.backend"):
        load_hardware_teleop_config(path)
