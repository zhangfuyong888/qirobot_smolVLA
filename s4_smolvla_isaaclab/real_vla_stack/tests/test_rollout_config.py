from __future__ import annotations

from pathlib import Path

from real_vla_stack.common.config import load_pipeline_config


def test_active_rollout_config_has_consistent_timing_and_safety() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_pipeline_config(root / "real_vla_stack/config/pipeline.yaml")
    assert cfg.robot["rollout"]["control_hz"] >= cfg.robot["rollout"]["policy_hz"]
    assert cfg.robot["freshness"]["max_chunk_age_ms"] >= 500
    assert cfg.robot["safety"]["max_command_tracking_error_rad"] == 0.18
