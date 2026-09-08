from __future__ import annotations

from pathlib import Path

from real_vla_stack.common.config import load_pipeline_config


def test_active_rollout_config_has_consistent_timing_and_safety() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_pipeline_config(root / "real_vla_stack/config/pipeline.yaml")
    assert cfg.robot["rollout"]["control_hz"] >= cfg.robot["rollout"]["policy_hz"]
    assert cfg.robot["rollout"]["camera_warmup_s"] == 2.0
    assert cfg.robot["rollout"]["execute_horizon"] == 35
    assert cfg.robot["rollout"]["replan_interval_steps"] == 10
    assert cfg.robot["freshness"]["max_response_age_ms"] == 500
    assert cfg.robot["freshness"]["max_motion_chunk_age_ms"] == 1800
    assert cfg.robot["freshness"]["max_chunk_age_ms"] == 2400
    assert cfg.robot["freshness"]["max_consecutive_policy_rejections"] == 2
    assert cfg.robot["safety"]["max_command_tracking_error_rad"] == 0.18
    assert cfg.robot["safety"]["max_rollout_joint_velocity_rad_s"] == [
        0.77, 0.50, 0.77, 0.91, 0.95, 0.70, 0.67
    ]
    assert cfg.robot["safety"]["chunk_blend_duration_ms"] == 0
