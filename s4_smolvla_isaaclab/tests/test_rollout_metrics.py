from pathlib import Path

from s4_pipeline import rollout_metrics
from s4_pipeline.rollout_metrics import (
    aggregate_rollout_summary,
    build_rollout_run_name,
    checkpoint_step_tag,
    default_summary_json_path,
    episode_artifact_paths,
    evaluate_drawer_success,
    make_can_grid_sampler,
    make_randomization_rng,
    resolve_randomization_cfg,
    resolve_rollout_run_dir,
    sample_randomization,
)
from s4_pipeline.drawer_distractors import GRASP_CAN_NOMINAL_POSITION


SCRIPTED = {
    "randomization": {
        "can_xy": {"enabled": True, "x_range": [-0.05, 0.05], "y_range": [-0.05, 0.05]},
        "drawer_initial_open": {"enabled": True, "range": [0.0, 0.05]},
        "distractor_cans": {
            "enabled": True,
            "ranges": [
                [[0.70, 1.00], [0.12, 0.30]],
                [[0.70, 1.00], [0.48, 0.66]],
                [[0.72, 1.00], [-0.68, -0.32]],
            ],
            "min_center_distance_m": 0.16,
        },
    },
    "success": {
        "drawer_open_abs_max": 0.04,
        "can_world_z": {"min_m": 1.00, "max_m": 1.04},
    },
}


def test_rollout_drawer_approach_and_pull_get_80_extension_frames_only():
    scripted = {
        "phases": [
            {"name": "left_approach_handle", "task": "approach"},
            {"name": "left_grasp_handle", "task": "grasp"},
            {"name": "pull_drawer", "task": "pull"},
            {"name": "right_pregrasp_can", "task": "pregrasp"},
        ]
    }

    assert rollout_metrics.rollout_phase_extension_frames({"task": "approach"}, scripted, 20) == 80
    assert rollout_metrics.rollout_phase_extension_frames({"task": "pull"}, scripted, 20) == 80
    assert rollout_metrics.rollout_phase_extension_frames({"task": "grasp"}, scripted, 20) == 20
    assert rollout_metrics.rollout_phase_extension_frames({"task": "pregrasp"}, scripted, 20) == 20


def test_macro_rollout_drawer_approach_and_pull_get_80_extension_frames_only():
    assert rollout_metrics.rollout_phase_extension_frames(
        {"language_phase_id": "approach_drawer_handle", "task": "macro approach"}, {}, 20
    ) == 80
    assert rollout_metrics.rollout_phase_extension_frames(
        {"language_phase_id": "pull_drawer", "task": "macro pull"}, {}, 20
    ) == 80
    assert rollout_metrics.rollout_phase_extension_frames(
        {"language_phase_id": "approach_can", "task": "macro can"}, {}, 20
    ) == 20


def test_resolve_randomization_respects_cli_and_disable():
    enabled = resolve_randomization_cfg(
        SCRIPTED,
        randomize_task=True,
        can_x_range=(-0.02, 0.02),
        drawer_open_range=(0.01, 0.03),
    )
    assert enabled["enabled"] is True
    assert enabled["can_xy"]["enabled"] is True
    assert enabled["can_xy"]["x_range"] == [-0.02, 0.02]
    assert enabled["can_xy"]["y_range"] == [-0.05, 0.05]
    assert enabled["drawer_initial_open"]["range"] == [0.01, 0.03]

    disabled = resolve_randomization_cfg(SCRIPTED, randomize_task=False)
    assert disabled["enabled"] is False
    assert disabled["can_xy"]["x_range"] == [0.0, 0.0]
    assert disabled["drawer_initial_open"]["range"] == [0.0, 0.0]


def test_yaml_can_xy_off_keeps_drawer_random_under_success_rate():
    scripted = {
        "randomization": {
            "can_xy": {"enabled": False, "x_range": [-0.05, 0.05], "y_range": [-0.05, 0.05]},
            "drawer_initial_open": {"enabled": True, "range": [0.0, 0.05]},
            "distractor_cans": {"enabled": False},
        }
    }
    cfg = resolve_randomization_cfg(scripted, randomize_task=True)
    assert cfg["can_xy"]["enabled"] is False
    assert cfg["drawer_initial_open"]["enabled"] is True
    assert make_can_grid_sampler(cfg, make_randomization_rng(42)) is None
    sample = sample_randomization(cfg, seed=42, rng=make_randomization_rng(42))
    assert sample["can_x_offset_m"] == 0.0
    assert sample["can_y_offset_m"] == 0.0
    assert 0.0 <= sample["drawer_open_m"] <= 0.05
    assert sample["distractor_can_xy"] == {}


def test_sample_randomization_keeps_fixed_seed_42():
    cfg = resolve_randomization_cfg(SCRIPTED, randomize_task=True)
    cfg["distractor_cans_enabled"] = True
    rng_a = make_randomization_rng(42)
    rng_b = make_randomization_rng(42)
    samples_a = [sample_randomization(cfg, seed=42, rng=rng_a) for _ in range(3)]
    samples_b = [sample_randomization(cfg, seed=42, rng=rng_b) for _ in range(3)]
    assert samples_a == samples_b
    assert all(sample["seed"] == 42 for sample in samples_a)
    assert samples_a[0] != samples_a[1]
    assert -0.05 <= samples_a[0]["can_x_offset_m"] <= 0.05
    assert 0.0 <= samples_a[0]["drawer_open_m"] <= 0.05
    assert set(samples_a[0]["distractor_can_xy"]) == {
        "distractor_master_chef_can",
        "distractor_mustard_bottle",
        "distractor_bleach_cleanser",
    }

    fixed = sample_randomization(resolve_randomization_cfg(SCRIPTED, randomize_task=False), seed=42)
    assert fixed["seed"] == 42
    assert fixed["can_x_offset_m"] == 0.0
    assert fixed["drawer_open_m"] == 0.0
    assert list(fixed["distractor_can_xy"]) == []


def test_fixed_rollout_keeps_distractor_assignment_stable():
    cfg = resolve_randomization_cfg(SCRIPTED, randomize_task=False)
    cfg["distractor_cans_enabled"] = True
    rng = make_randomization_rng(42)
    samples = [sample_randomization(cfg, seed=42, rng=rng) for _ in range(3)]
    assert samples[0]["distractor_can_xy"] == samples[1]["distractor_can_xy"] == samples[2]["distractor_can_xy"]


def test_rollout_stratified_grid_visits_every_cell_once():
    scripted = {
        "randomization": {
            "can_xy": {
                "enabled": True,
                "sampling": "stratified_grid",
                "grid_cells": [5, 5],
                "x_range": [-0.05, 0.05],
                "y_range": [-0.05, 0.05],
            },
            "drawer_initial_open": {"enabled": False, "range": [0.0, 0.0]},
        }
    }
    cfg = resolve_randomization_cfg(scripted, randomize_task=True)
    rng = make_randomization_rng(42)
    grid = make_can_grid_sampler(cfg, rng)
    samples = [sample_randomization(cfg, rng=rng, can_grid_sampler=grid) for _ in range(25)]
    assert len({tuple(sample["can_grid_cell"]) for sample in samples}) == 25
    assert {sample["can_grid_cycle"] for sample in samples} == {0}


def test_grasp_can_grid_is_shifted_to_right_hand_side():
    scripted = {
        **SCRIPTED,
        "randomization": {
            **SCRIPTED["randomization"],
            "can_xy": {
                "enabled": True,
                "sampling": "stratified_grid",
                "grid_cells": [5, 5],
                "x_range": [-0.025, -0.015],
                "y_range": [-0.06, 0.01],
            },
        },
    }
    cfg = resolve_randomization_cfg(scripted, randomize_task=True)
    rng = make_randomization_rng(42)
    grid = make_can_grid_sampler(cfg, rng)
    samples = [sample_randomization(cfg, rng=rng, can_grid_sampler=grid) for _ in range(25)]
    world_xy = [
        (
            GRASP_CAN_NOMINAL_POSITION[0] + sample["can_x_offset_m"],
            GRASP_CAN_NOMINAL_POSITION[1] + sample["can_y_offset_m"],
        )
        for sample in samples
    ]
    assert all(0.515 <= x <= 0.525 for x, _ in world_xy)
    assert all(-0.19 <= y <= -0.12 for _, y in world_xy)


def test_rollout_run_dir_and_episode_artifacts(tmp_path: Path):
    assert checkpoint_step_tag("outputs/train/x/checkpoints/360000/pretrained_model") == "ckpt360000"
    name = build_rollout_run_name(
        randomize_task=True,
        episodes=20,
        checkpoint="outputs/train/x/checkpoints/360000/pretrained_model",
        timestamp="20260808_113645",
    )
    assert name == "rollout_20260808_113645_rand20_ckpt360000"

    auto_dir = resolve_rollout_run_dir(
        eval_root=tmp_path / "eval",
        checkpoint=".../360000/pretrained_model",
        episodes=20,
        randomize_task=True,
        timestamp="20260808_113645",
    )
    assert auto_dir.name == "rollout_20260808_113645_rand20_ckpt360000"
    assert auto_dir.is_dir()

    from_video = resolve_rollout_run_dir(
        eval_root=tmp_path / "eval",
        checkpoint=".../360000/pretrained_model",
        episodes=1,
        randomize_task=False,
        output_video=tmp_path / "eval" / "my_det.avi",
    )
    assert from_video == (tmp_path / "eval" / "my_det").resolve()

    explicit = resolve_rollout_run_dir(
        eval_root=tmp_path / "eval",
        checkpoint=".../360000/pretrained_model",
        episodes=20,
        randomize_task=True,
        output_dir=tmp_path / "eval" / "custom_rand20",
    )
    assert explicit.name == "custom_rand20"

    single = episode_artifact_paths(auto_dir, episode_index=0, episodes=1)
    assert single["video"].name == "rollout.avi"
    assert single["diagnostics_csv"].name == "rollout_actions.csv"
    multi = episode_artifact_paths(auto_dir, episode_index=2, episodes=20)
    assert multi["video"].name == "ep003.avi"
    assert multi["diagnostics_csv"].name == "ep003_actions.csv"
    assert multi["video"].parent == auto_dir
    assert default_summary_json_path(auto_dir).name == "summary.json"


def test_aggregate_rollout_summary_rates():
    rows = [
        {"success": True, "complete": True},
        {"success": False, "complete": True},
        {"success": True, "complete": False},
        {"success": False, "complete": False},
    ]
    summary = aggregate_rollout_summary(
        rows,
        checkpoint="ckpt",
        randomize_task=True,
        base_seed=42,
        randomization=resolve_randomization_cfg(SCRIPTED, randomize_task=True),
        output_dir="/tmp/run",
    )
    assert summary["seed"] == 42
    assert summary["output_dir"] == "/tmp/run"
    assert summary["randomization"]["variables"] == [
        "can_xy_offset_m",
        "drawer_open_m",
        "distractor_can_xy",
    ]
    assert summary["episodes"] == 4
    assert summary["success_count"] == 2
    assert summary["complete_count"] == 2
    assert summary["complete_and_success_count"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["complete_and_success_rate"] == 0.25
