import copy

import pytest

from s4_pipeline.config import load_project_config
from scripts.dataset_check import _active_language_contract, _validate_portable_contract, _validate_task_sequences


def _valid_contract():
    cfg = load_project_config()
    language = _active_language_contract(cfg)
    return cfg, language, {
        "schema_version": cfg.dataset.schema_version,
        "action_semantics": cfg.dataset.action_semantics,
        "state_dim": cfg.features.state_dim,
        "action_dim": cfg.features.action_dim,
        "fps": cfg.dataset.fps,
        "camera_paths": list(cfg.raw["dataset"]["camera_paths"]),
        "language_contract_version": language.version,
        "language_phases": language.as_portable_records(),
        "distractor_cans_enabled": False,
        "distractor_assets": [],
        "grasp_can_nominal_position": [0.54, -0.13, 1.16],
        "grasp_can_scale": [1.0, 0.9, 1.0],
    }


def test_portable_dataset_contract_matches_active_task():
    cfg, language, contract = _valid_contract()
    _validate_portable_contract(contract, cfg, language)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("schema_version", "legacy", "dataset schema"),
        ("action_semantics", "delta", "action semantics"),
        ("state_dim", 13, "state_dim"),
        ("action_dim", 13, "action_dim"),
        ("fps", 30, "contract fps"),
        ("camera_paths", ["obs/chest_front_rgb"], "camera_paths"),
        ("language_contract_version", "legacy", "language contract"),
        ("language_phases", [], "language phase definitions"),
        ("grasp_can_scale", [1.0, 1.0, 1.0], "scene contract grasp_can_scale"),
    ),
)
def test_portable_dataset_contract_rejects_incompatible_fields(field, bad_value, message):
    cfg, language, contract = _valid_contract()
    invalid = copy.deepcopy(contract)
    invalid[field] = bad_value

    with pytest.raises(ValueError, match=message):
        _validate_portable_contract(invalid, cfg, language)


def test_task_sequence_accepts_unordered_categorical_indices():
    expected = ["prepare", "approach", "grasp"]
    task_pairs = [(7, "grasp"), (2, "prepare"), (9, "approach")]
    _validate_task_sequences(
        task_pairs,
        episode_indices=[0, 0, 0, 1, 1, 1],
        task_indices=[2, 9, 7, 2, 9, 7],
        expected_tasks=expected,
    )


def test_task_sequence_rejects_temporally_wrong_episode():
    with pytest.raises(ValueError, match="episode=1 language phase sequence mismatch"):
        _validate_task_sequences(
            [(0, "prepare"), (1, "approach"), (2, "grasp")],
            episode_indices=[0, 0, 0, 1, 1, 1],
            task_indices=[0, 1, 2, 0, 2, 1],
            expected_tasks=["prepare", "approach", "grasp"],
        )
