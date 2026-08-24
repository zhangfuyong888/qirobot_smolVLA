from pathlib import Path

from s4_pipeline.config import load_project_config, load_training_config
from s4_pipeline.paths import active_task_id


def test_active_task_config_resolves_paths():
    config = load_project_config()
    training = load_training_config()
    assert active_task_id() == "drawer_insert_close"
    assert config.dataset.task_id == "drawer_insert_close"
    assert config.features.state_dim == config.features.action_dim == 26
    assert config.dataset.schema_version == "s4_bimanual_v1"
    assert config.dataset.action_semantics == "absolute_joint_target"
    assert (config.dataset.fps, config.dataset.control_fps) == (20, 120)
    assert len(config.features.camera_keys) == 3
    assert config.dataset.repo_id == "local/s4_drawer_insert_close_v3_10phase_safe_handle_clear"
    assert training["dataset"] == "s4_drawer_insert_close_v3_10phase_safe_handle_clear"
    assert training["language_contract_version"] == "drawer_10phase_v3_safe_handle_clear"
    assert str(config.training.output_dir).endswith("smolvla_drawer_insert_close_v3_10phase_safe_handle_clear")
    assert "${" not in str(config.dataset.staging_root)
    assert "${" not in training["vlm_model_name"]
