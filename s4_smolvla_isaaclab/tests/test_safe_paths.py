from pathlib import Path

import pytest

from data.lerobot_conversion import safe_dataset_root, validate_overwrite_dataset_target
from scripts.safe_remove_train_output import validate_train_output


def test_dataset_target_is_a_named_strict_child(tmp_path: Path):
    root = tmp_path / "lerobot_data"
    assert safe_dataset_root(root, "local/drawer_v1") == root / "drawer_v1"


@pytest.mark.parametrize("repo_id", ("..", ".", "local/..", "local/bad name", "local/"))
def test_dataset_target_rejects_unsafe_leaf(tmp_path: Path, repo_id: str):
    with pytest.raises(ValueError, match="unsafe dataset"):
        safe_dataset_root(tmp_path / "lerobot_data", repo_id)


def test_dataset_target_rejects_symlinked_parent(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        safe_dataset_root(linked, "drawer_v1")


def test_dataset_target_rejects_project_and_data_roots(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("S4_DATA_ROOT", str(data_root))
    with pytest.raises(ValueError, match="broad dataset output root"):
        safe_dataset_root(data_root, "drawer_v1")


def test_dataset_overwrite_rejects_arbitrary_directory(tmp_path: Path):
    arbitrary = tmp_path / "important_dir"
    arbitrary.mkdir()
    (arbitrary / "do_not_delete.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not recognizably"):
        validate_overwrite_dataset_target(arbitrary)
    assert (arbitrary / "do_not_delete.txt").is_file()


def test_dataset_overwrite_accepts_lerobot_markers(tmp_path: Path):
    dataset = tmp_path / "drawer_v1"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (dataset / "data").mkdir()
    (dataset / "videos").mkdir()
    validate_overwrite_dataset_target(dataset)


def test_train_output_must_be_below_allowed_train_root(tmp_path: Path):
    project = tmp_path / "project"
    data = project / "datasets"
    allowed = project / "outputs" / "train"
    target = allowed / "drawer_v1"
    assert validate_train_output(target, allowed, project, data) == target


@pytest.mark.parametrize("which", ("allowed_root", "project", "data", "outside"))
def test_train_output_rejects_broad_or_outside_target(tmp_path: Path, which: str):
    project = tmp_path / "project"
    data = project / "datasets"
    allowed = project / "outputs" / "train"
    targets = {
        "allowed_root": allowed,
        "project": project,
        "data": data,
        "outside": tmp_path / "other" / "run",
    }
    with pytest.raises(ValueError):
        validate_train_output(targets[which], allowed, project, data)


def test_train_output_rejects_symlink_component(tmp_path: Path):
    project = tmp_path / "project"
    real = project / "real_outputs"
    real.mkdir(parents=True)
    linked = project / "outputs"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        validate_train_output(linked / "train" / "run", linked / "train", project, project / "datasets")
