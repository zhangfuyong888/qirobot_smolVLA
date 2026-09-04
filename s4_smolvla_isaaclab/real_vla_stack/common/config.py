from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contract import PolicyContract
from .errors import ContractError


STACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG = STACK_ROOT / "config" / "pipeline.yaml"


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a mapping")
    return payload


def _path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class PipelineConfig:
    source_path: Path
    task_path: Path
    host_path: Path
    robot_path: Path
    task: dict[str, Any]
    host: dict[str, Any]
    robot: dict[str, Any]
    contract: PolicyContract

    def host_path_value(self, key: str) -> Path:
        return _path(str(self.host["paths"][key]), self.host_path.parent)


def load_pipeline_config(path: Path = DEFAULT_PIPELINE_CONFIG) -> PipelineConfig:
    source = Path(path).expanduser().resolve()
    refs = _yaml(source)
    task_path = _path(refs["task"], source.parent)
    host_path = _path(refs["host"], source.parent)
    robot_path = _path(refs["robot"], source.parent)
    task = _yaml(task_path)
    host = _yaml(host_path)
    robot = _yaml(robot_path)
    state_names = tuple(str(v) for v in task["state"]["names"])
    action_names = tuple(str(v) for v in task["action"]["names"])
    camera_items = list(task["cameras"].values())
    contract = PolicyContract(
        contract_version=str(task["contract_version"]),
        raw_schema_version=str(task["dataset"]["raw_schema"]),
        task_id=str(task["task"]["id"]),
        task=str(task["task"]["instruction"]).strip(),
        robot_type=str(task["robot"]["type"]),
        active_arm=str(task["robot"]["active_arm"]),
        dataset_fps=int(task["dataset"]["fps"]),
        state_names=state_names,
        action_names=action_names,
        camera_keys=tuple(str(v["feature"]) for v in camera_items),
        camera_sources=tuple(str(v["source"]) for v in camera_items),
        phase_filter=tuple(str(v) for v in task["dataset"]["include_phases"]),
        max_camera_age_ms=float(task["alignment"]["max_camera_age_ms"]),
        max_cross_camera_skew_ms=float(task["alignment"]["max_cross_camera_skew_ms"]),
        lerobot_commit=str(task["dataset"]["lerobot_commit"]),
        state_semantics=str(task["state"]["semantics"]),
        action_semantics=str(task["action"]["semantics"]),
        units=str(task["action"]["units"]),
        gripper_semantics=str(task["gripper"]["semantics"]),
        gripper_open_threshold=float(task["gripper"]["open_threshold"]),
        gripper_grasp_threshold=float(task["gripper"]["grasp_threshold"]),
        alignment="causal_latest_before",
    )
    contract.validate()
    if robot["rollout"]["policy_hz"] != contract.dataset_fps:
        raise ContractError("robot rollout policy_hz must equal dataset_fps")
    return PipelineConfig(source, task_path, host_path, robot_path, task, host, robot, contract)
