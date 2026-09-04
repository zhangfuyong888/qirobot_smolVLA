from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ...common.config import PipelineConfig
from ...common.contract import PolicyContract
from ..inference.policy_runner import resolve_checkpoint


def resolve_deployment_checkpoint(config: PipelineConfig, value: str | None = None) -> Path:
    configured = value or str(config.host["deployment"]["checkpoint"])
    if configured != "latest":
        return resolve_checkpoint(Path(configured))
    candidates = sorted(config.host_path_value("output_root").glob("*/checkpoints/last"))
    if not candidates:
        raise FileNotFoundError("no checkpoints/last found below output_root")
    return resolve_checkpoint(max(candidates, key=lambda path: path.stat().st_mtime_ns))


def check_checkpoint(
    config: PipelineConfig,
    checkpoint: Path,
    *,
    run_inference: bool = False,
) -> dict[str, Any]:
    model = resolve_checkpoint(checkpoint)
    payload = json.loads((model / "config.json").read_text(encoding="utf-8"))
    inputs = payload["input_features"]
    outputs = payload["output_features"]
    if set(inputs) != {"observation.state", *config.contract.camera_keys}:
        raise ValueError(f"checkpoint input keys mismatch: {sorted(inputs)}")
    if inputs["observation.state"]["shape"] != [8] or outputs["action"]["shape"] != [8]:
        raise ValueError("checkpoint state/action dimensions are not 8D")
    required = (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint missing inference processors: {missing}")
    provenance = next(
        (parent / "s4_dataset_contract.json" for parent in [model, *model.parents] if (parent / "s4_dataset_contract.json").is_file()),
        None,
    )
    if provenance is None or PolicyContract.read(provenance).sha256 != config.contract.sha256:
        raise ValueError("checkpoint dataset contract provenance is missing or mismatched")
    repo_root = Path(__file__).resolve().parents[4]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "checkpoint": str(model),
        "contract_sha256": config.contract.sha256,
        "git_commit": git_commit,
        "lerobot_commit": config.contract.lerobot_commit,
        "state_dim": 8,
        "action_dim": 8,
        "image_keys": list(config.contract.camera_keys),
        "task": config.contract.task,
        "policy_fps": config.contract.dataset_fps,
        "chunk_size": int(config.host["model"]["chunk_size"]),
    }
    if run_inference:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from ..inference.policy_runner import PolicyRunner

        dataset_root = config.host_path_value("lerobot_root") / str(config.host["dataset"]["repo_id"])
        dataset = LeRobotDataset(repo_id=dataset_root.name, root=str(dataset_root), video_backend="pyav")
        sample = dataset[0]
        images = {}
        for key in config.contract.camera_keys:
            image = np.asarray(sample[key])
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
                image = image * 255.0
            images[key] = np.clip(image, 0, 255).astype(np.uint8)
        runner = PolicyRunner(model, config.contract, device=str(config.host["server"]["device"]))
        chunk = runner.predict_chunk(
            np.asarray(sample["observation.state"], dtype=np.float32), images, config.contract.task
        )
        expected = (int(config.host["model"]["chunk_size"]), 8)
        if chunk.shape != expected or not np.isfinite(chunk).all():
            raise ValueError(f"offline inference output shape={chunk.shape}, expected={expected}")
        manifest["offline_inference"] = {"passed": True, "action_chunk_shape": list(chunk.shape)}
    (model / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
