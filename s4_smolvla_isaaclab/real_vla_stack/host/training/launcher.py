from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from ...common.config import PipelineConfig
from ..dataset.lerobot_validator import validate_lerobot_dataset
from .preflight import preflight_pretrained_policy, strict_load_pretrained_policy


def training_command(config: PipelineConfig, *, profile: str | None = None) -> tuple[list[str], Path]:
    training = config.host["training"]
    model = config.host["model"]
    selected = profile or str(training["profile"])
    if selected not in training["profiles"]:
        raise ValueError(f"unknown training profile {selected!r}")
    steps = int(training["profiles"][selected])
    dataset_root = config.host_path_value("lerobot_root") / str(config.host["dataset"]["repo_id"])
    pretrained_value = model.get("pretrained_policy")
    pretrained = (
        Path(os.path.expandvars(os.path.expanduser(str(pretrained_value)))).resolve()
        if pretrained_value
        else None
    )
    mode = "smolvla_base_ft" if pretrained is not None else "scratch_expert_legacy"
    output = config.host_path_value("output_root") / f"{config.contract.task_id}_{mode}_{selected}"
    command = [
        "lerobot-train",
        f"--dataset.repo_id={config.host['dataset']['repo_id']}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        f"--output_dir={output}",
        f"--steps={steps}",
        f"--batch_size={int(training['batch_size'])}",
        f"--save_freq={min(int(training['save_freq']), steps)}",
        f"--num_workers={int(training['num_workers'])}",
        "--persistent_workers=false",
        f"--seed={int(training['seed'])}",
        "--resume=false",
        f"--policy.device={training['device']}",
        f"--policy.freeze_vision_encoder={str(model['freeze_vision_encoder']).lower()}",
        f"--policy.train_expert_only={str(model['train_expert_only']).lower()}",
        f"--policy.train_state_proj={str(model['train_state_proj']).lower()}",
        f"--policy.optimizer_lr={model['optimizer_lr']}",
        f"--policy.optimizer_weight_decay={model['optimizer_weight_decay']}",
        f"--policy.optimizer_grad_clip_norm={model['optimizer_grad_clip_norm']}",
        "--policy.push_to_hub=false",
    ]
    if pretrained is not None:
        command.insert(1, f"--policy.path={pretrained}")
        # smolvla_base ships generic 6D/three-camera feature metadata. Clearing
        # only this dataset contract lets upstream LeRobot infer the drawer 8D +
        # two-camera features without changing any latent architecture dimensions.
        command.insert(2, "--policy.input_features=null")
    else:
        model_root = config.host_path_value("model_root")
        vlm = model_root / str(model["vlm_model_name"])
        scratch_args = [
            "--policy.type=smolvla",
            f"--policy.chunk_size={int(model['chunk_size'])}",
            f"--policy.n_action_steps={int(model['chunk_size'])}",
            f"--policy.n_obs_steps={int(model['n_obs_steps'])}",
            f"--policy.max_state_dim={int(model['max_state_dim'])}",
            f"--policy.max_action_dim={int(model['max_action_dim'])}",
            f"--policy.resize_imgs_with_padding={model['resize_imgs_with_padding']}",
            f"--policy.load_vlm_weights={str(model['load_vlm_weights']).lower()}",
            f"--policy.vlm_model_name={vlm}",
        ]
        command[1:1] = scratch_args
    return command, output


def launch_training(config: PipelineConfig, *, profile: str | None = None, dry_run: bool = False) -> list[str]:
    dataset_root = config.host_path_value("lerobot_root") / str(config.host["dataset"]["repo_id"])
    validate_lerobot_dataset(dataset_root, config.contract)
    pretrained = config.host["model"].get("pretrained_policy")
    if pretrained:
        report = preflight_pretrained_policy(
            Path(os.path.expandvars(os.path.expanduser(str(pretrained)))).resolve(),
            config.contract,
            expected_chunk_size=int(config.host["model"]["chunk_size"]),
            expected_n_obs_steps=int(config.host["model"]["n_obs_steps"]),
            train_expert_only=bool(config.host["model"]["train_expert_only"]),
            train_state_proj=bool(config.host["model"]["train_state_proj"]),
        )
        print("[REAL-VLA-TRAIN] pretrained preflight", json.dumps(report, sort_keys=True), flush=True)
    command, output = training_command(config, profile=profile)
    adjacent_cli = Path(sys.executable).resolve().parent / command[0]
    resolved_cli = adjacent_cli if adjacent_cli.is_file() else shutil.which(command[0])
    if resolved_cli is None:
        raise FileNotFoundError("lerobot-train is not available in the active training environment")
    command[0] = str(resolved_cli)
    if output.exists():
        raise FileExistsError(f"fresh training output already exists: {output}")
    if dry_run:
        return command
    if pretrained:
        # Keep strict checkpoint loading in real_vla_stack rather than adding a
        # project-specific option to the pinned LeRobot submodule.
        strict_report = strict_load_pretrained_policy(
            Path(os.path.expandvars(os.path.expanduser(str(pretrained)))).resolve(),
            model_root=config.host_path_value("model_root"),
        )
        print(
            "[REAL-VLA-TRAIN] strict pretrained weight load "
            + json.dumps(strict_report, sort_keys=True),
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "training_runtime_preflight.py"),
            "--device",
            str(config.host["training"]["device"]),
            "--num-gpus",
            "1",
            "--gpu-ids",
            "",
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{output.name}.s4_dataset_contract.", dir=output.parent
    )
    os.close(descriptor)
    Path(stage_name).unlink()
    config.contract.write(Path(stage_name))
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "supervise_training.py"),
            "--output-dir",
            str(output),
            "--contract-stage",
            stage_name,
            "--",
            *command,
        ],
        check=True,
    )
    Path(stage_name).unlink(missing_ok=True)
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError(f"training finished without a valid output directory: {output}")
    shutil.copy2(config.source_path, output / "effective_pipeline.yaml")
    (output / "effective_config.json").write_text(
        json.dumps({"task": config.task, "host": config.host, "robot": config.robot}, indent=2) + "\n",
        encoding="utf-8",
    )
    return command
