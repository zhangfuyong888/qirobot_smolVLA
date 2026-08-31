#!/usr/bin/env python3
"""Check local paths, environments and the active task contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4_pipeline.config import load_project_config, load_training_config
from s4_pipeline.paths import PATH_DEFAULTS, REFERENCE_LEROBOT_DIR, active_task_id


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def _run_python(prefix: Path, code: str) -> tuple[bool, str]:
    python = prefix / "bin/python"
    if not python.is_file():
        return False, f"missing {python}"
    proc = subprocess.run([str(python), "-c", code], text=True, capture_output=True)
    output = (proc.stdout or proc.stderr).strip().splitlines()
    return proc.returncode == 0, output[-1] if output else f"exit={proc.returncode}"


def resolve_latest_checkpoint(output_dir: Path) -> Path:
    """Return the current complete pretrained-model path for strict checks."""
    checkpoints = Path(output_dir) / "checkpoints"
    last = checkpoints / "last" / "pretrained_model"
    if (last / "config.json").is_file():
        return last
    numeric = sorted(
        (
            item
            for item in checkpoints.iterdir()
            if item.is_dir() and item.name.isdigit() and (item / "pretrained_model/config.json").is_file()
        ),
        key=lambda item: int(item.name),
    ) if checkpoints.is_dir() else []
    if numeric:
        return numeric[-1] / "pretrained_model"
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate S4 project dependencies and active-task contract.")
    parser.add_argument("--strict", action="store_true", help="Also require dataset and checkpoint outputs.")
    args = parser.parse_args()
    cfg = load_project_config()
    train = load_training_config()
    print(f"S4 doctor | task={active_task_id()} | config={cfg.source}")
    print("Resolved paths:")
    for key in PATH_DEFAULTS:
        print(f"  {key}={os.environ[key]}")

    checks: list[bool] = []
    checks.append(_check("IsaacLab", (Path(os.environ["ISAACLAB_ROOT"]) / "isaaclab.sh").is_file(), os.environ["ISAACLAB_ROOT"]))
    checks.append(_check("project scene assets", Path(os.environ["S4_SCENE_ASSET_ROOT"]).is_dir(), os.environ["S4_SCENE_ASSET_ROOT"]))
    checks.append(_check("scene USD", cfg.scene.scene_usd.is_file(), str(cfg.scene.scene_usd)))
    scene_root = Path(os.environ["S4_SCENE_ASSET_ROOT"])
    implicit_assets = (
        "Isaac/Props/UIElements/frame_prim.usd",
        "Isaac/Props/UIElements/arrow_x.usd",
        "Isaac/Environments/Simple_Warehouse/Materials/OmniUe4Base.mdl",
        "Isaac/Environments/Simple_Warehouse/Materials/OmniUe4Function.mdl",
    )
    missing_implicit = [relative for relative in implicit_assets if not (scene_root / relative).is_file()]
    checks.append(
        _check(
            "implicit render assets",
            not missing_implicit,
            "complete" if not missing_implicit else ", ".join(missing_implicit),
        )
    )
    manifest_path = scene_root / "manifest.json"
    manifest_missing: list[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_missing = [
            str(item["path"])
            for item in manifest.get("files", [])
            if not (scene_root / str(item["path"])).is_file()
        ]
    checks.append(
        _check(
            "scene asset manifest",
            manifest_path.is_file() and not manifest_missing,
            (
                f"{len(manifest.get('files', []))} files complete"
                if manifest_path.is_file() and not manifest_missing
                else (", ".join(manifest_missing[:5]) or f"missing {manifest_path}")
            ),
        )
    )
    checks.append(
        _check(
            "LeRobot source",
            (REFERENCE_LEROBOT_DIR / "src" / "lerobot").is_dir(),
            str(REFERENCE_LEROBOT_DIR),
        )
    )
    checks.append(_check("base model", Path(train["vlm_model_name"]).is_dir(), str(train["vlm_model_name"])))
    checks.append(_check("26D contract", cfg.features.state_dim == cfg.features.action_dim == 26, f"state={cfg.features.state_dim} action={cfg.features.action_dim}"))
    checks.append(_check("schema", cfg.dataset.schema_version == "s4_bimanual_v1", cfg.dataset.schema_version))
    checks.append(_check("action semantics", cfg.dataset.action_semantics == "absolute_joint_target", cfg.dataset.action_semantics))
    checks.append(_check("camera contract", len(cfg.features.camera_keys) == 3, ", ".join(cfg.features.camera_keys)))
    checks.append(_check("frequency", cfg.dataset.fps == 20 and cfg.dataset.control_fps == 120, f"dataset={cfg.dataset.fps}Hz control={cfg.dataset.control_fps}Hz"))

    conda = Path(os.environ.get("CONDA_EXE", str(Path.home() / "miniconda3/bin/conda")))
    conda_root = conda.parents[1]
    isaac_prefix = Path(os.environ.get("S4_ISAACLAB_PREFIX", conda_root / "envs" / os.environ["S4_ISAACLAB_ENV"]))
    smol_prefix = Path(os.environ.get("S4_SMOLVLA_PREFIX", conda_root / "envs" / os.environ["S4_SMOLVLA_ENV"]))
    ok, detail = _run_python(isaac_prefix, "import sys,torch,isaaclab; print(sys.version.split()[0], torch.__version__, isaaclab.__version__)")
    checks.append(_check("env_isaaclab imports", ok, detail))
    ok, detail = _run_python(smol_prefix, "import sys,torch,lerobot,av,pyarrow; print(sys.version.split()[0], torch.__version__, lerobot.__version__, av.__version__)")
    checks.append(_check("smolvla imports", ok, detail))

    dataset = cfg.dataset.lerobot_root / cfg.dataset.repo_id.split("/")[-1]
    checkpoint = resolve_latest_checkpoint(cfg.training.output_dir)
    if args.strict or dataset.exists():
        checks.append(_check("LeRobotDataset", (dataset / "meta/info.json").is_file(), str(dataset)))
    if args.strict or checkpoint.exists():
        checks.append(_check("checkpoint", (checkpoint / "config.json").is_file(), str(checkpoint)))
    if args.strict:
        dataset_contract = dataset / "meta" / "s4_contract.json"
        training_contract = Path(train["output_dir"]) / "s4_dataset_contract.json"
        contracts_match = (
            dataset_contract.is_file()
            and training_contract.is_file()
            and json.loads(dataset_contract.read_text(encoding="utf-8"))
            == json.loads(training_contract.read_text(encoding="utf-8"))
        )
        checks.append(
            _check(
                "dataset/checkpoint contract",
                contracts_match,
                f"{dataset_contract} == {training_contract}",
            )
        )
    if checkpoint.joinpath("config.json").is_file():
        ckpt = json.loads(checkpoint.joinpath("config.json").read_text(encoding="utf-8"))
        checks.append(_check("checkpoint action", ckpt["output_features"]["action"]["shape"] == [26], str(ckpt["output_features"]["action"]["shape"])))
    if not all(checks):
        raise SystemExit(1)
    print("[OK] doctor completed")


if __name__ == "__main__":
    main()
