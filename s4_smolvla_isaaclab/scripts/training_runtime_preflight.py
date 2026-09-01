#!/usr/bin/env python3
"""Read-only dependency and GPU preflight for SmolVLA training."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path


EXPECTED_ACCELERATE = "1.14.0"
EXPECTED_TORCH_PREFIX = "2.7.0"
EXPECTED_CUDA = "12.8"


def _require_cli(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required training CLI is missing: {name}")
    executable = Path(resolved).resolve()
    environment_bin = Path(sys.executable).resolve().parent
    if executable.parent != environment_bin:
        raise RuntimeError(
            f"{name} resolves outside the active Python environment: "
            f"cli={executable}, python={sys.executable}"
        )
    return executable


def _require_output_parent_writable(output_dir: Path) -> Path:
    output = output_dir.expanduser().absolute()
    parent = output.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise RuntimeError(f"no existing parent directory for training output: {output}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"training output parent is not writable: {parent}")
    return parent


def run_preflight(
    *, device: str, num_gpus: int, gpu_ids: str, output_dir: Path
) -> None:
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")

    import accelerate
    import av
    import datasets
    import torch
    import transformers
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    accelerate_version = importlib.metadata.version("accelerate")
    if accelerate_version != EXPECTED_ACCELERATE:
        raise RuntimeError(
            f"accelerate version mismatch: expected {EXPECTED_ACCELERATE}, "
            f"got {accelerate_version}"
        )
    if not str(torch.__version__).startswith(EXPECTED_TORCH_PREFIX):
        raise RuntimeError(
            f"torch version mismatch: expected {EXPECTED_TORCH_PREFIX}.*, got {torch.__version__}"
        )

    accelerate_cli = _require_cli("accelerate")
    train_cli = _require_cli("lerobot-train")
    writable_parent = _require_output_parent_writable(output_dir)
    cuda_requested = device.lower().startswith("cuda")
    visible_gpus = int(torch.cuda.device_count()) if cuda_requested else 0
    if cuda_requested:
        if not torch.cuda.is_available():
            raise RuntimeError("training config requests CUDA but torch.cuda.is_available() is false")
        if torch.version.cuda != EXPECTED_CUDA:
            raise RuntimeError(
                f"PyTorch CUDA mismatch: expected {EXPECTED_CUDA}, got {torch.version.cuda}"
            )
        if visible_gpus < num_gpus:
            raise RuntimeError(
                f"requested {num_gpus} GPU process(es), but PyTorch sees {visible_gpus}"
            )
        selected = [int(value) for value in gpu_ids.split(",") if value] if gpu_ids else []
        invalid = [index for index in selected if index < 0 or index >= visible_gpus]
        if invalid:
            raise RuntimeError(
                f"GPU IDs are outside container-visible range [0, {visible_gpus - 1}]: {invalid}"
            )

    versions = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "accelerate": accelerate.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "av": av.__version__,
    }
    print(
        f"[TRAIN][PREFLIGHT] PASS versions={versions} visible_gpus={visible_gpus} "
        f"accelerate_cli={accelerate_cli} train_cli={train_cli} "
        f"output_parent={writable_parent}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_preflight(
            device=args.device,
            num_gpus=args.num_gpus,
            gpu_ids=args.gpu_ids,
            output_dir=args.output_dir,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"training runtime preflight failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
