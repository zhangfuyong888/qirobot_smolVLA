#!/usr/bin/env python3
"""Safely remove one configured training run below the allowed train root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _reject_symlink_components(path: Path, label: str) -> None:
    lexical = _absolute(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains symlink component: {current}")


def validate_train_output(output: Path, allowed_root: Path, project_root: Path, data_root: Path) -> Path:
    output = _absolute(output)
    allowed_root = _absolute(allowed_root)
    project_root = _absolute(project_root).resolve()
    data_root = _absolute(data_root).resolve()
    _reject_symlink_components(allowed_root, "allowed train root")
    _reject_symlink_components(output, "training output")
    allowed_resolved = allowed_root.resolve(strict=False)
    output_resolved = output.resolve(strict=False)
    dangerous = {Path("/").resolve(), Path.home().resolve(), project_root, data_root, allowed_resolved}
    if output_resolved in dangerous:
        raise ValueError(f"refusing broad destructive target: {output_resolved}")
    try:
        relative = output_resolved.relative_to(allowed_resolved)
    except ValueError as exc:
        raise ValueError(f"output {output_resolved} is outside {allowed_resolved}") from exc
    if not relative.parts:
        raise ValueError("training output must be a strict child of the allowed train root")
    return output_resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        target = validate_train_output(args.output, args.allowed_root, args.project_root, args.data_root)
    except ValueError as exc:
        parser.exit(2, f"unsafe training overwrite: {exc}\n")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
