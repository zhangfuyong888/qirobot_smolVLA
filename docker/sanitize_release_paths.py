#!/usr/bin/env python3
"""Rewrite workstation paths stored in copied training artifacts.

LeRobot serializes resolved paths into checkpoint JSON files.  Those paths are
valid on the training workstation but are metadata, not part of the learned
weights.  A release image must point them at the resources bundled in that
image.  This tool only edits the Docker build copy; the host artifacts remain
untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKSTATION_PREFIXES = ("/home/", "/Users/")


def _container_path(value: str, roots: dict[str, Path], json_path: Path, key: str | None) -> str:
    if not value.startswith("/"):
        return value

    # Saved policy configs should identify the checkpoint that contains them,
    # not the workstation's mutable checkpoints/last alias.
    if key == "pretrained_path" and json_path.parent.name == "pretrained_model":
        return str(json_path.parent)
    if key == "checkpoint_path" and json_path.parent.name == "pretrained_model":
        return str(json_path.parent.parent)

    markers = (
        ("/s4_smolvla_isaaclab/outputs", roots["output"]),
        ("/s4_smolvla_isaaclab/datasets", roots["data"]),
        ("/s4_smolvla_isaaclab/models", roots["model"]),
        ("/s4_smolvla_isaaclab", roots["project"]),
        ("/smolVLA/lerobot", roots["lerobot"]),
        ("/IsaacLab", roots["isaaclab"]),
    )
    for marker, target_root in markers:
        marker_at = value.find(marker)
        if marker_at < 0:
            continue
        suffix = value[marker_at + len(marker) :].lstrip("/")
        return str(target_root / suffix) if suffix else str(target_root)
    return value


def _rewrite(value: Any, roots: dict[str, Path], json_path: Path, key: str | None = None) -> tuple[Any, int]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        changes = 0
        for child_key, child_value in value.items():
            result[child_key], count = _rewrite(child_value, roots, json_path, child_key)
            changes += count
        return result, changes
    if isinstance(value, list):
        result = []
        changes = 0
        for child_value in value:
            rewritten, count = _rewrite(child_value, roots, json_path, key)
            result.append(rewritten)
            changes += count
        return result, changes
    if isinstance(value, str):
        rewritten = _container_path(value, roots, json_path, key)
        return rewritten, int(rewritten != value)
    return value, 0


def _workstation_values(value: Any, location: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}" if location else key
            found.extend(_workstation_values(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_workstation_values(child, f"{location}[{index}]"))
    elif isinstance(value, str) and value.startswith(WORKSTATION_PREFIXES):
        found.append((location, value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check", action="store_true", help="verify only; do not rewrite")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    roots = {
        "project": project_root,
        "lerobot": args.lerobot_root.resolve(),
        "isaaclab": args.isaaclab_root.resolve(),
        "data": (args.data_root or project_root / "datasets").resolve(),
        "model": (args.model_root or project_root / "models").resolve(),
        "output": (args.output_root or project_root / "outputs").resolve(),
    }
    artifact_root = roots["output"]
    # Only released policy metadata and completed evaluation summaries contain
    # resolved resource paths. Avoid touching live optimizer/scheduler state if
    # two containers intentionally share the same output volume.
    json_paths = []
    if artifact_root.is_dir():
        json_paths = sorted(
            set(artifact_root.glob("train/**/pretrained_model/*.json"))
            | set(artifact_root.glob("eval/**/summary.json"))
        )
    rewritten_files = 0
    rewritten_values = 0
    unresolved: list[tuple[Path, str, str]] = []

    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid JSON artifact {json_path}: {exc}") from exc

        rewritten_payload, changes = _rewrite(payload, roots, json_path)
        if not args.check:
            payload = rewritten_payload
            if changes:
                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                rewritten_files += 1
                rewritten_values += changes
        elif changes:
            # A path can be free of /home/<user> but still refer to another
            # container layout. Treat it as stale so custom mounted roots do
            # not silently use metadata from an older image.
            unresolved.append((json_path, "container_root", "path does not match the active container roots"))

        for location, value in _workstation_values(payload):
            unresolved.append((json_path, location, value))

    if unresolved:
        for json_path, location, value in unresolved:
            print(f"[FAIL] non-portable path remains: {json_path}:{location}={value}")
        raise SystemExit(1)

    mode = "verified" if args.check else "sanitized"
    print(
        f"[OK] {mode} release artifact paths: json_files={len(json_paths)} "
        f"rewritten_files={rewritten_files} rewritten_values={rewritten_values}"
    )


if __name__ == "__main__":
    main()
