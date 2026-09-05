#!/usr/bin/env python
"""Audit saved real_vla episodes and optionally quarantine unusable ones."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_vla.config_loader import load_collection_config  # noqa: E402
from real_vla.data.validation import validate_saved_episode  # noqa: E402


def _review_notes(trajectory: dict[str, np.ndarray]) -> list[str]:
    """Flag task-like behavior that needs video review rather than auto-rejection."""
    notes: list[str] = []
    gripper = np.asarray(
        trajectory.get("action_gripper_target", np.zeros((0,)))
    ).reshape(-1)
    closed_indices = np.flatnonzero(gripper > 0.5)
    if closed_indices.size == 0:
        notes.append("gripper never closed")
    elif not np.any(gripper[int(closed_indices[0]) :] < 0.5):
        notes.append("gripper did not reopen after closing")

    action_q = np.asarray(trajectory.get("action_arm_target_q", np.zeros((0, 7))))
    if action_q.ndim == 2 and action_q.shape[0] > 0 and action_q.shape[1] == 7:
        max_joint_range = float(np.max(np.ptp(action_q, axis=0)))
        if max_joint_range < 0.10:
            notes.append(f"very little arm motion (max joint range {max_joint_range:.3f}rad)")
    return notes


def _quarantine_target(root: Path, episode: Path, quarantine_root: Path) -> Path:
    relative = episode.resolve().relative_to(root.resolve())
    return quarantine_root / relative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fully validate all saved real_vla episodes. No data is moved by default."
    )
    parser.add_argument("root", type=Path, nargs="?", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="move INVALID/ERROR episodes into a timestamped quarantine tree",
    )
    parser.add_argument("--yes", action="store_true", help="confirm quarantine operation")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args(argv)

    config = load_collection_config(args.config)
    root = (args.root or config.storage.root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")
    if args.quarantine_invalid and not args.yes:
        parser.error("--quarantine-invalid requires --yes")

    episodes = sorted(
        path for path in root.glob("session_*/episodes/episode_*") if path.is_dir()
    )
    pending = sorted(path for path in root.glob("session_*/pending/*") if path.is_dir())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    quarantine_root = root / "quarantine" / stamp
    records: list[dict] = []
    counts: Counter[str] = Counter()

    print(f"Auditing {len(episodes)} saved episodes under {root}", flush=True)
    for index, episode in enumerate(episodes, 1):
        notes: list[str] = []
        try:
            result, meta, trajectory = validate_saved_episode(episode, config)
            notes.extend(result.notes)
            review = _review_notes(trajectory)
            if not result.valid:
                status = "INVALID"
            elif review:
                status = "REVIEW"
                notes.extend(review)
            elif result.warning:
                status = "WARN"
            else:
                status = "PASS"
            episode_id = meta.get("episode_id")
        except Exception as exc:
            status = "ERROR"
            episode_id = None
            notes.append(f"{type(exc).__name__}: {exc}")

        action = "kept"
        original = str(episode)
        if status in {"INVALID", "ERROR"} and args.quarantine_invalid:
            target = _quarantine_target(root, episode, quarantine_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(episode, target)
            action = f"quarantined:{target}"
        counts[status] += 1
        records.append(
            {
                "path": original,
                "episode_id": episode_id,
                "status": status,
                "notes": notes,
                "action": action,
            }
        )
        note_text = "; ".join(notes) if notes else "ok"
        print(
            f"[{index:04d}/{len(episodes):04d}] {status:7s} "
            f"{episode}: {note_text}",
            flush=True,
        )

    status_names = ("PASS", "WARN", "REVIEW", "INVALID", "ERROR")
    report = {
        "schema_version": "s4_real_vla_audit_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "counts": {name: counts[name] for name in status_names},
        "pending": [str(path) for path in pending],
        "quarantine_root": str(quarantine_root) if args.quarantine_invalid else None,
        "episodes": records,
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
        print(f"Report: {report_path}")

    summary = " ".join(f"{name}={counts[name]}" for name in status_names)
    print(f"SUMMARY {summary} PENDING={len(pending)}")
    unusable = counts["INVALID"] + counts["ERROR"]
    return 2 if args.fail_on_invalid and unusable else 0


if __name__ == "__main__":
    raise SystemExit(main())
