#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(root: Path) -> dict[str, Any]:
    events_path = Path(root).expanduser().resolve() / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    responses = [event for event in events if event.get("type") == "policy_response"]
    commands = [event for event in events if event.get("type") == "command"]
    rtc_chunks = [event for event in events if event.get("type") == "rtc_chunk"]
    aborted = next((event for event in reversed(events) if event.get("type") == "abort"), None)

    def values(records, key):
        return [float(record[key]) for record in records if record.get(key) is not None]

    rtt = values(responses, "rtt_ms")
    observation_age = values(responses, "observation_age_ms")
    inference = values(responses, "inference_ms")
    jumps = values(rtc_chunks, "first_target_jump_rad")
    tracking = []
    for command in commands:
        if "target" in command and "policy_target" in command:
            tracking.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(command["target"])[:7]
                            - np.asarray(command["policy_target"])[:7]
                        )
                    )
                )
            )
    gripper = values(commands, "policy_gripper_raw")
    return {
        "run": str(Path(root).expanduser().resolve()),
        "checkpoint": responses[-1].get("checkpoint", "") if responses else "",
        "aborted": aborted is not None,
        "abort_reason": None if aborted is None else aborted.get("reason"),
        "replans": len(responses),
        "rejected_chunks": sum(event.get("type") == "policy_rejected" for event in events),
        "mean_rtt_ms": float(np.mean(rtt)) if rtt else None,
        "p95_rtt_ms": float(np.percentile(rtt, 95)) if rtt else None,
        "mean_inference_ms": float(np.mean(inference)) if inference else None,
        "mean_observation_age_ms": float(np.mean(observation_age)) if observation_age else None,
        "rollout_limited_ratio": (
            float(np.mean([bool(event.get("rollout_limited")) for event in commands]))
            if commands
            else None
        ),
        "hardware_limited_ratio": (
            float(np.mean([bool(event.get("hardware_limited")) for event in commands]))
            if commands
            else None
        ),
        "gripper_raw_max": max(gripper) if gripper else None,
        "gripper_transitions": sum(bool(event.get("gripper_transition")) for event in commands),
        "mean_tracking_error_rad": float(np.mean(tracking)) if tracking else None,
        "p95_tracking_error_rad": float(np.percentile(tracking, 95)) if tracking else None,
        "mean_rtc_target_jump_rad": float(np.mean(jumps)) if jumps else None,
        "p95_rtc_target_jump_rad": float(np.percentile(jumps, 95)) if jumps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare S4 real-VLA rollout event logs")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.runs], indent=2))


if __name__ == "__main__":
    main()
