from __future__ import annotations

import json

from real_vla_stack.robot.rollout.logger import RolloutLogger


def test_logger_flushes_queued_events_and_shadow_snapshot(tmp_path) -> None:
    logger = RolloutLogger(tmp_path / "run", {"live": False})
    logger.event("first", value=1)
    logger.save_observation(
        request_id=2,
        state=[0.0] * 8,
        image_timestamps_ns=(10, 11),
        head_jpeg=b"head",
        wrist_jpeg=b"wrist",
    )
    logger.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "run/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["type"] for record in records] == ["first", "observation_snapshot"]
    assert all("timestamp_ns" in record for record in records)
    assert (tmp_path / "run/observations/000002_head.jpg").read_bytes() == b"head"
