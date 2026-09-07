from types import SimpleNamespace

import numpy as np

from real_vla.collection.schema import CameraFrame
from real_vla_stack.robot.rollout import observation


class _Buffer:
    def __init__(self, frame):
        self.frame = frame

    def snapshot_copy(self):
        return self.frame


def test_snapshot_timestamp_is_taken_after_frames_are_copied(monkeypatch):
    frames = [
        CameraFrame(1_000_000, 1, np.zeros((2, 2, 3), dtype=np.uint8), "head"),
        CameraFrame(1_200_000, 1, np.zeros((2, 2, 3), dtype=np.uint8), "wrist_right"),
    ]
    cameras = SimpleNamespace(
        names=("head", "wrist_right"),
        readers={
            frame.name: SimpleNamespace(buffer=_Buffer(frame)) for frame in frames
        },
    )
    monkeypatch.setattr(observation.time, "monotonic_ns", lambda: 1_300_000)

    images, timestamps, snapshot_ns = observation.snapshot_observation(
        cameras, max_age_ms=1.0, max_skew_ms=1.0
    )

    assert len(images) == 2
    assert timestamps == (1_000_000, 1_200_000)
    assert snapshot_ns == 1_300_000
