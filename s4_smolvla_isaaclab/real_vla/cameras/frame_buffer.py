"""Latest-frame buffer that never blocks the capture thread."""

from __future__ import annotations

import threading

import numpy as np

from real_vla.collection.schema import CameraFrame


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self.captured_frames = 0
        self.read_failures = 0
        self.writer_drops = 0
        self.last_timestamp_ns = 0
        self.max_interval_ns = 0

    def set_frame(self, frame: CameraFrame) -> None:
        with self._lock:
            if self.last_timestamp_ns > 0:
                gap = int(frame.timestamp_ns) - self.last_timestamp_ns
                if gap > self.max_interval_ns:
                    self.max_interval_ns = gap
            self._frame = frame
            self.captured_frames += 1
            self.last_timestamp_ns = int(frame.timestamp_ns)

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def snapshot_copy(self) -> CameraFrame | None:
        with self._lock:
            if self._frame is None:
                return None
            return CameraFrame(
                timestamp_ns=self._frame.timestamp_ns,
                capture_seq=self._frame.capture_seq,
                image_bgr=np.asarray(self._frame.image_bgr).copy(),
                name=self._frame.name,
            )

    def stats(self, now_ns: int) -> dict[str, float | int]:
        with self._lock:
            age_ms = 0.0
            if self.last_timestamp_ns > 0:
                age_ms = max(now_ns - self.last_timestamp_ns, 0) / 1.0e6
            return {
                "captured_frames": self.captured_frames,
                "read_failures": self.read_failures,
                "writer_drops": self.writer_drops,
                "last_frame_age_ms": age_ms,
                "max_frame_interval_ms": self.max_interval_ns / 1.0e6,
            }
