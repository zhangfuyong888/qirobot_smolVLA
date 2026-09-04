"""Open head + active wrist only. Cameras stay running in READY."""

from __future__ import annotations

import time

from real_vla.cameras.camera_device import CameraReader, FrameSource
from real_vla.config_loader import CamerasConfig


class CameraManager:
    def __init__(
        self,
        cameras: CamerasConfig,
        *,
        active_arm: str,
        sources: dict[str, FrameSource] | None = None,
    ) -> None:
        self.cameras = cameras
        self.active_arm = active_arm
        self.readers: dict[str, CameraReader] = {}
        for stream in cameras.enabled_streams(active_arm):
            source = None if sources is None else sources.get(stream.name)
            self.readers[stream.name] = CameraReader(stream, source=source)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.readers.keys())

    def start(self, warmup_s: float | None = None) -> None:
        for reader in self.readers.values():
            reader.start()
        wait_s = self.cameras.warmup_s if warmup_s is None else float(warmup_s)
        deadline = time.monotonic() + max(wait_s, 0.0)
        while time.monotonic() < deadline:
            if all(reader.buffer.captured_frames > 0 for reader in self.readers.values()):
                time.sleep(min(wait_s, 0.25))
                return
            time.sleep(0.02)
        missing = [name for name, reader in self.readers.items() if reader.buffer.captured_frames <= 0]
        if missing:
            raise RuntimeError(f"camera warmup produced no frames: {missing}")

    def health_line(self) -> str:
        now_ns = time.monotonic_ns()
        parts = []
        for name, reader in self.readers.items():
            stats = reader.buffer.stats(now_ns)
            captured = int(stats["captured_frames"])
            fps = 0.0
            if captured > 1 and stats["max_frame_interval_ms"]:
                fps = min(reader.config.fps * 1.5, 1.0e3 / max(float(stats["max_frame_interval_ms"]), 1.0))
            # Prefer captured/elapsed approximation from last timestamps is coarse;
            # report captured count and drops instead of a fake locked 30.0.
            drops = int(stats["writer_drops"])
            parts.append(f"{name.upper()} n={captured} drop={drops} age={stats['last_frame_age_ms']:.0f}ms")
            del fps
        return "  ".join(parts)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
