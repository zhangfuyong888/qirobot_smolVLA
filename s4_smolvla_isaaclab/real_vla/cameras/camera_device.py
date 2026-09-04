"""Independent RealSense capture thread. Never called from the robot loop."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from real_vla.cameras.frame_buffer import LatestFrameBuffer
from real_vla.collection.schema import CameraFrame
from real_vla.config_loader import CameraStreamConfig


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class RealSenseSource:
    def __init__(self, config: CameraStreamConfig) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(config.serial)
        rs_cfg.enable_stream(
            rs.stream.color,
            int(config.width),
            int(config.height),
            rs.format.bgr8,
            int(config.fps),
        )
        self._pipeline.start(rs_cfg)
        self._align = rs.align(rs.stream.color)

    def read(self) -> np.ndarray | None:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self._align.process(frames)
        color = aligned.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class CameraReader:
    def __init__(self, config: CameraStreamConfig, source: FrameSource | None = None) -> None:
        self.config = config
        self.buffer = LatestFrameBuffer()
        self._source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._writer = None

    @property
    def name(self) -> str:
        return self.config.name

    def start(self) -> None:
        if self._source is None:
            self._source = RealSenseSource(self.config)
        self._thread = threading.Thread(
            target=self._loop,
            name=f"camera-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def set_writer(self, writer) -> None:
        self._writer = writer

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                image = self._source.read() if self._source is not None else None
            except Exception:
                self.buffer.read_failures += 1
                time.sleep(0.01)
                continue
            if image is None:
                self.buffer.read_failures += 1
                continue
            self._seq += 1
            timestamp_ns = time.monotonic_ns()
            frame = CameraFrame(
                timestamp_ns=timestamp_ns,
                capture_seq=self._seq,
                image_bgr=image,
                name=self.config.name,
            )
            self.buffer.set_frame(frame)
            writer = self._writer
            if writer is not None:
                writer(
                    CameraFrame(
                        timestamp_ns=timestamp_ns,
                        capture_seq=self._seq,
                        image_bgr=np.asarray(image).copy(),
                        name=self.config.name,
                    )
                )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._source is not None:
            self._source.close()
            self._source = None
