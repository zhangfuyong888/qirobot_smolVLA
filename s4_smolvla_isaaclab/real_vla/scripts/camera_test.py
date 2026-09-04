#!/usr/bin/env python
"""Snapshot each configured RealSense camera to confirm serials and wrist sides."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_vla.config_loader import load_collection_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture one frame from each RealSense serial.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("/tmp/real_vla_camera_test"))
    args = parser.parse_args(argv)
    config = load_collection_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        import pyrealsense2 as rs
        import cv2
    except ImportError as exc:
        print(f"missing camera dependency: {exc}", file=sys.stderr)
        return 1

    ctx = rs.context()
    devices = list(ctx.query_devices())
    print(f"librealsense devices: {len(devices)}")
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f"  {name} serial={serial}")

    streams = [config.cameras.head, config.cameras.wrist_left, config.cameras.wrist_right]
    for stream in streams:
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(stream.serial)
        rs_cfg.enable_stream(
            rs.stream.color, stream.width, stream.height, rs.format.bgr8, stream.fps
        )
        try:
            pipeline.start(rs_cfg)
        except Exception as exc:
            print(f"{stream.name} serial={stream.serial} FAILED: {exc}")
            continue
        image = None
        for _ in range(30):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color:
                image = np.asanyarray(color.get_data())
            time.sleep(0.03)
        pipeline.stop()
        if image is None:
            print(f"{stream.name} serial={stream.serial} produced no color frame")
            continue
        path = args.out / f"{stream.name}_{stream.serial}.png"
        cv2.imwrite(str(path), image)
        print(f"wrote {path} shape={image.shape} model={stream.model}")
    print("Look at the PNGs to confirm which D405 is left vs right wrist, then edit real_vla/config/cameras.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
