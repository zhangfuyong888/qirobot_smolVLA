from __future__ import annotations

import numpy as np

from ...common.errors import DataValidationError


def snapshot_observation(cameras, *, max_age_ms: float, max_skew_ms: float, now_ns: int):
    frames = [cameras.readers[name].buffer.snapshot_copy() for name in cameras.names]
    if any(frame is None for frame in frames):
        raise DataValidationError("one or more rollout cameras have no frame")
    concrete = [frame for frame in frames if frame is not None]
    ages = [(now_ns - frame.timestamp_ns) / 1.0e6 for frame in concrete]
    if min(ages) < 0 or max(ages) > max_age_ms:
        raise DataValidationError(f"rollout camera age invalid: {ages}")
    skew = (max(frame.timestamp_ns for frame in concrete) - min(frame.timestamp_ns for frame in concrete)) / 1.0e6
    if skew > max_skew_ms:
        raise DataValidationError(f"rollout camera skew {skew:.1f}ms exceeds {max_skew_ms:.1f}ms")
    rgb = [np.asarray(frame.image_bgr)[:, :, ::-1].copy() for frame in concrete]
    return rgb, tuple(int(frame.timestamp_ns) for frame in concrete)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg", np.asarray(rgb)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG observation encoding failed")
    return encoded.tobytes()
