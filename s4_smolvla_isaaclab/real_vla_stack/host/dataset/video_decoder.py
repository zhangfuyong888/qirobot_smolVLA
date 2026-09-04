from __future__ import annotations

from pathlib import Path

import numpy as np

from ...common.errors import DataValidationError


def decode_selected_rgb(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    """Sequentially decode monotonic source indices as RGB uint8 HWC."""
    import av

    wanted = np.asarray(indices, dtype=np.int64)
    if wanted.ndim != 1 or wanted.size == 0 or np.any(np.diff(wanted) < 0):
        raise DataValidationError("selected video indices must be a non-empty monotonic vector")
    positions: dict[int, list[int]] = {}
    for output_index, source_index in enumerate(wanted.tolist()):
        positions.setdefault(int(source_index), []).append(output_index)
    result: list[np.ndarray | None] = [None] * len(wanted)
    with av.open(str(path)) as container:
        for source_index, frame in enumerate(container.decode(video=0)):
            if source_index in positions:
                rgb = frame.to_ndarray(format="rgb24")
                for output_index in positions[source_index]:
                    result[output_index] = rgb.copy()
            if source_index > int(wanted[-1]):
                break
    missing = [i for i, value in enumerate(result) if value is None]
    if missing:
        raise DataValidationError(f"{path}: could not decode selected frames at outputs {missing[:8]}")
    return [value for value in result if value is not None]
