"""Vendored from qiling_s4 for joint-state validation."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class JointStateValidation:
    accepted: bool
    reason: str = ""


class JointStateFrameGuard:
    """Reject invalid or isolated all-zero joint-state frames."""

    def __init__(
        self,
        monitored_names: Iterable[str],
        *,
        reject_zero_glitches: bool = True,
        zero_threshold: float = 1.0e-4,
        zero_previous_motion_threshold: float = 0.05,
        zero_confirmation_frames: int = 3,
        max_position_jump: float | None = None,
    ) -> None:
        self.monitored_names = tuple(monitored_names)
        self.reject_zero_glitches = bool(reject_zero_glitches)
        self.zero_threshold = max(float(zero_threshold), 0.0)
        self.zero_previous_motion_threshold = max(
            float(zero_previous_motion_threshold), self.zero_threshold
        )
        self.zero_confirmation_frames = max(int(zero_confirmation_frames), 1)
        self.max_position_jump = (
            None
            if max_position_jump is None
            else max(float(max_position_jump), 0.0)
        )
        self.last_accepted: Dict[str, float] = {}
        self._pending_zero_frames = 0

    def validate(self, positions: Mapping[str, float]) -> JointStateValidation:
        non_finite = [
            name for name, value in positions.items() if not math.isfinite(float(value))
        ]
        if non_finite:
            self._pending_zero_frames = 0
            return JointStateValidation(False, f"non-finite positions: {non_finite}")
        relevant = {
            name: float(positions[name])
            for name in self.monitored_names
            if name in positions
        }
        complete = len(relevant) == len(self.monitored_names)
        current_is_zero = complete and all(
            abs(value) <= self.zero_threshold for value in relevant.values()
        )
        previous_complete = all(name in self.last_accepted for name in self.monitored_names)
        previous_was_moving = previous_complete and any(
            abs(self.last_accepted[name]) >= self.zero_previous_motion_threshold
            for name in self.monitored_names
        )

        if complete and previous_complete and self.max_position_jump is not None:
            jumps = {
                name: abs(relevant[name] - self.last_accepted[name])
                for name in self.monitored_names
            }
            worst_name = max(jumps, key=jumps.get)
            worst_jump = jumps[worst_name]
            if worst_jump > self.max_position_jump:
                self._pending_zero_frames = 0
                return JointStateValidation(
                    False,
                    f"joint position jump for {worst_name}: {worst_jump:.6f} rad "
                    f"> {self.max_position_jump:.6f} rad",
                )

        if self.reject_zero_glitches and current_is_zero and previous_was_moving:
            self._pending_zero_frames += 1
            if self._pending_zero_frames < self.zero_confirmation_frames:
                return JointStateValidation(
                    False,
                    "unexpected all-zero frame "
                    f"({self._pending_zero_frames}/{self.zero_confirmation_frames})",
                )
        else:
            self._pending_zero_frames = 0

        self.last_accepted.update(relevant)
        if current_is_zero:
            self._pending_zero_frames = 0
        return JointStateValidation(True)

    def last_position(self, name: str) -> Optional[float]:
        return self.last_accepted.get(name)
