from __future__ import annotations

import numpy as np


class JointCommandFilter:
    """Rate- and acceleration-limit a seven-joint absolute target stream."""

    def __init__(
        self,
        *,
        control_hz: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> None:
        self.dt = 1.0 / float(control_hz)
        self.max_velocity = float(max_velocity_rad_s)
        self.max_acceleration = float(max_acceleration_rad_s2)
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(7, dtype=np.float64)

    def reset(self, position_q7: np.ndarray) -> None:
        self._position = np.asarray(position_q7, dtype=np.float64).reshape(7).copy()
        self._velocity.fill(0.0)

    def step(self, target_q7: np.ndarray) -> np.ndarray:
        target = np.asarray(target_q7, dtype=np.float64).reshape(7)
        if self._position is None:
            self.reset(target)
            return target.copy()
        error = target - self._position
        desired_velocity = np.clip(
            error / self.dt,
            -self.max_velocity,
            self.max_velocity,
        )
        max_dv = self.max_acceleration * self.dt
        velocity = np.clip(
            desired_velocity,
            self._velocity - max_dv,
            self._velocity + max_dv,
        )
        step = velocity * self.dt
        # Never pass through a nearby target while braking.
        overshoot = np.abs(step) > np.abs(error)
        step[overshoot] = error[overshoot]
        velocity[overshoot] = step[overshoot] / self.dt
        self._position = self._position + step
        self._velocity = velocity
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()
