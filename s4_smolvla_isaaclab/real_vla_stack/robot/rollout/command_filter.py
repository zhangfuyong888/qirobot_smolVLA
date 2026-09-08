from __future__ import annotations

import numpy as np


class JointCommandFilter:
    """Rate- and acceleration-limit a seven-joint absolute target stream."""

    def __init__(
        self,
        *,
        control_hz: float,
        max_velocity_rad_s: float | list[float] | np.ndarray,
        max_acceleration_rad_s2: float | list[float] | np.ndarray,
    ) -> None:
        self.dt = 1.0 / float(control_hz)
        self.max_velocity = self._limits(max_velocity_rad_s, "velocity")
        self.max_acceleration = self._limits(max_acceleration_rad_s2, "acceleration")
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(7, dtype=np.float64)
        self._limited = np.zeros(7, dtype=bool)
        self._velocity_limited = np.zeros(7, dtype=bool)
        self._acceleration_limited = np.zeros(7, dtype=bool)

    @staticmethod
    def _limits(values: float | list[float] | np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(values, dtype=np.float64)
        if value.ndim == 0:
            value = np.full(7, float(value), dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name} limits must be finite positive scalar or [7]")
        return value

    def reset(self, position_q7: np.ndarray) -> None:
        self._position = np.asarray(position_q7, dtype=np.float64).reshape(7).copy()
        self._velocity.fill(0.0)
        self._limited.fill(False)
        self._velocity_limited.fill(False)
        self._acceleration_limited.fill(False)

    def step(self, target_q7: np.ndarray) -> np.ndarray:
        target = np.asarray(target_q7, dtype=np.float64).reshape(7)
        if self._position is None:
            self.reset(target)
            return target.copy()
        error = target - self._position
        unconstrained_velocity = error / self.dt
        desired_velocity = np.clip(
            unconstrained_velocity,
            -self.max_velocity,
            self.max_velocity,
        )
        self._velocity_limited = ~np.isclose(
            desired_velocity, unconstrained_velocity, atol=1.0e-12
        )
        max_dv = self.max_acceleration * self.dt
        velocity = np.clip(
            desired_velocity,
            self._velocity - max_dv,
            self._velocity + max_dv,
        )
        self._acceleration_limited = ~np.isclose(
            velocity, desired_velocity, atol=1.0e-12
        )
        self._limited = self._velocity_limited | self._acceleration_limited
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

    @property
    def limited_joints(self) -> np.ndarray:
        return self._limited.copy()

    @property
    def velocity_limited_joints(self) -> np.ndarray:
        return self._velocity_limited.copy()

    @property
    def acceleration_limited_joints(self) -> np.ndarray:
        return self._acceleration_limited.copy()
