"""Quintic return-to-home. Publishes through the existing arm-replay limiter."""

from __future__ import annotations

import time

import numpy as np

from hardware_teleop.startup import (
    arm_home_error,
    interpolate_home_quintic,
    interpolate_toward_home,
)
from s4_robot.control_mapping import ACTION_SLICES


class HomeManager:
    def __init__(
        self,
        *,
        home_left_arm: tuple[float, ...] | np.ndarray,
        home_right_arm: tuple[float, ...] | np.ndarray,
        tolerance_rad: float = 0.05,
        stable_time_s: float = 0.3,
        duration_s: float = 6.0,
        max_joint_step_rad: float = 0.025,
        control_dt: float = 1.0 / 30.0,
    ) -> None:
        self.home_action = np.zeros(26, dtype=np.float32)
        self.home_action[ACTION_SLICES.left_arm] = np.asarray(home_left_arm, dtype=np.float32)
        self.home_action[ACTION_SLICES.right_arm] = np.asarray(home_right_arm, dtype=np.float32)
        self.tolerance_rad = float(tolerance_rad)
        self.stable_time_s = float(stable_time_s)
        self.duration_s = float(duration_s)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.control_dt = float(control_dt)
        self._active = False
        self._start_action: np.ndarray | None = None
        self._command: np.ndarray | None = None
        self._total_steps = 1
        self._step_index = 0
        self._stable_since: float | None = None
        self._started_s: float | None = None
        self.last_measured_error = float("inf")
        self.last_command_error = float("inf")
        self.arrived_by = ""

    @property
    def active(self) -> bool:
        return self._active

    @property
    def interpolation_done(self) -> bool:
        return self._step_index >= self._total_steps

    def status_line(self) -> str:
        return (
            f"home_meas={self.last_measured_error:.3f}rad "
            f"home_cmd={self.last_command_error:.3f}rad "
            f"step={self._step_index}/{self._total_steps}"
        )

    def request_home(self, command_action: np.ndarray, now_s: float | None = None) -> None:
        start = np.asarray(command_action, dtype=np.float32).copy()
        error = arm_home_error(start, self.home_action)
        requested_steps = max(1, int(np.ceil(self.duration_s / self.control_dt)))
        step_limited_steps = (
            int(np.ceil(1.875 * error / self.max_joint_step_rad))
            if self.max_joint_step_rad > 0.0
            else 1
        )
        self._start_action = start
        self._command = start.copy()
        self._total_steps = max(requested_steps, step_limited_steps, 1)
        self._step_index = 0
        self._stable_since = None
        self._started_s = time.monotonic() if now_s is None else float(now_s)
        self.last_measured_error = error
        self.last_command_error = error
        self.arrived_by = ""
        self._active = True

    def step(self) -> np.ndarray:
        if not self._active or self._start_action is None or self._command is None:
            raise RuntimeError("HomeManager.step() called before request_home()")
        tau = min(self._step_index / self._total_steps, 1.0)
        desired = interpolate_home_quintic(self._start_action, self.home_action, tau)
        self._command = interpolate_toward_home(
            self._command,
            desired,
            max_step_rad=self.max_joint_step_rad,
        )
        self._command[ACTION_SLICES.left_hand] = self._start_action[ACTION_SLICES.left_hand]
        self._command[ACTION_SLICES.right_hand] = self._start_action[ACTION_SLICES.right_hand]
        self._step_index += 1
        return self._command.copy()

    def is_home(
        self,
        measured_action: np.ndarray,
        now_s: float | None = None,
        *,
        require_measured: bool = False,
    ) -> bool:
        now = time.monotonic() if now_s is None else float(now_s)
        self.last_measured_error = arm_home_error(
            np.asarray(measured_action, dtype=np.float32), self.home_action
        )
        self.last_command_error = (
            arm_home_error(self._command, self.home_action)
            if self._command is not None
            else float("inf")
        )
        # Gravity-off arms often sit ~0.03–0.05 rad off the held command. Collection
        # must not wait forever for measured joints to match yaml home.
        arrived_by = ""
        if self.last_measured_error <= self.tolerance_rad:
            arrived_by = "measured"
        elif not require_measured and self.last_command_error <= self.tolerance_rad:
            arrived_by = "command"
        elif (
            self.interpolation_done
            and self._started_s is not None
            and now - self._started_s >= self.duration_s + max(self.stable_time_s, 1.0)
        ):
            arrived_by = "timeout"
        if not arrived_by:
            self._stable_since = None
            self.arrived_by = ""
            return False
        if self._stable_since is None:
            self._stable_since = now
            self.arrived_by = arrived_by
            return False
        if now - self._stable_since >= self.stable_time_s:
            self.arrived_by = arrived_by
            self._active = False
            return True
        self.arrived_by = arrived_by
        return False
