from __future__ import annotations

import math


class ExecutionSyncGuard:
    """Request a measured-state replan before execution drift becomes unsafe."""

    def __init__(
        self,
        *,
        policy_lag_rad: float,
        contact_tracking_error_rad: float,
        trigger_cycles: int,
    ) -> None:
        if not math.isfinite(policy_lag_rad) or policy_lag_rad <= 0:
            raise ValueError("policy_lag_rad must be finite and positive")
        if (
            not math.isfinite(contact_tracking_error_rad)
            or contact_tracking_error_rad <= 0
        ):
            raise ValueError(
                "contact_tracking_error_rad must be finite and positive"
            )
        if int(trigger_cycles) <= 0:
            raise ValueError("trigger_cycles must be positive")
        self.policy_lag_rad = float(policy_lag_rad)
        self.contact_tracking_error_rad = float(contact_tracking_error_rad)
        self.trigger_cycles = int(trigger_cycles)
        self.policy_lag_cycles = 0
        self.contact_cycles = 0
        self.hold = False
        self.reset_pending = False
        self.reason = ""

    def observe(
        self,
        *,
        policy_execution_lag_rad: float | None,
        command_tracking_error_rad: float | None,
        gripper_closed: bool,
    ) -> bool:
        """Update counters and return True only when a new hold is triggered."""
        policy_lagged = (
            policy_execution_lag_rad is not None
            and math.isfinite(policy_execution_lag_rad)
            and policy_execution_lag_rad > self.policy_lag_rad
        )
        contact_lagged = (
            gripper_closed
            and command_tracking_error_rad is not None
            and math.isfinite(command_tracking_error_rad)
            and command_tracking_error_rad > self.contact_tracking_error_rad
        )
        self.policy_lag_cycles = self.policy_lag_cycles + 1 if policy_lagged else 0
        self.contact_cycles = self.contact_cycles + 1 if contact_lagged else 0
        trigger_reason = ""
        if self.contact_cycles >= self.trigger_cycles:
            trigger_reason = "contact_tracking_lag"
        elif self.policy_lag_cycles >= self.trigger_cycles:
            trigger_reason = "policy_execution_lag"
        if not trigger_reason:
            return False
        newly_triggered = not self.hold
        self.hold = True
        self.reset_pending = True
        self.reason = trigger_reason
        return newly_triggered

    def acknowledge_resync(self) -> None:
        self.policy_lag_cycles = 0
        self.contact_cycles = 0
        self.hold = False
        self.reset_pending = False
        self.reason = ""

    def force_resync(self, reason: str) -> bool:
        """Enter hold immediately after an unsafe chunk or equivalent fault."""
        newly_triggered = not self.hold
        self.hold = True
        self.reset_pending = True
        self.reason = str(reason)
        return newly_triggered
