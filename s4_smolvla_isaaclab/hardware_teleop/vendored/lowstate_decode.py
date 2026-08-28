"""Parse robot lowstate into arm joint positions (vendored from qiling bridge logic)."""

from __future__ import annotations

import math
from typing import Mapping

from hardware_teleop.vendored.joint_layout import REAL_ROBOT_BODY_JOINT_ORDER, joint_order_for_body_dof
from hardware_teleop.vendored.joint_state_safety import JointStateFrameGuard, JointStateValidation


def arm_joint_names(joint_order: list[str]) -> list[str]:
    return [
        name
        for name in joint_order
        if "shoulder" in name or "elbow" in name or "wrist" in name
    ]


def decode_lowstate_arm_positions(
    motors: list,
    *,
    body_dof: int,
    sign_by_name: Mapping[str, float],
    guard: JointStateFrameGuard,
    arm_names: tuple[str, ...],
) -> tuple[dict[str, float] | None, JointStateValidation]:
    joint_order = joint_order_for_body_dof(body_dof)
    if not motors:
        return None, JointStateValidation(False, "empty lowstate motors")

    count = min(len(motors), len(joint_order))
    positions: dict[str, float] = {}
    for name, motor in zip(joint_order[:count], motors[:count], strict=False):
        sign = float(sign_by_name.get(name, 1.0))
        value = sign * float(motor.q)
        if not math.isfinite(value):
            return None, JointStateValidation(False, f"non-finite lowstate position for {name}")
        positions[name] = value

    validation = guard.validate(positions)
    if not validation.accepted:
        return None, validation

    try:
        arm_positions = {name: positions[name] for name in arm_names}
    except KeyError as exc:
        return None, JointStateValidation(False, f"missing arm joint in lowstate: {exc}")

    return arm_positions, JointStateValidation(True)
