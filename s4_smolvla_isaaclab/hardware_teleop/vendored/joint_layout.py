"""Vendored from qiling_s4 for real-robot lowcmd joint ordering."""

from typing import List


DEFAULT_JOINT_ORDER = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_foot_pitch_joint",
    "left_foot_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_foot_pitch_joint",
    "right_foot_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "LH_thumb_cmc_yaw",
    "LH_thumb_cmc_pitch",
    "LH_index_mcp_pitch",
    "LH_middle_mcp_pitch",
    "LH_ring_mcp_pitch",
    "LH_pinky_mcp_pitch",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "RH_thumb_cmc_yaw",
    "RH_thumb_cmc_pitch",
    "RH_index_mcp_pitch",
    "RH_middle_mcp_pitch",
    "RH_ring_mcp_pitch",
    "RH_pinky_mcp_pitch",
]

REAL_ROBOT_BODY_JOINT_ORDER = (
    DEFAULT_JOINT_ORDER[:12]
    + DEFAULT_JOINT_ORDER[12:19]
    + DEFAULT_JOINT_ORDER[25:32]
)

DEFAULT_REVERSED_JOINT_NAMES = (
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_yaw_joint",
)


def joint_order_for_body_dof(body_dof: int) -> List[str]:
    if int(body_dof) <= 26:
        return list(REAL_ROBOT_BODY_JOINT_ORDER)
    return list(DEFAULT_JOINT_ORDER)
