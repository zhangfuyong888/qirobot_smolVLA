"""Pinocchio gravity compensation for hardware lowcmd (vendored from qiling bridge logic)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

pin = None


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1] if "}" in tag else tag


def strip_urdf_for_dynamics(robot_description_xml: str) -> str:
    root = ET.fromstring(robot_description_xml)
    removed_tags = {"visual", "collision", "gazebo", "material", "transmission", "ros2_control"}
    for parent in root.iter():
        for child in list(parent):
            if _xml_local_name(child.tag) in removed_tags:
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


class ArmGravityCompensator:
    """Compute arm gravity torques from URDF using Pinocchio."""

    def __init__(
        self,
        urdf_path: Path,
        controlled_joints: List[str],
        *,
        gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81),
        scale: float = 0.6,
        sign: float = 1.0,
        tau_limit: float = 12.0,
        strip_visual_collision: bool = True,
    ) -> None:
        global pin
        if pin is None:
            numpy_major = int(str(np.__version__).split(".", maxsplit=1)[0])
            if numpy_major >= 2:
                raise RuntimeError(
                    "pinocchio gravity compensation requires numpy 1.x in this Isaac env; "
                    f"got numpy {np.__version__}"
                )
            try:
                import pinocchio as pinocchio_module
            except Exception as exc:
                raise RuntimeError(f"pinocchio is not available: {exc}") from exc
            pin = pinocchio_module

        urdf_path = urdf_path.resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"gravity compensation URDF not found: {urdf_path}")

        robot_description_xml = urdf_path.read_text(encoding="utf-8")
        if strip_visual_collision:
            robot_description_xml = strip_urdf_for_dynamics(robot_description_xml)

        self.model = pin.buildModelFromXML(robot_description_xml)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.scale = float(scale)
        self.sign = float(sign)
        self.tau_limit = abs(float(tau_limit))
        self.controlled_joints = list(controlled_joints)
        self.index_by_name: Dict[str, tuple[int, int]] = {}

        gravity = np.asarray(gravity_vector, dtype=float)
        if gravity.shape != (3,):
            raise ValueError(f"gravity_vector must have 3 values, got {gravity_vector}")
        self.model.gravity.linear = gravity

        missing = []
        for name in self.controlled_joints:
            joint_id = int(self.model.getJointId(name))
            if joint_id == 0 or joint_id >= len(self.model.joints):
                missing.append(name)
                continue
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                missing.append(name)
                continue
            self.index_by_name[name] = (joint.idx_q, joint.idx_v)
        if missing:
            raise RuntimeError(f"joints not usable in Pinocchio gravity model: {missing}")

    def compute(self, positions: Mapping[str, float], *, scale_multiplier: float = 1.0) -> Dict[str, float]:
        for name in self.controlled_joints:
            idx_q, _ = self.index_by_name[name]
            self.q[idx_q] = float(positions[name])

        tau = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        multiplier = self.scale * self.sign * float(scale_multiplier)
        output: Dict[str, float] = {}
        for name in self.controlled_joints:
            _, idx_v = self.index_by_name[name]
            value = float(tau[idx_v]) * multiplier
            if self.tau_limit > 0.0:
                value = max(-self.tau_limit, min(self.tau_limit, value))
            output[name] = value
        return output
