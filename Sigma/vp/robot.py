"""PyBullet PSM model helpers for the split controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


POSITION_JOINTS = ("yaw", "pitch", "insertion")
WRIST_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")
JAW_JOINT = "jaw"
MIMIC_JOINTS = {
    "pitch_1": ("pitch", 1.0),
    "pitch_2": ("pitch", 1.0),
    "pitch_3": ("pitch", -1.0),
    "pitch_4": ("pitch", -1.0),
    "pitch_5": ("pitch", 1.0),
    "jaw_mimic_1": ("jaw", 0.5),
    "jaw_mimic_2": ("jaw", -0.5),
}


@dataclass
class JointInfo:
    index: int
    lower: float
    upper: float
    max_force: float
    fixed: bool


@dataclass
class PyBulletPsmRobot:
    pybullet: object
    client_id: int
    body_id: int
    link_names: dict[str, int]
    joints: dict[str, JointInfo]
    tool_link_name: str = "PSM1_tool_tip_link"

    @classmethod
    def load(cls, pybullet, client_id: int, urdf_path: Path, position_force: float, jaw_force: float):
        body_id = pybullet.loadURDF(
            str(urdf_path),
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client_id,
        )

        link_names: dict[str, int] = {}
        joints: dict[str, JointInfo] = {}
        for joint_index in range(pybullet.getNumJoints(body_id, physicsClientId=client_id)):
            info = pybullet.getJointInfo(body_id, joint_index, physicsClientId=client_id)
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            lower = float(info[8])
            upper = float(info[9])
            if lower > upper:
                lower, upper = -np.inf, np.inf
            max_force = jaw_force if name.startswith("jaw") else position_force
            joints[name] = JointInfo(
                index=int(joint_index),
                lower=lower,
                upper=upper,
                max_force=float(max_force),
                fixed=info[2] == pybullet.JOINT_FIXED,
            )
            link_names[link_name] = int(joint_index)

        required = (*POSITION_JOINTS, *WRIST_JOINTS, JAW_JOINT)
        missing = [name for name in required if name not in joints]
        if missing:
            raise ValueError(f"PyBullet URDF missing required joints: {missing}")
        if "PSM1_tool_tip_link" not in link_names:
            raise ValueError("PyBullet URDF missing link: PSM1_tool_tip_link")

        robot = cls(pybullet=pybullet, client_id=client_id, body_id=int(body_id), link_names=link_names, joints=joints)
        robot.disable_default_motors()
        return robot

    @property
    def dt(self) -> float:
        return float(self.pybullet.getPhysicsEngineParameters(physicsClientId=self.client_id)["fixedTimeStep"])

    @property
    def position_limits(self) -> np.ndarray:
        return self.limits(POSITION_JOINTS)

    @property
    def wrist_limits(self) -> np.ndarray:
        return self.limits(WRIST_JOINTS)

    @property
    def jaw_limit(self) -> np.ndarray:
        return self.limits((JAW_JOINT,))[0]

    def limits(self, joint_names: tuple[str, ...]) -> np.ndarray:
        return np.array([[self.joints[name].lower, self.joints[name].upper] for name in joint_names], dtype=float)

    def disable_default_motors(self) -> None:
        for joint in self.joints.values():
            if joint.fixed:
                continue
            self.pybullet.setJointMotorControl2(
                self.body_id,
                joint.index,
                self.pybullet.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
                physicsClientId=self.client_id,
            )

    def reset_and_seed_home(self, home_insertion: float) -> None:
        targets = {
            "yaw": 0.0,
            "pitch": 0.0,
            "insertion": float(home_insertion),
            "roll": 0.0,
            "wrist_pitch": 0.0,
            "wrist_yaw": 0.0,
            "jaw": float(self.jaw_limit[0]),
        }
        self.reset_joint_targets(targets)
        for _ in range(120):
            self.apply_position_targets(targets)
            self.pybullet.stepSimulation(physicsClientId=self.client_id)

    def reset_joint_targets(self, targets: dict[str, float]) -> None:
        expanded = self.with_mimics(targets)
        for name, value in expanded.items():
            if name not in self.joints:
                continue
            joint = self.joints[name]
            if joint.fixed:
                continue
            self.pybullet.resetJointState(self.body_id, joint.index, float(value), physicsClientId=self.client_id)

    def with_mimics(self, targets: dict[str, float]) -> dict[str, float]:
        expanded = dict(targets)
        for mimic, (source, multiplier) in MIMIC_JOINTS.items():
            if source in expanded:
                expanded[mimic] = float(multiplier) * float(expanded[source])
        return expanded

    def apply_position_targets(self, targets: dict[str, float]) -> None:
        for name, value in self.with_mimics(targets).items():
            if name not in self.joints:
                continue
            joint = self.joints[name]
            if joint.fixed:
                continue
            self.pybullet.setJointMotorControl2(
                self.body_id,
                joint.index,
                self.pybullet.POSITION_CONTROL,
                targetPosition=float(np.clip(value, joint.lower, joint.upper)),
                force=joint.max_force,
                physicsClientId=self.client_id,
            )

    def joint_positions(self, joint_names: tuple[str, ...]) -> np.ndarray:
        values = []
        for name in joint_names:
            state = self.pybullet.getJointState(self.body_id, self.joints[name].index, physicsClientId=self.client_id)
            values.append(float(state[0]))
        return np.array(values, dtype=float)

    def tool_position(self) -> np.ndarray:
        state = self.pybullet.getLinkState(
            self.body_id,
            self.link_names[self.tool_link_name],
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.array(state[4], dtype=float)

    def finite_difference_position_jacobian(self, controlled_joints: tuple[str, ...], eps: float) -> np.ndarray:
        joint_names = tuple(name for name, joint in self.joints.items() if not joint.fixed)
        base_positions = self.joint_positions(joint_names)
        controlled_indices = [joint_names.index(name) for name in controlled_joints]
        current_tool = self.tool_position()
        jacobian = np.zeros((3, len(controlled_joints)), dtype=float)

        for col, all_joint_idx in enumerate(controlled_indices):
            perturbed = base_positions.copy()
            perturbed[all_joint_idx] += float(eps)
            self._reset_all_joint_states(joint_names, perturbed)
            jacobian[:, col] = (self.tool_position() - current_tool) / float(eps)

        self._reset_all_joint_states(joint_names, base_positions)
        return jacobian

    def _reset_all_joint_states(self, joint_names: tuple[str, ...], values: np.ndarray) -> None:
        for name, value in zip(joint_names, values):
            joint = self.joints[name]
            if joint.fixed:
                continue
            self.pybullet.resetJointState(self.body_id, joint.index, float(value), physicsClientId=self.client_id)
