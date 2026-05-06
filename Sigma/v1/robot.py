"""MuJoCo PSM model, joint, body, and actuator helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARM_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
POSITION_JOINTS = ("yaw", "pitch", "insertion")
DISTAL_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")
JAW_JOINT = "jaw"
TRACK_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0], dtype=float)


def body_rotation(data, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def point_world_pos(data, body_id: int, offset_local: np.ndarray = TRACK_OFFSET_LOCAL) -> np.ndarray:
    return data.xpos[body_id] + body_rotation(data, body_id) @ offset_local


def controlled_dof_columns(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    columns: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        columns.append(int(model.jnt_dofadr[joint_id]))
    return columns


def actuator_ids(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for name in joint_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ctrl_{name}")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: ctrl_{name}")
        ids.append(actuator_id)
    return ids


def actuator_targets_from_qpos(mujoco, model, data, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        values.append(float(data.qpos[qpos_address]))
    return np.array(values, dtype=float)


@dataclass
class PsmRobot:
    mujoco: object
    model: object
    data: object
    track_body_name: str
    track_body_id: int
    orientation_body_name: str
    orientation_body_id: int
    arm_actuators: list[int]
    jaw_actuator: int
    position_dof_columns: list[int]
    distal_dof_columns: list[int]
    arm_dof_columns: list[int]

    @classmethod
    def load(cls, mujoco, xml_path: Path, track_body_name: str, orientation_body_name: str | None):
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        track_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)
        if track_body_id < 0:
            raise ValueError(f"MuJoCo body not found: {track_body_name}")
        orientation_name = orientation_body_name or track_body_name
        orientation_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, orientation_name)
        if orientation_body_id < 0:
            raise ValueError(f"MuJoCo body not found: {orientation_name}")
        return cls(
            mujoco=mujoco,
            model=model,
            data=data,
            track_body_name=track_body_name,
            track_body_id=track_body_id,
            orientation_body_name=orientation_name,
            orientation_body_id=orientation_body_id,
            arm_actuators=actuator_ids(mujoco, model, ARM_JOINTS),
            jaw_actuator=actuator_ids(mujoco, model, (JAW_JOINT,))[0],
            position_dof_columns=controlled_dof_columns(mujoco, model, POSITION_JOINTS),
            distal_dof_columns=controlled_dof_columns(mujoco, model, DISTAL_JOINTS),
            arm_dof_columns=controlled_dof_columns(mujoco, model, ARM_JOINTS),
        )

    @property
    def arm_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.arm_actuators]

    @property
    def jaw_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.jaw_actuator]

    def reset_and_seed_home(self, home_insertion: float) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        desired = np.array([0.0, 0.0, home_insertion, 0.0, 0.0, 0.0], dtype=float)
        desired = np.clip(desired, self.arm_ctrl_range[:, 0], self.arm_ctrl_range[:, 1])
        self.data.ctrl[self.arm_actuators] = desired
        self.data.ctrl[self.jaw_actuator] = float(self.jaw_ctrl_range[0])
        for _ in range(400):
            self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)

    def arm_qpos(self) -> np.ndarray:
        return actuator_targets_from_qpos(self.mujoco, self.model, self.data, ARM_JOINTS)

    def track_position(self) -> np.ndarray:
        return point_world_pos(self.data, self.track_body_id)

    def track_rotation(self) -> np.ndarray:
        return body_rotation(self.data, self.track_body_id)

    def orientation_position(self) -> np.ndarray:
        return point_world_pos(self.data, self.orientation_body_id)

    def orientation_rotation(self) -> np.ndarray:
        return body_rotation(self.data, self.orientation_body_id)

    def jacobian(self, body_id: int, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mujoco.mj_jac(self.model, self.data, jacp, jacr, point, body_id)
        return jacp, jacr
