"""MuJoCo PSM helpers for the v2 split controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARM_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
POSITION_JOINTS = ("yaw", "pitch", "insertion")
WRIST_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")
JAW_JOINT = "jaw"


def body_position(data, body_id: int) -> np.ndarray:
    return data.xpos[body_id].copy()


def body_rotation(data, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def joint_dof_columns(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
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
        ids.append(int(actuator_id))
    return ids


@dataclass
class PsmRobot:
    mujoco: object
    model: object
    data: object
    track_body_name: str
    track_body_id: int
    arm_actuators: list[int]
    position_actuators: list[int]
    wrist_actuators: list[int]
    jaw_actuator: int
    position_dof_columns: list[int]

    @classmethod
    def load(cls, mujoco, xml_path: Path, track_body_name: str):
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        track_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)
        if track_body_id < 0:
            raise ValueError(f"MuJoCo body not found: {track_body_name}")
        return cls(
            mujoco=mujoco,
            model=model,
            data=data,
            track_body_name=track_body_name,
            track_body_id=int(track_body_id),
            arm_actuators=actuator_ids(mujoco, model, ARM_JOINTS),
            position_actuators=actuator_ids(mujoco, model, POSITION_JOINTS),
            wrist_actuators=actuator_ids(mujoco, model, WRIST_JOINTS),
            jaw_actuator=actuator_ids(mujoco, model, (JAW_JOINT,))[0],
            position_dof_columns=joint_dof_columns(mujoco, model, POSITION_JOINTS),
        )

    @property
    def arm_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.arm_actuators]

    @property
    def position_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.position_actuators]

    @property
    def wrist_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.wrist_actuators]

    @property
    def jaw_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.jaw_actuator]

    def reset_and_seed_home(self, home_insertion: float) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        desired = np.array([0.0, 0.0, home_insertion, 0.0, 0.0, 0.0], dtype=float)
        self.data.ctrl[self.arm_actuators] = np.clip(desired, self.arm_ctrl_range[:, 0], self.arm_ctrl_range[:, 1])
        self.data.ctrl[self.jaw_actuator] = float(self.jaw_ctrl_range[0])
        for _ in range(500):
            self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)

    def track_position(self) -> np.ndarray:
        return body_position(self.data, self.track_body_id)

    def track_rotation(self) -> np.ndarray:
        return body_rotation(self.data, self.track_body_id)

    def jacobian_at_track_body(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jac(self.model, self.data, jacp, jacr, self.track_position(), self.track_body_id)
        return jacp, jacr

