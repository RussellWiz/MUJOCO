"""v2-style split teleoperation logic for PyBullet."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .robot import JAW_JOINT, POSITION_JOINTS, WRIST_JOINTS, PyBulletPsmRobot
except ImportError:
    from robot import JAW_JOINT, POSITION_JOINTS, WRIST_JOINTS, PyBulletPsmRobot


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm > max_norm > 0.0:
        return v * (max_norm / norm)
    return v


def rotation_vector_from_delta(R_delta: np.ndarray) -> np.ndarray:
    cos_theta = float((np.trace(R_delta) - 1.0) * 0.5)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    skew = np.array(
        [
            R_delta[2, 1] - R_delta[1, 2],
            R_delta[0, 2] - R_delta[2, 0],
            R_delta[1, 0] - R_delta[0, 1],
        ],
        dtype=float,
    )
    if theta < 1.0e-6:
        return 0.5 * skew
    return theta / (2.0 * np.sin(theta)) * skew


def gripper_to_jaw(gripper_deg: float, close_deg: float, open_deg: float, jaw_range: np.ndarray, invert: bool) -> float:
    if np.isclose(open_deg, close_deg):
        normalized = 0.0
    else:
        normalized = (float(gripper_deg) - float(close_deg)) / (float(open_deg) - float(close_deg))
    normalized = float(np.clip(normalized, 0.0, 1.0))
    if invert:
        normalized = 1.0 - normalized
    return float(jaw_range[0] + normalized * (jaw_range[1] - jaw_range[0]))


def svd_adaptive_dls(
    J: np.ndarray,
    task_velocity: np.ndarray,
    damping_min: float,
    damping_max: float,
    singular_value_safe: float,
) -> tuple[np.ndarray, float]:
    U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros(J.shape[1], dtype=float), 0.0

    filtered = np.zeros_like(singular_values)
    sigma_min = float(np.min(singular_values))
    safe = max(float(singular_value_safe), 1.0e-9)
    for i, sigma in enumerate(singular_values):
        if sigma < safe:
            ratio = 1.0 - sigma / safe
            damping = float(damping_min) + float(damping_max) * ratio * ratio
        else:
            damping = float(damping_min)
        filtered[i] = sigma / (sigma * sigma + damping * damping)
    return Vt.T @ (filtered * (U.T @ task_velocity)), sigma_min


@dataclass
class MotionDebug:
    target: np.ndarray
    tip: np.ndarray
    error_norm: float
    sigma_delta: np.ndarray
    world_increment: np.ndarray
    wrist_increment: np.ndarray
    sigma_min: float
    jaw_target: float


class PyBulletSplitController:
    """Split PSM controller driven by sigma.7 increments."""

    def __init__(self, args, robot: PyBulletPsmRobot, master_state) -> None:
        self.args = args
        self.robot = robot
        self.prev_sigma_position = master_state.position.copy()
        self.prev_sigma_rotation = master_state.rotation.copy()
        self.command_target = robot.tool_position()
        self.filtered_target = self.command_target.copy()
        self.dq_filtered = np.zeros(3, dtype=float)
        self.position_targets = dict(zip(POSITION_JOINTS, robot.joint_positions(POSITION_JOINTS)))
        self.wrist_targets = dict(zip(WRIST_JOINTS, robot.joint_positions(WRIST_JOINTS)))
        self.jaw_target = float(robot.joint_positions((JAW_JOINT,))[0])
        self.home_jaw = self.jaw_target
        self.max_qstep = np.array([args.max_step_yaw, args.max_step_pitch, args.max_step_insertion], dtype=float)

    def reset(self, master_state) -> None:
        self.prev_sigma_position = master_state.position.copy()
        self.prev_sigma_rotation = master_state.rotation.copy()
        self.command_target = self.robot.tool_position()
        self.filtered_target = self.command_target.copy()
        self.dq_filtered[:] = 0.0
        self.position_targets = dict(zip(POSITION_JOINTS, self.robot.joint_positions(POSITION_JOINTS)))
        self.wrist_targets = dict(zip(WRIST_JOINTS, self.robot.joint_positions(WRIST_JOINTS)))
        self.jaw_target = float(self.robot.joint_positions((JAW_JOINT,))[0])
        self.home_jaw = self.jaw_target

    def update(self, master_state) -> MotionDebug:
        sigma_delta = np.asarray(master_state.position - self.prev_sigma_position, dtype=float)
        self.prev_sigma_position = master_state.position.copy()
        sigma_delta = np.where(np.abs(sigma_delta) < float(self.args.sigma_deadband), 0.0, sigma_delta)

        world_increment = np.array(
            [
                float(self.args.scale_x) * sigma_delta[0],
                float(self.args.scale_y) * sigma_delta[1],
                float(self.args.scale_z) * sigma_delta[2],
            ],
            dtype=float,
        )
        world_increment = clip_norm(world_increment, float(self.args.max_world_increment))

        current_pos = self.robot.tool_position()
        self.command_target += world_increment
        self.command_target = current_pos + clip_norm(self.command_target - current_pos, 3.0 * float(self.args.max_target_error))
        self.filtered_target += float(self.args.target_smooth) * (self.command_target - self.filtered_target)

        pos_error = clip_norm(self.filtered_target - current_pos, float(self.args.max_target_error))
        if float(np.linalg.norm(pos_error)) > float(self.args.deadband_pos):
            J = self.robot.finite_difference_position_jacobian(POSITION_JOINTS, float(self.args.fd_eps))
            task_velocity = float(self.args.position_gain) * pos_error
            dq, sigma_min = svd_adaptive_dls(
                J,
                task_velocity,
                float(self.args.damping_min),
                float(self.args.damping_max),
                float(self.args.singular_value_safe),
            )
            self.dq_filtered += float(self.args.dq_smooth) * (dq - self.dq_filtered)
            dq_step = np.clip(self.dq_filtered * self.robot.dt, -self.max_qstep, self.max_qstep)
            position_values = np.array([self.position_targets[name] for name in POSITION_JOINTS], dtype=float)
            position_values = np.clip(position_values + dq_step, self.robot.position_limits[:, 0], self.robot.position_limits[:, 1])
            self.position_targets.update(zip(POSITION_JOINTS, position_values))
        else:
            sigma_min = 0.0
            self.dq_filtered *= 1.0 - float(self.args.dq_smooth)

        R_delta = master_state.rotation @ self.prev_sigma_rotation.T
        self.prev_sigma_rotation = master_state.rotation.copy()
        wrist_increment = rotation_vector_from_delta(R_delta) * float(self.args.wrist_scale)
        wrist_increment = np.clip(wrist_increment, -float(self.args.max_wrist_step), float(self.args.max_wrist_step))
        wrist_values = np.array([self.wrist_targets[name] for name in WRIST_JOINTS], dtype=float)
        wrist_values = np.clip(wrist_values + wrist_increment, self.robot.wrist_limits[:, 0], self.robot.wrist_limits[:, 1])
        self.wrist_targets.update(zip(WRIST_JOINTS, wrist_values))

        if self.args.jaw_mode == "locked":
            target = self.home_jaw
        else:
            target = gripper_to_jaw(
                master_state.gripper_deg,
                float(self.args.gripper_close_deg),
                float(self.args.gripper_open_deg),
                self.robot.jaw_limit,
                bool(self.args.jaw_invert),
            )
        self.jaw_target += float(self.args.jaw_smooth) * (target - self.jaw_target)
        jaw_current = float(self.robot.joint_positions((JAW_JOINT,))[0])
        jaw_step = float(np.clip(self.jaw_target - jaw_current, -float(self.args.max_jaw_step), float(self.args.max_jaw_step)))
        self.jaw_target = float(np.clip(jaw_current + jaw_step, self.robot.jaw_limit[0], self.robot.jaw_limit[1]))

        targets = {}
        targets.update(self.position_targets)
        targets.update(self.wrist_targets)
        targets[JAW_JOINT] = self.jaw_target
        self.robot.apply_position_targets(targets)

        return MotionDebug(
            target=self.filtered_target.copy(),
            tip=current_pos.copy(),
            error_norm=float(np.linalg.norm(self.filtered_target - current_pos)),
            sigma_delta=sigma_delta.copy(),
            world_increment=world_increment.copy(),
            wrist_increment=wrist_increment.copy(),
            sigma_min=float(sigma_min),
            jaw_target=float(self.jaw_target),
        )

