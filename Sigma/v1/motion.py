"""Teleop target mapping, IK, and actuator filtering constraints."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .robot import ARM_JOINTS, DISTAL_JOINTS, POSITION_JOINTS, PsmRobot, body_rotation, point_world_pos
except ImportError:
    from robot import ARM_JOINTS, DISTAL_JOINTS, POSITION_JOINTS, PsmRobot, body_rotation, point_world_pos


@dataclass
class TeleopCalibration:
    master_position: np.ndarray
    master_rotation: np.ndarray
    target_position_home: np.ndarray
    target_rotation_home: np.ndarray
    ctrl_home: np.ndarray


@dataclass
class RpyDebug:
    sigma: np.ndarray
    target: np.ndarray
    bias: np.ndarray


@dataclass
class MotionTargets:
    position: np.ndarray
    rotation: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    task_idle: bool
    rpy_debug: RpyDebug


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if max_norm > 0.0 and norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    trace_value = float(np.trace(rotation))
    cos_angle = float(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    if abs(math.pi - angle) < 1e-5:
        diag = np.diag(rotation)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        axis = np.asarray(axis, dtype=float)
        if axis[0] > 1e-6:
            axis[1] = math.copysign(axis[1], rotation[0, 1])
            axis[2] = math.copysign(axis[2], rotation[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = math.copysign(axis[2], rotation[1, 2])
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis, angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(angle))
    return axis, angle


def scaled_relative_rotation(reference: np.ndarray, current: np.ndarray, scale: float) -> np.ndarray:
    relative_rotation = reference.T @ current
    axis, angle = rotation_matrix_to_axis_angle(relative_rotation)
    return axis_angle_to_matrix(axis, scale * angle)


def reexpress_rotation(rotation: np.ndarray, frame_rotation: np.ndarray) -> np.ndarray:
    return frame_rotation @ rotation @ frame_rotation.T


def master_local_axis_angle(master_rotation: np.ndarray, master_rotation_home: np.ndarray, axis: str) -> float:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_indices[axis]
    first_perp = (axis_index + 1) % 3
    second_perp = (axis_index + 2) % 3
    relative_local_rotation = master_rotation_home.T @ master_rotation
    return float(math.atan2(relative_local_rotation[second_perp, first_perp], relative_local_rotation[first_perp, first_perp]))


def master_local_rotation_vector(master_rotation: np.ndarray, master_rotation_home: np.ndarray) -> np.ndarray:
    relative_local_rotation = master_rotation_home.T @ master_rotation
    axis, angle = rotation_matrix_to_axis_angle(relative_local_rotation)
    return axis * angle


def master_axis_delta(master_rotation: np.ndarray, master_rotation_home: np.ndarray, axis: str, method: str) -> float:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    if method == "rotvec":
        return float(master_local_rotation_vector(master_rotation, master_rotation_home)[axis_indices[axis]])
    return master_local_axis_angle(master_rotation, master_rotation_home, axis)


def damped_least_squares_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    task_dim = jacobian.shape[0]
    regularized = jacobian @ jacobian.T + float(damping) * np.eye(task_dim)
    return jacobian.T @ np.linalg.solve(regularized, error)


def weighted_full_pose_step(
    jacp: np.ndarray,
    jacr: np.ndarray,
    dof_columns: list[int],
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    damping: float,
    joint_damping_scale: np.ndarray,
    position_task_weight: float,
    orientation_task_weight: float,
    position_axis_weights: np.ndarray,
) -> np.ndarray:
    Jp = jacp[:, dof_columns]
    Jr = jacr[:, dof_columns]
    Wp = np.diag(position_axis_weights)
    reg = float(damping) * np.diag(joint_damping_scale)
    wp = float(max(position_task_weight, 1e-6))
    wr = float(max(orientation_task_weight, 1e-6))
    lhs = wp * (Jp.T @ Wp @ Jp) + wr * (Jr.T @ Jr) + reg
    rhs = wp * (Jp.T @ Wp @ position_error) + wr * (Jr.T @ orientation_error)
    return np.linalg.solve(lhs, rhs)


def clipped_home_offset(home: float, value: float, max_offset: float) -> float:
    return float(np.clip(value, home - max_offset, home + max_offset))


def apply_deadband(values: np.ndarray, deadband: float) -> np.ndarray:
    if deadband <= 0.0:
        return values
    result = values.copy()
    result[np.abs(result) < float(deadband)] = 0.0
    return result


def joint_limit_slowdown(values: np.ndarray, ctrl_range: np.ndarray, margin: float, min_scale: float) -> np.ndarray:
    if margin <= 0.0:
        return np.ones_like(values, dtype=float)
    low = ctrl_range[:, 0]
    high = ctrl_range[:, 1]
    dist_to_limit = np.minimum(values - low, high - values)
    return np.clip(dist_to_limit / float(margin), float(min_scale), 1.0).astype(float)


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    return u @ vh


def master_to_task_rotation(args) -> np.ndarray:
    yaw = math.radians(args.master_yaw_deg)
    pitch = math.radians(args.master_pitch_deg)
    roll = math.radians(args.master_roll_deg)
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_r, s_r = math.cos(roll), math.sin(roll)
    rotate_z = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rotate_y = np.array([[c_p, 0.0, s_p], [0.0, 1.0, 0.0], [-s_p, 0.0, c_p]], dtype=float)
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, c_r, -s_r], [0.0, s_r, c_r]], dtype=float)
    return rotate_z @ rotate_y @ rotate_x


def make_calibration(master_state, robot: PsmRobot) -> TeleopCalibration:
    robot.mujoco.mj_forward(robot.model, robot.data)
    return TeleopCalibration(
        master_position=master_state.position.copy(),
        master_rotation=master_state.rotation.copy(),
        target_position_home=point_world_pos(robot.data, robot.track_body_id).copy(),
        target_rotation_home=body_rotation(robot.data, robot.track_body_id).copy(),
        ctrl_home=robot.data.ctrl.copy(),
    )


class TargetMapper:
    def __init__(self, args, calibration: TeleopCalibration, target_rotation_home: np.ndarray) -> None:
        self.args = args
        self.calibration = calibration
        self.target_rotation_home = target_rotation_home.copy()
        self.translation_scale = np.array([args.scale_x, args.scale_y, args.scale_z], dtype=float)
        self.master_to_task = master_to_task_rotation(args)
        self.target_position_cmd = calibration.target_position_home.copy()
        self.target_position_filt = calibration.target_position_home.copy()
        self.target_rotation_cmd = target_rotation_home.copy()
        self.target_rotation_filt = target_rotation_home.copy()

    def reset(self, calibration: TeleopCalibration, target_rotation_home: np.ndarray) -> None:
        self.calibration = calibration
        self.target_rotation_home = target_rotation_home.copy()
        self.target_position_cmd = calibration.target_position_home.copy()
        self.target_position_filt = calibration.target_position_home.copy()
        self.target_rotation_cmd = target_rotation_home.copy()
        self.target_rotation_filt = target_rotation_home.copy()

    def update(self, master_state) -> tuple[np.ndarray, np.ndarray]:
        master_delta_local = self.translation_scale * (master_state.position - self.calibration.master_position)
        master_delta_task = self.master_to_task @ master_delta_local
        if self.args.position_frame == "tool-home":
            self.target_position_cmd = self.calibration.target_position_home + self.calibration.target_rotation_home @ master_delta_task
        else:
            self.target_position_cmd = self.calibration.target_position_home + master_delta_task
        alpha = float(self.args.target_smooth)
        self.target_position_filt = self.target_position_filt + alpha * (self.target_position_cmd - self.target_position_filt)

        if self.args.orientation_mode in ("full", "hybrid", "split"):
            if self.args.orientation_mode in ("hybrid", "split"):
                relative_master_raw = master_state.rotation @ self.calibration.master_rotation.T
                axis, angle = rotation_matrix_to_axis_angle(relative_master_raw)
                relative_master_task = axis_angle_to_matrix(axis, float(self.args.orientation_scale) * angle)
            else:
                relative_master_rotation = scaled_relative_rotation(
                    self.calibration.master_rotation,
                    master_state.rotation,
                    float(self.args.orientation_scale),
                )
                relative_master_task = reexpress_rotation(relative_master_rotation, self.master_to_task)
            self.target_rotation_cmd = self.target_rotation_home @ relative_master_task
            self.target_rotation_filt = self.target_rotation_filt + alpha * (self.target_rotation_cmd - self.target_rotation_filt)
            self.target_rotation_filt = orthonormalize(self.target_rotation_filt)
        return self.target_position_filt, self.target_rotation_filt


class IkSolver:
    def __init__(self, args, robot: PsmRobot) -> None:
        self.args = args
        self.robot = robot
        self.hybrid_joint_damping_scale = np.array(
            [1.0, 1.0, 1.0, max(float(args.ik_wrist_damping_scale), 1.0), max(float(args.ik_wrist_damping_scale), 1.0), max(float(args.ik_wrist_damping_scale), 1.0)],
            dtype=float,
        )
        self.hybrid_position_axis_weights = np.array(
            [max(float(args.ik_pos_weight_x), 1e-6), max(float(args.ik_pos_weight_y), 1e-6), max(float(args.ik_pos_weight_z), 1e-6)],
            dtype=float,
        )

    def solve(self, position_error: np.ndarray, orientation_error: np.ndarray, task_idle: bool) -> np.ndarray:
        if task_idle:
            return np.zeros(len(ARM_JOINTS), dtype=float)
        jacp, jacr = self.robot.jacobian(self.robot.track_body_id, self.robot.track_position())
        if self.args.orientation_mode == "split":
            dq_position = damped_least_squares_step(jacp[:, self.robot.position_dof_columns], position_error, self.args.damping_pos)
            _, jacr_ori = self.robot.jacobian(self.robot.orientation_body_id, self.robot.orientation_position())
            dq_distal = damped_least_squares_step(jacr_ori[:, self.robot.distal_dof_columns], orientation_error, self.args.damping_full)
            dq = np.zeros(len(ARM_JOINTS), dtype=float)
            dq[:3] = dq_position
            dq[3:] = float(self.args.split_distal_step_scale) * dq_distal
            return dq
        if self.args.orientation_mode == "hybrid":
            return weighted_full_pose_step(
                jacp=jacp,
                jacr=jacr,
                dof_columns=self.robot.arm_dof_columns,
                position_error=position_error,
                orientation_error=orientation_error,
                damping=self.args.damping_full,
                joint_damping_scale=self.hybrid_joint_damping_scale,
                position_task_weight=self.args.ik_pos_weight,
                orientation_task_weight=self.args.ik_ori_weight,
                position_axis_weights=self.hybrid_position_axis_weights,
            )
        if self.args.orientation_mode == "full":
            task_jacobian = np.vstack([jacp[:, self.robot.arm_dof_columns], jacr[:, self.robot.arm_dof_columns]])
            task_error = np.concatenate([position_error, orientation_error])
            return damped_least_squares_step(task_jacobian, task_error, self.args.damping_full)

        dq_position = damped_least_squares_step(jacp[:, self.robot.position_dof_columns], position_error, self.args.damping_pos)
        dq = np.zeros(len(ARM_JOINTS), dtype=float)
        dq[:3] = dq_position
        return dq


class MotionController:
    def __init__(self, args, robot: PsmRobot, calibration: TeleopCalibration, target_rotation_home: np.ndarray) -> None:
        self.args = args
        self.robot = robot
        self.calibration = calibration
        self.mapper = TargetMapper(args, calibration, target_rotation_home)
        self.solver = IkSolver(args, robot)
        self.dq_filt = np.zeros(len(ARM_JOINTS), dtype=float)
        self.ctrl_filt = robot.data.ctrl[robot.arm_actuators].copy()
        self.max_ctrl_step = np.array([args.max_ctrl_step_pos, args.max_ctrl_step_pos, args.max_ctrl_step_ins, args.max_ctrl_step_roll, args.max_ctrl_step_pos, args.max_ctrl_step_pos], dtype=float)
        self.max_rpy_offset = np.array([math.radians(args.max_rpy_yaw_deg), math.radians(args.max_rpy_pitch_deg), math.radians(args.max_rpy_roll_deg)], dtype=float)
        self.rpy_debug = RpyDebug(
            sigma=np.zeros(3, dtype=float),
            target=calibration.ctrl_home[robot.arm_actuators][[0, 1, 3]].copy(),
            bias=np.zeros(2, dtype=float),
        )

    def reset(self, calibration: TeleopCalibration, target_rotation_home: np.ndarray) -> None:
        self.calibration = calibration
        self.mapper.reset(calibration, target_rotation_home)
        self.dq_filt[:] = 0.0
        self.ctrl_filt = self.robot.data.ctrl[self.robot.arm_actuators].copy()
        self.rpy_debug.sigma[:] = 0.0
        self.rpy_debug.target = calibration.ctrl_home[self.robot.arm_actuators][[0, 1, 3]].copy()
        self.rpy_debug.bias[:] = 0.0

    def compute_errors(self, master_state) -> tuple[np.ndarray, np.ndarray, bool]:
        target_position, target_rotation = self.mapper.update(master_state)
        current_position = self.robot.track_position()
        current_rotation = self.robot.orientation_rotation()
        position_error = clip_norm(self.args.position_gain * (target_position - current_position), self.args.max_position_error)

        if self.args.orientation_mode in ("full", "hybrid", "split"):
            orientation_error = clip_norm(
                self.args.orientation_gain * rotation_error(current_rotation, target_rotation),
                self.args.max_orientation_error,
            )
            task_idle = float(np.linalg.norm(position_error)) < float(self.args.deadband_pos) and float(np.linalg.norm(orientation_error)) < float(self.args.deadband_rot)
        elif self.args.orientation_mode == "roll":
            roll_delta = master_axis_delta(master_state.rotation, self.calibration.master_rotation, self.args.master_roll_axis, self.args.rpy_input_method)
            roll_target = float(self.calibration.ctrl_home[self.robot.arm_actuators[3]] + self.args.roll_scale * roll_delta)
            roll_target = clipped_home_offset(float(self.calibration.ctrl_home[self.robot.arm_actuators[3]]), roll_target, float(self.max_rpy_offset[2]))
            roll_target = float(np.clip(roll_target, self.robot.arm_ctrl_range[3, 0], self.robot.arm_ctrl_range[3, 1]))
            orientation_error = np.array([roll_target - float(self.robot.data.ctrl[self.robot.arm_actuators[3]])], dtype=float)
            task_idle = float(np.linalg.norm(position_error)) < float(self.args.deadband_pos) and abs(float(orientation_error[0])) < float(self.args.deadband_rot)
        elif self.args.orientation_mode == "rpy":
            yaw_target, pitch_target, roll_target, deltas = self.rpy_targets(master_state)
            orientation_error = np.array(
                [
                    yaw_target - float(self.robot.data.ctrl[self.robot.arm_actuators[0]]),
                    pitch_target - float(self.robot.data.ctrl[self.robot.arm_actuators[1]]),
                    roll_target - float(self.robot.data.ctrl[self.robot.arm_actuators[3]]),
                ],
                dtype=float,
            )
            self.rpy_debug.sigma = deltas
            self.rpy_debug.target = np.array([yaw_target, pitch_target, roll_target], dtype=float)
            task_idle = float(np.linalg.norm(position_error)) < float(self.args.deadband_pos) and float(np.linalg.norm(orientation_error)) < float(self.args.deadband_rot)
        else:
            orientation_error = np.zeros(1, dtype=float)
            task_idle = float(np.linalg.norm(position_error)) < float(self.args.deadband_pos)
        return position_error, orientation_error, task_idle

    def rpy_targets(self, master_state) -> tuple[float, float, float, np.ndarray]:
        yaw_delta = master_axis_delta(master_state.rotation, self.calibration.master_rotation, self.args.master_yaw_axis, self.args.rpy_input_method)
        pitch_delta = master_axis_delta(master_state.rotation, self.calibration.master_rotation, self.args.master_pitch_axis, self.args.rpy_input_method)
        roll_delta = master_axis_delta(master_state.rotation, self.calibration.master_rotation, self.args.master_roll_axis, self.args.rpy_input_method)
        yaw_target = float(self.calibration.ctrl_home[self.robot.arm_actuators[0]] + self.args.yaw_scale * yaw_delta)
        pitch_target = float(self.calibration.ctrl_home[self.robot.arm_actuators[1]] + self.args.pitch_scale * pitch_delta)
        roll_target = float(self.calibration.ctrl_home[self.robot.arm_actuators[3]] + self.args.roll_scale * roll_delta)
        yaw_target = clipped_home_offset(float(self.calibration.ctrl_home[self.robot.arm_actuators[0]]), yaw_target, float(self.max_rpy_offset[0]))
        pitch_target = clipped_home_offset(float(self.calibration.ctrl_home[self.robot.arm_actuators[1]]), pitch_target, float(self.max_rpy_offset[1]))
        roll_target = clipped_home_offset(float(self.calibration.ctrl_home[self.robot.arm_actuators[3]]), roll_target, float(self.max_rpy_offset[2]))
        return yaw_target, pitch_target, roll_target, np.array([yaw_delta, pitch_delta, roll_delta], dtype=float)

    def update_arm(self, master_state) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        position_error, orientation_error, task_idle = self.compute_errors(master_state)
        dq = self.solver.solve(position_error, orientation_error, task_idle)
        dq = clip_norm(dq, self.args.max_joint_step)
        dq = apply_deadband(dq, float(self.args.dq_deadband))
        dq *= joint_limit_slowdown(
            self.robot.data.ctrl[self.robot.arm_actuators],
            self.robot.arm_ctrl_range,
            float(self.args.limit_slowdown_margin),
            float(self.args.limit_slowdown_min_scale),
        )
        if self.args.orientation_mode == "split" and self.args.split_distal_smooth is not None:
            self.dq_filt[:3] = self.dq_filt[:3] + float(self.args.dq_smooth) * (dq[:3] - self.dq_filt[:3])
            self.dq_filt[3:] = self.dq_filt[3:] + float(self.args.split_distal_smooth) * (dq[3:] - self.dq_filt[3:])
        else:
            self.dq_filt = self.dq_filt + float(self.args.dq_smooth) * (dq - self.dq_filt)
        if self.args.orientation_mode in ("hybrid", "split"):
            self.dq_filt[2] = self.dq_filt[2] + float(self.args.hybrid_dq_smooth_ins) * (dq[2] - self.dq_filt[2])
            self.dq_filt[2] *= float(self.args.hybrid_insertion_dq_scale)

        measured_arm_qpos = self.robot.arm_qpos()
        insertion_lag = abs(float(measured_arm_qpos[2] - self.robot.data.ctrl[self.robot.arm_actuators[2]]))
        if self.args.orientation_mode not in ("hybrid", "split") or self.args.hybrid_use_servo_lag_guard:
            if insertion_lag > float(self.args.max_insertion_servo_lag):
                self.dq_filt[2] = 0.0
                self.ctrl_filt[2] = measured_arm_qpos[2]
        if self.args.orientation_mode in ("hybrid", "split"):
            desired_ctrl = self.robot.data.ctrl[self.robot.arm_actuators].copy()
        else:
            desired_ctrl = measured_arm_qpos.copy()
        desired_ctrl += self.dq_filt
        desired_ctrl = self.apply_historical_orientation_targets(master_state, desired_ctrl, measured_arm_qpos)
        desired_ctrl = np.clip(desired_ctrl, self.robot.arm_ctrl_range[:, 0], self.robot.arm_ctrl_range[:, 1])
        self.ctrl_filt = self.ctrl_filt + float(self.args.ctrl_smooth) * (desired_ctrl - self.ctrl_filt)
        delta = np.clip(self.ctrl_filt - self.robot.data.ctrl[self.robot.arm_actuators], -self.max_ctrl_step, self.max_ctrl_step)
        self.robot.data.ctrl[self.robot.arm_actuators] = np.clip(
            self.robot.data.ctrl[self.robot.arm_actuators] + delta,
            self.robot.arm_ctrl_range[:, 0],
            self.robot.arm_ctrl_range[:, 1],
        )
        return orientation_error, insertion_lag, measured_arm_qpos, self.mapper.target_position_filt

    def apply_historical_orientation_targets(self, master_state, desired_ctrl: np.ndarray, measured_arm_qpos: np.ndarray) -> np.ndarray:
        if self.args.orientation_mode == "none":
            desired_ctrl[3:] = self.calibration.ctrl_home[self.robot.arm_actuators][3:]
        elif self.args.orientation_mode == "roll":
            roll_delta = master_axis_delta(master_state.rotation, self.calibration.master_rotation, self.args.master_roll_axis, self.args.rpy_input_method)
            roll_target = float(self.calibration.ctrl_home[self.robot.arm_actuators[3]] + self.args.roll_scale * roll_delta)
            roll_target = clipped_home_offset(float(self.calibration.ctrl_home[self.robot.arm_actuators[3]]), roll_target, float(self.max_rpy_offset[2]))
            desired_ctrl[3] = roll_target
            desired_ctrl[4:] = self.calibration.ctrl_home[self.robot.arm_actuators][4:]
        elif self.args.orientation_mode == "rpy":
            yaw_target, pitch_target, roll_target, deltas = self.rpy_targets(master_state)
            yaw_pitch_target = np.array([yaw_target, pitch_target], dtype=float)
            if self.args.rpy_yaw_pitch_mode == "direct":
                yaw_pitch_bias = np.clip(yaw_pitch_target - measured_arm_qpos[:2], -float(self.args.max_rpy_bias_step), float(self.args.max_rpy_bias_step))
                desired_ctrl[:2] = measured_arm_qpos[:2] + yaw_pitch_bias
            elif self.args.rpy_yaw_pitch_mode == "blend":
                yaw_pitch_step = np.clip(yaw_pitch_target - measured_arm_qpos[:2], -float(self.args.max_rpy_bias_step), float(self.args.max_rpy_bias_step))
                yaw_pitch_direct = measured_arm_qpos[:2] + yaw_pitch_step
                direct_weight = float(np.clip(self.args.rpy_direct_weight, 0.0, 1.0))
                position_desired = desired_ctrl[:2].copy()
                desired_ctrl[:2] = (1.0 - direct_weight) * position_desired + direct_weight * yaw_pitch_direct
                yaw_pitch_bias = desired_ctrl[:2] - position_desired
            else:
                yaw_pitch_bias = np.clip(float(self.args.rpy_bias_gain) * (yaw_pitch_target - desired_ctrl[:2]), -float(self.args.max_rpy_bias_step), float(self.args.max_rpy_bias_step))
                desired_ctrl[:2] += yaw_pitch_bias
            self.rpy_debug.sigma = deltas
            self.rpy_debug.target = np.array([yaw_target, pitch_target, roll_target], dtype=float)
            self.rpy_debug.bias = yaw_pitch_bias.copy()
            desired_ctrl[3] = roll_target
            desired_ctrl[4:] = self.calibration.ctrl_home[self.robot.arm_actuators][4:]
        return desired_ctrl
