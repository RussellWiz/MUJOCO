"""Teleoperate the MuJoCo dVRK PSM with a Force Dimension sigma.7.

Run from the project root:
    python Sigma/sigma7_psm_teleop.py

If the SDK DLLs are not on PATH, pass the directory containing
`dhd64.dll` and `drd64.dll`:
    python Sigma/sigma7_psm_teleop.py --sdk-bin "C:\\Program Files\\Force Dimension\\sdk-3.17.6\\bin"

Controls:
    q       quit
    r       recalibrate the sigma.7 neutral pose and reset teleop targets
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from force_dimension_sdk import ForceDimensionSDK

try:
    from config import DEFAULT_PROFILE, TELEOP_CONFIGS
except ImportError:
    DEFAULT_PROFILE = None
    TELEOP_CONFIGS = {}


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"

ARM_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
POSITION_JOINTS = ("yaw", "pitch", "insertion")
DISTAL_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")
JAW_JOINT = "jaw"
TRACK_BODY = "tool_tip"
TRACK_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0], dtype=float)


@dataclass
class TeleopCalibration:
    master_position: np.ndarray
    master_rotation: np.ndarray
    target_position_home: np.ndarray
    target_rotation_home: np.ndarray
    ctrl_home: np.ndarray


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if max_norm > 0.0 and norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def body_rotation(data, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def point_world_pos(data, body_id: int, offset_local: np.ndarray) -> np.ndarray:
    rotation = body_rotation(data, body_id)
    return data.xpos[body_id] + rotation @ offset_local


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
    """Re-express a relative rotation in a rotated task frame."""
    return frame_rotation @ rotation @ frame_rotation.T


def master_local_axis_angle(master_rotation: np.ndarray, master_rotation_home: np.ndarray, axis: str) -> float:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_indices[axis]
    first_perp = (axis_index + 1) % 3
    second_perp = (axis_index + 2) % 3
    relative_local_rotation = master_rotation_home.T @ master_rotation
    return float(
        math.atan2(
            relative_local_rotation[second_perp, first_perp],
            relative_local_rotation[first_perp, first_perp],
        )
    )


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


def seed_home_configuration(mujoco, model, data, arm_actuators: list[int], jaw_actuator: int, home_insertion: float) -> None:
    arm_ctrl_range = model.actuator_ctrlrange[arm_actuators]
    jaw_ctrl_range = model.actuator_ctrlrange[jaw_actuator]

    desired = np.array([0.0, 0.0, home_insertion, 0.0, 0.0, 0.0], dtype=float)
    desired = np.clip(desired, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
    data.ctrl[arm_actuators] = desired
    data.ctrl[jaw_actuator] = float(jaw_ctrl_range[0])

    for _ in range(400):
        mujoco.mj_step(model, data)


def make_calibration(device: ForceDimensionSDK, mujoco, model, data, body_id: int) -> TeleopCalibration:
    master_state = device.read_state()
    mujoco.mj_forward(model, data)
    return TeleopCalibration(
        master_position=master_state.position.copy(),
        master_rotation=master_state.rotation.copy(),
        target_position_home=point_world_pos(data, body_id, TRACK_OFFSET_LOCAL).copy(),
        target_rotation_home=body_rotation(data, body_id).copy(),
        ctrl_home=data.ctrl.copy(),
    )


def keyboard_command() -> str | None:
    if sys.platform != "win32":
        return None

    import msvcrt

    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0") and msvcrt.kbhit():
        key = msvcrt.getwch()
    return key.lower()


def gripper_to_jaw(gripper_deg: float, close_deg: float, open_deg: float, jaw_range: np.ndarray, invert: bool) -> float:
    if math.isclose(open_deg, close_deg):
        normalized = 0.0
    else:
        normalized = (gripper_deg - close_deg) / (open_deg - close_deg)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    if invert:
        normalized = 1.0 - normalized
    return float(jaw_range[0] + normalized * (jaw_range[1] - jaw_range[0]))


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
    scale = np.clip(dist_to_limit / float(margin), float(min_scale), 1.0)
    return scale.astype(float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate the MuJoCo dVRK PSM with a Force Dimension sigma.7")
    parser.add_argument("--config-profile", default=None, help="Load defaults from Sigma/config.py TELEOP_CONFIGS")
    parser.add_argument("--list-config-profiles", action="store_true", help="List available config profiles and exit")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo PSM XML")
    parser.add_argument("--sdk-bin", default=None, help="Directory containing dhd64.dll and drd64.dll")
    parser.add_argument("--no-drd-init", action="store_true", help="Skip DRD auto-init and only use DHD open/read calls")
    parser.add_argument("--orientation-mode", choices=("none", "roll", "rpy", "full", "hybrid", "split"), default="none", help="How sigma.7 orientation drives the PSM")
    parser.add_argument("--track-body", default=TRACK_BODY, help="MuJoCo body used as the Cartesian control point")
    parser.add_argument("--orientation-body", default=None, help="MuJoCo body used as the orientation control point; defaults to --track-body")
    parser.add_argument("--position-frame", choices=("world", "tool-home"), default="world", help="Frame used to map sigma.7 translation into MuJoCo target motion")
    parser.add_argument("--scale-x", type=float, default=0.55, help="Sigma X meters to teleop X meters")
    parser.add_argument("--scale-y", type=float, default=0.55, help="Sigma Y meters to teleop Y meters")
    parser.add_argument("--scale-z", type=float, default=0.25, help="Sigma Z meters to teleop Z meters")
    parser.add_argument("--master-yaw-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local Z before mapping")
    parser.add_argument("--master-pitch-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local Y before mapping")
    parser.add_argument("--master-roll-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local X before mapping")
    parser.add_argument("--home-insertion", type=float, default=0.12, help="Initial PSM insertion target in meters before calibration")
    parser.add_argument("--damping-pos", type=float, default=3e-4, help="Damped least-squares IK damping for position-only mode")
    parser.add_argument("--damping-full", type=float, default=3e-3, help="Damped least-squares IK damping for full-pose mode")
    parser.add_argument("--ik-pos-weight", type=float, default=12.0, help="Hybrid IK task weight for Cartesian position")
    parser.add_argument("--ik-ori-weight", type=float, default=1.0, help="Hybrid IK task weight for orientation")
    parser.add_argument("--ik-pos-weight-x", type=float, default=4.0, help="Hybrid IK extra task weight on world X")
    parser.add_argument("--ik-pos-weight-y", type=float, default=4.0, help="Hybrid IK extra task weight on world Y")
    parser.add_argument("--ik-pos-weight-z", type=float, default=1.0, help="Hybrid IK extra task weight on world Z")
    parser.add_argument("--ik-wrist-damping-scale", type=float, default=2.0, help="Hybrid IK extra damping on roll/wrist_pitch/wrist_yaw")
    parser.add_argument("--position-gain", type=float, default=1.0, help="Task-space gain applied to position error")
    parser.add_argument("--orientation-gain", type=float, default=0.35, help="Task-space gain applied to orientation error in full mode")
    parser.add_argument("--orientation-scale", type=float, default=1.0, help="Scale factor for sigma orientation relative motion")
    parser.add_argument("--master-yaw-axis", choices=("x", "y", "z"), default="z", help="sigma.7 local rotation axis mapped to the PSM yaw joint in rpy mode")
    parser.add_argument("--master-pitch-axis", choices=("x", "y", "z"), default="y", help="sigma.7 local rotation axis mapped to the PSM pitch joint in rpy mode")
    parser.add_argument("--master-roll-axis", choices=("x", "y", "z"), default="z", help="sigma.7 local rotation axis mapped to the PSM roll joint in roll mode")
    parser.add_argument("--rpy-input-method", choices=("local-angle", "rotvec"), default="local-angle", help="How sigma.7 relative rotation is decomposed for rpy mode")
    parser.add_argument("--yaw-scale", type=float, default=0.35, help="Scale sigma yaw to PSM yaw bias in rpy mode")
    parser.add_argument("--pitch-scale", type=float, default=0.35, help="Scale sigma pitch to PSM pitch bias in rpy mode")
    parser.add_argument("--roll-scale", type=float, default=1.0, help="Scale sigma roll to PSM roll in roll mode")
    parser.add_argument("--rpy-yaw-pitch-mode", choices=("blend", "direct", "bias"), default="blend", help="How rpy mode applies sigma yaw/pitch to the PSM main-chain joints")
    parser.add_argument("--rpy-direct-weight", type=float, default=0.35, help="Blend weight for direct yaw/pitch targets in rpy blend mode")
    parser.add_argument("--max-rpy-yaw-deg", type=float, default=20.0, help="Max yaw offset from calibration home in rpy mode")
    parser.add_argument("--max-rpy-pitch-deg", type=float, default=20.0, help="Max pitch offset from calibration home in rpy mode")
    parser.add_argument("--max-rpy-roll-deg", type=float, default=45.0, help="Max roll offset from calibration home in rpy/roll mode")
    parser.add_argument("--rpy-bias-gain", type=float, default=0.25, help="Blend gain for yaw/pitch joint bias after position IK in rpy mode")
    parser.add_argument("--max-rpy-bias-step", type=float, default=0.006, help="Max yaw/pitch bias step per frame in rpy mode")
    parser.add_argument("--debug-rpy", action="store_true", help="Print sigma rpy deltas and mapped PSM yaw/pitch/roll targets")
    parser.add_argument("--max-position-error", type=float, default=0.012, help="Max Cartesian correction per frame in meters")
    parser.add_argument("--max-orientation-error", type=float, default=0.20, help="Max orientation error vector norm per frame in radians")
    parser.add_argument("--max-joint-step", type=float, default=0.025, help="Max joint-space IK step norm per frame")
    parser.add_argument("--deadband-pos", type=float, default=2e-4, help="Ignore smaller position errors in meters")
    parser.add_argument("--deadband-rot", type=float, default=2e-3, help="Ignore smaller rotation errors in radians")
    parser.add_argument("--target-smooth", type=float, default=0.45, help="Low-pass alpha for target position/orientation (0..1)")
    parser.add_argument("--dq-smooth", type=float, default=0.55, help="Low-pass alpha for joint-space IK steps (0..1)")
    parser.add_argument("--dq-deadband", type=float, default=0.0, help="Zero IK joint steps smaller than this value")
    parser.add_argument("--split-distal-smooth", type=float, default=None, help="Optional split-mode low-pass alpha for roll/wrist_pitch/wrist_yaw dq")
    parser.add_argument("--ctrl-smooth", type=float, default=0.55, help="Low-pass alpha for actuator targets (0..1)")
    parser.add_argument("--max-ctrl-step-pos", type=float, default=0.018, help="Max per-frame change for yaw/pitch/wrist rotary joints in radians")
    parser.add_argument("--max-ctrl-step-ins", type=float, default=0.0006, help="Max per-frame change for insertion in meters")
    parser.add_argument("--max-ctrl-step-roll", type=float, default=0.006, help="Max per-frame change for roll in radians")
    parser.add_argument("--max-insertion-servo-lag", type=float, default=0.008, help="Freeze insertion IK when measured qpos and actuator target differ by more than this many meters")
    parser.add_argument("--hybrid-insertion-dq-scale", type=float, default=0.35, help="Scale hybrid IK insertion dq to reduce chatter")
    parser.add_argument("--hybrid-dq-smooth-ins", type=float, default=0.18, help="Extra low-pass alpha for hybrid insertion dq")
    parser.add_argument("--hybrid-use-servo-lag-guard", action="store_true", help="Apply insertion servo-lag guard in hybrid mode")
    parser.add_argument("--split-distal-step-scale", type=float, default=1.0, help="Scale split-mode distal orientation dq before filtering")
    parser.add_argument("--limit-slowdown-margin", type=float, default=0.08, help="Slow joints near actuator limits within this margin")
    parser.add_argument("--limit-slowdown-min-scale", type=float, default=0.15, help="Minimum joint step scale near actuator limits")
    parser.add_argument("--jaw-mode", choices=("locked", "follow"), default="locked", help="Lock jaw at calibration home or follow sigma.7 gripper")
    parser.add_argument("--jaw-lock-value", type=float, default=None, help="Optional fixed jaw actuator target in radians when --jaw-mode locked")
    parser.add_argument("--gripper-close-deg", type=float, default=0.0, help="sigma.7 gripper angle treated as closed")
    parser.add_argument("--gripper-open-deg", type=float, default=30.0, help="sigma.7 gripper angle treated as open")
    parser.add_argument("--jaw-invert", action="store_true", help="Invert sigma.7 gripper to jaw opening direction")
    parser.add_argument("--jaw-smooth", type=float, default=0.25, help="Low-pass alpha for jaw actuator target in follow mode")
    parser.add_argument("--jaw-deadband", type=float, default=0.002, help="Ignore smaller jaw actuator target changes in radians")
    parser.add_argument("--max-ctrl-step-jaw", type=float, default=0.02, help="Max per-frame jaw actuator target change in radians")
    parser.add_argument("--print-every", type=float, default=0.25, help="Status print period in seconds")
    parser.add_argument("--log-csv", default=None, help="Optional CSV path for teleop debug logs")
    parser.add_argument("--log-every", type=float, default=None, help="CSV log period in seconds; defaults to --print-every")
    profile_args, _ = parser.parse_known_args()
    if profile_args.list_config_profiles:
        if TELEOP_CONFIGS:
            print("Available config profiles:")
            for name in sorted(TELEOP_CONFIGS):
                default_mark = " (default)" if name == DEFAULT_PROFILE else ""
                print(f"  {name}{default_mark}")
        else:
            print("No config profiles found.")
        raise SystemExit(0)

    profile_name = profile_args.config_profile
    if profile_name:
        if profile_name not in TELEOP_CONFIGS:
            available = ", ".join(sorted(TELEOP_CONFIGS)) or "none"
            raise ValueError(f"Unknown config profile '{profile_name}'. Available profiles: {available}")
        parser.set_defaults(**TELEOP_CONFIGS[profile_name])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The active Python environment does not have the 'mujoco' package. "
            "Install it with 'pip install mujoco' or run this script from the Python "
            "environment you use for the existing MuJoCo examples."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    track_body_name = str(args.track_body)
    track_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)
    if track_body_id < 0:
        raise ValueError(f"MuJoCo body not found: {track_body_name}")
    orientation_body_name = str(args.orientation_body) if args.orientation_body else track_body_name
    orientation_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, orientation_body_name)
    if orientation_body_id < 0:
        raise ValueError(f"MuJoCo body not found: {orientation_body_name}")

    arm_actuators = actuator_ids(mujoco, model, ARM_JOINTS)
    jaw_actuator = actuator_ids(mujoco, model, (JAW_JOINT,))[0]
    position_dof_columns = controlled_dof_columns(mujoco, model, POSITION_JOINTS)
    distal_dof_columns = controlled_dof_columns(mujoco, model, DISTAL_JOINTS)
    arm_dof_columns = controlled_dof_columns(mujoco, model, ARM_JOINTS)

    mujoco.mj_resetData(model, data)
    seed_home_configuration(mujoco, model, data, arm_actuators, jaw_actuator, args.home_insertion)
    mujoco.mj_forward(model, data)

    arm_ctrl_range = model.actuator_ctrlrange[arm_actuators]
    jaw_ctrl_range = model.actuator_ctrlrange[jaw_actuator]
    translation_scale = np.array([args.scale_x, args.scale_y, args.scale_z], dtype=float)

    yaw = math.radians(args.master_yaw_deg)
    pitch = math.radians(args.master_pitch_deg)
    roll = math.radians(args.master_roll_deg)
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    c_r, s_r = math.cos(roll), math.sin(roll)
    rotate_z = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rotate_y = np.array([[c_p, 0.0, s_p], [0.0, 1.0, 0.0], [-s_p, 0.0, c_p]], dtype=float)
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, c_r, -s_r], [0.0, s_r, c_r]], dtype=float)
    master_to_task_rotation = rotate_z @ rotate_y @ rotate_x

    max_ctrl_step = np.array(
        [
            args.max_ctrl_step_pos,
            args.max_ctrl_step_pos,
            args.max_ctrl_step_ins,
            args.max_ctrl_step_roll,
            args.max_ctrl_step_pos,
            args.max_ctrl_step_pos,
        ],
        dtype=float,
    )
    max_rpy_offset = np.array(
        [
            math.radians(args.max_rpy_yaw_deg),
            math.radians(args.max_rpy_pitch_deg),
            math.radians(args.max_rpy_roll_deg),
        ],
        dtype=float,
    )
    hybrid_joint_damping_scale = np.array(
        [1.0, 1.0, 1.0, max(float(args.ik_wrist_damping_scale), 1.0), max(float(args.ik_wrist_damping_scale), 1.0), max(float(args.ik_wrist_damping_scale), 1.0)],
        dtype=float,
    )
    hybrid_position_axis_weights = np.array(
        [
            max(float(args.ik_pos_weight_x), 1e-6),
            max(float(args.ik_pos_weight_y), 1e-6),
            max(float(args.ik_pos_weight_z), 1e-6),
        ],
        dtype=float,
    )

    device = ForceDimensionSDK(args.sdk_bin, use_drd_init=not args.no_drd_init)
    log_file = None
    log_writer = None

    try:
        print("Opening sigma.7...")
        device.open()

        calibration = make_calibration(device, mujoco, model, data, track_body_id)
        target_rotation_home = body_rotation(data, orientation_body_id).copy()
        target_position_cmd = calibration.target_position_home.copy()
        target_position_filt = calibration.target_position_home.copy()
        target_rotation_cmd = target_rotation_home.copy()
        target_rotation_filt = target_rotation_home.copy()
        dq_filt = np.zeros(len(ARM_JOINTS), dtype=float)
        ctrl_filt = data.ctrl[arm_actuators].copy()
        jaw_filt = float(data.ctrl[jaw_actuator])
        jaw_target = jaw_filt
        rpy_debug = {
            "sigma": np.zeros(3, dtype=float),
            "target": calibration.ctrl_home[arm_actuators][[0, 1, 3]].copy(),
            "bias": np.zeros(2, dtype=float),
        }

        print(f"Loaded MuJoCo XML: {xml_path}")
        print(f"SDK DLL directory: {device.sdk_bin if device.sdk_bin else 'PATH'}")
        print(
            "Teleop ready. "
            f"orientation-mode={args.orientation_mode}, "
            f"position-frame={args.position_frame}, "
            f"track-body={track_body_name}, "
            f"orientation-body={orientation_body_name}, "
            f"jaw-mode={args.jaw_mode}. "
            "Press r to recalibrate, q to quit."
        )
        if args.orientation_mode == "none":
            print("Engineering mode: distal roll/wrist are held at calibration home; x/y/z are solved by yaw/pitch/insertion.")

        if args.log_csv:
            log_path = Path(args.log_csv)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.DictWriter(
                log_file,
                fieldnames=[
                    "t",
                    "orientation_mode",
                    "track_body",
                    "orientation_body",
                    "jaw_mode",
                    "rpy_mode",
                    "rpy_input_method",
                    "master_yaw_axis",
                    "master_pitch_axis",
                    "master_roll_axis",
                    "sigma_x",
                    "sigma_y",
                    "sigma_z",
                    "target_x",
                    "target_y",
                    "target_z",
                    "tip_x",
                    "tip_y",
                    "tip_z",
                    "err_m",
                    "ori_err_norm",
                    "ins_q",
                    "ins_ctrl",
                    "ins_lag",
                    "jaw_deg",
                    "jaw_target",
                    "jaw_ctrl",
                    "freq_khz",
                    "rpy_in_yaw_deg",
                    "rpy_in_pitch_deg",
                    "rpy_in_roll_deg",
                    "rpy_tgt_yaw",
                    "rpy_tgt_pitch",
                    "rpy_tgt_roll",
                    "rpy_q_yaw",
                    "rpy_q_pitch",
                    "rpy_q_roll",
                    "rpy_bias_yaw",
                    "rpy_bias_pitch",
                    "ctrl_yaw",
                    "ctrl_pitch",
                    "ctrl_insertion",
                    "ctrl_roll",
                    "ctrl_wrist_pitch",
                    "ctrl_wrist_yaw",
                ],
            )
            log_writer.writeheader()
            print(f"CSV logging: {log_path}")

        last_print = time.perf_counter()
        last_log = last_print
        log_every = float(args.print_every if args.log_every is None else args.log_every)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                command = keyboard_command()
                if command == "q":
                    break
                if command == "r":
                    data.ctrl[:] = calibration.ctrl_home
                    for _ in range(150):
                        mujoco.mj_step(model, data)
                    calibration = make_calibration(device, mujoco, model, data, track_body_id)
                    target_rotation_home = body_rotation(data, orientation_body_id).copy()
                    target_position_cmd = calibration.target_position_home.copy()
                    target_position_filt = calibration.target_position_home.copy()
                    target_rotation_cmd = target_rotation_home.copy()
                    target_rotation_filt = target_rotation_home.copy()
                    dq_filt[:] = 0.0
                    ctrl_filt = data.ctrl[arm_actuators].copy()
                    jaw_filt = float(data.ctrl[jaw_actuator])
                    jaw_target = jaw_filt
                    rpy_debug["sigma"][:] = 0.0
                    rpy_debug["target"] = calibration.ctrl_home[arm_actuators][[0, 1, 3]].copy()
                    rpy_debug["bias"][:] = 0.0
                    print("\nRecalibrated neutral pose.")

                master_state = device.read_state()
                device.set_zero_force()

                master_delta_local = translation_scale * (master_state.position - calibration.master_position)
                master_delta_task = master_to_task_rotation @ master_delta_local
                if args.position_frame == "tool-home":
                    target_position_cmd = calibration.target_position_home + calibration.target_rotation_home @ master_delta_task
                else:
                    target_position_cmd = calibration.target_position_home + master_delta_task
                target_position_filt = target_position_filt + float(args.target_smooth) * (target_position_cmd - target_position_filt)

                if args.orientation_mode in ("full", "hybrid", "split"):
                    if args.orientation_mode in ("hybrid", "split"):
                        relative_master_raw = master_state.rotation @ calibration.master_rotation.T
                        axis, angle = rotation_matrix_to_axis_angle(relative_master_raw)
                        relative_master_task = axis_angle_to_matrix(axis, float(args.orientation_scale) * angle)
                    else:
                        relative_master_rotation = scaled_relative_rotation(
                            calibration.master_rotation,
                            master_state.rotation,
                            float(args.orientation_scale),
                        )
                        relative_master_task = reexpress_rotation(relative_master_rotation, master_to_task_rotation)
                    target_rotation_cmd = target_rotation_home @ relative_master_task
                    target_rotation_filt = target_rotation_filt + float(args.target_smooth) * (
                        target_rotation_cmd - target_rotation_filt
                    )
                    u, _, vh = np.linalg.svd(target_rotation_filt)
                    target_rotation_filt = u @ vh

                current_position = point_world_pos(data, track_body_id, TRACK_OFFSET_LOCAL)
                current_rotation = body_rotation(data, orientation_body_id)
                position_error = clip_norm(
                    args.position_gain * (target_position_filt - current_position),
                    args.max_position_error,
                )

                if args.orientation_mode in ("full", "hybrid", "split"):
                    orientation_error = clip_norm(
                        args.orientation_gain * rotation_error(current_rotation, target_rotation_filt),
                        args.max_orientation_error,
                    )
                    task_idle = (
                        float(np.linalg.norm(position_error)) < float(args.deadband_pos)
                        and float(np.linalg.norm(orientation_error)) < float(args.deadband_rot)
                    )
                elif args.orientation_mode == "roll":
                    roll_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_roll_axis,
                        args.rpy_input_method,
                    )
                    roll_target = float(calibration.ctrl_home[arm_actuators[3]] + args.roll_scale * roll_delta)
                    roll_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[3]]),
                        roll_target,
                        float(max_rpy_offset[2]),
                    )
                    roll_target = float(np.clip(roll_target, arm_ctrl_range[3, 0], arm_ctrl_range[3, 1]))
                    orientation_error = np.array([roll_target - float(data.ctrl[arm_actuators[3]])], dtype=float)
                    task_idle = (
                        float(np.linalg.norm(position_error)) < float(args.deadband_pos)
                        and abs(float(orientation_error[0])) < float(args.deadband_rot)
                    )
                elif args.orientation_mode == "rpy":
                    yaw_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_yaw_axis,
                        args.rpy_input_method,
                    )
                    pitch_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_pitch_axis,
                        args.rpy_input_method,
                    )
                    roll_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_roll_axis,
                        args.rpy_input_method,
                    )
                    yaw_target = float(calibration.ctrl_home[arm_actuators[0]] + args.yaw_scale * yaw_delta)
                    pitch_target = float(calibration.ctrl_home[arm_actuators[1]] + args.pitch_scale * pitch_delta)
                    roll_target = float(calibration.ctrl_home[arm_actuators[3]] + args.roll_scale * roll_delta)
                    yaw_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[0]]),
                        yaw_target,
                        float(max_rpy_offset[0]),
                    )
                    pitch_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[1]]),
                        pitch_target,
                        float(max_rpy_offset[1]),
                    )
                    roll_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[3]]),
                        roll_target,
                        float(max_rpy_offset[2]),
                    )
                    orientation_error = np.array(
                        [
                            yaw_target - float(data.ctrl[arm_actuators[0]]),
                            pitch_target - float(data.ctrl[arm_actuators[1]]),
                            roll_target - float(data.ctrl[arm_actuators[3]]),
                        ],
                        dtype=float,
                    )
                    task_idle = (
                        float(np.linalg.norm(position_error)) < float(args.deadband_pos)
                        and float(np.linalg.norm(orientation_error)) < float(args.deadband_rot)
                    )
                else:
                    orientation_error = np.zeros(1, dtype=float)
                    task_idle = float(np.linalg.norm(position_error)) < float(args.deadband_pos)

                if task_idle:
                    dq = np.zeros(len(ARM_JOINTS), dtype=float)
                else:
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jac(
                        model,
                        data,
                        jacp,
                        jacr,
                        current_position,
                        track_body_id,
                    )

                    if args.orientation_mode == "split":
                        dq_position = damped_least_squares_step(
                            jacp[:, position_dof_columns],
                            position_error,
                            args.damping_pos,
                        )
                        jacp_ori = np.zeros((3, model.nv))
                        jacr_ori = np.zeros((3, model.nv))
                        mujoco.mj_jac(
                            model,
                            data,
                            jacp_ori,
                            jacr_ori,
                            point_world_pos(data, orientation_body_id, TRACK_OFFSET_LOCAL),
                            orientation_body_id,
                        )
                        dq_distal = damped_least_squares_step(
                            jacr_ori[:, distal_dof_columns],
                            orientation_error,
                            args.damping_full,
                        )
                        dq = np.zeros(len(ARM_JOINTS), dtype=float)
                        dq[:3] = dq_position
                        dq[3:] = float(args.split_distal_step_scale) * dq_distal
                    elif args.orientation_mode == "hybrid":
                        dq = weighted_full_pose_step(
                            jacp=jacp,
                            jacr=jacr,
                            dof_columns=arm_dof_columns,
                            position_error=position_error,
                            orientation_error=orientation_error,
                            damping=args.damping_full,
                            joint_damping_scale=hybrid_joint_damping_scale,
                            position_task_weight=args.ik_pos_weight,
                            orientation_task_weight=args.ik_ori_weight,
                            position_axis_weights=hybrid_position_axis_weights,
                        )
                    elif args.orientation_mode == "full":
                        task_jacobian = np.vstack(
                            [
                                jacp[:, arm_dof_columns],
                                jacr[:, arm_dof_columns],
                            ]
                        )
                        task_error = np.concatenate([position_error, orientation_error])
                        dq = damped_least_squares_step(task_jacobian, task_error, args.damping_full)
                    else:
                        dq_position = damped_least_squares_step(
                            jacp[:, position_dof_columns],
                            position_error,
                            args.damping_pos,
                        )
                        dq = np.zeros(len(ARM_JOINTS), dtype=float)
                        dq[:3] = dq_position

                dq = clip_norm(dq, args.max_joint_step)
                dq = apply_deadband(dq, float(args.dq_deadband))
                slowdown = joint_limit_slowdown(
                    data.ctrl[arm_actuators],
                    arm_ctrl_range,
                    float(args.limit_slowdown_margin),
                    float(args.limit_slowdown_min_scale),
                )
                dq *= slowdown
                if args.orientation_mode == "split" and args.split_distal_smooth is not None:
                    dq_filt[:3] = dq_filt[:3] + float(args.dq_smooth) * (dq[:3] - dq_filt[:3])
                    dq_filt[3:] = dq_filt[3:] + float(args.split_distal_smooth) * (dq[3:] - dq_filt[3:])
                else:
                    dq_filt = dq_filt + float(args.dq_smooth) * (dq - dq_filt)
                if args.orientation_mode in ("hybrid", "split"):
                    dq_filt[2] = dq_filt[2] + float(args.hybrid_dq_smooth_ins) * (dq[2] - dq_filt[2])
                    dq_filt[2] *= float(args.hybrid_insertion_dq_scale)

                measured_arm_qpos = actuator_targets_from_qpos(mujoco, model, data, ARM_JOINTS)
                insertion_lag = abs(float(measured_arm_qpos[2] - data.ctrl[arm_actuators[2]]))
                if args.orientation_mode not in ("hybrid", "split") or args.hybrid_use_servo_lag_guard:
                    if insertion_lag > float(args.max_insertion_servo_lag):
                        dq_filt[2] = 0.0
                        ctrl_filt[2] = measured_arm_qpos[2]
                if args.orientation_mode in ("hybrid", "split"):
                    desired_ctrl = data.ctrl[arm_actuators].copy()
                else:
                    desired_ctrl = measured_arm_qpos.copy()
                desired_ctrl += dq_filt

                if args.orientation_mode == "none":
                    desired_ctrl[3:] = calibration.ctrl_home[arm_actuators][3:]
                elif args.orientation_mode == "roll":
                    roll_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_roll_axis,
                        args.rpy_input_method,
                    )
                    roll_target = float(calibration.ctrl_home[arm_actuators[3]] + args.roll_scale * roll_delta)
                    roll_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[3]]),
                        roll_target,
                        float(max_rpy_offset[2]),
                    )
                    desired_ctrl[3] = roll_target
                    desired_ctrl[4:] = calibration.ctrl_home[arm_actuators][4:]
                elif args.orientation_mode == "rpy":
                    yaw_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_yaw_axis,
                        args.rpy_input_method,
                    )
                    pitch_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_pitch_axis,
                        args.rpy_input_method,
                    )
                    roll_delta = master_axis_delta(
                        master_state.rotation,
                        calibration.master_rotation,
                        args.master_roll_axis,
                        args.rpy_input_method,
                    )
                    yaw_target = float(calibration.ctrl_home[arm_actuators[0]] + args.yaw_scale * yaw_delta)
                    pitch_target = float(calibration.ctrl_home[arm_actuators[1]] + args.pitch_scale * pitch_delta)
                    roll_target = float(calibration.ctrl_home[arm_actuators[3]] + args.roll_scale * roll_delta)
                    yaw_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[0]]),
                        yaw_target,
                        float(max_rpy_offset[0]),
                    )
                    pitch_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[1]]),
                        pitch_target,
                        float(max_rpy_offset[1]),
                    )
                    roll_target = clipped_home_offset(
                        float(calibration.ctrl_home[arm_actuators[3]]),
                        roll_target,
                        float(max_rpy_offset[2]),
                    )
                    yaw_pitch_target = np.array([yaw_target, pitch_target], dtype=float)
                    if args.rpy_yaw_pitch_mode == "direct":
                        yaw_pitch_bias = np.clip(
                            yaw_pitch_target - measured_arm_qpos[:2],
                            -float(args.max_rpy_bias_step),
                            float(args.max_rpy_bias_step),
                        )
                        desired_ctrl[:2] = measured_arm_qpos[:2] + yaw_pitch_bias
                    elif args.rpy_yaw_pitch_mode == "blend":
                        yaw_pitch_step = np.clip(
                            yaw_pitch_target - measured_arm_qpos[:2],
                            -float(args.max_rpy_bias_step),
                            float(args.max_rpy_bias_step),
                        )
                        yaw_pitch_direct = measured_arm_qpos[:2] + yaw_pitch_step
                        direct_weight = float(np.clip(args.rpy_direct_weight, 0.0, 1.0))
                        position_desired = desired_ctrl[:2].copy()
                        desired_ctrl[:2] = (1.0 - direct_weight) * position_desired + direct_weight * yaw_pitch_direct
                        yaw_pitch_bias = desired_ctrl[:2] - position_desired
                    else:
                        yaw_pitch_bias = np.clip(
                            float(args.rpy_bias_gain) * (yaw_pitch_target - desired_ctrl[:2]),
                            -float(args.max_rpy_bias_step),
                            float(args.max_rpy_bias_step),
                        )
                        desired_ctrl[:2] += yaw_pitch_bias
                    rpy_debug["sigma"] = np.array([yaw_delta, pitch_delta, roll_delta], dtype=float)
                    rpy_debug["target"] = np.array([yaw_target, pitch_target, roll_target], dtype=float)
                    rpy_debug["bias"] = yaw_pitch_bias.copy()
                    desired_ctrl[3] = roll_target
                    desired_ctrl[4:] = calibration.ctrl_home[arm_actuators][4:]

                desired_ctrl = np.clip(desired_ctrl, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
                ctrl_filt = ctrl_filt + float(args.ctrl_smooth) * (desired_ctrl - ctrl_filt)
                delta = np.clip(ctrl_filt - data.ctrl[arm_actuators], -max_ctrl_step, max_ctrl_step)
                data.ctrl[arm_actuators] = np.clip(
                    data.ctrl[arm_actuators] + delta,
                    arm_ctrl_range[:, 0],
                    arm_ctrl_range[:, 1],
                )
                if args.jaw_mode == "locked":
                    jaw_target = (
                        float(calibration.ctrl_home[jaw_actuator])
                        if args.jaw_lock_value is None
                        else float(args.jaw_lock_value)
                    )
                else:
                    jaw_target = gripper_to_jaw(
                        master_state.gripper_deg,
                        args.gripper_close_deg,
                        args.gripper_open_deg,
                        jaw_ctrl_range,
                        args.jaw_invert,
                    )
                jaw_target = float(np.clip(jaw_target, jaw_ctrl_range[0], jaw_ctrl_range[1]))
                if abs(jaw_target - jaw_filt) >= float(args.jaw_deadband):
                    jaw_filt = jaw_filt + float(args.jaw_smooth) * (jaw_target - jaw_filt)
                jaw_delta = float(np.clip(jaw_filt - data.ctrl[jaw_actuator], -float(args.max_ctrl_step_jaw), float(args.max_ctrl_step_jaw)))
                data.ctrl[jaw_actuator] = float(np.clip(data.ctrl[jaw_actuator] + jaw_delta, jaw_ctrl_range[0], jaw_ctrl_range[1]))

                mujoco.mj_step(model, data)
                viewer.sync()

                now = time.perf_counter()
                print_due = now - last_print >= args.print_every
                log_due = log_writer is not None and now - last_log >= log_every
                if print_due or log_due:
                    tip_position = point_world_pos(data, track_body_id, TRACK_OFFSET_LOCAL)
                    position_err_norm = float(np.linalg.norm(target_position_filt - tip_position))
                    freq_khz = float(device.com_freq())
                if log_due:
                    last_log = now
                    log_writer.writerow(
                        {
                            "t": f"{now:.6f}",
                            "orientation_mode": args.orientation_mode,
                            "track_body": track_body_name,
                            "orientation_body": orientation_body_name,
                            "jaw_mode": args.jaw_mode,
                            "rpy_mode": args.rpy_yaw_pitch_mode,
                            "rpy_input_method": args.rpy_input_method,
                            "master_yaw_axis": args.master_yaw_axis,
                            "master_pitch_axis": args.master_pitch_axis,
                            "master_roll_axis": args.master_roll_axis,
                            "sigma_x": f"{master_state.position[0]:.6f}",
                            "sigma_y": f"{master_state.position[1]:.6f}",
                            "sigma_z": f"{master_state.position[2]:.6f}",
                            "target_x": f"{target_position_filt[0]:.6f}",
                            "target_y": f"{target_position_filt[1]:.6f}",
                            "target_z": f"{target_position_filt[2]:.6f}",
                            "tip_x": f"{tip_position[0]:.6f}",
                            "tip_y": f"{tip_position[1]:.6f}",
                            "tip_z": f"{tip_position[2]:.6f}",
                            "err_m": f"{position_err_norm:.6f}",
                            "ori_err_norm": f"{float(np.linalg.norm(orientation_error)):.6f}",
                            "ins_q": f"{measured_arm_qpos[2]:.6f}",
                            "ins_ctrl": f"{data.ctrl[arm_actuators[2]]:.6f}",
                            "ins_lag": f"{insertion_lag:.6f}",
                            "jaw_deg": f"{master_state.gripper_deg:.6f}",
                            "jaw_target": f"{jaw_target:.6f}",
                            "jaw_ctrl": f"{data.ctrl[jaw_actuator]:.6f}",
                            "freq_khz": f"{freq_khz:.6f}",
                            "rpy_in_yaw_deg": f"{math.degrees(rpy_debug['sigma'][0]):.6f}",
                            "rpy_in_pitch_deg": f"{math.degrees(rpy_debug['sigma'][1]):.6f}",
                            "rpy_in_roll_deg": f"{math.degrees(rpy_debug['sigma'][2]):.6f}",
                            "rpy_tgt_yaw": f"{rpy_debug['target'][0]:.6f}",
                            "rpy_tgt_pitch": f"{rpy_debug['target'][1]:.6f}",
                            "rpy_tgt_roll": f"{rpy_debug['target'][2]:.6f}",
                            "rpy_q_yaw": f"{measured_arm_qpos[0]:.6f}",
                            "rpy_q_pitch": f"{measured_arm_qpos[1]:.6f}",
                            "rpy_q_roll": f"{measured_arm_qpos[3]:.6f}",
                            "rpy_bias_yaw": f"{rpy_debug['bias'][0]:.6f}",
                            "rpy_bias_pitch": f"{rpy_debug['bias'][1]:.6f}",
                            "ctrl_yaw": f"{data.ctrl[arm_actuators[0]]:.6f}",
                            "ctrl_pitch": f"{data.ctrl[arm_actuators[1]]:.6f}",
                            "ctrl_insertion": f"{data.ctrl[arm_actuators[2]]:.6f}",
                            "ctrl_roll": f"{data.ctrl[arm_actuators[3]]:.6f}",
                            "ctrl_wrist_pitch": f"{data.ctrl[arm_actuators[4]]:.6f}",
                            "ctrl_wrist_yaw": f"{data.ctrl[arm_actuators[5]]:.6f}",
                        }
                    )
                    log_file.flush()
                if print_due:
                    last_print = now
                    print(
                        "\r"
                        f"sigma=({master_state.position[0]:+.3f},{master_state.position[1]:+.3f},{master_state.position[2]:+.3f}) m "
                        f"target=({target_position_filt[0]:+.3f},{target_position_filt[1]:+.3f},{target_position_filt[2]:+.3f}) m "
                        f"{track_body_name}=({tip_position[0]:+.3f},{tip_position[1]:+.3f},{tip_position[2]:+.3f}) m "
                        f"err={position_err_norm:.4f} m "
                        f"ori_err={float(np.linalg.norm(orientation_error)):.4f} "
                        f"ins(q/c)=({measured_arm_qpos[2]:+.4f}/{data.ctrl[arm_actuators[2]]:+.4f}) "
                        f"jaw={master_state.gripper_deg:5.1f} deg/{data.ctrl[jaw_actuator]:+.3f} ctrl "
                        f"freq={freq_khz:.2f} kHz"
                        + (
                            " "
                            f"rpy_in=({math.degrees(rpy_debug['sigma'][0]):+.1f},"
                            f"{math.degrees(rpy_debug['sigma'][1]):+.1f},"
                            f"{math.degrees(rpy_debug['sigma'][2]):+.1f})deg "
                            f"rpy_tgt=({rpy_debug['target'][0]:+.3f},"
                            f"{rpy_debug['target'][1]:+.3f},"
                            f"{rpy_debug['target'][2]:+.3f}) "
                            f"rpy_q=({measured_arm_qpos[0]:+.3f},"
                            f"{measured_arm_qpos[1]:+.3f},"
                            f"{measured_arm_qpos[3]:+.3f}) "
                            f"rpy_mode={args.rpy_yaw_pitch_mode} "
                            f"bias=({rpy_debug['bias'][0]:+.4f},{rpy_debug['bias'][1]:+.4f})"
                            if args.debug_rpy
                            else ""
                        ),
                        end="",
                        flush=True,
                    )
    finally:
        if log_file is not None:
            log_file.close()
        device.close()
        print("\nDevice closed.")


if __name__ == "__main__":
    main()
