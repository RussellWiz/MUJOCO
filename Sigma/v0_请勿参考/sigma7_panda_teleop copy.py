"""
Use a Force Dimension sigma.7 to teleoperate the MuJoCo Franka Panda scene.

Run from the project root:
    python Sigma/sigma7_panda_teleop.py --sdk-bin "D:\\DVRK\\MUJOCO\\Sigma\\sdk-3.17.6\\bin"

Controls:
    q       quit
    r       reset the sigma.7 neutral pose and Panda home target
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigma7_psm_teleop import ForceDimensionSDK, keyboard_command


mujoco = None


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_DIR / "Assest" / "franka_emika_panda" / "scene32.xml"

PANDA_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
PANDA_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
EE_BODY = "hand"
TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=float)


@dataclass
class TeleopCalibration:
    master_position: np.ndarray
    master_rotation: np.ndarray
    target_position_home: np.ndarray
    target_rotation_home: np.ndarray
    ctrl_home: np.ndarray


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > max_norm > 0.0:
        return vector * (max_norm / norm)
    return vector


def rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def body_rotation(data, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def tcp_world_position(data, hand_body_id: int) -> np.ndarray:
    rotation = body_rotation(data, hand_body_id)
    return data.xpos[hand_body_id] + rotation @ TCP_OFFSET


def vertical_desired_rotation(data, hand_body_id: int) -> np.ndarray:
    """
    Keep the gripper strictly vertical-down (hand Z axis = world -Z),
    while preserving the current yaw around world Z.
    """
    r_cur = body_rotation(data, hand_body_id)
    z_des = np.array([0.0, 0.0, -1.0])

    x_proj = r_cur[:, 0].copy()
    x_proj[2] = 0.0
    norm = np.linalg.norm(x_proj)
    if norm < 1e-6:
        x_proj = np.array([1.0, 0.0, 0.0])
    else:
        x_proj /= norm

    y_des = np.cross(z_des, x_proj)
    return np.column_stack([x_proj, y_des, z_des])


def actuator_ids(model, actuator_names: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {name}")
        ids.append(actuator_id)
    return ids


def controlled_dof_columns(model, joint_names: tuple[str, ...]) -> list[int]:
    columns: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        columns.append(int(model.jnt_dofadr[joint_id]))
    return columns


def qpos_for_joints(model, data, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        values.append(float(data.qpos[qpos_address]))
    return np.array(values, dtype=float)


def hold_current_arm_pose(model, data, arm_actuators: list[int]) -> None:
    """Freeze the arm by setting actuator targets to current joint positions."""
    data.ctrl[arm_actuators] = qpos_for_joints(model, data, PANDA_JOINTS)


def find_torus_qpos(model) -> tuple[int, np.ndarray] | None:
    torus_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torus")
    if torus_body_id < 0 or model.body_jntnum[torus_body_id] == 0:
        return None

    joint_id = int(model.body_jntadr[torus_body_id])
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        return None

    qpos_address = int(model.jnt_qposadr[joint_id])
    return qpos_address, model.qpos0[qpos_address:qpos_address + 7].copy()


def reset_scene(model, data, torus_qpos: tuple[int, np.ndarray] | None) -> None:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    if torus_qpos is not None:
        qpos_address, qpos_value = torus_qpos
        data.qpos[qpos_address:qpos_address + 7] = qpos_value

    mujoco.mj_forward(model, data)


def compute_ik_step(
    model,
    data,
    hand_body_id: int,
    dof_columns: list[int],
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    damping: float,
    orientation_gain: float,
    track_orientation: bool,
    max_position_error: float,
    max_step: float,
) -> np.ndarray:
    current_position = tcp_world_position(data, hand_body_id)
    current_rotation = body_rotation(data, hand_body_id)

    position_error = clip_norm(target_position - current_position, max_position_error)
    if track_orientation:
        angular_error = orientation_gain * rotation_error(current_rotation, target_rotation)
        task_error = np.concatenate([position_error, angular_error])
    else:
        task_error = position_error

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, current_position, hand_body_id)
    if track_orientation:
        jacobian = np.vstack([jacp[:, dof_columns], jacr[:, dof_columns]])
        regularized = jacobian @ jacobian.T + damping * np.eye(6)
    else:
        jacobian = jacp[:, dof_columns]
        regularized = jacobian @ jacobian.T + damping * np.eye(3)
    dq = jacobian.T @ np.linalg.solve(regularized, task_error)
    return clip_norm(dq, max_step)


def make_calibration(
    device: ForceDimensionSDK,
    model,
    data,
    hand_body_id: int,
    arm_actuators: list[int],
    gripper_actuator: int,
    home_offset: np.ndarray,
) -> TeleopCalibration:
    master_position, master_rotation, _, _ = device.read_state()
    mujoco.mj_forward(model, data)

    ctrl_home = data.ctrl.copy()
    data.ctrl[arm_actuators] = qpos_for_joints(model, data, PANDA_JOINTS)
    data.ctrl[gripper_actuator] = model.actuator_ctrlrange[gripper_actuator, 1]

    return TeleopCalibration(
        master_position=master_position,
        master_rotation=master_rotation,
        target_position_home=tcp_world_position(data, hand_body_id) + home_offset,
        target_rotation_home=body_rotation(data, hand_body_id),
        ctrl_home=ctrl_home.copy(),
    )


def gripper_to_ctrl(gripper_deg: float, close_deg: float, open_deg: float, ctrl_range: np.ndarray) -> float:
    if math.isclose(open_deg, close_deg):
        normalized = 0.0
    else:
        normalized = (gripper_deg - close_deg) / (open_deg - close_deg)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return float(ctrl_range[0] + normalized * (ctrl_range[1] - ctrl_range[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate the MuJoCo Panda arm with a Force Dimension sigma.7")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo Panda scene XML")
    parser.add_argument("--sdk-bin", default=None, help="Directory containing dhd64.dll and drd64.dll")
    parser.add_argument("--no-drd-init", action="store_true", help="Skip DRD auto-init and only use DHD open/read calls")
    parser.add_argument("--scale-x", type=float, default=1.8, help="Master X meters to MuJoCo world X meters")
    parser.add_argument("--scale-y", type=float, default=1.8, help="Master Y meters to MuJoCo world Y meters")
    parser.add_argument("--scale-z", type=float, default=1.8, help="Master Z meters to MuJoCo world Z meters (legacy)")
    parser.add_argument("--scale-z-up", type=float, default=None, help="Master +Z scaling (override scale-z)")
    parser.add_argument("--scale-z-down", type=float, default=None, help="Master -Z scaling (override scale-z). Increase to reach ground without huge home offset.")
    parser.add_argument("--scale-z-up-exp", type=float, default=1.0, help="Exponent for +Z mapping. 1.0 = linear.")
    parser.add_argument("--scale-z-down-exp", type=float, default=0.7, help="Exponent for -Z mapping (<1 amplifies small motions). 1.0 = linear.")
    parser.add_argument("--scale-z-exp-ref", type=float, default=0.05, help="Reference length (m) to keep exponent mapping dimensionally consistent.")
    # More aggressive defaults: less smoothing, smaller deadbands, larger per-frame steps.
    parser.add_argument("--pos-deadband", type=float, default=0.0015, help="Deadband for master position delta (m) to reduce jitter")
    parser.add_argument("--pos-filter", type=float, default=0.35, help="EMA low-pass alpha for master position delta (0..1). Smaller = smoother.")
    parser.add_argument("--deadband-pos", type=float, default=1e-4, help="Cartesian error deadband in meters (skip IK when within)")
    parser.add_argument("--deadband-rot", type=float, default=1.2e-3, help="Orientation deadband (norm) in radians (skip IK when within)")
    parser.add_argument("--target-smooth", type=float, default=0.38, help="Low-pass alpha for target position (0..1)")
    parser.add_argument("--dq-smooth", type=float, default=0.55, help="Low-pass alpha for IK dq (0..1)")
    parser.add_argument("--ctrl-smooth", type=float, default=0.4, help="Low-pass alpha for actuator targets (0..1)")
    parser.add_argument("--max-ctrl-step", type=float, default=0.012, help="Max per-frame change for Panda joints (radians)")
    parser.add_argument("--home-offset-x", type=float, default=0.0, help="Shift MuJoCo TCP home target in X (m)")
    parser.add_argument("--home-offset-y", type=float, default=0.0, help="Shift MuJoCo TCP home target in Y (m)")
    parser.add_argument("--home-offset-z", type=float, default=-0.18, help="Shift MuJoCo TCP home target in Z (m). Negative moves closer to ground.")
    parser.add_argument("--damping", type=float, default=4e-3, help="Damped least-squares IK damping")
    parser.add_argument("--orientation-gain", type=float, default=0.25, help="IK orientation error gain")
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Track position only (ignore sigma.7 orientation)",
    )
    parser.add_argument(
        "--vertical-orientation",
        action="store_true",
        help="Override orientation target to keep the gripper vertical-down (legacy behavior)",
    )
    parser.add_argument("--max-position-error", type=float, default=0.01, help="Max Cartesian IK correction per frame")
    parser.add_argument("--max-joint-step", type=float, default=0.009, help="Max arm joint target update per frame")
    parser.add_argument("--gripper-close-deg", type=float, default=0.0, help="sigma.7 gripper angle treated as closed")
    parser.add_argument("--gripper-open-deg", type=float, default=30.0, help="sigma.7 gripper angle treated as open")
    parser.add_argument("--print-every", type=float, default=0.25, help="Status print period in seconds")
    return parser.parse_args()


def main() -> None:
    global mujoco

    args = parse_args()
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    try:
        import mujoco as mujoco_module
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The active Python environment does not have the 'mujoco' package. "
            "Install it with 'pip install mujoco' or run this script from the environment "
            "you use for the existing MuJoCo examples."
        ) from exc

    mujoco = mujoco_module
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    torus_qpos = find_torus_qpos(model)
    reset_scene(model, data, torus_qpos)

    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if hand_body_id < 0:
        raise ValueError(f"MuJoCo body not found: {EE_BODY}")

    arm_actuators = actuator_ids(model, PANDA_ACTUATORS)
    gripper_actuator = actuator_ids(model, (GRIPPER_ACTUATOR,))[0]
    arm_dof_columns = controlled_dof_columns(model, PANDA_JOINTS)
    arm_ctrl_range = model.actuator_ctrlrange[arm_actuators]
    gripper_ctrl_range = model.actuator_ctrlrange[gripper_actuator]
    scale_z_up = args.scale_z if args.scale_z_up is None else float(args.scale_z_up)
    scale_z_down = args.scale_z if args.scale_z_down is None else float(args.scale_z_down)
    scale_xy = np.array([args.scale_x, args.scale_y], dtype=float)
    home_offset = np.array([args.home_offset_x, args.home_offset_y, args.home_offset_z], dtype=float)

    device = ForceDimensionSDK(args.sdk_bin, use_drd_init=not args.no_drd_init)
    try:
        print("Opening sigma.7...")
        device.open()
        calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator, home_offset)

        print(f"Loaded MuJoCo XML: {xml_path}")
        print(f"SDK DLL directory: {device.sdk_bin if device.sdk_bin else 'PATH'}")
        print("Move the sigma.7 to control the Panda TCP. Press r to reset/recalibrate, q to quit.")

        last_print = time.perf_counter()
        previous_time = data.time
        filtered_master_delta_raw = np.zeros(3, dtype=float)
        filter_alpha = float(np.clip(args.pos_filter, 0.0, 1.0))
        target_position_filt = calibration.target_position_home.copy()
        target_rotation = calibration.target_rotation_home.copy()
        dq_filt = np.zeros(len(arm_dof_columns), dtype=float)
        ctrl_filt = data.ctrl[arm_actuators].copy()
        max_ctrl_step = float(max(0.0, args.max_ctrl_step))
        track_orientation = (not args.no_orientation) and (not args.vertical_orientation)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if data.time < previous_time:
                    reset_scene(model, data, torus_qpos)
                    calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator, home_offset)
                    target_position_filt = calibration.target_position_home.copy()
                    target_rotation = calibration.target_rotation_home.copy()
                    dq_filt[:] = 0.0
                    ctrl_filt = data.ctrl[arm_actuators].copy()
                    print("\nViewer reset detected; scene and neutral pose restored.")
                previous_time = data.time

                command = keyboard_command()
                if command == "q":
                    break
                if command == "r":
                    reset_scene(model, data, torus_qpos)
                    calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator, home_offset)
                    filtered_master_delta_raw[:] = 0.0
                    target_position_filt = calibration.target_position_home.copy()
                    target_rotation = calibration.target_rotation_home.copy()
                    dq_filt[:] = 0.0
                    ctrl_filt = data.ctrl[arm_actuators].copy()
                    print("\nRecalibrated neutral pose.")

                master_position, master_rotation, gripper_deg, _ = device.read_state()
                device.set_zero_force()

                master_delta_raw = master_position - calibration.master_position
                # Deadband + low-pass filter on the raw master delta (reduces high-frequency jitter).
                if np.linalg.norm(master_delta_raw) < args.pos_deadband:
                    master_delta_raw = np.zeros(3, dtype=float)
                filtered_master_delta_raw = (
                    filter_alpha * master_delta_raw + (1.0 - filter_alpha) * filtered_master_delta_raw
                )

                master_delta = np.zeros(3, dtype=float)
                master_delta[:2] = filtered_master_delta_raw[:2] * scale_xy
                # Piecewise Z scaling with optional exponent curve.
                dz = float(filtered_master_delta_raw[2])
                ref = max(1e-6, float(args.scale_z_exp_ref))
                if dz >= 0.0:
                    dz_norm = dz / ref
                    master_delta[2] = scale_z_up * ref * (dz_norm ** float(args.scale_z_up_exp))
                else:
                    dz_norm = (-dz) / ref
                    master_delta[2] = -scale_z_down * ref * (dz_norm ** float(args.scale_z_down_exp))
                target_position_cmd = calibration.target_position_home + master_delta
                target_position_filt = target_position_filt + float(args.target_smooth) * (target_position_cmd - target_position_filt)

                if args.vertical_orientation:
                    target_rotation = vertical_desired_rotation(data, hand_body_id)
                elif track_orientation:
                    master_rotation_delta = master_rotation @ calibration.master_rotation.T
                    target_rotation = calibration.target_rotation_home @ master_rotation_delta

                # Deadband on task-space error: ignore tiny noise to prevent chatter.
                current_position = tcp_world_position(data, hand_body_id)
                current_rotation = body_rotation(data, hand_body_id)
                e_pos = target_position_filt - current_position
                if track_orientation or args.vertical_orientation:
                    e_rot = rotation_error(current_rotation, target_rotation)
                    within_rot = float(np.linalg.norm(e_rot)) < float(args.deadband_rot)
                else:
                    within_rot = True
                if float(np.linalg.norm(e_pos)) < float(args.deadband_pos) and within_rot:
                    data.ctrl[gripper_actuator] = gripper_to_ctrl(
                        gripper_deg,
                        args.gripper_close_deg,
                        args.gripper_open_deg,
                        gripper_ctrl_range,
                    )
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    continue

                dq = compute_ik_step(
                    model=model,
                    data=data,
                    hand_body_id=hand_body_id,
                    dof_columns=arm_dof_columns,
                    target_position=target_position_filt,
                    target_rotation=target_rotation,
                    damping=args.damping,
                    orientation_gain=args.orientation_gain,
                    track_orientation=(track_orientation or args.vertical_orientation),
                    max_position_error=args.max_position_error,
                    max_step=args.max_joint_step,
                )
                dq_filt = dq_filt + float(args.dq_smooth) * (dq - dq_filt)
                desired_ctrl = data.ctrl[arm_actuators] + dq_filt
                desired_ctrl = np.clip(desired_ctrl, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
                ctrl_filt = ctrl_filt + float(args.ctrl_smooth) * (desired_ctrl - ctrl_filt)
                delta = np.clip(ctrl_filt - data.ctrl[arm_actuators], -max_ctrl_step, max_ctrl_step)
                data.ctrl[arm_actuators] = np.clip(
                    data.ctrl[arm_actuators] + delta, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1]
                )
                data.ctrl[gripper_actuator] = gripper_to_ctrl(
                    gripper_deg,
                    args.gripper_close_deg,
                    args.gripper_open_deg,
                    gripper_ctrl_range,
                )

                mujoco.mj_step(model, data)
                viewer.sync()

                now = time.perf_counter()
                if now - last_print >= args.print_every:
                    last_print = now
                    tcp = tcp_world_position(data, hand_body_id)
                    print(
                        "\r"
                        f"sigma=({master_position[0]:+.3f},{master_position[1]:+.3f},{master_position[2]:+.3f}) m "
                        f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) m "
                        f"grip={gripper_deg:5.1f} deg "
                        f"freq={device.com_freq():.2f} kHz",
                        end="",
                        flush=True,
                    )
    finally:
        device.close()
        print("\nDevice closed.")


if __name__ == "__main__":
    main()
