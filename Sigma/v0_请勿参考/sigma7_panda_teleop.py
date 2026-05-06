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
    max_position_error: float,
    max_step: float,
) -> np.ndarray:
    current_position = tcp_world_position(data, hand_body_id)
    current_rotation = body_rotation(data, hand_body_id)

    position_error = clip_norm(target_position - current_position, max_position_error)
    angular_error = orientation_gain * rotation_error(current_rotation, target_rotation)
    task_error = np.concatenate([position_error, angular_error])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, current_position, hand_body_id)
    jacobian = np.vstack([jacp[:, dof_columns], jacr[:, dof_columns]])
    regularized = jacobian @ jacobian.T + damping * np.eye(6)
    dq = jacobian.T @ np.linalg.solve(regularized, task_error)
    return clip_norm(dq, max_step)


def make_calibration(
    device: ForceDimensionSDK,
    model,
    data,
    hand_body_id: int,
    arm_actuators: list[int],
    gripper_actuator: int,
) -> TeleopCalibration:
    master_position, master_rotation, _, _ = device.read_state()
    mujoco.mj_forward(model, data)

    ctrl_home = data.ctrl.copy()
    data.ctrl[arm_actuators] = qpos_for_joints(model, data, PANDA_JOINTS)
    data.ctrl[gripper_actuator] = model.actuator_ctrlrange[gripper_actuator, 1]

    return TeleopCalibration(
        master_position=master_position,
        master_rotation=master_rotation,
        target_position_home=tcp_world_position(data, hand_body_id),
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
    parser.add_argument("--scale-z", type=float, default=1.8, help="Master Z meters to MuJoCo world Z meters")
    parser.add_argument("--damping", type=float, default=1e-2, help="Damped least-squares IK damping")
    parser.add_argument("--orientation-gain", type=float, default=0.25, help="IK orientation error gain")
    parser.add_argument("--max-position-error", type=float, default=0.004, help="Max Cartesian IK correction per frame")
    parser.add_argument("--max-joint-step", type=float, default=0.003, help="Max arm joint target update per frame")
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
    scale = np.array([args.scale_x, args.scale_y, args.scale_z], dtype=float)

    device = ForceDimensionSDK(args.sdk_bin, use_drd_init=not args.no_drd_init)
    try:
        print("Opening sigma.7...")
        device.open()
        calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator)

        print(f"Loaded MuJoCo XML: {xml_path}")
        print(f"SDK DLL directory: {device.sdk_bin if device.sdk_bin else 'PATH'}")
        print("Move the sigma.7 to control the Panda TCP. Press r to reset/recalibrate, q to quit.")

        last_print = time.perf_counter()
        previous_time = data.time
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if data.time < previous_time:
                    reset_scene(model, data, torus_qpos)
                    calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator)
                    print("\nViewer reset detected; scene and neutral pose restored.")
                previous_time = data.time

                command = keyboard_command()
                if command == "q":
                    break
                if command == "r":
                    reset_scene(model, data, torus_qpos)
                    calibration = make_calibration(device, model, data, hand_body_id, arm_actuators, gripper_actuator)
                    print("\nRecalibrated neutral pose.")

                master_position, master_rotation, gripper_deg, _ = device.read_state()
                device.set_zero_force()

                master_delta = (master_position - calibration.master_position) * scale
                target_position = calibration.target_position_home + master_delta

                master_rotation_delta = master_rotation @ calibration.master_rotation.T
                target_rotation = calibration.target_rotation_home @ master_rotation_delta

                dq = compute_ik_step(
                    model=model,
                    data=data,
                    hand_body_id=hand_body_id,
                    dof_columns=arm_dof_columns,
                    target_position=target_position,
                    target_rotation=target_rotation,
                    damping=args.damping,
                    orientation_gain=args.orientation_gain,
                    max_position_error=args.max_position_error,
                    max_step=args.max_joint_step,
                )
                next_arm_ctrl = data.ctrl[arm_actuators] + dq
                data.ctrl[arm_actuators] = np.clip(next_arm_ctrl, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
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
