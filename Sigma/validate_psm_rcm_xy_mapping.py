"""Offline check for PSM tool_tip x/y/z mapping through the RCM main chain.

This script does not use the sigma.7 hardware. It applies synthetic Cartesian
targets to `tool_tip` and solves only `yaw/pitch/insertion`, leaving
`roll/wrist_pitch/wrist_yaw` fixed at home. That matches the current engineering
teleop objective: preserve the RCM mechanism and make x/y translation visibly
come from the PSM main chain.

Run from the project root with the MuJoCo Python environment:
    D:\\anaconda3\\envs\\dvrk\\python.exe Sigma\\validate_psm_rcm_xy_mapping.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"

ARM_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
POSITION_JOINTS = ("yaw", "pitch", "insertion")
TRACK_BODY = "tool_tip"


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if max_norm > 0.0 and norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def actuator_ids(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for name in joint_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ctrl_{name}")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: ctrl_{name}")
        ids.append(int(actuator_id))
    return ids


def dof_columns(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    columns: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        columns.append(int(model.jnt_dofadr[joint_id]))
    return columns


def qpos_addresses(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def solve_axis(mujoco, model, data, body_id: int, position_dofs: list[int], position_qpos: list[int], target: np.ndarray, args) -> None:
    joint_range = model.jnt_range[
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POSITION_JOINTS]
    ]

    for _ in range(args.steps):
        mujoco.mj_forward(model, data)
        current = data.xpos[body_id].copy()
        error = clip_norm(target - current, args.max_position_error)
        if float(np.linalg.norm(error)) < args.tolerance:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, current, body_id)
        jacobian = jacp[:, position_dofs]
        regularized = jacobian @ jacobian.T + args.damping * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(regularized, error)
        dq = clip_norm(dq, args.max_joint_step)

        qpos = data.qpos[position_qpos].copy()
        data.qpos[position_qpos] = np.clip(qpos + dq, joint_range[:, 0], joint_range[:, 1])
        data.qvel[:] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RCM main-chain x/y/z mapping for the MuJoCo PSM.")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to psm_control.xml")
    parser.add_argument("--home-insertion", type=float, default=0.12, help="Initial insertion in meters")
    parser.add_argument("--delta", type=float, default=0.015, help="Synthetic Cartesian target step in meters")
    parser.add_argument("--steps", type=int, default=220, help="Solver iterations per target")
    parser.add_argument("--damping", type=float, default=3e-4, help="DLS damping")
    parser.add_argument("--max-position-error", type=float, default=0.012, help="Max Cartesian correction per step")
    parser.add_argument("--max-joint-step", type=float, default=0.025, help="Max IK joint step per iteration")
    parser.add_argument("--tolerance", type=float, default=2e-4, help="Stop when position error is below this value")
    args = parser.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    arm_actuators = actuator_ids(mujoco, model, ARM_JOINTS)
    position_dofs = dof_columns(mujoco, model, POSITION_JOINTS)
    arm_qpos = qpos_addresses(mujoco, model, ARM_JOINTS)
    position_qpos = qpos_addresses(mujoco, model, POSITION_JOINTS)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {TRACK_BODY}")

    mujoco.mj_resetData(model, data)
    data.ctrl[arm_actuators] = np.array([0.0, 0.0, args.home_insertion, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    home_ctrl = data.ctrl[arm_actuators].copy()
    home_qpos = data.qpos[arm_qpos].copy()
    home_tip = data.xpos[body_id].copy()

    print(f"home tip: {home_tip[0]:+.5f} {home_tip[1]:+.5f} {home_tip[2]:+.5f}")
    print("axis    target_delta               actual_delta               yaw/pitch/ins delta")

    axes = {
        "+X": np.array([args.delta, 0.0, 0.0], dtype=float),
        "+Y": np.array([0.0, args.delta, 0.0], dtype=float),
        "+Z": np.array([0.0, 0.0, args.delta], dtype=float),
    }
    for name, delta in axes.items():
        data.ctrl[arm_actuators] = home_ctrl.copy()
        data.qpos[arm_qpos] = home_qpos.copy()
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        target = home_tip + delta
        solve_axis(mujoco, model, data, body_id, position_dofs, position_qpos, target, args)
        mujoco.mj_forward(model, data)
        actual_delta = data.xpos[body_id].copy() - home_tip
        ctrl_delta = data.qpos[position_qpos] - home_qpos[:3]
        print(
            f"{name:>3}    "
            f"{delta[0]:+8.5f} {delta[1]:+8.5f} {delta[2]:+8.5f}    "
            f"{actual_delta[0]:+8.5f} {actual_delta[1]:+8.5f} {actual_delta[2]:+8.5f}    "
            f"{ctrl_delta[0]:+8.5f} {ctrl_delta[1]:+8.5f} {ctrl_delta[2]:+8.5f}"
        )


if __name__ == "__main__":
    main()
