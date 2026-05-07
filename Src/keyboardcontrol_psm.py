from __future__ import annotations

import os
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard


PROJECT_DIR = Path(__file__).resolve().parents[1]
# XML_PATH = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"
XML_PATH = PROJECT_DIR / "Src" / "scene" / "psm_control_capsule32_torus.xml"
# The translational task is solved only with these three joints.
POSITION_JOINTS = ("yaw", "pitch", "insertion")

# Wrist orientation is deliberately decoupled from the translational IK.
WRIST_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")
JAW_JOINT = "jaw"
EE_BODY = "tool_tip"

WORLD_LINEAR_SPEED = 0.015  # m/s
WRIST_SPEED = 0.35  # rad/s
JAW_SPEED = 0.45  # rad/s

TARGET_FILTER = 0.25
DQ_FILTER = 0.35
KP_TASK = 5.0
MAX_TARGET_ERROR = 0.012  # m
MAX_QSTEP = np.array([0.004, 0.004, 0.0007])  # yaw, pitch, insertion per sim step

DAMPING_MIN = 1.0e-4
DAMPING_MAX = 8.0e-2
SINGULAR_VALUE_SAFE = 0.035
POS_DEADBAND = 2.0e-5

_keys: set[object] = set()
_lock = threading.Lock()
_quit = False


CTRL_HINT = """
dVRK PSM keyboard control

World-frame translation target:
  W/S       +X / -X
  A/D       +Y / -Y
  PgUp/Dn   +Z / -Z

Decoupled wrist:
  Z/X       roll - / +
  R/F       wrist pitch + / -
  T/G       wrist yaw + / -

Jaw:
  O/P       open / close

Esc closes the script.
"""


def _on_press(key: object) -> None:
    global _quit
    if key == keyboard.Key.esc:
        _quit = True
    with _lock:
        try:
            _keys.add(key.char.lower())  # type: ignore[attr-defined]
        except AttributeError:
            _keys.add(key)


def _on_release(key: object) -> None:
    with _lock:
        try:
            _keys.discard(key.char.lower())  # type: ignore[attr-defined]
        except AttributeError:
            _keys.discard(key)


def body_pos(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xpos[body_id].copy()


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm > max_norm > 0.0:
        return v * (max_norm / norm)
    return v


def joint_dof_indices(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing joint in XML: {name}")
        indices.append(int(model.jnt_dofadr[jid]))
    return indices


def actuator_indices(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    actuator_ids: list[int] = []
    for joint_name in joint_names:
        target_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if target_joint_id < 0:
            raise ValueError(f"Missing joint in XML: {joint_name}")

        for actuator_id in range(model.nu):
            trn_id = int(model.actuator_trnid[actuator_id, 0])
            if trn_id == target_joint_id:
                actuator_ids.append(actuator_id)
                break
        else:
            raise ValueError(f"Missing actuator for joint: {joint_name}")
    return actuator_ids


def get_keyboard_velocity() -> tuple[np.ndarray, np.ndarray, float]:
    linear = np.zeros(3, dtype=float)
    wrist = np.zeros(3, dtype=float)
    jaw = 0.0

    with _lock:
        if "w" in _keys:
            linear[0] += WORLD_LINEAR_SPEED
        if "s" in _keys:
            linear[0] -= WORLD_LINEAR_SPEED
        if "a" in _keys:
            linear[1] += WORLD_LINEAR_SPEED
        if "d" in _keys:
            linear[1] -= WORLD_LINEAR_SPEED
        if keyboard.Key.page_up in _keys:
            linear[2] += WORLD_LINEAR_SPEED
        if keyboard.Key.page_down in _keys:
            linear[2] -= WORLD_LINEAR_SPEED

        if "x" in _keys:
            wrist[0] += WRIST_SPEED
        if "z" in _keys:
            wrist[0] -= WRIST_SPEED
        if "r" in _keys:
            wrist[1] += WRIST_SPEED
        if "f" in _keys:
            wrist[1] -= WRIST_SPEED
        if "t" in _keys:
            wrist[2] += WRIST_SPEED
        if "g" in _keys:
            wrist[2] -= WRIST_SPEED

        if "o" in _keys:
            jaw += JAW_SPEED
        if "p" in _keys:
            jaw -= JAW_SPEED

    return linear, wrist, jaw


def svd_adaptive_dls(J: np.ndarray, task_velocity: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve dq = J# v with singular-value-dependent damped reciprocals."""
    U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros(J.shape[1], dtype=float), 0.0

    filtered = np.zeros_like(singular_values)
    sigma_min = float(np.min(singular_values))
    for i, sigma in enumerate(singular_values):
        if sigma < SINGULAR_VALUE_SAFE:
            ratio = 1.0 - sigma / SINGULAR_VALUE_SAFE
            damping = DAMPING_MIN + DAMPING_MAX * ratio * ratio
        else:
            damping = DAMPING_MIN
        filtered[i] = sigma / (sigma * sigma + damping * damping)

    dq = Vt.T @ (filtered * (U.T @ task_velocity))
    return dq, sigma_min


def main() -> None:
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"MuJoCo XML not found: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if ee_body_id < 0:
        raise ValueError(f"Missing body in XML: {EE_BODY}")

    pos_dofs = joint_dof_indices(model, POSITION_JOINTS)
    pos_actuators = actuator_indices(model, POSITION_JOINTS)
    wrist_actuators = actuator_indices(model, WRIST_JOINTS)
    jaw_actuator = actuator_indices(model, (JAW_JOINT,))[0]

    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    mujoco.mj_resetData(model, data)
    data.ctrl[pos_actuators[0]] = 0.0
    data.ctrl[pos_actuators[1]] = 0.0
    data.ctrl[pos_actuators[2]] = 0.12
    home_ctrl = data.ctrl.copy()

    for _ in range(500):
        mujoco.mj_step(model, data)

    command_target = body_pos(data, ee_body_id)
    filtered_target = command_target.copy()
    dq_filtered = np.zeros(len(POSITION_JOINTS), dtype=float)
    dt = float(model.opt.timestep)

    print(CTRL_HINT)
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    step_count = 0
    prev_time = data.time

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not _quit:
                if data.time < prev_time:
                    data.ctrl[:] = home_ctrl
                    command_target = body_pos(data, ee_body_id)
                    filtered_target = command_target.copy()
                    dq_filtered[:] = 0.0
                    print("[Reset] controller target returned to current tool pose.")
                prev_time = data.time

                linear_v, wrist_v, jaw_v = get_keyboard_velocity()
                command_target += linear_v * dt

                current_pos = body_pos(data, ee_body_id)
                command_target = current_pos + clip_norm(command_target - current_pos, 3.0 * MAX_TARGET_ERROR)
                filtered_target += TARGET_FILTER * (command_target - filtered_target)
                pos_error = clip_norm(filtered_target - current_pos, MAX_TARGET_ERROR)

                if float(np.linalg.norm(pos_error)) > POS_DEADBAND:
                    jacp = np.zeros((3, model.nv), dtype=float)
                    jacr = np.zeros((3, model.nv), dtype=float)
                    mujoco.mj_jac(model, data, jacp, jacr, current_pos, ee_body_id)
                    J = jacp[:, pos_dofs]

                    task_v = KP_TASK * pos_error
                    dq, sigma_min = svd_adaptive_dls(J, task_v)
                    dq_filtered += DQ_FILTER * (dq - dq_filtered)
                    dq_step = np.clip(dq_filtered * dt, -MAX_QSTEP, MAX_QSTEP)

                    for actuator_id, qstep in zip(pos_actuators, dq_step):
                        data.ctrl[actuator_id] = np.clip(
                            data.ctrl[actuator_id] + qstep,
                            ctrl_lo[actuator_id],
                            ctrl_hi[actuator_id],
                        )
                else:
                    sigma_min = 0.0
                    dq_filtered *= 1.0 - DQ_FILTER

                for actuator_id, velocity in zip(wrist_actuators, wrist_v):
                    data.ctrl[actuator_id] = np.clip(
                        data.ctrl[actuator_id] + velocity * dt,
                        ctrl_lo[actuator_id],
                        ctrl_hi[actuator_id],
                    )

                if jaw_v:
                    data.ctrl[jaw_actuator] = np.clip(
                        data.ctrl[jaw_actuator] + jaw_v * dt,
                        ctrl_lo[jaw_actuator],
                        ctrl_hi[jaw_actuator],
                    )

                mujoco.mj_step(model, data)
                viewer.sync()

                step_count += 1
                if step_count % 500 == 0:
                    p = body_pos(data, ee_body_id)
                    print(
                        f"[tool_tip] x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f} "
                        f"sigma_min={sigma_min:.4f} jaw={data.ctrl[jaw_actuator]:.2f}"
                    )
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
