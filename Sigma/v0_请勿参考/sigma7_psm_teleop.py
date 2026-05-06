"""
Use a Force Dimension sigma.7 to teleoperate the MuJoCo dVRK PSM model on Windows.

Run from the project root:
    python Sigma/sigma7_psm_teleop.py

If the SDK DLLs are not on PATH, pass the directory containing dhd64.dll/drd64.dll:
    python Sigma/sigma7_psm_teleop.py --sdk-bin "C:\\Program Files\\Force Dimension\\sdk-3.17.6\\bin"

Controls:
    q       quit
    r       reset the sigma.7 neutral pose and PSM home target
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


mujoco = None


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"
LOCAL_SDK_DIR = Path(__file__).resolve().parent / "sdk-3.17.6"

CONTROL_JOINTS = ("yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
POSITION_JOINTS = ("yaw", "pitch", "insertion")
JAW_JOINT = "jaw"
# Main Cartesian point: insertion / wrist junction — child of `insertion` on `tool_main`.
# Using `tool_main` instead makes horizontal motion resolve almost entirely in yaw (pitch
# column of the position Jacobian is tiny there); `tool_wrist` restores meaningful pitch.
TRACK_BODY = "tool_wrist"
TRACK_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0], dtype=float)


def point_world_pos(data: "mujoco.MjData", body_id: int, offset_local: np.ndarray) -> np.ndarray:
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ offset_local


def _dll_names() -> tuple[str, str]:
    return ("dhd64.dll", "drd64.dll") if ctypes.sizeof(ctypes.c_void_p) == 8 else ("dhd.dll", "drd.dll")


def _candidate_sdk_dirs(explicit: str | None) -> list[Path | None]:
    candidates: list[Path | None] = []
    if explicit:
        candidates.append(Path(explicit))

    for env_name in ("FDSDK", "FORCE_DIMENSION_SDK", "FORCEDIMENSION_SDK"):
        value = os.environ.get(env_name)
        if value:
            root = Path(value)
            candidates.extend([root / "bin", root])

    candidates.extend([
        LOCAL_SDK_DIR / "bin",
        Path(r"C:\Program Files\Force Dimension\sdk-3.17.6\bin"),
        Path(r"C:\Program Files (x86)\Force Dimension\sdk-3.17.6\bin"),
        Path(r"C:\Force Dimension\sdk-3.17.6\bin"),
        None,  # Let the Windows loader search PATH.
    ])
    return candidates


def _find_sdk_bin(explicit: str | None) -> Path | None:
    dhd_name, drd_name = _dll_names()

    for candidate in _candidate_sdk_dirs(explicit):
        if candidate is None:
            continue
        if (candidate / dhd_name).exists() and (candidate / drd_name).exists():
            return candidate

    for root in (LOCAL_SDK_DIR, Path(r"C:\Program Files\Force Dimension"), Path(r"C:\Force Dimension")):
        if not root.exists():
            continue
        for dhd_path in root.rglob(dhd_name):
            candidate = dhd_path.parent
            if (candidate / drd_name).exists():
                return candidate

    return None


class ForceDimensionSDK:
    """Small ctypes wrapper for the SDK functions used by this teleop script."""

    DHD_ON = 1
    DHD_OFF = 0
    DEFAULT_DEVICE_ID = ctypes.c_byte(-1)

    def __init__(self, sdk_bin: str | None = None, use_drd_init: bool = True) -> None:
        self.sdk_bin = _find_sdk_bin(sdk_bin)
        self._dll_dir_handle = None
        if sys.platform == "win32" and self.sdk_bin is not None:
            self._dll_dir_handle = os.add_dll_directory(str(self.sdk_bin))

        dhd_name, drd_name = _dll_names()
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        try:
            dhd_path = str(self.sdk_bin / dhd_name) if self.sdk_bin else dhd_name
            drd_path = str(self.sdk_bin / drd_name) if self.sdk_bin else drd_name
            self.dhd = loader(dhd_path)
            self.drd = loader(drd_path)
        except OSError as exc:
            searched = "\n  ".join(str(p) for p in _candidate_sdk_dirs(sdk_bin) if p is not None)
            raise RuntimeError(
                f"Could not load Force Dimension SDK DLLs ({dhd_name}, {drd_name}).\n"
                f"Pass --sdk-bin or add the SDK bin directory to PATH.\nSearched:\n  {searched}"
            ) from exc

        self._configure_signatures()
        self._opened = False
        self._use_drd_init = use_drd_init
        self._force_output_enabled = False
        self.device_id = self.DEFAULT_DEVICE_ID

    def _configure_signatures(self) -> None:
        c_byte = ctypes.c_byte
        c_bool = ctypes.c_bool
        c_double = ctypes.c_double
        c_int = ctypes.c_int
        c_ubyte = ctypes.c_ubyte

        self.dhd.dhdOpen.restype = c_int
        self.dhd.dhdGetDeviceID.restype = c_int
        self.dhd.dhdClose.argtypes = [c_byte]
        self.dhd.dhdClose.restype = c_int
        self.dhd.dhdErrorGetLastStr.restype = ctypes.c_char_p
        self.dhd.dhdGetComFreq.argtypes = [c_byte]
        self.dhd.dhdGetComFreq.restype = c_double
        self.dhd.dhdGetTime.restype = c_double
        self.dhd.dhdSleep.argtypes = [c_double]
        self.dhd.dhdEnableForce.argtypes = [c_ubyte, c_byte]
        self.dhd.dhdEnableForce.restype = c_int
        self.dhd.dhdSetGravityCompensation.argtypes = [c_int, c_byte]
        self.dhd.dhdSetGravityCompensation.restype = c_int
        self.dhd.dhdGetGripperAngleDeg.argtypes = [ctypes.POINTER(c_double), c_byte]
        self.dhd.dhdGetGripperAngleDeg.restype = c_int
        self.dhd.dhdGetLinearVelocity.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.dhd.dhdGetLinearVelocity.restype = c_int
        self.dhd.dhdGetPositionAndOrientationFrame.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.dhd.dhdGetPositionAndOrientationFrame.restype = c_int
        self.dhd.dhdSetForceAndTorqueAndGripperForce.argtypes = [
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_byte,
        ]
        self.dhd.dhdSetForceAndTorqueAndGripperForce.restype = c_int

        self.drd.drdOpen.restype = c_int
        self.drd.drdGetDeviceID.restype = c_int
        self.drd.drdClose.argtypes = [c_byte]
        self.drd.drdClose.restype = c_int
        self.drd.drdIsInitialized.argtypes = [c_byte]
        self.drd.drdIsInitialized.restype = c_bool
        self.drd.drdAutoInit.argtypes = [c_byte]
        self.drd.drdAutoInit.restype = c_int
        self.drd.drdStart.argtypes = [c_byte]
        self.drd.drdStart.restype = c_int
        self.drd.drdGetCtrlFreq.argtypes = [c_byte]
        self.drd.drdGetCtrlFreq.restype = c_double
        self.drd.drdGetPositionAndOrientation.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.drd.drdGetPositionAndOrientation.restype = c_int
        self.drd.drdGetVelocity.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.drd.drdGetVelocity.restype = c_int
        self.drd.drdStop.argtypes = [c_bool, c_byte]
        self.drd.drdStop.restype = c_int
        self.drd.drdMoveToPos.argtypes = [c_double, c_double, c_double, c_bool, c_byte]
        self.drd.drdMoveToPos.restype = c_int
        self.drd.drdMoveToRot.argtypes = [c_double, c_double, c_double, c_bool, c_byte]
        self.drd.drdMoveToRot.restype = c_int

    def error(self) -> str:
        raw = self.dhd.dhdErrorGetLastStr()
        return raw.decode("utf-8", errors="replace") if raw else "unknown SDK error"

    def _check(self, code: int, action: str) -> None:
        if code < 0:
            raise RuntimeError(f"{action} failed: {self.error()}")

    def _set_device_id(self, device_id: int) -> None:
        if device_id >= 0:
            self.device_id = ctypes.c_byte(device_id)

    def _enable_optional_device_features(self) -> None:
        gravity_result = self.dhd.dhdSetGravityCompensation(self.DHD_ON, self.device_id)
        if gravity_result < 0:
            print(f"Warning: gravity compensation unavailable: {self.error()}")

        force_result = self.dhd.dhdEnableForce(self.DHD_ON, self.device_id)
        if force_result < 0:
            print(f"Warning: force output unavailable; continuing read-only: {self.error()}")
            self._force_output_enabled = False
        else:
            self._force_output_enabled = True

    def open(self) -> None:
        if self._use_drd_init:
            self._check(self.drd.drdOpen(), "drdOpen")
            self._opened = True
            self._set_device_id(self.drd.drdGetDeviceID())
            if not self.drd.drdIsInitialized(self.device_id):
                self._check(self.drd.drdAutoInit(self.device_id), "drdAutoInit")
            self._check(self.drd.drdStart(self.device_id), "drdStart")
            self._check(self.drd.drdMoveToPos(0.0, 0.0, 0.0, True, self.device_id), "drdMoveToPos")
            self._check(self.drd.drdMoveToRot(0.0, 0.0, 0.0, True, self.device_id), "drdMoveToRot")
            self._check(self.drd.drdStop(True, self.device_id), "drdStop")
            self._set_device_id(self.drd.drdGetDeviceID())
        else:
            self._check(self.dhd.dhdOpen(), "dhdOpen")
            self._opened = True
            self._set_device_id(self.dhd.dhdGetDeviceID())

        self._enable_optional_device_features()

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.set_zero_force()
            if self._force_output_enabled:
                self.dhd.dhdEnableForce(self.DHD_OFF, self.device_id)
            self.dhd.dhdClose(self.device_id)
            self.drd.drdClose(self.device_id)
        finally:
            self._opened = False
            self._force_output_enabled = False

    def set_zero_force(self) -> None:
        if not self._force_output_enabled:
            return
        self.dhd.dhdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.device_id)

    def read_state(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        state = self._read_state_dhd()
        if state is not None:
            return state

        state = self._read_state_drd()
        if state is not None:
            return state

        raise RuntimeError(f"Could not read device state with DHD or DRD: {self.error()}")

    def _read_state_dhd(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
        px = ctypes.c_double()
        py = ctypes.c_double()
        pz = ctypes.c_double()
        matrix_buffer = (ctypes.c_double * 9)()
        result = self.dhd.dhdGetPositionAndOrientationFrame(
            ctypes.byref(px),
            ctypes.byref(py),
            ctypes.byref(pz),
            matrix_buffer,
            self.device_id,
        )
        if result < 0:
            return None

        gripper = ctypes.c_double()
        if self.dhd.dhdGetGripperAngleDeg(ctypes.byref(gripper), self.device_id) < 0:
            gripper.value = 0.0

        vx = ctypes.c_double()
        vy = ctypes.c_double()
        vz = ctypes.c_double()
        if self.dhd.dhdGetLinearVelocity(ctypes.byref(vx), ctypes.byref(vy), ctypes.byref(vz), self.device_id) < 0:
            vx.value = vy.value = vz.value = 0.0

        position = np.array([px.value, py.value, pz.value], dtype=float)
        rotation = np.array(matrix_buffer, dtype=float).reshape(3, 3)
        linear_velocity = np.array([vx.value, vy.value, vz.value], dtype=float)
        return position, rotation, float(gripper.value), linear_velocity

    def _read_state_drd(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
        px = ctypes.c_double()
        py = ctypes.c_double()
        pz = ctypes.c_double()
        oa = ctypes.c_double()
        ob = ctypes.c_double()
        og = ctypes.c_double()
        pg = ctypes.c_double()
        matrix_buffer = (ctypes.c_double * 9)()
        result = self.drd.drdGetPositionAndOrientation(
            ctypes.byref(px),
            ctypes.byref(py),
            ctypes.byref(pz),
            ctypes.byref(oa),
            ctypes.byref(ob),
            ctypes.byref(og),
            ctypes.byref(pg),
            matrix_buffer,
            self.device_id,
        )
        if result < 0:
            return None

        vx = ctypes.c_double()
        vy = ctypes.c_double()
        vz = ctypes.c_double()
        wx = ctypes.c_double()
        wy = ctypes.c_double()
        wz = ctypes.c_double()
        vg = ctypes.c_double()
        velocity_result = self.drd.drdGetVelocity(
            ctypes.byref(vx),
            ctypes.byref(vy),
            ctypes.byref(vz),
            ctypes.byref(wx),
            ctypes.byref(wy),
            ctypes.byref(wz),
            ctypes.byref(vg),
            self.device_id,
        )
        if velocity_result < 0:
            vx.value = vy.value = vz.value = 0.0

        position = np.array([px.value, py.value, pz.value], dtype=float)
        rotation = np.array(matrix_buffer, dtype=float).reshape(3, 3)
        linear_velocity = np.array([vx.value, vy.value, vz.value], dtype=float)
        return position, rotation, math.degrees(pg.value), linear_velocity

    def com_freq(self) -> float:
        freq = float(self.dhd.dhdGetComFreq(self.device_id))
        if freq > 0.0:
            return freq
        return float(self.drd.drdGetCtrlFreq(self.device_id))


@dataclass
class TeleopCalibration:
    master_position: np.ndarray
    master_rotation: np.ndarray
    target_position_home: np.ndarray
    ctrl_home: np.ndarray


def rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > max_norm > 0.0:
        return vector * (max_norm / norm)
    return vector


def body_rotation(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def controlled_dof_columns(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    columns: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        columns.append(int(model.jnt_dofadr[joint_id]))
    return columns


def actuator_ids(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    for name in joint_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ctrl_{name}")
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: ctrl_{name}")
        ids.append(actuator_id)
    return ids


def actuator_home_from_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        values.append(float(data.qpos[qpos_address]))
    return np.array(values, dtype=float)


def damped_least_squares_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    rows = jacobian.shape[0]
    regularized = jacobian @ jacobian.T + float(damping) * np.eye(rows)
    return jacobian.T @ np.linalg.solve(regularized, error)


def master_local_axis_angle(master_rotation: np.ndarray, master_rotation_home: np.ndarray, axis: str) -> float:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_indices[axis]
    first_perp = (axis_index + 1) % 3
    second_perp = (axis_index + 2) % 3
    relative_local_rotation = master_rotation_home.T @ master_rotation
    return float(math.atan2(relative_local_rotation[second_perp, first_perp], relative_local_rotation[first_perp, first_perp]))


def compute_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    offset_local: np.ndarray,
    position_dof_columns: list[int],
    target_position: np.ndarray,
    position_damping: float,
    max_position_error: float,
    max_position_step: float,
) -> np.ndarray:
    """Damped LS in joint space with per-joint damping and separate position vs orientation weights.

    Without weighting, a 6×6 position+orientation stack often spends most joint motion on the
    wrist for orientation, and lateral position error is cleared almost only with yaw — pitch
    barely moves. Larger pos_task_weight fixes that for PSM teleop.
    """
    current_position = point_world_pos(data, body_id, offset_local)

    position_error = clip_norm(target_position - current_position, max_position_error)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, current_position, body_id)

    dq_position = damped_least_squares_step(jacp[:, position_dof_columns], position_error, position_damping)
    dq_position = clip_norm(dq_position, max_position_step)

    return clip_norm(dq_position, max_position_step)


def make_calibration(
    device: ForceDimensionSDK,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    offset_local: np.ndarray,
    arm_actuators: list[int],
    jaw_actuator: int,
) -> TeleopCalibration:
    master_position, master_rotation, _, _ = device.read_state()
    mujoco.mj_forward(model, data)

    ctrl_home = data.ctrl.copy()
    data.ctrl[arm_actuators] = actuator_home_from_qpos(model, data, CONTROL_JOINTS)
    jaw_home = actuator_home_from_qpos(model, data, (JAW_JOINT,))[0]
    data.ctrl[jaw_actuator] = jaw_home

    return TeleopCalibration(
        master_position=master_position,
        master_rotation=master_rotation,
        target_position_home=point_world_pos(data, body_id, offset_local),
        ctrl_home=ctrl_home.copy(),
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


def gripper_to_jaw(gripper_deg: float, close_deg: float, open_deg: float, jaw_range: np.ndarray) -> float:
    if math.isclose(open_deg, close_deg):
        normalized = 0.0
    else:
        normalized = (gripper_deg - close_deg) / (open_deg - close_deg)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return float(jaw_range[0] + normalized * (jaw_range[1] - jaw_range[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate the MuJoCo dVRK PSM with a Force Dimension sigma.7")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo PSM XML")
    parser.add_argument("--sdk-bin", default=None, help="Directory containing dhd64.dll and drd64.dll")
    parser.add_argument("--no-drd-init", action="store_true", help="Skip DRD auto-init and only use DHD open/read calls")
    parser.add_argument("--scale-x", type=float, default=0.5, help="Master X meters to MuJoCo world X meters")
    parser.add_argument("--scale-y", type=float, default=0.5, help="Master Y meters to MuJoCo world Y meters")
    parser.add_argument("--scale-z", type=float, default=0.5, help="Master Z meters to MuJoCo world Z meters")
    parser.add_argument("--damping", type=float, default=1e-3, help="Damped least-squares IK damping for RCM position")
    parser.add_argument(
        "--master-planar-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate master (sigma) x/y/z displacement about world +Z before adding to PSM tracking target (deg). "
        "Tune if device axes are misaligned with MuJoCo world so x/y motion uses pitch as expected.",
    )
    parser.add_argument(
        "--no-roll",
        action="store_true",
        help="Track position only and keep PSM roll at calibration value",
    )
    parser.add_argument(
        "--master-roll-axis",
        choices=("x", "y", "z"),
        default="z",
        help="sigma.7 local rotation axis mapped to the PSM roll joint",
    )
    parser.add_argument("--roll-scale", type=float, default=1.0, help="Scale master roll angle to PSM roll angle")
    parser.add_argument("--max-position-error", type=float, default=0.004, help="Max Cartesian IK correction per frame")
    parser.add_argument("--max-joint-step", type=float, default=0.003, help="Max RCM joint target update per frame")
    parser.add_argument("--max-roll-step", type=float, default=0.004, help="Max roll target update per frame")
    parser.add_argument("--deadband-pos", type=float, default=2e-4, help="Position deadband in meters")
    parser.add_argument("--deadband-roll", type=float, default=2e-3, help="Roll deadband in radians")
    parser.add_argument("--target-smooth", type=float, default=0.25, help="Low-pass alpha for target position (0..1)")
    parser.add_argument("--dq-smooth", type=float, default=0.35, help="Low-pass alpha for IK dq (0..1)")
    parser.add_argument("--ctrl-smooth", type=float, default=0.25, help="Low-pass alpha for actuator targets (0..1)")
    parser.add_argument(
        "--max-ctrl-step-rot",
        type=float,
        default=0.003,
        help="Max per-frame change for rotary joints (radians)",
    )
    parser.add_argument(
        "--max-ctrl-step-ins",
        type=float,
        default=0.0006,
        help="Max per-frame change for insertion joint (meters)",
    )
    parser.add_argument(
        "--insertion-dq-scale",
        type=float,
        default=0.35,
        help="Scale factor applied to insertion component of IK dq (0..1)",
    )
    parser.add_argument(
        "--dq-smooth-ins",
        type=float,
        default=0.18,
        help="Low-pass alpha for insertion dq (0..1). Smaller = more smoothing.",
    )
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
            "Install it with 'pip install mujoco' or run this script from the Conda/Python "
            "environment you use for the existing MuJoCo examples."
        ) from exc

    mujoco = mujoco_module
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    track_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY)
    if track_body_id < 0:
        raise ValueError(f"MuJoCo body not found: {TRACK_BODY}")

    arm_actuators = actuator_ids(model, CONTROL_JOINTS)
    jaw_actuator = actuator_ids(model, (JAW_JOINT,))[0]
    position_dof_columns = controlled_dof_columns(model, POSITION_JOINTS)
    arm_ctrl_range = model.actuator_ctrlrange[arm_actuators]
    jaw_ctrl_range = model.actuator_ctrlrange[jaw_actuator]
    scale = np.clip(np.array([args.scale_x, args.scale_y, args.scale_z], dtype=float), 1e-6, 1.0)
    if not np.allclose(scale, np.array([args.scale_x, args.scale_y, args.scale_z], dtype=float)):
        print("Warning: dVRK PSM translation scale is clamped to the official (0, 1] range.")
    planar_yaw = math.radians(float(args.master_planar_yaw_deg))
    c_z, s_z = math.cos(planar_yaw), math.sin(planar_yaw)
    master_delta_to_world = np.array(
        [[c_z, -s_z, 0.0], [s_z, c_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    device = ForceDimensionSDK(args.sdk_bin, use_drd_init=not args.no_drd_init)
    try:
        print("Opening sigma.7...")
        device.open()
        calibration = make_calibration(device, model, data, track_body_id, TRACK_OFFSET_LOCAL, arm_actuators, jaw_actuator)
        track_roll = not args.no_roll
        target_position_filt = calibration.target_position_home.copy()
        dq_filt = np.zeros(len(POSITION_JOINTS), dtype=float)
        ctrl_filt = data.ctrl[arm_actuators].copy()
        roll_home = float(ctrl_filt[3])
        locked_wrist_home = ctrl_filt[4:6].copy()
        # Per-actuator rate limits: yaw, pitch, insertion, roll, wrist_pitch, wrist_yaw
        max_ctrl_step = np.array(
            [
                args.max_ctrl_step_rot,
                args.max_ctrl_step_rot,
                args.max_ctrl_step_ins,
                args.max_roll_step,
                args.max_ctrl_step_rot,
                args.max_ctrl_step_rot,
            ],
            dtype=float,
        )

        print(f"Loaded MuJoCo XML: {xml_path}")
        print(f"SDK DLL directory: {device.sdk_bin if device.sdk_bin else 'PATH'}")
        print("Move the sigma.7 to control the PSM. Press r to recalibrate, q to quit.")

        last_print = time.perf_counter()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                command = keyboard_command()
                if command == "q":
                    break
                if command == "r":
                    calibration = make_calibration(
                        device, model, data, track_body_id, TRACK_OFFSET_LOCAL, arm_actuators, jaw_actuator
                    )
                    target_position_filt = calibration.target_position_home.copy()
                    dq_filt[:] = 0.0
                    ctrl_filt = data.ctrl[arm_actuators].copy()
                    roll_home = float(ctrl_filt[3])
                    locked_wrist_home = ctrl_filt[4:6].copy()
                    print("\nRecalibrated neutral pose.")

                master_position, master_rotation, gripper_deg, _ = device.read_state()
                device.set_zero_force()

                master_delta = (master_position - calibration.master_position) * scale
                master_delta = master_delta_to_world @ master_delta
                # For the current setup we only want the dVRK tracking point to
                # follow sigma.7 motion along the master Y direction.
                master_delta_y_only = np.array([0.0, master_delta[1], 0.0], dtype=float)
                target_position_cmd = calibration.target_position_home + master_delta_y_only
                target_position_filt = target_position_filt + float(args.target_smooth) * (target_position_cmd - target_position_filt)

                roll_delta = master_local_axis_angle(master_rotation, calibration.master_rotation, args.master_roll_axis)
                target_roll_cmd = roll_home + float(args.roll_scale) * roll_delta
                target_roll_cmd = float(np.clip(target_roll_cmd, arm_ctrl_range[3, 0], arm_ctrl_range[3, 1]))

                # Deadband: ignore tiny noise to prevent chatter.
                current_position = point_world_pos(data, track_body_id, TRACK_OFFSET_LOCAL)
                e_pos = target_position_filt - current_position
                e_roll = target_roll_cmd - float(data.ctrl[arm_actuators[3]])
                position_idle = float(np.linalg.norm(e_pos)) < float(args.deadband_pos)
                roll_idle = (not track_roll) or abs(e_roll) < float(args.deadband_roll)
                if position_idle and roll_idle:
                    dq = np.zeros(len(POSITION_JOINTS), dtype=float)
                else:
                    dq = compute_ik_step(
                        model=model,
                        data=data,
                        body_id=track_body_id,
                        offset_local=TRACK_OFFSET_LOCAL,
                        position_dof_columns=position_dof_columns,
                        target_position=target_position_filt,
                        position_damping=args.damping,
                        max_position_error=args.max_position_error,
                        max_position_step=args.max_joint_step,
                    )
                dq_cmd = dq.copy()
                dq_cmd[2] *= float(args.insertion_dq_scale)
                dq_alpha = np.full(len(POSITION_JOINTS), float(args.dq_smooth), dtype=float)
                dq_alpha[2] = float(args.dq_smooth_ins)
                dq_filt = dq_filt + dq_alpha * (dq_cmd - dq_filt)
                # Compute desired actuator targets then filter + rate-limit them.
                desired_ctrl = data.ctrl[arm_actuators].copy()
                desired_ctrl[:3] = desired_ctrl[:3] + dq_filt
                desired_ctrl[3] = target_roll_cmd if track_roll else roll_home
                desired_ctrl[4:6] = locked_wrist_home
                desired_ctrl = np.clip(desired_ctrl, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
                ctrl_filt = ctrl_filt + float(args.ctrl_smooth) * (desired_ctrl - ctrl_filt)
                delta = np.clip(ctrl_filt - data.ctrl[arm_actuators], -max_ctrl_step, max_ctrl_step)
                data.ctrl[arm_actuators] = np.clip(data.ctrl[arm_actuators] + delta, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
                data.ctrl[jaw_actuator] = gripper_to_jaw(
                    gripper_deg,
                    args.gripper_close_deg,
                    args.gripper_open_deg,
                    jaw_ctrl_range,
                )

                mujoco.mj_step(model, data)
                viewer.sync()

                now = time.perf_counter()
                if now - last_print >= args.print_every:
                    last_print = now
                    track_point = point_world_pos(data, track_body_id, TRACK_OFFSET_LOCAL)
                    print(
                        "\r"
                        f"sigma=({master_position[0]:+.3f},{master_position[1]:+.3f},{master_position[2]:+.3f}) m "
                        f"wrist=({track_point[0]:+.3f},{track_point[1]:+.3f},{track_point[2]:+.3f}) m "
                        f"roll={data.ctrl[arm_actuators[3]]:+.3f} rad "
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
