"""Minimal Force Dimension sigma.7 SDK wrapper for Windows/Linux ctypes use.

This module only wraps the SDK calls needed by the MuJoCo teleoperation
examples in this repository.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOCAL_SDK_DIR = Path(__file__).resolve().parent / "sdk-3.17.6"


def _dll_names() -> tuple[str, str]:
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        return "dhd64.dll", "drd64.dll"
    return "dhd.dll", "drd.dll"


def candidate_sdk_dirs(explicit: str | None = None) -> list[Path | None]:
    candidates: list[Path | None] = []
    if explicit:
        candidates.append(Path(explicit))

    for env_name in ("FDSDK", "FORCE_DIMENSION_SDK", "FORCEDIMENSION_SDK"):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value)
        candidates.extend([root / "bin", root])

    candidates.extend(
        [
            LOCAL_SDK_DIR / "bin",
            Path(r"C:\Program Files\Force Dimension\sdk-3.17.6\bin"),
            Path(r"C:\Program Files (x86)\Force Dimension\sdk-3.17.6\bin"),
            Path(r"C:\Force Dimension\sdk-3.17.6\bin"),
            None,
        ]
    )
    return candidates


def find_sdk_bin(explicit: str | None = None) -> Path | None:
    dhd_name, drd_name = _dll_names()

    for candidate in candidate_sdk_dirs(explicit):
        if candidate is None:
            continue
        if (candidate / dhd_name).exists() and (candidate / drd_name).exists():
            return candidate

    for root in (
        LOCAL_SDK_DIR,
        Path(r"C:\Program Files\Force Dimension"),
        Path(r"C:\Program Files (x86)\Force Dimension"),
        Path(r"C:\Force Dimension"),
    ):
        if not root.exists():
            continue
        for dhd_path in root.rglob(dhd_name):
            candidate = dhd_path.parent
            if (candidate / drd_name).exists():
                return candidate

    return None


@dataclass
class ForceDimensionState:
    position: np.ndarray
    rotation: np.ndarray
    gripper_deg: float
    linear_velocity: np.ndarray
    angular_velocity_deg: np.ndarray


class ForceDimensionSDK:
    """Small ctypes wrapper around the Force Dimension SDK."""

    DHD_ON = 1
    DHD_OFF = 0
    DEFAULT_DEVICE_ID = ctypes.c_byte(-1)

    def __init__(self, sdk_bin: str | None = None, use_drd_init: bool = True) -> None:
        self.sdk_bin = find_sdk_bin(sdk_bin)
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
            searched = "\n  ".join(str(path) for path in candidate_sdk_dirs(sdk_bin) if path is not None)
            raise RuntimeError(
                f"Could not load Force Dimension SDK libraries ({dhd_name}, {drd_name}).\n"
                f"Pass --sdk-bin or add the SDK bin directory to PATH.\n"
                f"Searched:\n  {searched}"
            ) from exc

        self._configure_signatures()
        self._opened = False
        self._use_drd_init = use_drd_init
        self._force_output_enabled = False
        self._opened_via_drd = False
        self._gravity_comp_enabled = False
        self.device_id = self.DEFAULT_DEVICE_ID

    def _configure_signatures(self) -> None:
        c_byte = ctypes.c_byte
        c_bool = ctypes.c_bool
        c_double = ctypes.c_double
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_ubyte = ctypes.c_ubyte

        self.dhd.dhdOpen.restype = c_int
        self.dhd.dhdGetDeviceID.restype = c_int
        self.dhd.dhdClose.argtypes = [c_byte]
        self.dhd.dhdClose.restype = c_int
        self.dhd.dhdErrorGetLastStr.restype = ctypes.c_char_p
        self.dhd.dhdGetComFreq.argtypes = [c_byte]
        self.dhd.dhdGetComFreq.restype = c_double
        self.dhd.dhdEnableForce.argtypes = [c_ubyte, c_byte]
        self.dhd.dhdEnableForce.restype = c_int
        self.dhd.dhdSetGravityCompensation.argtypes = [c_int, c_byte]
        self.dhd.dhdSetGravityCompensation.restype = c_int
        self.dhd.dhdGetButton.argtypes = [c_int, c_byte]
        self.dhd.dhdGetButton.restype = c_int
        self.dhd.dhdGetButtonMask.argtypes = [c_byte]
        self.dhd.dhdGetButtonMask.restype = c_uint
        self.dhd.dhdHasActiveGripper.argtypes = [c_byte]
        self.dhd.dhdHasActiveGripper.restype = c_bool
        self.dhd.dhdEmulateButton.argtypes = [c_ubyte, c_byte]
        self.dhd.dhdEmulateButton.restype = c_int
        self.dhd.dhdGetGripperAngleDeg.argtypes = [ctypes.POINTER(c_double), c_byte]
        self.dhd.dhdGetGripperAngleDeg.restype = c_int
        self.dhd.dhdGetLinearVelocity.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.dhd.dhdGetLinearVelocity.restype = c_int
        self.dhd.dhdGetAngularVelocityDeg.argtypes = [
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            ctypes.POINTER(c_double),
            c_byte,
        ]
        self.dhd.dhdGetAngularVelocityDeg.restype = c_int
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
        self.drd.drdIsRunning.argtypes = [c_byte]
        self.drd.drdIsRunning.restype = c_bool
        self.drd.drdAutoInit.argtypes = [c_byte]
        self.drd.drdAutoInit.restype = c_int
        self.drd.drdStart.argtypes = [c_byte]
        self.drd.drdStart.restype = c_int
        self.drd.drdStop.argtypes = [c_bool, c_byte]
        self.drd.drdStop.restype = c_int
        self.drd.drdGetCtrlFreq.argtypes = [c_byte]
        self.drd.drdGetCtrlFreq.restype = c_double
        self.drd.drdMoveToPos.argtypes = [c_double, c_double, c_double, c_bool, c_byte]
        self.drd.drdMoveToPos.restype = c_int
        self.drd.drdMoveToRot.argtypes = [c_double, c_double, c_double, c_bool, c_byte]
        self.drd.drdMoveToRot.restype = c_int
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
        force_result = self.dhd.dhdEnableForce(self.DHD_ON, self.device_id)
        if force_result < 0:
            print(f"Warning: force output unavailable; continuing read-only: {self.error()}")
            self._force_output_enabled = False
        else:
            self._force_output_enabled = True

        gravity_result = self.dhd.dhdSetGravityCompensation(self.DHD_ON, self.device_id)
        if gravity_result < 0:
            print(f"Warning: gravity compensation unavailable: {self.error()}")
            self._gravity_comp_enabled = False
        else:
            self._gravity_comp_enabled = True

        if self.dhd.dhdHasActiveGripper(self.device_id):
            emulate_result = self.dhd.dhdEmulateButton(self.DHD_ON, self.device_id)
            if emulate_result < 0:
                print(f"Warning: button emulation unavailable: {self.error()}")

    def open(self) -> None:
        if self._use_drd_init:
            self._check(self.drd.drdOpen(), "drdOpen")
            self._opened = True
            self._opened_via_drd = True
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
            self._opened_via_drd = False
            self._set_device_id(self.dhd.dhdGetDeviceID())

        self._enable_optional_device_features()
        self.set_zero_force()

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
            self._gravity_comp_enabled = False
            self._opened_via_drd = False

    def __enter__(self) -> "ForceDimensionSDK":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_zero_force(self) -> None:
        if not self._force_output_enabled:
            return
        self.dhd.dhdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.device_id)

    def set_force(self, force: np.ndarray, torque: np.ndarray | None = None, gripper_force: float = 0.0) -> None:
        if not self._force_output_enabled:
            return
        torque = np.zeros(3, dtype=float) if torque is None else np.asarray(torque, dtype=float)
        force = np.asarray(force, dtype=float)
        self.dhd.dhdSetForceAndTorqueAndGripperForce(
            float(force[0]),
            float(force[1]),
            float(force[2]),
            float(torque[0]),
            float(torque[1]),
            float(torque[2]),
            float(gripper_force),
            self.device_id,
        )

    def user_button_pressed(self, index: int = 0) -> bool:
        return bool(self.dhd.dhdGetButton(int(index), self.device_id))

    def button_mask(self) -> int:
        return int(self.dhd.dhdGetButtonMask(self.device_id))

    def read_state(self) -> ForceDimensionState:
        state = self._read_state_dhd()
        if state is not None:
            return state

        state = self._read_state_drd()
        if state is not None:
            return state

        raise RuntimeError(f"Could not read device state with DHD or DRD: {self.error()}")

    def _read_state_dhd(self) -> ForceDimensionState | None:
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

        wx = ctypes.c_double()
        wy = ctypes.c_double()
        wz = ctypes.c_double()
        if self.dhd.dhdGetAngularVelocityDeg(ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(wz), self.device_id) < 0:
            wx.value = wy.value = wz.value = 0.0

        return ForceDimensionState(
            position=np.array([px.value, py.value, pz.value], dtype=float),
            rotation=np.array(matrix_buffer, dtype=float).reshape(3, 3),
            gripper_deg=float(gripper.value),
            linear_velocity=np.array([vx.value, vy.value, vz.value], dtype=float),
            angular_velocity_deg=np.array([wx.value, wy.value, wz.value], dtype=float),
        )

    def _read_state_drd(self) -> ForceDimensionState | None:
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
        if self.drd.drdGetVelocity(
            ctypes.byref(vx),
            ctypes.byref(vy),
            ctypes.byref(vz),
            ctypes.byref(wx),
            ctypes.byref(wy),
            ctypes.byref(wz),
            ctypes.byref(vg),
            self.device_id,
        ) < 0:
            vx.value = vy.value = vz.value = 0.0
            wx.value = wy.value = wz.value = 0.0

        return ForceDimensionState(
            position=np.array([px.value, py.value, pz.value], dtype=float),
            rotation=np.array(matrix_buffer, dtype=float).reshape(3, 3),
            gripper_deg=math.degrees(pg.value),
            linear_velocity=np.array([vx.value, vy.value, vz.value], dtype=float),
            angular_velocity_deg=np.array([wx.value, wy.value, wz.value], dtype=float),
        )

    def com_freq(self) -> float:
        dhd_freq = float(self.dhd.dhdGetComFreq(self.device_id))
        if dhd_freq > 0.0:
            return dhd_freq
        return float(self.drd.drdGetCtrlFreq(self.device_id))

    def debug_status(self) -> dict[str, object]:
        drd_initialized = False
        drd_running = False
        if self._opened:
            try:
                drd_initialized = bool(self.drd.drdIsInitialized(self.device_id))
            except Exception:
                drd_initialized = False
            try:
                drd_running = bool(self.drd.drdIsRunning(self.device_id))
            except Exception:
                drd_running = False

        return {
            "opened": self._opened,
            "opened_via_drd": self._opened_via_drd,
            "use_drd_init": self._use_drd_init,
            "device_id": int(self.device_id.value),
            "force_output_enabled": self._force_output_enabled,
            "gravity_comp_enabled": self._gravity_comp_enabled,
            "drd_initialized": drd_initialized,
            "drd_running": drd_running,
            "com_freq_khz": self.com_freq() if self._opened else 0.0,
        }
