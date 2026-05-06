"""sigma.7 master-device wrapper and master-side mappings."""
from __future__ import annotations

import math
import sys

import numpy as np

from force_dimension_sdk import ForceDimensionSDK


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


class SigmaMaster:
    """Thin owner for Force Dimension I/O.

    Force Dimension SDK details stay in force_dimension_sdk.py; this class only
    centralizes teleop-side constraints and device lifecycle.
    """

    def __init__(
        self,
        sdk_bin: str | None,
        use_drd_init: bool,
    ) -> None:
        self.device = ForceDimensionSDK(sdk_bin, use_drd_init=use_drd_init)

    @property
    def sdk_bin(self):
        return self.device.sdk_bin

    def open(self) -> None:
        self.device.open()

    def close(self) -> None:
        self.device.close()

    def read_state(self):
        return self.device.read_state()

    def set_zero_force(self) -> None:
        self.device.set_zero_force()

    def com_freq(self) -> float:
        return float(self.device.com_freq())
