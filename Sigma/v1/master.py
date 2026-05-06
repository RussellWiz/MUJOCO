"""sigma.7 master-device wrapper and master-side mappings."""
from __future__ import annotations

import math
import sys
import threading
import time

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
        self._state_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._stream_ready = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._latest_state = None
        self._stream_error: Exception | None = None
        self._poll_period_s = 0.001

    @property
    def sdk_bin(self):
        return self.device.sdk_bin

    def open(self) -> None:
        self.device.open()
        self._start_streaming()

    def close(self) -> None:
        self._stop_streaming()
        self.device.close()

    def read_state(self):
        if self._stream_thread is None:
            return self.device.read_state()

        if not self._stream_ready.wait(timeout=2.0):
            raise RuntimeError("sigma.7 background polling did not produce an initial sample")

        if self._stream_error is not None:
            raise RuntimeError(f"sigma.7 background polling failed: {self._stream_error}") from self._stream_error

        with self._state_lock:
            if self._latest_state is None:
                raise RuntimeError("sigma.7 background polling has no sample available")
            return self._latest_state

    def set_zero_force(self) -> None:
        self.device.set_zero_force()

    def com_freq(self) -> float:
        return float(self.device.com_freq())

    def debug_status(self):
        return self.device.debug_status()

    def _start_streaming(self) -> None:
        if self._stream_thread is not None:
            return
        self._stream_stop.clear()
        self._stream_ready.clear()
        self._stream_error = None
        self._latest_state = None
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="sigma7-master-loop",
            daemon=True,
        )
        self._stream_thread.start()

    def _stop_streaming(self) -> None:
        thread = self._stream_thread
        if thread is None:
            return
        self._stream_stop.set()
        thread.join(timeout=2.0)
        self._stream_thread = None
        self._stream_ready.clear()

    def _stream_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stream_stop.is_set():
            try:
                state = self.device.read_state()
                self.device.set_zero_force()
            except Exception as exc:
                self._stream_error = exc
                self._stream_ready.set()
                return

            with self._state_lock:
                self._latest_state = state
            self._stream_ready.set()

            next_tick += self._poll_period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()
