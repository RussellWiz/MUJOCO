"""sigma.7 master-device wrapper for the PyBullet controller."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

VP_DIR = Path(__file__).resolve().parent
SIGMA_DIR = VP_DIR.parent
if str(SIGMA_DIR) not in sys.path:
    sys.path.insert(0, str(SIGMA_DIR))

from force_dimension_sdk import ForceDimensionSDK


class SigmaMaster:
    """Threaded sigma.7 reader with zero-force output."""

    def __init__(self, sdk_bin: str | None, use_drd_init: bool) -> None:
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
            raise RuntimeError("sigma.7 polling did not produce an initial sample")
        if self._stream_error is not None:
            raise RuntimeError(f"sigma.7 polling failed: {self._stream_error}") from self._stream_error
        with self._state_lock:
            if self._latest_state is None:
                raise RuntimeError("sigma.7 polling has no sample available")
            return self._latest_state

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
        self._stream_thread = threading.Thread(target=self._stream_loop, name="sigma7-pybullet-loop", daemon=True)
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

