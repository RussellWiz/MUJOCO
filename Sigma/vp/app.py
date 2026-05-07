"""PyBullet entry point for sigma.7 driven PSM teleoperation."""
from __future__ import annotations

from pathlib import Path
import sys
import time

VP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VP_DIR.parent.parent
if __package__ in (None, ""):
    for path in (str(VP_DIR), str(PROJECT_DIR), str(VP_DIR.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from configs import parse_args
    from master import SigmaMaster
    from motion import PyBulletSplitController
    from robot import PyBulletPsmRobot
else:
    from .configs import parse_args
    from .master import SigmaMaster
    from .motion import PyBulletSplitController
    from .robot import PyBulletPsmRobot


class TeleopApp:
    def __init__(self, args) -> None:
        self.args = args
        self.urdf_path = Path(args.urdf)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"PyBullet URDF not found: {self.urdf_path}")
        self.pb = self._load_pybullet()
        mode = self.pb.DIRECT if args.headless else self.pb.GUI
        self.client_id = self.pb.connect(mode)
        self.pb.setTimeStep(float(args.time_step), physicsClientId=self.client_id)
        self.pb.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        self.robot = PyBulletPsmRobot.load(
            self.pb,
            self.client_id,
            self.urdf_path,
            float(args.position_force),
            float(args.jaw_force),
        )
        self.robot.reset_and_seed_home(float(args.home_insertion))
        self.master = SigmaMaster(args.sdk_bin, use_drd_init=not args.no_drd_init)
        self.motion: PyBulletSplitController | None = None

    def _load_pybullet(self):
        try:
            import pybullet as pybullet
        except ModuleNotFoundError as exc:
            raise RuntimeError("The active Python environment does not have 'pybullet'. Install it with 'pip install pybullet'.") from exc
        return pybullet

    def print_banner(self) -> None:
        print(f"Loaded PyBullet URDF: {self.urdf_path}")
        print(f"SDK DLL directory: {self.master.sdk_bin if self.master.sdk_bin else 'PATH'}")
        print(
            "PyBullet split control: sigma increments -> world target increments; "
            "yaw/pitch/insertion solve position with adaptive SVD-DLS; wrist is decoupled. "
            "Close the GUI or press Ctrl+C to quit."
        )

    def run(self) -> None:
        try:
            print("Opening sigma.7...")
            self.master.open()
            if self.args.debug_sdk:
                print(f"[sdk] {self.master.debug_status()}")
            first_state = self.master.read_state()
            self.motion = PyBulletSplitController(self.args, self.robot, first_state)
            self.print_banner()

            last_print = time.perf_counter()
            while self.pb.isConnected(self.client_id):
                state = self.master.read_state()
                debug = self.motion.update(state)
                self.pb.stepSimulation(physicsClientId=self.client_id)
                if not self.args.headless:
                    time.sleep(float(self.args.time_step))

                now = time.perf_counter()
                if now - last_print >= float(self.args.print_every):
                    last_print = now
                    self.print_status(debug)
        finally:
            self.master.close()
            if getattr(self, "pb", None) is not None and self.pb.isConnected(self.client_id):
                self.pb.disconnect(self.client_id)
            print("\nDevice closed.")

    def print_status(self, debug) -> None:
        print(
            "\r"
            f"sigma_d=({debug.sigma_delta[0]:+.5f},{debug.sigma_delta[1]:+.5f},{debug.sigma_delta[2]:+.5f}) "
            f"target=({debug.target[0]:+.3f},{debug.target[1]:+.3f},{debug.target[2]:+.3f}) "
            f"tool_tip=({debug.tip[0]:+.3f},{debug.tip[1]:+.3f},{debug.tip[2]:+.3f}) "
            f"err={debug.error_norm:.4f} "
            f"smin={debug.sigma_min:.4f} "
            f"jaw={debug.jaw_target:+.3f}",
            end="",
            flush=True,
        )


def main() -> None:
    TeleopApp(parse_args()).run()


if __name__ == "__main__":
    main()

