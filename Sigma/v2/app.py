"""Entry point for sigma.7 driven v2 PSM teleoperation."""
from __future__ import annotations

from pathlib import Path
import sys
import time

V2_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    project_dir = str(V2_DIR.parent.parent)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from Sigma.v2.configs import parse_args
    from Sigma.v2.master import SigmaMaster
    from Sigma.v2.motion import SplitTeleopController
    from Sigma.v2.robot import PsmRobot
else:
    from .configs import parse_args
    from .master import SigmaMaster
    from .motion import SplitTeleopController
    from .robot import PsmRobot


class TeleopApp:
    def __init__(self, args) -> None:
        self.args = args
        self.xml_path = Path(args.xml)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")
        self.mujoco = self._load_mujoco()
        self.robot = PsmRobot.load(self.mujoco, self.xml_path, str(args.track_body))
        self.robot.reset_and_seed_home(float(args.home_insertion))
        self.master = SigmaMaster(args.sdk_bin, use_drd_init=not args.no_drd_init)
        self.motion: SplitTeleopController | None = None

    def _load_mujoco(self):
        try:
            import mujoco
            import mujoco.viewer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The active Python environment does not have the 'mujoco' package. "
                "Install it or run this from the environment used by the MuJoCo examples."
            ) from exc
        return mujoco

    def recalibrate(self) -> None:
        self.robot.reset_and_seed_home(float(self.args.home_insertion))
        state = self.master.read_state()
        self.motion.reset(state)
        print("\nRecalibrated sigma and PSM home.")

    def print_banner(self) -> None:
        print(f"Loaded MuJoCo XML: {self.xml_path}")
        print(f"SDK DLL directory: {self.master.sdk_bin if self.master.sdk_bin else 'PATH'}")
        print(
            "v2 split control: sigma increments -> world target increments; "
            "yaw/pitch/insertion solve position with adaptive SVD-DLS; wrist is decoupled. "
            "Close the viewer or press Ctrl+C to quit."
        )

    def run(self) -> None:
        try:
            print("Opening sigma.7...")
            self.master.open()
            if self.args.debug_sdk:
                print(f"[sdk] {self.master.debug_status()}")
            first_state = self.master.read_state()
            self.motion = SplitTeleopController(self.args, self.robot, first_state)
            self.print_banner()

            last_print = time.perf_counter()
            with self.mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
                while viewer.is_running():
                    state = self.master.read_state()
                    debug = self.motion.update(state)
                    self.mujoco.mj_step(self.robot.model, self.robot.data)
                    viewer.sync()

                    now = time.perf_counter()
                    if now - last_print >= float(self.args.print_every):
                        last_print = now
                        self.print_status(debug)
        finally:
            self.master.close()
            print("\nDevice closed.")

    def print_status(self, debug) -> None:
        arm = self.robot.arm_actuators
        print(
            "\r"
            f"sigma_d=({debug.sigma_delta[0]:+.5f},{debug.sigma_delta[1]:+.5f},{debug.sigma_delta[2]:+.5f}) "
            f"target=({debug.target[0]:+.3f},{debug.target[1]:+.3f},{debug.target[2]:+.3f}) "
            f"{self.robot.track_body_name}=({debug.tip[0]:+.3f},{debug.tip[1]:+.3f},{debug.tip[2]:+.3f}) "
            f"err={debug.error_norm:.4f} "
            f"smin={debug.sigma_min:.4f} "
            f"ctrl=({self.robot.data.ctrl[arm[0]]:+.3f},{self.robot.data.ctrl[arm[1]]:+.3f},{self.robot.data.ctrl[arm[2]]:+.4f},"
            f"{self.robot.data.ctrl[arm[3]]:+.3f},{self.robot.data.ctrl[arm[4]]:+.3f},{self.robot.data.ctrl[arm[5]]:+.3f}) "
            f"jaw={self.robot.data.ctrl[self.robot.jaw_actuator]:+.3f}",
            end="",
            flush=True,
        )


def main() -> None:
    TeleopApp(parse_args()).run()


if __name__ == "__main__":
    main()
