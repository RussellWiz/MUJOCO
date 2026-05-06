"""GUI/viewer entry point and main teleop application loop."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import time

import numpy as np

V1_DIR = Path(__file__).resolve().parent
SIGMA_DIR = V1_DIR.parent
PROJECT_DIR = SIGMA_DIR.parent

if __package__ in (None, ""):
    for path in (str(V1_DIR), str(SIGMA_DIR), str(PROJECT_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from configs import parse_args
    from master import SigmaMaster, gripper_to_jaw, keyboard_command
    from motion import MotionController, make_calibration
    from robot import PsmRobot, body_rotation, point_world_pos
else:
    from .configs import parse_args
    from .master import SigmaMaster, gripper_to_jaw, keyboard_command
    from .motion import MotionController, make_calibration
    from .robot import PsmRobot, body_rotation, point_world_pos


CSV_FIELDNAMES = [
    "t",
    "orientation_mode",
    "track_body",
    "orientation_body",
    "jaw_mode",
    "rpy_mode",
    "rpy_input_method",
    "master_yaw_axis",
    "master_pitch_axis",
    "master_roll_axis",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "target_x",
    "target_y",
    "target_z",
    "tip_x",
    "tip_y",
    "tip_z",
    "err_m",
    "ori_err_norm",
    "ins_q",
    "ins_ctrl",
    "ins_lag",
    "jaw_deg",
    "jaw_target",
    "jaw_ctrl",
    "freq_khz",
    "rpy_in_yaw_deg",
    "rpy_in_pitch_deg",
    "rpy_in_roll_deg",
    "rpy_tgt_yaw",
    "rpy_tgt_pitch",
    "rpy_tgt_roll",
    "rpy_q_yaw",
    "rpy_q_pitch",
    "rpy_q_roll",
    "rpy_bias_yaw",
    "rpy_bias_pitch",
    "ctrl_yaw",
    "ctrl_pitch",
    "ctrl_insertion",
    "ctrl_roll",
    "ctrl_wrist_pitch",
    "ctrl_wrist_yaw",
]


class CsvTeleopLogger:
    def __init__(self, path: str | None) -> None:
        self.file = None
        self.writer = None
        if path:
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = log_path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDNAMES)
            self.writer.writeheader()
            print(f"CSV logging: {log_path}")

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def close(self) -> None:
        if self.file is not None:
            self.file.close()

    def write(self, row: dict[str, str]) -> None:
        if self.writer is None:
            return
        self.writer.writerow(row)
        self.file.flush()


class JawController:
    def __init__(self, args, robot: PsmRobot, calibration) -> None:
        self.args = args
        self.robot = robot
        self.calibration = calibration
        self.jaw_filt = float(robot.data.ctrl[robot.jaw_actuator])
        self.jaw_target = self.jaw_filt

    def reset(self, calibration) -> None:
        self.calibration = calibration
        self.jaw_filt = float(self.robot.data.ctrl[self.robot.jaw_actuator])
        self.jaw_target = self.jaw_filt

    def update(self, master_state) -> float:
        if self.args.jaw_mode == "locked":
            self.jaw_target = (
                float(self.calibration.ctrl_home[self.robot.jaw_actuator])
                if self.args.jaw_lock_value is None
                else float(self.args.jaw_lock_value)
            )
        else:
            self.jaw_target = gripper_to_jaw(
                master_state.gripper_deg,
                self.args.gripper_close_deg,
                self.args.gripper_open_deg,
                self.robot.jaw_ctrl_range,
                self.args.jaw_invert,
            )
        self.jaw_target = float(np.clip(self.jaw_target, self.robot.jaw_ctrl_range[0], self.robot.jaw_ctrl_range[1]))
        if abs(self.jaw_target - self.jaw_filt) >= float(self.args.jaw_deadband):
            self.jaw_filt = self.jaw_filt + float(self.args.jaw_smooth) * (self.jaw_target - self.jaw_filt)
        jaw_delta = float(
            np.clip(
                self.jaw_filt - self.robot.data.ctrl[self.robot.jaw_actuator],
                -float(self.args.max_ctrl_step_jaw),
                float(self.args.max_ctrl_step_jaw),
            )
        )
        self.robot.data.ctrl[self.robot.jaw_actuator] = float(
            np.clip(
                self.robot.data.ctrl[self.robot.jaw_actuator] + jaw_delta,
                self.robot.jaw_ctrl_range[0],
                self.robot.jaw_ctrl_range[1],
            )
        )
        return self.jaw_target


class TeleopApp:
    def __init__(self, args) -> None:
        self.args = args
        self.xml_path = Path(args.xml)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")
        self.mujoco = self._load_mujoco()
        self.robot = PsmRobot.load(self.mujoco, self.xml_path, str(args.track_body), args.orientation_body)
        self.robot.reset_and_seed_home(args.home_insertion)
        self.master = SigmaMaster(
            args.sdk_bin,
            use_drd_init=not args.no_drd_init,
        )
        self.logger = CsvTeleopLogger(args.log_csv)
        self.calibration = None
        self.motion = None
        self.jaw = None

    def _load_mujoco(self):
        try:
            import mujoco
            import mujoco.viewer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The active Python environment does not have the 'mujoco' package. "
                "Install it with 'pip install mujoco' or run this script from the Python "
                "environment you use for the existing MuJoCo examples."
            ) from exc
        return mujoco

    def initialize_teleop_state(self) -> None:
        master_state = self.master.read_state()
        self.calibration = make_calibration(master_state, self.robot)
        target_rotation_home = body_rotation(self.robot.data, self.robot.orientation_body_id).copy()
        self.motion = MotionController(self.args, self.robot, self.calibration, target_rotation_home)
        self.jaw = JawController(self.args, self.robot, self.calibration)

    def recalibrate(self) -> None:
        self.robot.data.ctrl[:] = self.calibration.ctrl_home
        for _ in range(150):
            self.mujoco.mj_step(self.robot.model, self.robot.data)
        master_state = self.master.read_state()
        self.calibration = make_calibration(master_state, self.robot)
        target_rotation_home = body_rotation(self.robot.data, self.robot.orientation_body_id).copy()
        self.motion.reset(self.calibration, target_rotation_home)
        self.jaw.reset(self.calibration)
        print("\nRecalibrated neutral pose.")

    def print_banner(self) -> None:
        print(f"Loaded MuJoCo XML: {self.xml_path}")
        print(f"SDK DLL directory: {self.master.sdk_bin if self.master.sdk_bin else 'PATH'}")
        print(
            "Teleop ready. "
            f"orientation-mode={self.args.orientation_mode}, "
            f"position-frame={self.args.position_frame}, "
            f"track-body={self.robot.track_body_name}, "
            f"orientation-body={self.robot.orientation_body_name}, "
            f"jaw-mode={self.args.jaw_mode}. "
            "Press r to recalibrate, q to quit."
        )
        if self.args.orientation_mode == "none":
            print("Engineering mode: distal roll/wrist are held at calibration home; x/y/z are solved by yaw/pitch/insertion.")

    def run(self) -> None:
        try:
            print("Opening sigma.7...")
            self.master.open()
            self.initialize_teleop_state()
            self.print_banner()
            last_print = time.perf_counter()
            last_log = last_print
            log_every = float(self.args.print_every if self.args.log_every is None else self.args.log_every)
            with self.mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
                while viewer.is_running():
                    command = keyboard_command()
                    if command == "q":
                        break
                    if command == "r":
                        self.recalibrate()

                    master_state = self.master.read_state()
                    self.master.set_zero_force()
                    orientation_error, insertion_lag, measured_arm_qpos, target_position_filt = self.motion.update_arm(master_state)
                    jaw_target = self.jaw.update(master_state)

                    self.mujoco.mj_step(self.robot.model, self.robot.data)
                    viewer.sync()

                    now = time.perf_counter()
                    print_due = now - last_print >= self.args.print_every
                    log_due = self.logger.enabled and now - last_log >= log_every
                    if print_due or log_due:
                        tip_position = point_world_pos(self.robot.data, self.robot.track_body_id)
                        position_err_norm = float(np.linalg.norm(target_position_filt - tip_position))
                        freq_khz = self.master.com_freq()
                    if log_due:
                        last_log = now
                        self.logger.write(
                            self.log_row(
                                now,
                                master_state,
                                target_position_filt,
                                tip_position,
                                position_err_norm,
                                orientation_error,
                                measured_arm_qpos,
                                insertion_lag,
                                jaw_target,
                                freq_khz,
                            )
                        )
                    if print_due:
                        last_print = now
                        self.print_status(
                            master_state,
                            target_position_filt,
                            tip_position,
                            position_err_norm,
                            orientation_error,
                            measured_arm_qpos,
                            freq_khz,
                        )
        finally:
            self.logger.close()
            self.master.close()
            print("\nDevice closed.")

    def log_row(
        self,
        now: float,
        master_state,
        target_position_filt: np.ndarray,
        tip_position: np.ndarray,
        position_err_norm: float,
        orientation_error: np.ndarray,
        measured_arm_qpos: np.ndarray,
        insertion_lag: float,
        jaw_target: float,
        freq_khz: float,
    ) -> dict[str, str]:
        arm = self.robot.arm_actuators
        rpy = self.motion.rpy_debug
        return {
            "t": f"{now:.6f}",
            "orientation_mode": self.args.orientation_mode,
            "track_body": self.robot.track_body_name,
            "orientation_body": self.robot.orientation_body_name,
            "jaw_mode": self.args.jaw_mode,
            "rpy_mode": self.args.rpy_yaw_pitch_mode,
            "rpy_input_method": self.args.rpy_input_method,
            "master_yaw_axis": self.args.master_yaw_axis,
            "master_pitch_axis": self.args.master_pitch_axis,
            "master_roll_axis": self.args.master_roll_axis,
            "sigma_x": f"{master_state.position[0]:.6f}",
            "sigma_y": f"{master_state.position[1]:.6f}",
            "sigma_z": f"{master_state.position[2]:.6f}",
            "target_x": f"{target_position_filt[0]:.6f}",
            "target_y": f"{target_position_filt[1]:.6f}",
            "target_z": f"{target_position_filt[2]:.6f}",
            "tip_x": f"{tip_position[0]:.6f}",
            "tip_y": f"{tip_position[1]:.6f}",
            "tip_z": f"{tip_position[2]:.6f}",
            "err_m": f"{position_err_norm:.6f}",
            "ori_err_norm": f"{float(np.linalg.norm(orientation_error)):.6f}",
            "ins_q": f"{measured_arm_qpos[2]:.6f}",
            "ins_ctrl": f"{self.robot.data.ctrl[arm[2]]:.6f}",
            "ins_lag": f"{insertion_lag:.6f}",
            "jaw_deg": f"{master_state.gripper_deg:.6f}",
            "jaw_target": f"{jaw_target:.6f}",
            "jaw_ctrl": f"{self.robot.data.ctrl[self.robot.jaw_actuator]:.6f}",
            "freq_khz": f"{freq_khz:.6f}",
            "rpy_in_yaw_deg": f"{math.degrees(rpy.sigma[0]):.6f}",
            "rpy_in_pitch_deg": f"{math.degrees(rpy.sigma[1]):.6f}",
            "rpy_in_roll_deg": f"{math.degrees(rpy.sigma[2]):.6f}",
            "rpy_tgt_yaw": f"{rpy.target[0]:.6f}",
            "rpy_tgt_pitch": f"{rpy.target[1]:.6f}",
            "rpy_tgt_roll": f"{rpy.target[2]:.6f}",
            "rpy_q_yaw": f"{measured_arm_qpos[0]:.6f}",
            "rpy_q_pitch": f"{measured_arm_qpos[1]:.6f}",
            "rpy_q_roll": f"{measured_arm_qpos[3]:.6f}",
            "rpy_bias_yaw": f"{rpy.bias[0]:.6f}",
            "rpy_bias_pitch": f"{rpy.bias[1]:.6f}",
            "ctrl_yaw": f"{self.robot.data.ctrl[arm[0]]:.6f}",
            "ctrl_pitch": f"{self.robot.data.ctrl[arm[1]]:.6f}",
            "ctrl_insertion": f"{self.robot.data.ctrl[arm[2]]:.6f}",
            "ctrl_roll": f"{self.robot.data.ctrl[arm[3]]:.6f}",
            "ctrl_wrist_pitch": f"{self.robot.data.ctrl[arm[4]]:.6f}",
            "ctrl_wrist_yaw": f"{self.robot.data.ctrl[arm[5]]:.6f}",
        }

    def print_status(
        self,
        master_state,
        target_position_filt: np.ndarray,
        tip_position: np.ndarray,
        position_err_norm: float,
        orientation_error: np.ndarray,
        measured_arm_qpos: np.ndarray,
        freq_khz: float,
    ) -> None:
        arm = self.robot.arm_actuators
        rpy = self.motion.rpy_debug
        print(
            "\r"
            f"sigma=({master_state.position[0]:+.3f},{master_state.position[1]:+.3f},{master_state.position[2]:+.3f}) m "
            f"target=({target_position_filt[0]:+.3f},{target_position_filt[1]:+.3f},{target_position_filt[2]:+.3f}) m "
            f"{self.robot.track_body_name}=({tip_position[0]:+.3f},{tip_position[1]:+.3f},{tip_position[2]:+.3f}) m "
            f"err={position_err_norm:.4f} m "
            f"ori_err={float(np.linalg.norm(orientation_error)):.4f} "
            f"ins(q/c)=({measured_arm_qpos[2]:+.4f}/{self.robot.data.ctrl[arm[2]]:+.4f}) "
            f"jaw={master_state.gripper_deg:5.1f} deg/{self.robot.data.ctrl[self.robot.jaw_actuator]:+.3f} ctrl "
            f"freq={freq_khz:.2f} kHz"
            + (
                " "
                f"rpy_in=({math.degrees(rpy.sigma[0]):+.1f},"
                f"{math.degrees(rpy.sigma[1]):+.1f},"
                f"{math.degrees(rpy.sigma[2]):+.1f})deg "
                f"rpy_tgt=({rpy.target[0]:+.3f},"
                f"{rpy.target[1]:+.3f},"
                f"{rpy.target[2]:+.3f}) "
                f"rpy_q=({measured_arm_qpos[0]:+.3f},"
                f"{measured_arm_qpos[1]:+.3f},"
                f"{measured_arm_qpos[3]:+.3f}) "
                f"rpy_mode={self.args.rpy_yaw_pitch_mode} "
                f"bias=({rpy.bias[0]:+.4f},{rpy.bias[1]:+.4f})"
                if self.args.debug_rpy
                else ""
            ),
            end="",
            flush=True,
        )


def main() -> None:
    TeleopApp(parse_args()).run()


if __name__ == "__main__":
    main()
