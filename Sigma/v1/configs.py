"""CLI and config profile handling for the v1 teleop app."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


V1_DIR = Path(__file__).resolve().parent
SIGMA_DIR = V1_DIR.parent
PROJECT_DIR = SIGMA_DIR.parent
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"

if str(SIGMA_DIR) not in sys.path:
    sys.path.insert(0, str(SIGMA_DIR))

try:
    from config import DEFAULT_PROFILE, TELEOP_CONFIGS
except ImportError:
    DEFAULT_PROFILE = None
    TELEOP_CONFIGS = {}

# Default v1 strategy. Change this value in configs.py to switch the app's
# normal behavior without changing the launch command.
ACTIVE_PROFILE = DEFAULT_PROFILE or "split_antijitter_keep_rot"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate the MuJoCo dVRK PSM with a Force Dimension sigma.7")
    parser.add_argument("--config-profile", default=None, help="Override ACTIVE_PROFILE with a Sigma/config.py TELEOP_CONFIGS profile")
    parser.add_argument("--list-config-profiles", action="store_true", help="List available config profiles and exit")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo PSM XML")
    parser.add_argument("--sdk-bin", default=None, help="Directory containing dhd64.dll and drd64.dll")
    parser.add_argument("--no-drd-init", action="store_true", help="Skip DRD auto-init and only use DHD open/read calls")
    parser.add_argument("--orientation-mode", choices=("none", "roll", "rpy", "full", "hybrid", "split"), default="none", help="How sigma.7 orientation drives the PSM")
    parser.add_argument("--track-body", default="tool_tip", help="MuJoCo body used as the Cartesian control point")
    parser.add_argument("--orientation-body", default=None, help="MuJoCo body used as the orientation control point; defaults to --track-body")
    parser.add_argument("--position-frame", choices=("world", "tool-home"), default="world", help="Frame used to map sigma.7 translation into MuJoCo target motion")
    parser.add_argument("--scale-x", type=float, default=0.55, help="Sigma X meters to teleop X meters")
    parser.add_argument("--scale-y", type=float, default=0.55, help="Sigma Y meters to teleop Y meters")
    parser.add_argument("--scale-z", type=float, default=0.25, help="Sigma Z meters to teleop Z meters")
    parser.add_argument("--master-yaw-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local Z before mapping")
    parser.add_argument("--master-pitch-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local Y before mapping")
    parser.add_argument("--master-roll-deg", type=float, default=0.0, help="Rotate sigma translation/orientation frame around local X before mapping")
    parser.add_argument("--home-insertion", type=float, default=0.12, help="Initial PSM insertion target in meters before calibration")
    parser.add_argument("--damping-pos", type=float, default=3e-4, help="Damped least-squares IK damping for position-only mode")
    parser.add_argument("--damping-full", type=float, default=3e-3, help="Damped least-squares IK damping for full-pose mode")
    parser.add_argument("--ik-pos-weight", type=float, default=12.0, help="Hybrid IK task weight for Cartesian position")
    parser.add_argument("--ik-ori-weight", type=float, default=1.0, help="Hybrid IK task weight for orientation")
    parser.add_argument("--ik-pos-weight-x", type=float, default=4.0, help="Hybrid IK extra task weight on world X")
    parser.add_argument("--ik-pos-weight-y", type=float, default=4.0, help="Hybrid IK extra task weight on world Y")
    parser.add_argument("--ik-pos-weight-z", type=float, default=1.0, help="Hybrid IK extra task weight on world Z")
    parser.add_argument("--ik-wrist-damping-scale", type=float, default=2.0, help="Hybrid IK extra damping on roll/wrist_pitch/wrist_yaw")
    parser.add_argument("--position-gain", type=float, default=1.0, help="Task-space gain applied to position error")
    parser.add_argument("--orientation-gain", type=float, default=0.35, help="Task-space gain applied to orientation error in full mode")
    parser.add_argument("--orientation-scale", type=float, default=1.0, help="Scale factor for sigma orientation relative motion")
    parser.add_argument("--master-yaw-axis", choices=("x", "y", "z"), default="z", help="sigma.7 local rotation axis mapped to the PSM yaw joint in rpy mode")
    parser.add_argument("--master-pitch-axis", choices=("x", "y", "z"), default="y", help="sigma.7 local rotation axis mapped to the PSM pitch joint in rpy mode")
    parser.add_argument("--master-roll-axis", choices=("x", "y", "z"), default="z", help="sigma.7 local rotation axis mapped to the PSM roll joint in roll mode")
    parser.add_argument("--rpy-input-method", choices=("local-angle", "rotvec"), default="local-angle", help="How sigma.7 relative rotation is decomposed for rpy mode")
    parser.add_argument("--yaw-scale", type=float, default=0.35, help="Scale sigma yaw to PSM yaw bias in rpy mode")
    parser.add_argument("--pitch-scale", type=float, default=0.35, help="Scale sigma pitch to PSM pitch bias in rpy mode")
    parser.add_argument("--roll-scale", type=float, default=1.0, help="Scale sigma roll to PSM roll in roll mode")
    parser.add_argument("--rpy-yaw-pitch-mode", choices=("blend", "direct", "bias"), default="blend", help="How rpy mode applies sigma yaw/pitch to the PSM main-chain joints")
    parser.add_argument("--rpy-direct-weight", type=float, default=0.35, help="Blend weight for direct yaw/pitch targets in rpy blend mode")
    parser.add_argument("--max-rpy-yaw-deg", type=float, default=20.0, help="Max yaw offset from calibration home in rpy mode")
    parser.add_argument("--max-rpy-pitch-deg", type=float, default=20.0, help="Max pitch offset from calibration home in rpy mode")
    parser.add_argument("--max-rpy-roll-deg", type=float, default=45.0, help="Max roll offset from calibration home in rpy/roll mode")
    parser.add_argument("--rpy-bias-gain", type=float, default=0.25, help="Blend gain for yaw/pitch joint bias after position IK in rpy mode")
    parser.add_argument("--max-rpy-bias-step", type=float, default=0.006, help="Max yaw/pitch bias step per frame in rpy mode")
    parser.add_argument("--debug-rpy", action="store_true", help="Print sigma rpy deltas and mapped PSM yaw/pitch/roll targets")
    parser.add_argument("--max-position-error", type=float, default=0.012, help="Max Cartesian correction per frame in meters")
    parser.add_argument("--max-orientation-error", type=float, default=0.20, help="Max orientation error vector norm per frame in radians")
    parser.add_argument("--max-joint-step", type=float, default=0.025, help="Max joint-space IK step norm per frame")
    parser.add_argument("--deadband-pos", type=float, default=2e-4, help="Ignore smaller position errors in meters")
    parser.add_argument("--deadband-rot", type=float, default=2e-3, help="Ignore smaller rotation errors in radians")
    parser.add_argument("--target-smooth", type=float, default=0.45, help="Low-pass alpha for target position/orientation (0..1)")
    parser.add_argument("--dq-smooth", type=float, default=0.55, help="Low-pass alpha for joint-space IK steps (0..1)")
    parser.add_argument("--dq-deadband", type=float, default=0.0, help="Zero IK joint steps smaller than this value")
    parser.add_argument("--split-distal-smooth", type=float, default=None, help="Optional split-mode low-pass alpha for roll/wrist_pitch/wrist_yaw dq")
    parser.add_argument("--ctrl-smooth", type=float, default=0.55, help="Low-pass alpha for actuator targets (0..1)")
    parser.add_argument("--max-ctrl-step-pos", type=float, default=0.018, help="Max per-frame change for yaw/pitch/wrist rotary joints in radians")
    parser.add_argument("--max-ctrl-step-ins", type=float, default=0.0006, help="Max per-frame change for insertion in meters")
    parser.add_argument("--max-ctrl-step-roll", type=float, default=0.006, help="Max per-frame change for roll in radians")
    parser.add_argument("--max-insertion-servo-lag", type=float, default=0.008, help="Freeze insertion IK when measured qpos and actuator target differ by more than this many meters")
    parser.add_argument("--hybrid-insertion-dq-scale", type=float, default=0.35, help="Scale hybrid/split insertion dq to reduce chatter")
    parser.add_argument("--hybrid-dq-smooth-ins", type=float, default=0.18, help="Extra low-pass alpha for hybrid/split insertion dq")
    parser.add_argument("--hybrid-use-servo-lag-guard", action="store_true", help="Apply insertion servo-lag guard in hybrid/split mode")
    parser.add_argument("--split-distal-step-scale", type=float, default=1.0, help="Scale split-mode distal orientation dq before filtering")
    parser.add_argument("--limit-slowdown-margin", type=float, default=0.08, help="Slow joints near actuator limits within this margin")
    parser.add_argument("--limit-slowdown-min-scale", type=float, default=0.15, help="Minimum joint step scale near actuator limits")
    parser.add_argument("--jaw-mode", choices=("locked", "follow"), default="locked", help="Lock jaw at calibration home or follow sigma.7 gripper")
    parser.add_argument("--jaw-lock-value", type=float, default=None, help="Optional fixed jaw actuator target in radians when --jaw-mode locked")
    parser.add_argument("--gripper-close-deg", type=float, default=0.0, help="sigma.7 gripper angle treated as closed")
    parser.add_argument("--gripper-open-deg", type=float, default=30.0, help="sigma.7 gripper angle treated as open")
    parser.add_argument("--jaw-invert", action="store_true", help="Invert sigma.7 gripper to jaw opening direction")
    parser.add_argument("--jaw-smooth", type=float, default=0.25, help="Low-pass alpha for jaw actuator target in follow mode")
    parser.add_argument("--jaw-deadband", type=float, default=0.002, help="Ignore smaller jaw actuator target changes in radians")
    parser.add_argument("--max-ctrl-step-jaw", type=float, default=0.02, help="Max per-frame jaw actuator target change in radians")
    parser.add_argument("--print-every", type=float, default=0.25, help="Status print period in seconds")
    parser.add_argument("--log-csv", default=None, help="Optional CSV path for teleop debug logs")
    parser.add_argument("--log-every", type=float, default=None, help="CSV log period in seconds; defaults to --print-every")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    profile_args, _ = parser.parse_known_args()
    if profile_args.list_config_profiles:
        if TELEOP_CONFIGS:
            print("Available config profiles:")
            for name in sorted(TELEOP_CONFIGS):
                default_mark = " (default)" if name == DEFAULT_PROFILE else ""
                active_mark = " [active]" if name == ACTIVE_PROFILE else ""
                print(f"  {name}{default_mark}{active_mark}")
        else:
            print("No config profiles found.")
        raise SystemExit(0)

    profile_name = profile_args.config_profile or ACTIVE_PROFILE
    if profile_name:
        if profile_name not in TELEOP_CONFIGS:
            available = ", ".join(sorted(TELEOP_CONFIGS)) or "none"
            raise ValueError(f"Unknown config profile '{profile_name}'. Available profiles: {available}")
        parser.set_defaults(**TELEOP_CONFIGS[profile_name])

    return parser.parse_args()
