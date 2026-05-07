"""CLI configuration for the v2 sigma.7 split controller."""
from __future__ import annotations

import argparse
from pathlib import Path


V2_DIR = Path(__file__).resolve().parent
PROJECT_DIR = V2_DIR.parent.parent
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate MuJoCo dVRK PSM with sigma.7 using v2 split IK")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo PSM XML")
    parser.add_argument("--sdk-bin", default=None, help="Directory containing dhd64.dll and drd64.dll")
    parser.add_argument("--no-drd-init", action="store_true", help="Skip DRD auto-init and only use DHD open/read calls")
    parser.add_argument("--debug-sdk", action="store_true", help="Print Force Dimension SDK status after opening")
    parser.add_argument("--track-body", default="tool_tip", help="MuJoCo body used as the translational control point")
    parser.add_argument("--home-insertion", type=float, default=0.12, help="Initial insertion actuator target")

    parser.add_argument("--scale-x", type=float, default=0.55, help="Sigma X delta to world X delta scale")
    parser.add_argument("--scale-y", type=float, default=0.55, help="Sigma Y delta to world Y delta scale")
    parser.add_argument("--scale-z", type=float, default=0.45, help="Sigma Z delta to world Z delta scale")
    parser.add_argument("--sigma-deadband", type=float, default=0.00008, help="Ignore smaller sigma translation increments")
    parser.add_argument("--max-world-increment", type=float, default=0.0015, help="Max world target increment per sim step")
    parser.add_argument("--target-smooth", type=float, default=0.25, help="Low-pass alpha for target position")
    parser.add_argument("--position-gain", type=float, default=8.0, help="Task-space proportional gain")
    parser.add_argument("--max-target-error", type=float, default=0.012, help="Max Cartesian correction vector norm")
    parser.add_argument("--deadband-pos", type=float, default=2.0e-5, help="Ignore smaller tool position errors")

    parser.add_argument("--damping-min", type=float, default=1.0e-4, help="DLS damping away from singularities")
    parser.add_argument("--damping-max", type=float, default=8.0e-2, help="DLS damping near singularities")
    parser.add_argument("--singular-value-safe", type=float, default=0.035, help="Singular value threshold for adaptive damping")
    parser.add_argument("--dq-smooth", type=float, default=0.35, help="Low-pass alpha for IK joint velocity")
    parser.add_argument("--max-step-yaw", type=float, default=0.010, help="Max yaw actuator step per frame")
    parser.add_argument("--max-step-pitch", type=float, default=0.010, help="Max pitch actuator step per frame")
    parser.add_argument("--max-step-insertion", type=float, default=0.0015, help="Max insertion actuator step per frame")

    parser.add_argument("--wrist-scale", type=float, default=1.0, help="Scale sigma orientation increments to PSM wrist increments")
    parser.add_argument("--max-wrist-step", type=float, default=0.006, help="Max wrist actuator step per frame")
    parser.add_argument("--jaw-mode", choices=("locked", "follow"), default="follow", help="Lock jaw or follow sigma.7 gripper")
    parser.add_argument("--gripper-close-deg", type=float, default=0.0, help="sigma.7 gripper angle treated as closed")
    parser.add_argument("--gripper-open-deg", type=float, default=1.6, help="sigma.7 gripper angle treated as open")
    parser.add_argument("--jaw-invert", action="store_true", help="Invert gripper-to-jaw mapping")
    parser.add_argument("--jaw-smooth", type=float, default=0.20, help="Low-pass alpha for jaw target")
    parser.add_argument("--max-jaw-step", type=float, default=0.03, help="Max jaw actuator step per frame")

    parser.add_argument("--print-every", type=float, default=0.10, help="Status print period in seconds")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()

