"""Visualize the official dVRK PSM teleoperation setpoint in MuJoCo.

This viewer highlights:
  - official teleop setpoint equivalent in this MuJoCo model: `tool_tip`
  - V0 pivot point used in earlier experiments: `tool_wrist`
  - insertion body origin: `tool_main`

Run from the project root:
    python Sigma/visualize_official_psm_setpoint.py

If your environment has an alternate XML:
    python Sigma/visualize_official_psm_setpoint.py --xml Assest/psm_official/psm_control_dynamics.xml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"

OFFICIAL_POINT_BODY = "tool_tip"
V0_POINT_BODY = "tool_wrist"
INSERTION_BODY = "tool_main"


def body_world_pos(data, body_id: int) -> np.ndarray:
    return data.xpos[body_id].copy()


def body_world_rot(data, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


def make_rotation_columns(scale: float = 1.0) -> np.ndarray:
    return np.eye(3, dtype=float) * scale


def set_geom_sphere(mujoco, geom, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=float),
        pos,
        np.eye(3, dtype=float).reshape(-1),
        rgba,
    )


def set_geom_capsule_between(mujoco, geom, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    midpoint = 0.5 * (start + end)
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-9:
        set_geom_sphere(mujoco, geom, midpoint, radius, rgba)
        return

    z_axis = segment / length
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(z_axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = np.cross(helper, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.5 * length, 0.0], dtype=float),
        midpoint,
        rotation.reshape(-1),
        rgba,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the dVRK official teleop setpoint in MuJoCo")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo PSM XML")
    parser.add_argument("--home-insertion", type=float, default=0.12, help="Initial insertion in meters")
    parser.add_argument("--official-radius", type=float, default=0.006, help="Marker radius for official setpoint")
    parser.add_argument("--v0-radius", type=float, default=0.005, help="Marker radius for V0 point")
    parser.add_argument("--main-radius", type=float, default=0.008, help="Marker radius for insertion body origin")
    parser.add_argument("--fade-robot-alpha", type=float, default=0.45, help="Alpha applied to robot visual geoms so markers are easier to see")
    parser.add_argument("--print-every", type=float, default=0.5, help="Status print period in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The active Python environment does not have the 'mujoco' package. "
            "Install it with 'pip install mujoco' or switch to the environment used for your MuJoCo scripts."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Fade the robot visuals so point markers are easier to see through the mesh.
    if model.ngeom > 0:
        model.geom_rgba[:, 3] = np.minimum(model.geom_rgba[:, 3], float(np.clip(args.fade_robot_alpha, 0.05, 1.0)))

    body_ids = {}
    for name in (OFFICIAL_POINT_BODY, V0_POINT_BODY, INSERTION_BODY):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {name}")
        body_ids[name] = body_id

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ctrl_insertion")
    if actuator_id >= 0:
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = float(np.clip(args.home_insertion, ctrl_range[0], ctrl_range[1]))

    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    print(f"Loaded MuJoCo XML: {xml_path}")
    print("MuJoCo marker legend:")
    print("  red sphere    = official dVRK teleop setpoint equivalent (`tool_tip`)")
    print("  blue sphere   = your V0 control point (`tool_wrist`)")
    print("  yellow sphere = insertion body origin (`tool_main`)")
    print("  yellow capsule= from tool_main to tool_wrist")
    print("  white capsule = from tool_wrist to tool_tip")
    print("Controls:")
    print("  q             quit")
    print("  r             reset insertion to the requested home value")

    last_print = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if hasattr(viewer, "cam"):
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.15
            viewer.cam.lookat[:] = np.array([-0.05, 0.05, 0.53], dtype=float)

        while viewer.is_running():
            command = keyboard_command()
            if command == "q":
                break
            if command == "r" and actuator_id >= 0:
                ctrl_range = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = float(np.clip(args.home_insertion, ctrl_range[0], ctrl_range[1]))

            mujoco.mj_forward(model, data)

            official_pos = body_world_pos(data, body_ids[OFFICIAL_POINT_BODY])
            v0_pos = body_world_pos(data, body_ids[V0_POINT_BODY])
            main_pos = body_world_pos(data, body_ids[INSERTION_BODY])

            viewer.user_scn.ngeom = 0

            set_geom_sphere(
                mujoco,
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                official_pos,
                args.official_radius,
                np.array([1.0, 0.1, 0.1, 0.95], dtype=float),
            )
            viewer.user_scn.ngeom += 1

            set_geom_sphere(
                mujoco,
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                v0_pos,
                args.v0_radius,
                np.array([0.1, 0.45, 1.0, 0.95], dtype=float),
            )
            viewer.user_scn.ngeom += 1

            set_geom_sphere(
                mujoco,
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                main_pos,
                args.main_radius,
                np.array([1.0, 0.95, 0.05, 1.0], dtype=float),
            )
            viewer.user_scn.ngeom += 1

            set_geom_capsule_between(
                mujoco,
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                main_pos,
                v0_pos,
                0.0020,
                np.array([1.0, 0.9, 0.1, 0.9], dtype=float),
            )
            viewer.user_scn.ngeom += 1

            set_geom_capsule_between(
                mujoco,
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                v0_pos,
                official_pos,
                0.0015,
                np.array([1.0, 1.0, 1.0, 0.8], dtype=float),
            )
            viewer.user_scn.ngeom += 1

            viewer.sync()

            now = time.perf_counter()
            if now - last_print >= args.print_every:
                last_print = now
                wrist_to_tip = float(np.linalg.norm(official_pos - v0_pos))
                main_to_wrist = float(np.linalg.norm(v0_pos - main_pos))
                print(
                    "\r"
                    f"tool_main=({main_pos[0]:+.3f},{main_pos[1]:+.3f},{main_pos[2]:+.3f}) "
                    f"tool_wrist=({v0_pos[0]:+.3f},{v0_pos[1]:+.3f},{v0_pos[2]:+.3f}) "
                    f"tool_tip=({official_pos[0]:+.3f},{official_pos[1]:+.3f},{official_pos[2]:+.3f}) "
                    f"| wrist->tip={wrist_to_tip:.4f} m "
                    f"| main->wrist={main_to_wrist:.4f} m",
                    end="",
                    flush=True,
                )


if __name__ == "__main__":
    main()
