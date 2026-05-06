"""
Generate and load a stable, actuator-controlled dVRK PSM model in MuJoCo.

Outputs:
    Assest/psm_official/psm_control.xml

Run:
    python Src/loadpsm_control.py
    python Src/loadpsm_control.py --no-gui

The MuJoCo GUI exposes one control slider per position actuator, so each PSM
joint can be moved directly from the viewer.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_DIR / "Assest" / "psm_official"
DEBUG_URDF = ASSET_DIR / "psm1_sca_mujoco.urdf"
CONTROL_XML = ASSET_DIR / "psm_control.xml"

VISUAL_ALPHA = "1"
DEFAULT_BODY_MASS = "0.02"
DEFAULT_BODY_INERTIA = "0.0001 0.0001 0.0001"

# Official xacro uses URDF mimic tags for the parallel-link visuals and jaw
# fingers. MuJoCo does not import those tags from URDF, so we recreate the
# relationships as equality constraints in the generated MJCF.
MIMIC_JOINTS = {
    "pitch_1": ("pitch", 1.0),
    "pitch_2": ("pitch", 1.0),
    "pitch_3": ("pitch", -1.0),
    "pitch_4": ("pitch", -1.0),
    "pitch_5": ("pitch", 1.0),
    "jaw_mimic_1": ("jaw", 0.5),
    "jaw_mimic_2": ("jaw", -0.5),
}


def fvec(text: str | None, default: str = "0 0 0") -> str:
    return text if text else default


def urdf_rpy_to_mujoco_quat(rpy: str | None) -> str:
    """Convert URDF fixed-axis RPY to MuJoCo's w x y z quaternion."""
    roll, pitch, yaw = (float(value) for value in fvec(rpy).split())

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    # URDF RPY is fixed-axis X-Y-Z, equivalent to Rz(yaw) * Ry(pitch) * Rx(roll).
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return f"{w / norm:.12g} {x / norm:.12g} {y / norm:.12g} {z / norm:.12g}"


def parse_xyz_quat(origin: ET.Element | None) -> tuple[str, str]:
    if origin is None:
        return "0 0 0", "1 0 0 0"
    return fvec(origin.get("xyz")), urdf_rpy_to_mujoco_quat(origin.get("rpy"))


def clean_name(name: str) -> str:
    return (
        name.replace("PSM1_", "")
        .replace("_link", "")
        .replace("tool_wrist_sca_", "sca_")
        .replace("tool_wrist_", "wrist_")
    )


def collect_model_data() -> tuple[dict[str, ET.Element], list[dict], dict[str, list[dict]]]:
    if not DEBUG_URDF.exists():
        raise FileNotFoundError(
            f"Missing {DEBUG_URDF}. Run Src/load_dvrk_psm.py --no-gui first."
        )

    root = ET.parse(DEBUG_URDF).getroot()

    links = {link.get("name"): link for link in root.findall("link") if link.get("name")}
    joints = []
    child_to_joint = {}

    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type", "fixed")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        pos, quat = parse_xyz_quat(joint.find("origin"))
        axis = fvec(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, "0 0 1")
        limit = joint.find("limit")
        lower = limit.get("lower") if limit is not None and limit.get("lower") is not None else None
        upper = limit.get("upper") if limit is not None and limit.get("upper") is not None else None
        effort = limit.get("effort") if limit is not None and limit.get("effort") is not None else "50"

        info = {
            "name": name,
            "type": jtype,
            "parent": parent,
            "child": child,
            "pos": pos,
            "quat": quat,
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "effort": effort,
        }
        joints.append(info)
        child_to_joint[child] = info

    children: dict[str, list[dict]] = {}
    for joint in joints:
        children.setdefault(joint["parent"], []).append(joint)

    return links, children, child_to_joint


def link_meshes(link: ET.Element) -> list[dict[str, str]]:
    geoms = []
    for visual in link.findall("visual"):
        geom = visual.find("geometry")
        mesh = geom.find("mesh") if geom is not None else None
        if mesh is None or mesh.get("filename") is None:
            continue
        pos, quat = parse_xyz_quat(visual.find("origin"))
        geoms.append({
            "file": mesh.get("filename"),
            "pos": pos,
            "quat": quat,
        })
    return geoms


def mj_joint_type(urdf_type: str) -> str | None:
    if urdf_type in ("fixed",):
        return None
    if urdf_type == "prismatic":
        return "slide"
    return "hinge"


def joint_range(joint: dict) -> tuple[str, str]:
    if joint["lower"] is not None and joint["upper"] is not None:
        return joint["lower"], joint["upper"]
    if joint["type"] == "continuous":
        return str(-math.pi), str(math.pi)
    if joint["type"] == "prismatic":
        return "0", "0.24"
    return str(-math.pi), str(math.pi)


def actuator_params(joint: dict) -> tuple[str, str]:
    if joint["type"] == "prismatic":
        return "400", "120"
    if "jaw" in joint["name"]:
        return "15", "20"
    return "60", "80"


def generate_control_xml() -> Path:
    links, children, _ = collect_model_data()

    mesh_files = []
    for link in links.values():
        for geom in link_meshes(link):
            if geom["file"] not in mesh_files:
                mesh_files.append(geom["file"])

    mesh_names = {file: f"mesh_{idx}_{Path(file).stem}" for idx, file in enumerate(mesh_files)}
    actuated_joints: list[dict] = []

    lines: list[str] = []
    lines.append('<mujoco model="dvrk_psm_official_control">')
    lines.append('  <compiler angle="radian" meshdir="." autolimits="true"/>')
    lines.append('  <option timestep="0.002" iterations="80" integrator="implicitfast"/>')
    lines.append('  <statistic center="-0.25 0 0.5" extent="1.2"/>')
    lines.append("")
    lines.append("  <visual>")
    lines.append('    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>')
    lines.append('    <global azimuth="135" elevation="-20"/>')
    lines.append("  </visual>")
    lines.append("")
    lines.append("  <asset>")
    lines.append('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.45 0.65" rgb2="0 0 0" width="512" height="3072"/>')
    lines.append('    <material name="psm_mat" rgba="0.8 0.8 0.78 1" specular="0.25" shininess="0.4"/>')
    for file in mesh_files:
        lines.append(f'    <mesh name="{mesh_names[file]}" file="{file}"/>')
    lines.append("  </asset>")
    lines.append("")
    lines.append("  <worldbody>")
    lines.append('    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>')
    lines.append('    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.25 0.3 0.35 1"/>')

    def write_body(link_name: str, joint: dict | None, indent: int) -> None:
        link = links[link_name]
        pad = " " * indent
        body_name = clean_name(link_name)
        pos = joint["pos"] if joint else "0 0 0"
        quat = joint["quat"] if joint else "1 0 0 0"
        lines.append(f'{pad}<body name="{body_name}" pos="{pos}" quat="{quat}">')
        lines.append(
            f'{pad}  <inertial pos="0 0 0" mass="{DEFAULT_BODY_MASS}" '
            f'diaginertia="{DEFAULT_BODY_INERTIA}"/>'
        )

        if joint is not None:
            mj_type = mj_joint_type(joint["type"])
            if mj_type:
                lo, hi = joint_range(joint)
                lines.append(
                    f'{pad}  <joint name="{joint["name"]}" type="{mj_type}" axis="{joint["axis"]}" '
                    f'range="{lo} {hi}" damping="2" armature="0.02" frictionloss="0.01"/>'
                )
                if joint["name"] not in MIMIC_JOINTS:
                    actuated_joints.append(joint)

        for idx, geom in enumerate(link_meshes(link)):
            lines.append(
                f'{pad}  <geom name="{body_name}_visual_{idx}" type="mesh" '
                f'mesh="{mesh_names[geom["file"]]}" material="psm_mat" '
                f'pos="{geom["pos"]}" quat="{geom["quat"]}" '
                f'contype="0" conaffinity="0" group="2" density="0"/>'
            )

        for child_joint in children.get(link_name, []):
            write_body(child_joint["child"], child_joint, indent + 2)

        lines.append(f"{pad}</body>")

    # The generated official debug URDF uses `world` as root.
    for root_joint in children.get("world", []):
        write_body(root_joint["child"], root_joint, 4)

    lines.append("  </worldbody>")
    lines.append("")
    lines.append("  <equality>")
    for mimic_name, (source_name, multiplier) in MIMIC_JOINTS.items():
        lines.append(
            f'    <joint name="eq_{mimic_name}" joint1="{mimic_name}" joint2="{source_name}" '
            f'polycoef="0 {multiplier:g} 0 0 0" solref="0.005 1"/>'
        )
    lines.append("  </equality>")
    lines.append("")
    lines.append("  <actuator>")
    for joint in actuated_joints:
        lo, hi = joint_range(joint)
        kp, force = actuator_params(joint)
        lines.append(
            f'    <position name="ctrl_{joint["name"]}" joint="{joint["name"]}" '
            f'kp="{kp}" ctrlrange="{lo} {hi}" forcelimited="true" forcerange="-{force} {force}"/>'
        )
    lines.append("  </actuator>")
    lines.append("")
    lines.append("</mujoco>")

    CONTROL_XML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CONTROL_XML


def load_model(xml_path: Path, show_gui: bool) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"Loaded control XML: {xml_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}")
    print("Actuator sliders:")
    for aid in range(model.nu):
        print(f"  {aid:02d}: {model.actuator(aid).name}")

    if show_gui:
        print("\nOpen the MuJoCo Control panel to move each joint with sliders.")
        mujoco.viewer.launch(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load stable actuator-controlled dVRK PSM model")
    parser.add_argument("--no-gui", action="store_true", help="Only compile the XML and print joint controls")
    parser.add_argument("--use-existing", action="store_true", help="Do not regenerate psm_control.xml")
    args = parser.parse_args()

    if args.use_existing and CONTROL_XML.exists():
        xml_path = CONTROL_XML
    else:
        xml_path = generate_control_xml()
        print(f"Generated: {xml_path}")

    load_model(xml_path, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
