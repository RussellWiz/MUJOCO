from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, cast


mujoco = cast(Any, import_module("mujoco"))


DEFAULT_COMPILER_ATTRS = {
    "autolimits": "true",
    "balanceinertia": "true",
    "inertiafromgeom": "true",
    "alignfree": "true",
    "saveinertial": "true",
    "discardvisual": "false",
    "boundmass": "1e-8",
    "boundinertia": "1e-10",
}


MESH_ALIASES = {
    "meshes/visual/CDF_base_no_shaft.obj": "meshes/visual/tool_roll_link.STL",
    "meshes/collision/CDF_base_no_shaft.obj": "meshes/collision/tool_roll_link.STL",
    "meshes/visual/CDF_base.obj": "meshes/visual/main_insertion_link_3.obj",
}


def build_augmented_urdf(source_urdf: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Expected a URDF robot root element, got {root.tag!r}")

    mujoco_block = root.find("mujoco")
    if mujoco_block is None:
        mujoco_block = ET.Element("mujoco")
        root.insert(0, mujoco_block)

    default_block = mujoco_block.find("default")
    if default_block is None:
        default_block = ET.SubElement(mujoco_block, "default")

    mesh_default = default_block.find("mesh")
    if mesh_default is None:
        mesh_default = ET.SubElement(default_block, "mesh")
    if "inertia" not in mesh_default.attrib:
        mesh_default.set("inertia", "shell")

    compiler = mujoco_block.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_block, "compiler")

    for key, value in DEFAULT_COMPILER_ATTRS.items():
        if key not in compiler.attrib:
            compiler.set(key, value)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue

        resolved = source_urdf.parent / filename
        if resolved.exists():
            continue

        alias = MESH_ALIASES.get(filename)
        if alias is not None:
            mesh.set("filename", alias)

    temp_file = tempfile.NamedTemporaryFile(
        dir=source_urdf.parent, prefix=f"{source_urdf.stem}_mujoco_", suffix=".urdf", delete=False
    )
    augmented_urdf = Path(temp_file.name)
    temp_file.close()
    tree.write(augmented_urdf, encoding="utf-8", xml_declaration=True)
    return augmented_urdf


def convert_urdf_to_mjcf(source_urdf: Path, output_xml: Path) -> None:
    source_urdf = source_urdf.resolve()
    output_xml = output_xml.resolve()

    augmented_urdf = build_augmented_urdf(source_urdf)

    try:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        model = mujoco.MjModel.from_xml_path(str(augmented_urdf))
        mujoco.mj_saveLastXML(str(output_xml), model)
    finally:
        if "model" in locals() and hasattr(model, "delete"):
            model.delete()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert psm.urdf to MJCF using MuJoCo's official compiler path."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "Assest" / "psm" / "psm.urdf",
        type=Path,
        help="Input URDF file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "Assest" / "psm_mjcf" / "psm" / "psm.xml",
        type=Path,
        help="Output MJCF file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_urdf_to_mjcf(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()