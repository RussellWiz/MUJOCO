"""
Convert the official dVRK Classic PSM xacro assets into a MuJoCo-loadable
debug URDF, then optionally open the MuJoCo GUI.

The conversion result is written to:
    Assest/psm_official/

This script intentionally keeps the conversion steps in Python so the process
is reproducible on Windows without a ROS workspace.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "dvrk_model-main" / "dvrk_model-main"
TARGET_DIR = PROJECT_DIR / "Assest" / "psm_official"

CLASSIC_URDF_DIR = SOURCE_DIR / "urdf" / "Classic"
SOURCE_MESH_DIR = SOURCE_DIR / "meshes" / "Classic" / "PSM"

CONVERTED_URDF = TARGET_DIR / "psm1_sca_mujoco.urdf"

PSM_PREFIX = "PSM1_"
PSM_XYZ = "-0.25 0.0 0.5"
PSM_RPY = f"0.0 0.0 {math.pi}"


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_asset_folder() -> None:
    """Copy official source xacro and PSM meshes into Assest/psm_official."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    copytree_replace(CLASSIC_URDF_DIR, TARGET_DIR / "urdf" / "Classic")
    copytree_replace(SOURCE_MESH_DIR, TARGET_DIR / "meshes" / "Classic" / "PSM")
    shutil.copy2(SOURCE_DIR / "package.xml", TARGET_DIR / "package.xml")


def macro_body(text: str, macro_name: str) -> str:
    pattern = re.compile(
        rf"<xacro:macro\s+name=[\"']{re.escape(macro_name)}[\"'][^>]*>(.*?)</xacro:macro>",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Cannot find xacro macro: {macro_name}")
    return match.group(1)


def expand_xacro_exprs(text: str, values: dict[str, str]) -> str:
    """Expand simple ${...} expressions used by the official Classic PSM xacros."""
    env: dict[str, object] = {"PI": math.pi, **values}

    def replace_expr(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if expr in values:
            return values[expr]
        try:
            return str(eval(expr, {"__builtins__": {}}, env))
        except Exception as exc:
            raise ValueError(f"Unsupported xacro expression: {expr}") from exc

    return re.sub(r"\$\{([^}]+)\}", replace_expr, text)


def remove_xacro_leftovers(text: str) -> str:
    text = re.sub(r"\s*<xacro:[^>]+/>\s*", "\n", text)
    text = re.sub(r"\s*<xacro:[^>]+>.*?</xacro:[^>]+>\s*", "\n", text, flags=re.DOTALL)
    return text


def remove_mimic_tags(text: str) -> str:
    # MuJoCo does not preserve URDF mimic behavior reliably. For this visual
    # debug model, keep those joints independent and drive them manually later.
    return re.sub(r"\s*<mimic\b[^>]*/>\s*", "\n", text)


def replace_zero_boxes(text: str) -> str:
    return re.sub(r"<box\s+size=[\"']0\s+0\s+0[\"']\s*/>", '<box size="0.001 0.001 0.001"/>', text)


def inertial_block() -> str:
    return (
        '\n      <inertial>\n'
        '        <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '        <mass value="0.01"/>\n'
        '        <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>\n'
        '      </inertial>'
    )


def add_inertials(text: str) -> str:
    """Add simple positive inertials so MuJoCo can compile moving links."""

    def expand_self_closing(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "world":
            return match.group(0)
        return f'<link name="{name}">{inertial_block()}\n    </link>'

    text = re.sub(r"<link\s+name=[\"']([^\"']+)[\"']\s*/>", expand_self_closing, text)

    def inject(match: re.Match[str]) -> str:
        open_tag = match.group(1)
        name = match.group(2)
        rest = match.group(3)
        if name == "world" or "<inertial" in rest:
            return match.group(0)
        return f"{open_tag}{inertial_block()}{rest}</link>"

    return re.sub(r"(<link\s+name=[\"']([^\"']+)[\"']>)(.*?)</link>", inject, text, flags=re.DOTALL)


def duplicate_visuals_as_collisions(text: str) -> str:
    """Add collision tags from visual tags when the official xacro only has visuals."""

    def replace_link(match: re.Match[str]) -> str:
        link_text = match.group(0)
        if "<collision" in link_text or "<visual" not in link_text:
            return link_text

        visual_match = re.search(r"<visual>(.*?)</visual>", link_text, flags=re.DOTALL)
        if not visual_match:
            return link_text
        collision = "<collision>" + visual_match.group(1) + "</collision>"
        return link_text.replace("</visual>", "</visual>\n" + collision, 1)

    return re.sub(r"<link\s+name=[\"'][^\"']+[\"']>.*?</link>", replace_link, text, flags=re.DOTALL)


def build_flat_urdf() -> str:
    """Expand the Classic PSM1 + SCA xacro subset used in this project."""
    psm_base = macro_body((CLASSIC_URDF_DIR / "psm_base.urdf.xacro").read_text(encoding="utf-8"), "psm_base")
    psm_tool_sca = macro_body((CLASSIC_URDF_DIR / "psm_tool_sca.urdf.xacro").read_text(encoding="utf-8"), "psm_tool_sca")

    values = {
        "prefix": PSM_PREFIX,
        "parent_link": "world",
        "xyz": PSM_XYZ,
        "rpy": PSM_RPY,
    }

    body = "\n".join([
        '  <link name="world"/>',
        expand_xacro_exprs(psm_base, values),
        expand_xacro_exprs(psm_tool_sca, values),
    ])

    body = remove_xacro_leftovers(body)
    body = remove_mimic_tags(body)
    body = replace_zero_boxes(body)
    body = add_inertials(body)
    body = duplicate_visuals_as_collisions(body)

    return f'<?xml version="1.0"?>\n<robot name="dvrk_psm1_sca_official">\n{body}\n</robot>\n'


def dae_to_obj(src: Path, dst: Path) -> None:
    import trimesh

    loaded = trimesh.load(src, force="scene")
    if hasattr(loaded, "to_geometry"):
        mesh = loaded.to_geometry()
    elif hasattr(loaded, "dump"):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh after loading {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)


def convert_mesh_references(urdf_text: str) -> str:
    """Copy/convert referenced meshes and rewrite package:// paths to local paths."""
    converted_root = TARGET_DIR / "converted_meshes"
    converted_root.mkdir(parents=True, exist_ok=True)

    mesh_re = re.compile(r'filename="package://dvrk_model/([^"]+)"')
    replacements: dict[str, str] = {}

    for rel in sorted(set(mesh_re.findall(urdf_text))):
        src = SOURCE_DIR / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing mesh referenced by xacro: {src}")

        rel_path = Path(rel)
        if rel_path.suffix.lower() == ".dae":
            out_rel = rel_path.with_suffix(".obj")
            dst = converted_root / out_rel
            dae_to_obj(src, dst)
        else:
            out_rel = rel_path
            dst = converted_root / out_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # URDF paths are relative to the URDF file location.
        replacements[rel] = str(Path("converted_meshes") / out_rel).replace("\\", "/")

    def replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        return f'filename="{replacements[rel]}"'

    return mesh_re.sub(replace, urdf_text)


def write_debug_readme(mesh_count: int) -> None:
    readme = TARGET_DIR / "README.md"
    readme.write_text(
        "# psm_official\n\n"
        "This folder is generated by `Src/load_dvrk_psm.py`.\n\n"
        "Contents:\n"
        "- `urdf/Classic/`: copied official dVRK Classic xacro files.\n"
        "- `meshes/Classic/PSM/`: copied official PSM source meshes.\n"
        "- `converted_meshes/`: MuJoCo-friendly mesh files referenced by the generated URDF.\n"
        "- `psm1_sca_mujoco.urdf`: flattened Classic PSM1 + SCA tool URDF for MuJoCo debug loading.\n\n"
        f"Converted mesh references: {mesh_count}\n\n"
        "Notes:\n"
        "- ROS xacro `mimic` tags are removed for MuJoCo loading; mimic joints become independent debug joints.\n"
        "- Simple positive inertials are injected so MuJoCo can compile moving links.\n"
        "- Visual geometry is duplicated as collision geometry for first-pass model debugging.\n",
        encoding="utf-8",
    )


def convert_model() -> Path:
    prepare_asset_folder()
    urdf = build_flat_urdf()
    mesh_refs_before = len(set(re.findall(r'filename="package://dvrk_model/([^"]+)"', urdf)))
    urdf = convert_mesh_references(urdf)
    CONVERTED_URDF.write_text(urdf, encoding="utf-8")
    write_debug_readme(mesh_refs_before)
    return CONVERTED_URDF


def load_in_mujoco(urdf_path: Path, show_gui: bool) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)

    print(f"Loaded: {urdf_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}")
    print("Joints:")
    for jid in range(model.njnt):
        print(f"  {jid:02d}: {model.joint(jid).name}")

    if show_gui:
        mujoco.viewer.launch(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert and load official dVRK Classic PSM1 in MuJoCo")
    parser.add_argument("--convert-only", action="store_true", help="Only generate files under Assest/psm_official")
    parser.add_argument("--no-gui", action="store_true", help="Compile model but do not open the MuJoCo viewer")
    args = parser.parse_args()

    urdf_path = convert_model()
    print(f"Generated MuJoCo debug URDF: {urdf_path}")

    if args.convert_only:
        return

    load_in_mujoco(urdf_path, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
