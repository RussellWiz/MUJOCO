from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml"
OUT_DIR = PROJECT_DIR / "Src" / "scene"
OUT_XML = OUT_DIR / "psm_control_capsule32_torus.xml"

NUM_CAPSULES = 32
RING_RADIUS = 0.018
TUBE_RADIUS = 0.003
TOTAL_MASS = 0.0025
MASS_PER_CAPSULE = TOTAL_MASS / NUM_CAPSULES

PSM_BASE_POS = (-0.25, 0.0, 0.088)
SCENE_CENTER = (-0.25, 0.0, 0.12)
TORUS_BODY_POS = (-0.25, 0.006, 0.028)
TORUS_RGBA = "0.9 0.35 0.12 1"


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _circle_point(index: int) -> tuple[float, float, float]:
    theta = 2.0 * math.pi * index / NUM_CAPSULES
    return (
        RING_RADIUS * math.cos(theta),
        RING_RADIUS * math.sin(theta),
        0.0,
    )


def _build_torus_body() -> ET.Element:
    body = ET.Element(
        "body",
        {
            "name": "capsule32_torus",
            "pos": f"{_fmt(TORUS_BODY_POS[0])} {_fmt(TORUS_BODY_POS[1])} {_fmt(TORUS_BODY_POS[2])}",
        },
    )
    ET.SubElement(body, "freejoint")

    for i in range(NUM_CAPSULES):
        p0 = _circle_point(i)
        p1 = _circle_point((i + 1) % NUM_CAPSULES)
        fromto = (
            f"{_fmt(p0[0])} {_fmt(p0[1])} {_fmt(p0[2])} "
            f"{_fmt(p1[0])} {_fmt(p1[1])} {_fmt(p1[2])}"
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"torus_capsule_{i:02d}",
                "type": "capsule",
                "fromto": fromto,
                "size": _fmt(TUBE_RADIUS),
                "rgba": TORUS_RGBA,
                "mass": _fmt(MASS_PER_CAPSULE),
                "contype": "1",
                "conaffinity": "1",
                "friction": "1.5 0.2 0.01",
                "solref": "0.01 1",
                "solimp": "0.9 0.95 0.001",
            },
        )
    return body


def _ensure_end_effector_collision(root: ET.Element) -> None:
    # Match the validated URDFStudio MuJoCo tool setup in Assest/psm_urdfstudio:
    # each jaw uses a slim box collider near the fingertip.
    tool_tip = root.find(".//body[@name='tool_tip']")
    if tool_tip is not None:
        for geom_name in ("tool_tip_col",):
            geom = tool_tip.find(f"./geom[@name='{geom_name}']")
            if geom is not None:
                tool_tip.remove(geom)

    for body_name, pos in (
        ("sca_ee_1", "0.0007 0.0051 0"),
        ("sca_ee_2", "-0.0007 0.0051 0"),
    ):
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Missing {body_name} body in source XML")
        geom_name = f"{body_name}_col"
        old_geom = body.find(f"./geom[@name='{geom_name}']")
        if old_geom is not None:
            body.remove(old_geom)
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": "box",
                "pos": pos,
                "size": "0.0014 0.0102 0.0018",
                "rgba": "0.2 0.9 0.2 0.35",
                "mass": "0.0005",
                "contype": "1",
                "conaffinity": "1",
                "condim": "4",
                "friction": "1.0 1.0 0.01",
                "solref": "0.008 1",
                "solimp": "0.95 0.99 0.001 0.5 2",
            },
        )


def _ensure_contact_excludes(root: ET.Element) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    for body1, body2, name in (
        ("sca_ee_1", "sca_ee_2", "exclude_jaw_pair"),
    ):
        exists = False
        for exclude in contact.findall("exclude"):
            if exclude.get("body1") == body1 and exclude.get("body2") == body2:
                exists = True
                if exclude.get("name") is None:
                    exclude.set("name", name)
                break
        if not exists:
            ET.SubElement(
                contact,
                "exclude",
                {
                    "name": name,
                    "body1": body1,
                    "body2": body2,
                },
            )


def main() -> None:
    if not SOURCE_XML.exists():
        raise FileNotFoundError(f"Source XML not found: {SOURCE_XML}")

    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Missing <compiler> in source XML")
    compiler.set("meshdir", "../../Assest/psm_official")

    statistic = root.find("statistic")
    if statistic is not None:
        statistic.set(
            "center",
            f"{_fmt(SCENE_CENTER[0])} {_fmt(SCENE_CENTER[1])} {_fmt(SCENE_CENTER[2])}",
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing <worldbody> in source XML")

    psm_base = worldbody.find("./body[@name='psm_base']")
    if psm_base is None:
        raise ValueError("Missing psm_base body in source XML")
    psm_base.set(
        "pos",
        f"{_fmt(PSM_BASE_POS[0])} {_fmt(PSM_BASE_POS[1])} {_fmt(PSM_BASE_POS[2])}",
    )

    old_torus = worldbody.find("./body[@name='capsule32_torus']")
    if old_torus is not None:
        worldbody.remove(old_torus)
    worldbody.append(_build_torus_body())
    _ensure_end_effector_collision(root)
    _ensure_contact_excludes(root)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=False)

    print(f"Generated: {OUT_XML}")
    print(f"Capsules: {NUM_CAPSULES}, RING_RADIUS: {RING_RADIUS}, TUBE_RADIUS: {TUBE_RADIUS}")
    print(f"TOTAL_MASS: {TOTAL_MASS}, MASS_PER_CAPSULE: {MASS_PER_CAPSULE}")


if __name__ == "__main__":
    main()
