"""
Load the MuJoCo dVRK PSM model that uses identified (real) dynamics parameters.

Default XML:
    Assest/psm_official/psm_control_dynamics.xml

Run from the project root:
    python Src/load_realmodel_dvrk.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_DIR / "Assest" / "psm_official" / "psm_control_dynamics.xml"


def load_in_mujoco(xml_path: Path, show_gui: bool) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"Loaded: {xml_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}")

    if show_gui:
        mujoco.viewer.launch(model, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the MuJoCo dVRK PSM real-dynamics XML")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Path to the MuJoCo XML to load")
    parser.add_argument("--no-gui", action="store_true", help="Compile model but do not open the MuJoCo viewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    load_in_mujoco(xml_path, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()

