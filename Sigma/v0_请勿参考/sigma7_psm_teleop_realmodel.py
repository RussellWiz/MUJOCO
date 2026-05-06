"""
Use a Force Dimension sigma.7 to teleoperate the MuJoCo dVRK PSM model (real dynamics XML).

Default XML:
    Assest/psm_official/psm_control_dynamics.xml

Run from the project root:
    python Sigma/teleop_realmodel.py

If the SDK DLLs are not on PATH, pass the directory containing dhd64.dll/drd64.dll:
    python Sigma/teleop_realmodel.py --sdk-bin "C:\\Program Files\\Force Dimension\\sdk-3.17.6\\bin"

Controls:
    q       quit
    r       reset the sigma.7 neutral pose and PSM home target
"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML_REAL = PROJECT_DIR / "Assest" / "psm_official" / "psm_control_dynamics.xml"
SIGMA7_TELEOP = PROJECT_DIR / "Sigma" / "sigma7_psm_teleop.py"


def main() -> None:
    # The underlying script uses argparse; inject the default if user didn't specify --xml.
    # We run it by path so Sigma/ doesn't need to be a Python package.
    import sys
    import runpy

    argv = sys.argv[1:]
    if "--xml" not in argv:
        sys.argv = [sys.argv[0], "--xml", str(DEFAULT_XML_REAL), *argv]
    runpy.run_path(str(SIGMA7_TELEOP), run_name="__main__")


if __name__ == "__main__":
    main()
