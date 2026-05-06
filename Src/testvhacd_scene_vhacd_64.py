"""
Load the finer VHACD-style torus scene with the same keyboard controller.

This is the 64-convex-hull version, useful as a baseline against:
- testscene.py: direct torus mesh sector decomposition
- testscene2.py: 32-capsule mesh sector decomposition
"""
import testscene as base


base.XML_PATH = r"d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_vhacd64.xml"
base.CTRL_HINT = base.CTRL_HINT.replace("精细分解圆环", "VHACD 64凸包圆环")


if __name__ == "__main__":
    base.main()
