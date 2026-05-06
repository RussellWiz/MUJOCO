"""
Load the torus generated from the 32-capsule ring and then decomposed again.
"""
import testscene as base


base.XML_PATH = r"d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_capsule32_decomp.xml"
base.CTRL_HINT = base.CTRL_HINT.replace("精细分解圆环", "32-capsule 分解圆环")


if __name__ == "__main__":
    base.main()
