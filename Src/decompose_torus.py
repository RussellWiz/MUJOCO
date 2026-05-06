"""
将 torus.obj 按角度切成 N 个扇区并取凸包，生成独立 OBJ 文件，
再自动生成 MuJoCo 场景 XML。

比 VHACD 更适合圆环：扇区凸包保留了空心结构的内外表面。

用法:
    python Src/decompose_torus.py
    python Src/decompose_torus.py --sectors 24 --overlap 20
    python Src/decompose_torus.py --sectors 16 --overlap 10   # 粗一些
"""
import argparse
import os
import shutil
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TORUS_OBJ   = os.path.join(PROJECT_DIR, "dvrk_hanoi", "torus", "torus.obj")
ASSETS_DIR  = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda", "assets")
SCENE_DIR   = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda")
SCENE_XML   = os.path.join(SCENE_DIR, "scene_vhacd_fine.xml")

PREFIX = "torus_fine"


def sector_decompose(obj_path, n_sectors=20, overlap_deg=15):
    """
    将圆环 mesh 按角度切成 n_sectors 个扇区，
    相邻扇区有 overlap_deg 度的重叠以确保碰撞无缝。
    返回 [(verts_array, faces_array), ...] 列表。
    """
    mesh = trimesh.load(obj_path, force="mesh")
    verts = np.array(mesh.vertices)
    print(f"输入 mesh: {len(verts)} vertices, {len(mesh.faces)} faces")
    print(f"bounds: {mesh.bounds[0]} -> {mesh.bounds[1]}")

    # OBJ torus: 圆环平面在 XZ，管截面沿 Y
    angles = np.arctan2(verts[:, 2], verts[:, 0])
    sector_rad = 2 * np.pi / n_sectors
    overlap_rad = np.radians(overlap_deg)

    parts = []
    for i in range(n_sectors):
        center = -np.pi + (i + 0.5) * sector_rad
        half = sector_rad / 2 + overlap_rad
        diff = (angles - center + np.pi) % (2 * np.pi) - np.pi
        mask = np.abs(diff) <= half

        sec_verts = verts[mask]
        if len(sec_verts) < 4:
            print(f"  Sector {i}: too few verts ({len(sec_verts)}), skip")
            continue

        hull = ConvexHull(sec_verts)
        hull_verts = sec_verts[hull.vertices]
        hull_faces = np.zeros_like(hull.simplices)
        remap = {old: new for new, old in enumerate(hull.vertices)}
        for fi, face in enumerate(hull.simplices):
            hull_faces[fi] = [remap[v] for v in face]

        parts.append((hull_verts, hull_faces))
        print(f"  Sector {i:2d}: {mask.sum()} verts -> hull {len(hull_verts)} verts, {len(hull_faces)} faces")

    print(f"\n共 {len(parts)} 个凸包扇区")
    return parts


def save_convex_parts(parts, out_dir, prefix):
    """保存凸包为独立 OBJ，返回文件名列表。"""
    os.makedirs(out_dir, exist_ok=True)
    filenames = []
    for i, (verts, faces) in enumerate(parts):
        fname = f"{prefix}_convex_{i}.obj"
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            f.write(f"o {prefix}_{i}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for fc in faces:
                f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
        filenames.append(fname)
    print(f"写入 {len(filenames)} 个 OBJ 文件到 {out_dir}")
    return filenames


def generate_scene_xml(convex_files, out_path, scale=0.07):
    """根据凸包文件列表自动生成 MuJoCo 场景 XML。"""
    n = len(convex_files)
    # Keep the ring light so the gripper can lift stably.
    total_mass = 0.012
    mass_per = round(total_mass / n, 6)
    s = f"{scale} {scale} {scale}"

    mesh_decls = []
    mesh_decls.append(f'    <mesh name="torus_visual" file="torus_visual.obj" scale="{s}"/>')
    for i, fn in enumerate(convex_files):
        mesh_decls.append(f'    <mesh name="torus_ch{i}" file="{fn}" scale="{s}"/>')

    col_geoms = []
    for i in range(n):
        col_geoms.append(
            f'      <geom name="torus_col{i}" type="mesh" mesh="torus_ch{i}" group="3"\n'
            f'            contype="1" conaffinity="1" condim="4" friction="8 3 0.4"\n'
            f'            solref="0.015 1" solimp="0.95 0.99 0.001 0.5 2"\n'
            f'            margin="0.0002" mass="{mass_per}"/>'
        )

    xml = f'''<mujoco model="panda scene with fine decomposed torus">
  <include file="panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>
  <option iterations="50" noslip_iterations="15"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="torus_mat" rgba="0.8 0.45 0.15 1" specular="0.5" shininess="0.5"/>

{chr(10).join(mesh_decls)}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          solref="0.001 1" solimp="0.99 0.999 0.001" priority="1"/>

    <!-- Sector-decomposed torus: {n} convex hulls, total mass ~0.02 kg -->
    <body name="torus" pos="0.6 0 0.25" quat="0.707 0.707 0 0">
      <freejoint/>

      <geom name="torus_visual" type="mesh" mesh="torus_visual" material="torus_mat"
            contype="0" conaffinity="0" group="2" density="0"/>

{chr(10).join(col_geoms)}
    </body>
  </worldbody>

</mujoco>
'''
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"\n已生成场景: {out_path}")
    print(f"  凸包数: {n},  每块质量: {mass_per} kg,  总质量: {mass_per * n:.4f} kg")


def main():
    ap = argparse.ArgumentParser(description="圆环扇区凸分解 + 自动生成 MuJoCo 场景")
    ap.add_argument("--sectors", type=int, default=20,
                    help="扇区数量 (default 20)")
    ap.add_argument("--overlap", type=float, default=15,
                    help="相邻扇区重叠角度 (degrees, default 15)")
    ap.add_argument("--scale", type=float, default=0.07,
                    help="MuJoCo mesh scale (default 0.07)")
    args = ap.parse_args()

    if not os.path.isfile(TORUS_OBJ):
        raise FileNotFoundError(f"找不到: {TORUS_OBJ}")

    # 1. 扇区凸分解
    parts = sector_decompose(TORUS_OBJ, n_sectors=args.sectors, overlap_deg=args.overlap)

    # 2. 保存为 OBJ
    convex_files = save_convex_parts(parts, ASSETS_DIR, PREFIX)

    # 3. 确保视觉 OBJ 存在
    visual_dst = os.path.join(ASSETS_DIR, "torus_visual.obj")
    if not os.path.isfile(visual_dst):
        shutil.copy2(TORUS_OBJ, visual_dst)
        print(f"已复制视觉 mesh: {visual_dst}")

    # 4. 生成场景 XML
    generate_scene_xml(convex_files, SCENE_XML, scale=args.scale)

    print(f"\n完成！运行以下命令测试:")
    print(f"  python Src/testscene.py")


if __name__ == "__main__":
    main()
