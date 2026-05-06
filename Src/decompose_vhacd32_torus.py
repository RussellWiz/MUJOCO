"""
Generate a fine convex-hull torus scene for `testvhacd.py`.

This keeps the VHACD test path separate from the stable `testscene.py`
20-hull scene, so the two variants can be compared independently.
"""
import os
import shutil

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TORUS_OBJ = os.path.join(PROJECT_DIR, "dvrk_hanoi", "torus", "torus.obj")
ASSETS_DIR = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda", "assets")
SCENE_XML = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda", "scene_vhacd64.xml")

PREFIX = "torus_vhacd64"
NUM_HULLS = 64
OVERLAP_DEG = 4.0
SCALE = 0.07
TOTAL_MASS = 0.008


def decompose_torus(obj_path, num_hulls=NUM_HULLS, overlap_deg=OVERLAP_DEG):
    mesh = trimesh.load(obj_path, force="mesh")
    verts = np.asarray(mesh.vertices)

    # Original torus lies in the XZ plane, Y is the tube thickness direction.
    angles = np.arctan2(verts[:, 2], verts[:, 0])
    sector_rad = 2.0 * np.pi / num_hulls
    overlap_rad = np.radians(overlap_deg)

    parts = []
    print(f"input: {len(verts)} vertices, {len(mesh.faces)} faces")
    for i in range(num_hulls):
        center = -np.pi + (i + 0.5) * sector_rad
        half = sector_rad * 0.5 + overlap_rad
        diff = (angles - center + np.pi) % (2.0 * np.pi) - np.pi
        mask = np.abs(diff) <= half

        sector_verts = verts[mask]
        hull = ConvexHull(sector_verts)
        hull_verts = sector_verts[hull.vertices]
        remap = {old: new for new, old in enumerate(hull.vertices)}
        hull_faces = np.array([[remap[idx] for idx in face] for face in hull.simplices], dtype=int)
        parts.append((hull_verts, hull_faces))
        print(f"  hull {i:02d}: {mask.sum()} verts -> {len(hull_verts)} hull verts")

    return parts


def save_parts(parts):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    filenames = []
    for i, (verts, faces) in enumerate(parts):
        filename = f"{PREFIX}_convex_{i}.obj"
        path = os.path.join(ASSETS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"o {PREFIX}_{i}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        filenames.append(filename)
    return filenames


def generate_xml(convex_files):
    scale = f"{SCALE} {SCALE} {SCALE}"
    mass_per = round(TOTAL_MASS / len(convex_files), 6)

    mesh_lines = [f'    <mesh name="torus_visual" file="torus_visual.obj" scale="{scale}"/>']
    for i, filename in enumerate(convex_files):
        mesh_lines.append(f'    <mesh name="torus_ch{i}" file="{filename}" scale="{scale}"/>')

    geom_lines = []
    for i in range(len(convex_files)):
        geom_lines.append(
            f'      <geom name="torus_col{i}" type="mesh" mesh="torus_ch{i}" group="3"\n'
            f'            contype="1" conaffinity="1" condim="6" friction="24 10 2"\n'
            f'            solref="0.012 1" solimp="0.97 0.995 0.0005 0.5 2"\n'
            f'            margin="0.0002" mass="{mass_per}"/>'
        )

    xml = f"""<mujoco model="panda scene with VHACD64 torus">
  <include file="panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>
  <option iterations="100" noslip_iterations="50"/>

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

{chr(10).join(mesh_lines)}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          solref="0.001 1" solimp="0.99 0.999 0.001" priority="1"/>

    <body name="torus" pos="0.6 0 0.25" quat="0.707 0.707 0 0">
      <freejoint/>

      <geom name="torus_visual" type="mesh" mesh="torus_visual" material="torus_mat"
            contype="0" conaffinity="0" group="2" density="0"/>

{chr(10).join(geom_lines)}
    </body>
  </worldbody>

</mujoco>
"""

    with open(SCENE_XML, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"generated: {SCENE_XML}")
    print(f"hulls: {len(convex_files)}, total mass: {mass_per * len(convex_files):.4f} kg")


def main():
    visual_dst = os.path.join(ASSETS_DIR, "torus_visual.obj")
    if not os.path.isfile(visual_dst):
        shutil.copy2(TORUS_OBJ, visual_dst)

    parts = decompose_torus(TORUS_OBJ)
    convex_files = save_parts(parts)
    generate_xml(convex_files)


if __name__ == "__main__":
    main()
