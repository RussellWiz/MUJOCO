"""
Build a torus from 32 capsules, then decompose it into convex sectors.

This gives a second collision model to compare against the direct torus mesh
decomposition. The generated scene is loaded by `Src/testscene2.py`.
"""
import argparse
import math
import os

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda", "assets")
SCENE_DIR = os.path.join(PROJECT_DIR, "Assest", "franka_emika_panda")

VISUAL_OBJ = os.path.join(ASSETS_DIR, "torus_capsule32_visual.obj")
SCENE_XML = os.path.join(SCENE_DIR, "scene_capsule32_decomp.xml")
PREFIX = "torus_capsule32"

RING_RADIUS = 0.075
TUBE_RADIUS = 0.016
NUM_CAPSULES = 32
# Keep this version lighter than the direct 32-capsule ring so the gripper
# can lift it more easily during comparison tests.
TOTAL_MASS = 0.012


def build_capsule_ring_mesh(num_capsules=NUM_CAPSULES, ring_radius=RING_RADIUS, tube_radius=TUBE_RADIUS):
    """Create a watertight mesh by concatenating 32 capsule meshes on a ring."""
    points = []
    for i in range(num_capsules):
        angle = 2.0 * math.pi * i / num_capsules
        points.append(np.array([
            ring_radius * math.cos(angle),
            ring_radius * math.sin(angle),
            0.0,
        ]))

    meshes = []
    for i in range(num_capsules):
        p0 = points[i]
        p1 = points[(i + 1) % num_capsules]
        axis = p1 - p0
        length = np.linalg.norm(axis)
        direction = axis / length

        capsule = trimesh.creation.capsule(radius=tube_radius, height=length, count=[10, 16])
        capsule.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
        capsule.apply_translation((p0 + p1) * 0.5)
        meshes.append(capsule)

    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    return mesh


def sector_decompose_xy(mesh_path, n_sectors=32, overlap_deg=8.0):
    """
    Decompose a torus lying in the XY plane into convex sectors.

    The ring is built from capsules in the XY plane, so we use atan2(y, x).
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    verts = np.asarray(mesh.vertices)
    angles = np.arctan2(verts[:, 1], verts[:, 0])
    sector_rad = 2.0 * np.pi / n_sectors
    overlap_rad = np.radians(overlap_deg)

    print(f"input mesh: {len(verts)} vertices, {len(mesh.faces)} faces")
    print(f"bounds: {mesh.bounds[0]} -> {mesh.bounds[1]}")

    parts = []
    for i in range(n_sectors):
        center = -np.pi + (i + 0.5) * sector_rad
        half = sector_rad * 0.5 + overlap_rad
        diff = (angles - center + np.pi) % (2.0 * np.pi) - np.pi
        mask = np.abs(diff) <= half

        sector_verts = verts[mask]
        if len(sector_verts) < 4:
            print(f"  sector {i}: skip, only {len(sector_verts)} vertices")
            continue

        hull = ConvexHull(sector_verts)
        hull_verts = sector_verts[hull.vertices]
        remap = {old: new for new, old in enumerate(hull.vertices)}
        hull_faces = np.array([[remap[idx] for idx in face] for face in hull.simplices], dtype=int)
        parts.append((hull_verts, hull_faces))
        print(f"  sector {i:02d}: {mask.sum()} verts -> hull {len(hull_verts)} verts, {len(hull_faces)} faces")

    print(f"generated {len(parts)} convex sectors")
    return parts


def save_convex_parts(parts, out_dir, prefix):
    filenames = []
    for i, (verts, faces) in enumerate(parts):
        filename = f"{prefix}_convex_{i}.obj"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"o {prefix}_{i}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        filenames.append(filename)
    print(f"wrote {len(filenames)} convex OBJ files")
    return filenames


def generate_scene_xml(convex_files, out_path):
    n = len(convex_files)
    mass_per = round(TOTAL_MASS / n, 6)

    mesh_lines = ['    <mesh name="torus_visual" file="torus_capsule32_visual.obj"/>']
    for i, filename in enumerate(convex_files):
        mesh_lines.append(f'    <mesh name="torus_ch{i}" file="{filename}"/>')

    geom_lines = []
    for i in range(n):
        geom_lines.append(
            f'      <geom name="torus_col{i}" type="mesh" mesh="torus_ch{i}" group="3"\n'
            f'            contype="1" conaffinity="1" condim="4" friction="10 4 0.5"\n'
            f'            solref="0.015 1" solimp="0.95 0.99 0.001 0.5 2"\n'
            f'            margin="0.0003" mass="{mass_per}"/>'
        )

    xml = f"""<mujoco model="panda scene with capsule32 decomposed torus">
  <include file="panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>
  <option iterations="60" noslip_iterations="20"/>

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
    <material name="torus_mat" rgba="0.8 0.35 0.1 1" specular="0.5" shininess="0.5"/>

{chr(10).join(mesh_lines)}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          solref="0.001 1" solimp="0.99 0.999 0.001" priority="1"/>

    <!-- Derived from the 32-capsule torus, then split into convex sectors. -->
    <body name="torus" pos="0.7 0 0.25">
      <freejoint/>

      <geom name="torus_visual" type="mesh" mesh="torus_visual" material="torus_mat"
            contype="0" conaffinity="0" group="2" density="0"/>

{chr(10).join(geom_lines)}
    </body>
  </worldbody>

</mujoco>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"generated scene: {out_path}")
    print(f"  convex hulls: {n}, mass per geom: {mass_per}, total mass: {mass_per * n:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Build and decompose a 32-capsule torus")
    parser.add_argument("--sectors", type=int, default=32, help="number of convex sectors")
    parser.add_argument("--overlap", type=float, default=8.0, help="sector overlap in degrees")
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)

    visual_mesh = build_capsule_ring_mesh()
    visual_mesh.export(VISUAL_OBJ)
    print(f"wrote visual mesh: {VISUAL_OBJ}")

    parts = sector_decompose_xy(VISUAL_OBJ, n_sectors=args.sectors, overlap_deg=args.overlap)
    convex_files = save_convex_parts(parts, ASSETS_DIR, PREFIX)
    generate_scene_xml(convex_files, SCENE_XML)

    print("\nready:")
    print(f"  python {os.path.join('Src', 'testscene2.py')}")


if __name__ == "__main__":
    main()
