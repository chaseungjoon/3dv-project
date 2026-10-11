"""Collision model of the ParaHome book + the desk-top box for place_book (project root SKILLMIMIC_NOTES.md 4).

    .venv/bin/python experiments/tools/make_book_collision.py

The book mesh (148k vertices) is a closed hardcover, 26.9 x 17.5 x 4.7 cm. Upstream loads it through a 10k-voxel
automatic decomposition at density 2500 (7.1 kg in the sim, held one-handed from frame 70 on) and the desk through the same
coarse decomposition (the book rests ~3.8 cm above its reference). Here:
  - book: its oriented bounding box as one convex 8-vertex mesh (box volume 2197 cm^3 vs hull 2078 cm^3), so the
    collision shape is the book itself; density printed for a 1.0 kg book
  - desk: prints the staticBoxes entry (desk top fitted to the mesh in the scene pose, top at the book's reference
    bottom so the book rests where the reference has it)

Writes skillmimic/data/assets/obj/book_fix/box.obj and skillmimic/data/assets/obj/book_fix.urdf (visual = book.obj).
"""
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBJ_DIR = os.path.join(REPO, "skillmimic/data/assets/obj")
MOTION = os.path.join(REPO, "skillmimic/data/motions/ParaHome/place_book/002_s6_place_book_2460_2610.pt")
SCENE = os.path.join(REPO, "skillmimic/data/motions/ParaHome/s10/object_transformations.pkl")  # the env places the desk from s10 frame 0 (same pose as s6)
MASS = 1.0  # kg


def verts(name):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(OBJ_DIR, f"{name}.obj")) if l.startswith("v ")])


def obb(P):
    """principal axes (rows, right-handed), min and max along them, and the mean"""
    c = P.mean(0)
    vt = np.linalg.svd(P - c, full_matrices=False)[2]
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    Q = (P - c) @ vt.T
    return vt, Q.min(0), Q.max(0), c


def main():
    # ---- book: oriented box ----
    vt, lo, hi, c = obb(verts("book"))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ vt + c
    # faces of the box with corner index i = 4*ix + 2*iy + iz, oriented outward below
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    centre = corners.mean(0)
    out_dir = os.path.join(OBJ_DIR, "book_fix")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "box.obj"), "w") as f:
        f.write("# oriented bounding box of book.obj (experiments/tools/make_book_collision.py)\n")
        for v in corners:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
        for q in quads:
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                p = corners[list(tri)]
                n = np.cross(p[1] - p[0], p[2] - p[0])
                if np.dot(n, p[0] - centre) < 0:
                    tri = (tri[0], tri[2], tri[1])
                f.write("f {} {} {}\n".format(*(i + 1 for i in tri)))
    size = hi - lo
    vol = float(np.prod(size))
    with open(os.path.join(OBJ_DIR, "book_fix.urdf"), "w") as f:
        f.write(f'''<?xml version="1.0" ?>
<!-- book collision = its oriented bounding box (experiments/tools/make_book_collision.py); load with vhacd off -->
<robot name="book_fix.urdf">
  <link name="baseLink">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="book.obj" scale="1. 1. 1."/></geometry>
      <material name="red"><color rgba="1 0.4 0.4 1"/></material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="book_fix/box.obj" scale="1. 1. 1."/></geometry>
    </collision>
  </link>
</robot>
''')
    print(f"[book] box {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm, volume {vol*1e6:.0f} cm^3 "
          f"-> ballDensity for {MASS} kg: {MASS / vol:.0f}")

    # ---- desk: box fitted to the top surface in the scene pose ----
    M = pickle.load(open(SCENE, "rb"))[0]["desk_base"]
    D = verts("desk") @ M[:3, :3].T + M[:3, 3]
    top = D[D[:, 2] > D[:, 2].max() - 0.02][:, :2]       # top slab (the mesh top is ~2 cm thick, 0.70-0.73 m)
    c2 = top.mean(0)
    vt2 = np.linalg.svd(top - c2, full_matrices=False)[2]
    Q = (top - c2) @ vt2.T
    ctr = c2 + ((Q.min(0) + Q.max(0)) / 2) @ vt2
    yaw = np.degrees(np.arctan2(vt2[0, 1], vt2[0, 0])) % 180.0
    # the box top = the book's lowest point at reference frame 0 (the book rests where the reference has it)
    h = torch.load(MOTION, map_location="cpu").numpy()
    o = 9 + 180 + 213
    book0 = R.from_rotvec(h[0, o + 3:o + 6]).apply(corners) + h[0, o:o + 3]
    sz = Q.max(0) - Q.min(0)
    print(f"[desk] staticBoxes: {{desk: {{size: [{sz[0]:.3f}, {sz[1]:.3f}, 0.05], center: [{ctr[0]:.3f}, {ctr[1]:.3f}], "
          f"top: {book0[:, 2].min():.3f}, yawDeg: {yaw:.2f}}}}}")


if __name__ == "__main__":
    main()
