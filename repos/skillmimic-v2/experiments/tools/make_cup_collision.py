"""Build a collision model of the ParaHome cup that keeps the handle loop open (project root SKILLMIMIC_NOTES.md 4).

    .venv/bin/python experiments/tools/make_cup_collision.py

The reference drink_cup grasp puts the fingers THROUGH the handle loop (fingertips 5.3-5.9 cm from the cup axis,
inside the 4.3-5.8 cm opening). Any automatic convex decomposition we can afford at 2048 envs fills that loop, so
fingers in the reference pose sit inside the collision shape and the cup is ejected. Here the mesh is split by
hand into convex pieces: the body (one hull) and the D-shaped handle as four bars (top, bottom, outer upper,
outer lower), leaving the opening empty.

Writes skillmimic/data/assets/obj/cup_fix/*.obj and skillmimic/data/assets/obj/cup_fix.urdf (visual = cup.obj).
Prints the total collision volume (to choose the density for a target mass).
"""
import os

import numpy as np
from scipy.spatial import ConvexHull

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBJ_DIR = os.path.join(REPO, "skillmimic/data/assets/obj")
UP = np.array([0.71, -0.05, -0.71])   # cup axis in mesh coordinates (world up at reference frame 0)
HANDLE_AZ = -113.0                    # handle direction around the axis, degrees (frame below)


MAX_VERTS = {"body": 64}   # handle bars: 24. PhysX contact cost grows with hull size (2048 envs, 12 GB)


def fps(P, k):
    """farthest-point subset of k points (keeps the extremes, so the hull shrinks by ~1 mm at most)"""
    if len(P) <= k:
        return P
    sel = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[sel]


def box_piece(c_h, c_r, half_h, half_r, half_t, up, u, v):
    """8 corners of a box in cup coordinates: centre at height c_h and radius c_r along the handle direction u"""
    c = c_h * up + c_r * u
    return np.array([c + sh * half_h * up + sr * half_r * u + st * half_t * v
                     for sh in (-1, 1) for sr in (-1, 1) for st in (-1, 1)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", default="mesh", choices=["thin", "mesh"],
                    help="mesh: hulls of the mesh handle bars (default, used by drink_cup_fix.yaml); thin: straps of --t thickness on the centreline (tested, not better)")
    ap.add_argument("--t", type=float, default=0.006, help="strap thickness (m) for --handle thin")
    args = ap.parse_args()
    V = np.array([[float(x) for x in l.split()[1:4]] for l in open(os.path.join(OBJ_DIR, "cup.obj")) if l.startswith("v ")])
    up = UP / np.linalg.norm(UP)
    a = np.cross(up, [0, 1, 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    h, x, y = V @ up, V @ a, V @ b
    r = np.hypot(x, y)
    az = np.degrees(np.arctan2(y, x))
    sector = np.abs(((az - HANDLE_AZ + 180) % 360) - 180) < 35
    handle = sector & (r > 0.044)
    pieces = {
        "body": r <= 0.044,
        "handle_top": handle & (h > 0.031),
        "handle_bottom": handle & (h < -0.017),
        "handle_outer_up": handle & (r > 0.057) & (h >= 0.005) & (h <= 0.036),
        "handle_outer_lo": handle & (r > 0.057) & (h >= -0.022) & (h < 0.005),
    }
    pts = {k: V[m] for k, m in pieces.items()}
    if args.handle == "thin":
        # centreline of the D-shaped handle (measured from the mesh, cup frame): top bar h 3.5-4.7 cm, outer bar
        # r 5.8-7.0 cm, bottom bar h -3.1..-1.9 cm, width ~1.8 cm. Straps of thickness t on the centreline leave
        # room for the noisy reference fingers (pinch: thumb on top of the top bar, index under it).
        u = np.cos(np.radians(HANDLE_AZ)) * a + np.sin(np.radians(HANDLE_AZ)) * b
        v = np.cross(up, u)
        t2, w2 = args.t / 2, 0.009
        r_in, r_out, h_top, h_bot = 0.040, 0.064, 0.041, -0.025
        pts = {"body": V[pieces["body"]],
               "handle_top": box_piece(h_top, (r_in + r_out) / 2, t2, (r_out - r_in) / 2 + t2, w2, up, u, v),
               "handle_bottom": box_piece(h_bot, (r_in + r_out) / 2, t2, (r_out - r_in) / 2 + t2, w2, up, u, v),
               "handle_outer_up": box_piece((h_top + 0.008) / 2 + 0.004, r_out, (h_top - 0.008) / 2 + t2, t2, w2, up, u, v),
               "handle_outer_lo": box_piece((h_bot + 0.008) / 2, r_out, (0.008 - h_bot) / 2, t2, w2, up, u, v)}
        print(f"[cup] thin handle straps, thickness {args.t * 100:.1f} cm")
    out_dir = os.path.join(OBJ_DIR, "cup_fix")
    os.makedirs(out_dir, exist_ok=True)
    total = 0.0
    for name, P in pts.items():
        P = fps(P[ConvexHull(P).vertices], MAX_VERTS.get(name, 24))
        hull = ConvexHull(P)
        total += hull.volume
        idx = {v: i for i, v in enumerate(hull.vertices)}
        with open(os.path.join(out_dir, f"{name}.obj"), "w") as f:
            f.write(f"# convex piece '{name}' of cup.obj (experiments/tools/make_cup_collision.py)\n")
            for v in hull.vertices:
                f.write("v {:.6f} {:.6f} {:.6f}\n".format(*P[v]))
            for s in hull.simplices:
                # orient faces outward
                p = P[s]; n = np.cross(p[1] - p[0], p[2] - p[0])
                if np.dot(n, p[0] - P[hull.vertices].mean(0)) < 0:
                    s = s[[0, 2, 1]]
                f.write("f {} {} {}\n".format(*(idx[v] + 1 for v in s)))
        print(f"[cup] {name:16s} -> hull {len(hull.vertices):3d} verts, volume {hull.volume * 1e6:7.1f} cm^3")
    cols = "\n".join(f'''    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="cup_fix/{n}.obj" scale="1. 1. 1."/></geometry>
    </collision>''' for n in pieces)
    urdf = f'''<?xml version="1.0" ?>
<!-- cup with an open handle loop for collision (experiments/tools/make_cup_collision.py); load with vhacd off -->
<robot name="cup_fix.urdf">
  <link name="baseLink">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="cup.obj" scale="1. 1. 1."/></geometry>
      <material name="red"><color rgba="1 0.4 0.4 1"/></material>
    </visual>
{cols}
  </link>
</robot>
'''
    with open(os.path.join(OBJ_DIR, "cup_fix.urdf"), "w") as f:
        f.write(urdf)
    print(f"[cup] total collision volume {total * 1e6:.1f} cm^3 -> density for 0.35 kg: {0.35 / total:.0f} kg/m^3")


if __name__ == "__main__":
    main()
