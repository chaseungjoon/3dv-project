"""Build the missing WristMimic bookshelf meshes from the ParaHome scan.

The WristMimic release ships `sim_object/bookshelf/bookshelf.urdf` (and its code lists bookshelf as a fixed,
80-hull object) but not the two meshes the URDF points to, so every bookshelf scene (both Place Book clips)
fails with `Failed to resolve collision mesh 'bookshelf/base_collision.obj'`. For every object that *is* shipped,
`base_visual.obj` is byte-identical to ParaHome `scan/<obj>/simplified/base.obj` and `base_collision.obj` is the
same mesh decimated to ~14.8% of its faces (desk 0.1482, book 0.1482, diningtable 0.1492, kettle 0.1416).
This rebuilds the bookshelf the same way.

    python experiments/tools/make_bookshelf_asset.py      # writes ../t09-wristmimic/.../sim_object/bookshelf/base_{visual,collision}.obj
"""
import os
import shutil

import fast_simplification
import numpy as np
import trimesh

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "data", "scan", "bookshelf", "simplified", "base.obj")
DST = os.path.join(REPO, "..", "t09-wristmimic", "intermimic", "data", "assets", "parahome", "sim_object", "bookshelf")
RATIO = 0.148  # median face ratio of the shipped collision meshes


def main():
    vis, col = os.path.join(DST, "base_visual.obj"), os.path.join(DST, "base_collision.obj")
    if os.path.isfile(vis) and os.path.isfile(col):
        print(f"[bookshelf] already there: {DST}")
        return
    shutil.copyfile(SRC, vis)
    m = trimesh.load(SRC, force="mesh", process=False)
    v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64), target_reduction=1 - RATIO)
    out = trimesh.Trimesh(v, f, process=False)
    out.export(col)
    print(f"[bookshelf] visual {len(m.faces)} faces (copy of the scan), collision {len(f)} faces "
          f"({len(f) / len(m.faces):.4f}), bounds diff {np.abs(out.bounds - m.bounds).max() * 1000:.1f} mm -> {DST}")


if __name__ == "__main__":
    main()
