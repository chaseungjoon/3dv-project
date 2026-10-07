"""T08 check: DextER's DexGYS success-rate stack (Isaac Gym validator + ShadowHand FK + csdf) on sm_120.

No dataset needed: uses a synthetic 6 cm box as the object. Two "grasps" are simulated with the real
IsaacValidator (6 shake directions each): hand far from the object (must FAIL) and hand touching the
object (contact kept -> passes the benchmark's contact criterion). This checks the pipeline runs, not
grasp quality.

    cd repos/t08-dexter && .venv-isaacgym/bin/python ../../isaacgym-env/scripts/dexgys_validator_check.py
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "benchmark" / "dexgys"))
sys.path.insert(0, str(REPO / "src"))

from validator import IsaacValidator  # noqa: E402  (imports isaacgym, must precede torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from csdf import compute_sdf, index_vertices_by_faces  # noqa: E402
from dexter.utils.shadowhand import ShadowHandModel  # noqa: E402

dev = "cuda:0"

# 1) csdf CUDA extension: signed distance of points to a box mesh
box = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
verts = torch.tensor(box.vertices, dtype=torch.float, device=dev)
faces = torch.tensor(box.faces, dtype=torch.long, device=dev)
pts = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], device=dev)
dist, *_ = compute_sdf(pts, index_vertices_by_faces(verts, faces))
print(f"csdf squared dist (inside centre, 7cm outside): {dist.tolist()}")
assert abs(dist[1].item() - 0.07 ** 2) < 1e-5

# 2) ShadowHand kinematics on the GPU (pytorch_kinematics)
hand = ShadowHandModel(base_dir=str(REPO / "assets" / "shadowhand"), device=dev)
print(f"ShadowHandModel on {dev}: {len(hand.mesh)} link meshes")

# 3) Isaac Gym validator with the benchmark's hand asset
tmp = tempfile.mkdtemp()
box.export(os.path.join(tmp, "box.obj"))
Path(tmp, "coacd.urdf").write_text("""<?xml version="1.0"?>
<robot name="box"><link name="base">
  <visual><geometry><mesh filename="box.obj"/></geometry></visual>
  <collision><geometry><mesh filename="box.obj"/></geometry></collision>
</link></robot>
""")
sim = IsaacValidator(gpu=0)
sim.set_asset(str(REPO / "assets" / "shadowhand_openai"), "hand/shadow_hand.xml", tmp, "coacd.urdf")
qpos = np.zeros(22)
ident = [1.0, 0.0, 0.0, 0.0]  # wxyz
sim.add_env(ident, [1.0, 1.0, 1.0], qpos, 1)   # grasp 0: hand 1.7 m away
sim.add_env(ident, [0.0, 0.0, 0.0], qpos, 1)   # grasp 1: hand overlapping the box
res = sim.run_sim()
sim.destroy()
per_grasp = [sum(res[i * 6:(i + 1) * 6]) >= 1 for i in range(2)]
print(f"raw per-direction contact flags: {res}")
print(f"grasp success (>=1 of 6 directions): far={per_grasp[0]}, touching={per_grasp[1]}")
assert per_grasp == [False, True]
print("T08 DEXGYS SUCCESS-RATE STACK OK")
