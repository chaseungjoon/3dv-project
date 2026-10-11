"""Physics probe for the ParaHome scenes (no humanoid): where do objects come to rest, how heavy are they.

    .venv/bin/python experiments/probe/scene_probe.py drink_cup
For each collision variant of the static furniture and the dynamic object, drops (a) a 1 cm sphere on the furniture
under the reference object xy (measures the effective support surface) and (b) the object itself at its reference
pose of frame 0, then reports rest heights vs the reference and the object mass.
"""
from isaacgym import gymapi, gymtorch  # noqa: E402  (must precede torch)
import sys, pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

CLIP = sys.argv[1] if len(sys.argv) > 1 else "drink_cup"
SCENE = {"drink_cup": ("cup", "diningtable", "001_s6_drink_1860_2040.pt"),
         "place_book": ("book", "desk", "002_s6_place_book_2460_2610.pt"),
         "place_kettle": ("kettle", "diningtable", None)}
OBJ, STATIC, PT = SCENE[CLIP]
ROOT = "skillmimic/data/assets/obj/"

h = torch.load(f"skillmimic/data/motions/ParaHome/{CLIP}/{PT}", map_location="cpu").numpy()
o = 9 + 180 + 213
ref_pos, ref_rotvec = h[0, o:o + 3], h[0, o + 3:o + 6]
ref_q = R.from_rotvec(ref_rotvec).as_quat()  # xyzw
tf = pickle.load(open("skillmimic/data/motions/ParaHome/s10/object_transformations.pkl", "rb"))[0][f"{STATIC}_base"]
st_q = R.from_matrix(tf[:3, :3]).as_quat()

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 30.0; sp.substeps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.use_gpu_pipeline = True
px = sp.physx; px.use_gpu = True; px.solver_type = 1; px.num_position_iterations = 10; px.num_velocity_iterations = 0
px.contact_offset = 0.02; px.rest_offset = 0.0; px.bounce_threshold_velocity = 0.2; px.max_depenetration_velocity = 10.0
px.default_buffer_size_multiplier = 10.0; px.max_gpu_contact_pairs = 8 * 1024 * 1024; px.num_threads = 4
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1); pp.static_friction = pp.dynamic_friction = 1.0
gym.add_ground(sim, pp)

def static_asset(res, hulls, verts):
    a = gymapi.AssetOptions(); a.fix_base_link = True
    a.vhacd_enabled = True; a.vhacd_params.resolution = res; a.vhacd_params.max_convex_hulls = hulls
    a.vhacd_params.max_num_vertices_per_ch = verts
    return gym.load_asset(sim, ROOT, f"{STATIC}.urdf", a)

def dyn_asset(res, hulls, verts, density=2500.0):
    a = gymapi.AssetOptions(); a.angular_damping = 0.01; a.linear_damping = 0.01; a.max_angular_velocity = 100.0
    a.density = density; a.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    a.vhacd_enabled = True; a.vhacd_params.resolution = res; a.vhacd_params.max_convex_hulls = hulls
    a.vhacd_params.max_num_vertices_per_ch = verts
    return gym.load_asset(sim, ROOT, f"{OBJ}.urdf", a)

sphere = gym.create_sphere(sim, 0.01, gymapi.AssetOptions())
STATIC_VARIANTS = {"upstream res1e4 h10 v32": (10000, 10, 32), "res1e6 h64 v64": (1000000, 64, 64)}
DYN_VARIANTS = {"upstream res1e4 h10 v32": (10000, 10, 32), "res1e6 h32 v64": (1000000, 32, 64)}
s_assets = {k: static_asset(*v) for k, v in STATIC_VARIANTS.items()}
d_assets = {k: dyn_asset(*v) for k, v in DYN_VARIANTS.items()}

envs, rows = [], []
lower, upper = gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3)
for sk, sa in s_assets.items():
    for dk, da in list(d_assets.items()) + [("sphere1cm", sphere)]:
        env = gym.create_env(sim, lower, upper, 4)
        t = gymapi.Transform(); t.p = gymapi.Vec3(*tf[:3, 3]); t.r = gymapi.Quat(*st_q)
        gym.create_actor(env, sa, t, "static", len(envs), 0)
        t2 = gymapi.Transform()
        if dk == "sphere1cm":
            t2.p = gymapi.Vec3(ref_pos[0], ref_pos[1], ref_pos[2] + 0.15)
        else:
            t2.p = gymapi.Vec3(*ref_pos); t2.r = gymapi.Quat(*ref_q)
        a = gym.create_actor(env, da, t2, "obj", len(envs), 0)
        mass = sum(p.mass for p in gym.get_actor_rigid_body_properties(env, a))
        envs.append(env); rows.append((sk, dk, mass))
gym.prepare_sim(sim)
rs = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(len(envs), 2, 13)
traj = []
for step in range(90):
    gym.simulate(sim); gym.fetch_results(sim, True); gym.refresh_actor_root_state_tensor(sim)
    traj.append(rs[:, 1, :3].cpu().numpy().copy())
traj = np.stack(traj)
print(f"clip {CLIP}: object {OBJ} on {STATIC}; reference object center z = {ref_pos[2]:.4f}  (substeps {sp.substeps})")
for i, (sk, dk, mass) in enumerate(rows):
    z1, z5, zend = traj[0, i, 2], traj[4, i, 2], traj[-1, i, 2]
    dxy = np.linalg.norm(traj[-1, i, :2] - ref_pos[:2])
    if dk == "sphere1cm":
        print(f"  static[{sk:24s}] support surface z = {zend - 0.01:.4f}")
    else:
        print(f"  static[{sk:24s}] obj[{dk:24s}] mass {mass:6.3f} kg | z after 1 step {z1:.4f} (+{(z1-ref_pos[2])*100:.1f} cm) "
              f"5 steps {z5:.4f} | rest {zend:.4f} (+{(zend-ref_pos[2])*100:.1f} cm), xy drift {dxy*100:.1f} cm")
