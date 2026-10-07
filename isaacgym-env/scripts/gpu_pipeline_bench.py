"""Headless Isaac Gym benchmark for the full GPU pipeline (PhysX GPU + tensor API + torch CUDA).

Spawns N cartpoles, drives them with random efforts computed by torch on the GPU, and reports
simulation throughput and VRAM. This is the code path used by RL training (e.g. WristMimic/rl_games).

    uv run python scripts/gpu_pipeline_bench.py --num_envs 4096 --steps 1000
"""
import argparse
import os
import time

from isaacgym import gymapi, gymtorch  # must be imported before torch
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 60.0
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sp.use_gpu_pipeline = True
sp.physx.use_gpu = True
sp.physx.num_threads = 4
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
assert sim is not None, "create_sim failed"

asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "isaacgym", "assets")
opts = gymapi.AssetOptions()
opts.fix_base_link = True
cartpole = gym.load_asset(sim, asset_root, "urdf/cartpole.urdf", opts)
num_dofs = gym.get_asset_dof_count(cartpole)

spacing = 2.0
per_row = int(args.num_envs ** 0.5)
for i in range(args.num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), per_row)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
    actor = gym.create_actor(env, cartpole, pose, "cartpole", i, 1)
    props = gym.get_actor_dof_properties(env, actor)
    props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
    props["stiffness"][:] = 0.0
    props["damping"][:] = 0.0
    gym.set_actor_dof_properties(env, actor, props)

gym.prepare_sim(sim)
dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(args.num_envs, num_dofs, 2)
assert dof_state.is_cuda

efforts = torch.zeros(args.num_envs, num_dofs, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(args.steps):
    efforts[:, 0] = torch.randn(args.num_envs, device="cuda") * 50.0  # push the cart
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(efforts))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    reward = torch.cos(dof_state[:, 1, 0]).mean()  # pole angle -> "reward", a torch CUDA kernel on sim state
torch.cuda.synchronize()
dt = time.time() - t0

print(f"GPU            : {torch.cuda.get_device_name(0)} (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
print(f"torch          : {torch.__version__}, arch list {torch.cuda.get_arch_list()}")
print(f"envs x steps   : {args.num_envs} x {args.steps}")
print(f"throughput     : {args.num_envs * args.steps / dt:,.0f} env-steps/s ({dt:.2f}s)")
print(f"cart pos range : [{dof_state[:, 0, 0].min().item():.3f}, {dof_state[:, 0, 0].max().item():.3f}]")
print(f"mean cos(pole) : {reward.item():.3f}")
free, total = torch.cuda.mem_get_info()
print(f"GPU mem in use : {(total - free) / 2**20:,.0f} / {total / 2**20:,.0f} MiB (whole device, incl. PhysX and desktop)")
print("GPU PIPELINE OK")
gym.destroy_sim(sim)
