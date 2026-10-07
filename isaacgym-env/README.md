# Isaac Gym Preview 4 on RTX 5070 (Blackwell, sm_120)

Verified 2026-09-29 on Ubuntu 24.04, driver 595.71, RTX 5070 12GB.

**Verdict: Isaac Gym runs on the RTX 5070, including the GPU pipeline, the viewer, and RL training.**
The one blocker was PyTorch: the last cp38 wheel (torch 2.4.1) has no sm_120 kernels, and Isaac Gym
only has Python 3.6–3.8 bindings. The fix is torch 2.4.1 built from source for sm_120 (`wheels/`).

| Layer | Status on sm_120 | Why |
|---|---|---|
| PhysX GPU (`libPhysXGpu_64.so`) | works as shipped | contains sm_86 PTX, JIT-compiled by the driver |
| `rlgpu`, Flex | works as shipped | contain PTX |
| Viewer (Vulkan) | works | NVIDIA ICD, X11 |
| torch 2.4.1+cu124 (PyPI) | **fails** | `no kernel image is available` (sm_50..sm_90 only) |
| torch 2.4.1 built here (CUDA 12.8, sm_120) | works | `.toolchain/build_torch.sh` |

## Commands

```bash
cd ~/code/3dv-project/isaacgym-env
uv sync                                                    # one-time; env already built

# Headless GPU pipeline benchmark (PhysX GPU + tensor API + gymtorch + torch CUDA)
uv run python scripts/gpu_pipeline_bench.py --num_envs 4096 --steps 1000

# Simple environment with the viewer (run from examples/, assets are relative)
cd isaacgym/python/examples
uv run python 1080_balls_of_solitude.py                    # falling balls, GUI
uv run python joint_monkey.py                              # articulated assets, GUI
uv run python franka_cube_ik_osc.py                        # Franka pick with IK, GPU pipeline (default)
```

Import order: `from isaacgym import ...` must come before `import torch`.

## T09 WristMimic (`../repos/t09-wristmimic`)

```bash
cd ~/code/3dv-project/repos/t09-wristmimic
WANDB_MODE=disabled uv run python intermimic/run.py --task InterMimic_MULTI_OBJ \
  --cfg_env intermimic/data/cfg/parahome_train_12gb.yaml \
  --cfg_train intermimic/data/cfg/train/rlg/parahome.yaml \
  --output checkpoints --num_envs 1024 --minibatch_size 16384 --max_iterations 20 \
  --motion_file InterAct/Parahome/s110_0_kettle_table2desk \
  --robot_type sim_human/s110_intermimic_ROM.xml --experiment s110_kettle \
  --headless --num_position_iterations 20 --num_velocity_iterations 0
```

| num_envs | result | env-steps/s | peak GPU mem (incl. ~0.5GB desktop) |
|---|---|---|---|
| 512 | trains | ~4,700 | 6.5 GB |
| 1024 | trains | ~5,700 | 8.2 GB |
| 2048 (paper) | CUDA OOM | ~6,400 before OOM | >11.5 GB |

Local changes to the released repo:
- `pyproject.toml` (uv env). `numpy==1.23.5` instead of 1.21.1: the torch wheel is built against the 1.23 C-API.
- `intermimic/learning/intermimic_network_builder.py`: register `silu` (the released `parahome.yaml` uses it; PyPI `rl-games==1.1.4` lacks it → `ValueError: silu`).
- `intermimic/data/cfg/parahome_train_12gb.yaml`: PhysX `default_buffer_size_multiplier` 20→5, `max_gpu_contact_pairs` 33.5M→8.4M. The originals are sized for 24GB and fail with `PxgCudaDeviceMemoryAllocator fail to allocate memory` → illegal memory access.

## T08 DextER DexGYS success rate (`../repos/t08-dexter`)

```bash
cd ~/code/3dv-project/repos/t08-dexter
# env: benchmark/dexgys/isaacgym-env/pyproject.toml now points torch at isaacgym-env/wheels
CUDA_HOME=~/code/3dv-project/isaacgym-env/.toolchain/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 \
  UV_PROJECT_ENVIRONMENT=$PWD/.venv-isaacgym uv sync --project benchmark/dexgys/isaacgym-env
.venv-isaacgym/bin/python ../../isaacgym-env/scripts/dexgys_validator_check.py
```

This checks the real `IsaacValidator` (6-direction shake test), the ShadowHand FK on CUDA, and the `csdf` CUDA
extension compiled for sm_120, using a synthetic box. It does not use the dataset (121GB). The DextER model env
(py3.12, torch 2.8/cu128) was not installed here.
The DexGYS objects (`coacd.urdf`) come from the dataset. The Dexonomy benchmark uses MuJoCo, not Isaac Gym.

## Layout

- `isaacgym/`: Isaac Gym Preview 4 (from `https://developer.nvidia.com/isaac-gym-preview-4`). A `libpython3.8.so.1.0` symlink added to `_bindings/linux-x86_64/` removes the need for `LD_LIBRARY_PATH`.
- `wheels/torch-2.4.1+cu128sm120-cp38-*.whl`: the source-built torch. Its RUNPATH and nvrtc preload point at `.toolchain/cuda-12.8`, so it only works on this machine unless the toolkit sits at the same path.
- `.toolchain/`: user-local CUDA 12.8 toolkit, pytorch v2.4.1 source + `sm120-pytorch-2.4.1.patch`, `build_torch.sh` (~25 min on 32 cores).
- `scripts/`: `gpu_pipeline_bench.py` (headless PhysX GPU + torch benchmark), `dexgys_validator_check.py` (T08).

### Moving this directory

The wheel stores absolute paths to `.toolchain/cuda-12.8` (RUNPATH of 8 libs, the nvrtc preload in `torch/__init__.py`,
two cmake files). The 2026-09-29 rename from `isaacgym-smoke/` was handled like this, and a later move needs the same:

```bash
uvx --from wheel wheel unpack wheels/torch-*.whl -d /tmp/w
for f in $(find /tmp/w -name '*.so*'); do r=$(patchelf --print-rpath $f); \
  [[ $r == *$OLD* ]] && patchelf --set-rpath "${r//$OLD/$NEW}" $f; done   # patchelf: uv tool install patchelf
grep -rIl "$OLD" /tmp/w | xargs sed -i "s|$OLD|$NEW|g"
uvx --from wheel wheel pack /tmp/w/torch-* -d wheels/
# then fix the torch path in each pyproject.toml, `uv lock`, delete and re-sync the three venvs
```
Checked after the rename: `gpu_pipeline_bench.py` (4096 envs, ~3M env-steps/s), T09 5 PPO epochs at 1024 envs (~5,500 env-steps/s),
T08 validator check.
