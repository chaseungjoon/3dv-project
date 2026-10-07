# 3DV Project

## Layout

| Path | What it is |
| --- | --- |
| `repos/t09-wristmimic/` | WristMimic (humanoid manipulation, Isaac Gym). Our experiments are in `experiments/` |
| `repos/t10-crossformer/` | CrossFormer (cross-embodiment robot policy). Our experiments are in `experiments/` |
| `isaacgym-env/` | Isaac Gym + PyTorch 2.4.1 built for RTX 50-series (sm_120) |
| `PROPOSAL_T09.md` | T09 research plan |
| `BASELINE_T09.md`, `BASELINE_T10.md` | Baseline results summaries |

## Getting started

1. **Isaac Gym env** (needed for T09): see `isaacgym-env/README.md`. Isaac Gym itself and the torch wheel are not in git.
2. **T09**: `repos/t09-wristmimic/experiments/README.md`
3. **T10**: `repos/t10-crossformer/experiments/QUICKSTART.md`

Tested on Ubuntu 24.04 with an RTX 5070 (12 GB). Python envs are managed with `uv`.
