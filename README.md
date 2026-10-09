# 3DV Project

## Layout

| Path | What it is |
| --- | --- |
| `repos/t09-wristmimic/` | WristMimic (humanoid manipulation, Isaac Gym). Our experiments are in `experiments/` |
| `repos/t10-crossformer/` | CrossFormer (cross-embodiment robot policy). Our experiments are in `experiments/` |
| `repos/skillmimic-v2/` | SkillMimic-V2 (Isaac Gym) on ParaHome Place Book / Drink Cup / Place Kettle. T09 viability check |
| `repos/parahome/` | ParaHome dataset: clip audit + the T09 WristMimic setup on cup/book scenes. T09 viability check |
| `repos/humanoidmimicgen/` | HumanoidMimicGen (MuJoCo, G1) Push Button with Diffusion Policy. T09 viability check |
| `isaacgym-env/` | Isaac Gym + PyTorch 2.4.1 built for RTX 50-series (sm_120) |
| `PROPOSAL_T09.md` | T09 research plan |
| `BASELINE_T09.md`, `BASELINE_T10.md` | Baseline results summaries |
| `T09_VIABILITY_CHECK.md` | Is T09's 0% a hardware limit or a task/method issue? Plan, run order, decision table |

## Getting started

1. **Isaac Gym env** (needed for T09): see `isaacgym-env/README.md`. Isaac Gym itself and the torch wheel are not in git.
2. **T09**: `repos/t09-wristmimic/experiments/README.md`
3. **T10**: `repos/t10-crossformer/experiments/QUICKSTART.md`
4. **T09 viability check**: `T09_VIABILITY_CHECK.md`, then `repos/{skillmimic-v2,parahome,humanoidmimicgen}/QUICKSTART.md`

Tested on Ubuntu 24.04 with an RTX 5070 (12 GB). Python envs are managed with `uv`.
