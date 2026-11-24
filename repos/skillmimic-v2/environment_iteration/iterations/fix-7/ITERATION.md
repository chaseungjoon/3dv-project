# fix-7

Parent: fix-6 | Created: 2026-11-24

Run in tmux from the repository root with `bash environment_iteration/scripts/train.sh fix-7`.
Protocol v1 starts from scratch at seed 0 for 3,000 epochs with 2,048 environments. Re-run the same command to resume.

## 1. Changes vs fix-6

The requested `FIX7_PLAN.md` bundles two reward changes. This is a human-directed reward-method deviation under LOOP §6 rule 3. Both changes share the same training run, so the final comparison measures their combined effect. Fix-7 inherits fix-6's physics, ATS, own starts, PPO settings, observations, termination, and evaluation protocol.

| Change | Implementation | Fix-6 failure it targets |
|---|---|---|
| Increase direct book-position weight from 30 to 60 | `env.rewardWeights.op` in [env.yaml](env.yaml) | The median live position error was 14.1 cm during the walk, with 76% of live frames more than 10 cm off. At 14.1 cm, the position factor falls from 0.82 to 0.67. |
| Add a sign-invariant book-rotation factor with weight 1.0 | `env.objRotWeight` and [obj_rot.py](../../src/tasks/obj_rot.py) | Median book angle error was 28° during the lift and 37° during the walk. Matching the reference pivot may also reduce the lift's three-frame lead. |

`SkillMimicParahomeIterObjRot` extends fix-6's own-start task. It takes the simulated and reference book quaternions from the same observation tensors used by the position reward. It computes `θ = 2 acos(clamp(abs(dot(q_sim, q_ref)), 0, 1))` in radians and multiplies each raw reward by `exp(-objRotWeight × θ²)` before buffer-node averaging. The upstream component-wise quaternion weight `rewardWeights.or` stays at 0. When `objRotWeight` is omitted or zero, the task delegates the full reward calculation to its parent.

## 2. Hypothesis

At epoch 3,000, compare with fix-6 on the 96 clean stochastic episodes. Use live frames for the lift and carry windows. The pivot's effect on lift timing is a hypothesis.

| Metric | Fix-6 | Fix-7 support criterion |
|---|---:|---:|
| Carry frames outside 10 cm, t85–147 | 76% | ≤ 50% |
| Median carry position error, t85–147 | 14.1 cm | ≤ 10 cm |
| Median carry rotation error, t85–147 | 37° | ≤ 20° |
| Median nearest-reference lead, t55–80 | +3 frames | ≤ +1 frame |
| Episodes with whole-clip maximum error ≤ 20 cm | 27% | ≥ 50% |
| Lift success / upright carry walk | 100% / 90% | ≥ 90% / ≥ 80% |
| Strict task success (95% Wilson CI) | 1.0% (0–6%) | Increase |

Lift below 80%, a return to ignoring the book, or more than 40% of episodes falling in the clip would count against the combined reward change. Report strict and paper success alongside whole-clip pass rates at 10, 15, 20, and 30 cm. Fix-6 baselines come from the plan's trace analysis and the [fix-6 iteration report](../fix-6/ITERATION.md).

## 3. Validation before training (agent, each < 30 min)

- [x] Compared the parsed YAML and resolved arguments against fix-6. The only training differences are `op`, `objRotWeight`, and the selected task and config path. `fingerprint.py diff fix-7` has no production fingerprint until first launch. The recorded fix-6 fingerprint flags the new task file and registry entry because it hashes the whole task directory. Fix-6's own task and other recorded inputs match.
- [x] The CPU portion of `check.sh` passed shell and Python syntax, metrics regression, and protocol and iteration loading. Three CPU reward tests passed for quaternion signs and known angles, weighting before buffer averaging, and delegation with the weight omitted.
- [x] Ran a live GPU reward probe in fix-7 with the frozen fix-6 epoch-3000 checkpoint, 8 clean stochastic environments, seed 101. For each step, the probe evaluated both reward definitions on the same tensors. The reconstructed reward matched fix-7 to `1.79e-7` absolute error. Simulated and reference book quaternion norms were within `1.8e-7` of 1.
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-7 --keep` passed. The scratch run interrupted at epoch 10 with exit code 130, restored full training state, reached epoch 20, evaluated both cells, and generated the report. A third run skipped training. The video check rendered two clips, and the resume verified the recorded scratch fingerprint.
- [x] Timed a scratch run from seed 0 with 2,048 environments on the RTX 5070. Epochs 1–40 averaged **10.60 s/epoch** from the trainer's total throughput, or **about 8.8 hours** for 3,000 epochs before evaluation. Fix-6 averaged 10.92 s/epoch over its first 40 epochs and 9.91 s/epoch across its full run. The scratch launcher took 443 s wall time, including setup and the trainer's exit pass after the epoch-40 checkpoint. The ATS sampler updated at step 750.

Live probe medians over pooled live frames. Negative dots measure opposite quaternion signs.

| Window | Live samples | Negative dots | Position factor, fix-6 → fix-7 | Rotation factor | Total reward, fix-6 → fix-7 |
|---|---:|---:|---:|---:|---:|
| t45–49 | 40 | 0% | 0.997 → 0.993 | 0.997 | 0.828 → 0.814 |
| t50–54 | 40 | 80% | 0.998 → 0.997 | 0.986 | 0.792 → 0.779 |
| t55–80 | 208 | 100% | 0.978 → 0.955 | 0.807 | 0.675 → 0.485 |
| t85–147 | 504 | 100% | 0.842 → 0.708 | 0.680 | 0.493 → 0.266 |

The carry median retained 53.9% of the paired fix-6 reward, above the plan's one-third cutoff. This keeps `objRotWeight: 1.0`. The plan's 0.367 reference value came from the earlier fix-5 checkpoint probe during t45–85. The paired carry comparison uses the fix-6 checkpoint.

## 4. Results (from analysis/e3000/REPORT.md)

| Metric (96 episodes per cell) | fix-6 | fix-7 |
|---|---:|---:|
| Strict task success, clean (95% Wilson CI) | 1.0% (0–6%) | Pending |
| Strict task success, perturbed (95% Wilson CI) | 0.0% (0–4%) | Pending |
| Paper success, clean | 93.8% | Pending |
| Lift success, clean | 100.0% | Pending |
| Task progress, clean | 48.4% | Pending |
| Fell in clip, clean | 10.4% | Pending |
| Dominant first failure, clean | off_path 86% @ t68 | Pending |

## 5. Failure analysis (agent)

Pending epoch-3000 evaluation.

## 6. Videos

Select the clips after epoch-3000 analysis.

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
