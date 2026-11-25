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

## 4. Results (epoch 3000)

From [analysis/e3000/REPORT.md](analysis/e3000/REPORT.md) and the fix-6 / fix-5 reports. 96 episodes per cell.

| Metric | fix-5 | fix-6 | fix-7 |
|---|---:|---:|---:|
| Strict task success, clean (95% Wilson CI) | 0.0% (0–4%) | 1.0% (0–6%) | **0.0% (0–4%)** |
| Strict task success, perturbed (95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) | 0.0% (0–4%) |
| Paper success, clean | 19.8% | 93.8% | 15.6% |
| Lift success, clean | 96.9% | 100.0% | 77.1% |
| Task progress, clean | 43.0% | 48.4% | 47.7% |
| Fell in clip, clean | 58.3% | 10.4% | **93.8%** |
| Dominant first failure, clean | lead 59% @ t62 | off_path 86% @ t68 | off_path 91% @ t67 |
| Late events, clean (median frame) | drop 77% @117, fall 97% @131 | drop 92% @190, fall 100% @210 | drop 61% @126, fall 96% @130 |

Additional measures (agent, from the stored epoch-3000 traces; `metrics.body_outcome` for the walk classes; "live" = frames before the fall; t55–80 lead = nearest reference frame along the book path minus t). Script: the per-episode loop over `eval/<cell>/e3000_s*.npz` with `metrics.body_outcome` / `metrics.rot_err_deg`; fix-6 values reproduce its §4 table exactly.

| Metric (clean stoch, 96 episodes) | fix-5 | fix-6 | fix-7 | fix-7 §2 criterion | met? |
|---|---:|---:|---:|---:|:---:|
| walk_up / stand_squat | 18% / 24% | 90% / 0% | **0% / 6%** | walk_up ≥ 80% | no |
| fall_fwd / fall_back | 17% / 42% | 10% / 0% | **94% / 0%** | – | – |
| Median root progress along the walk | 0.00 m | 0.77 m | 1.12 m | – | – |
| Carry live frames outside 10 cm, t85–147 | 98% | 76% | 84% | ≤ 50% | no |
| Median carry position error, t85–147 (live) | 44.8 cm | 14.1 cm | 16.0 cm | ≤ 10 cm | no |
| Median carry rotation error, t85–147 (live) | 53° | 37° | **21°** | ≤ 20° | almost (1° short) |
| Median lift rotation error, t55–80 (live) | 38° | 28° | **20°** | – | – |
| Median nearest-reference lead, t55–80 | +4 | +3 | +2 | ≤ +1 | no |
| Median whole-clip max book error | 1.57 m | 0.25 m | 1.02 m | – | – |
| Whole-clip max error ≤ 10 / 15 / 20 / 30 cm | 0 / 0 / 0 / 1% | 1 / 4 / 27 / 76% | 0 / 0 / 0 / 0% | ≤ 20 cm in ≥ 50% | no |
| Episodes within 10 cm at t 70 / 90 / 110 / 130 | 5 / 2 / 6 / 0% | 27 / 42 / 33 / 16% | 23 / 19 / 17 / 0% | – | – |
| Median in-clip fall frame (episodes) | 112 (56) | 134 (10) | 130 (90) | – | – |
| Perturbed: lift / walk_up / fall in clip | 56% / 12% / 51% | 54% / 73% / 27% | 47% / 2% / 70% | – | – |
| Perturbed: paper success | 11.5% | 35.4% | 4.2% | – | – |

`book_tracking.py fix-6 fix-7` (p10 / p50 / p90):

| t45–85 | fix-6 clean | fix-7 clean | fix-6 perturbed | fix-7 perturbed |
|---|---|---|---|---|
| max 3D error (m) | 0.080 / 0.172 / 0.266 | 0.105 / 0.167 / 0.292 | 0.112 / 0.477 / 0.810 | 0.127 / 0.482 / 0.830 |
| peak vertical lead (m) | 0.025 / 0.094 / 0.149 | 0.021 / 0.087 / 0.149 | -0.002 / 0.005 / 0.121 | -0.002 / 0.004 / 0.106 |
| first >10 cm while alive (t) | 63.4 / 67 / 90.2 | 64 / 67 / 77.5 | 48 / 64 / 73 | 51 / 65 / 71 |
| final 3D error, survivors (m) | 0.136 / 0.210 / 0.306 (86) | 1.665 / 1.742 / 1.761 (6) | 0.205 / 0.462 / 1.782 (70) | 1.694 / 1.817 / 1.850 (29) |

Learning curve (det, 8 envs, trend only): lift appears at e750–1000 and stays at 87–100% from e1000. Fell in clip is 100% at every point from e1250 except e1500 (12.5%), so the forward fall was learned early and kept. Single training seed.

## 5. Failure analysis (agent)

Hypothesis (§2): **falsified.** Only the rotation measures moved as intended; every position and walk criterion got worse, and two of the stated counter-indicators fired (lift 77% < 80%, fell in clip 94% > 40%).

What the numbers show:

1. **The rotation reward worked on its own target.** Book orientation error dropped during the lift (t55–80 median 28° → 20°) and the carry (37° → 21°, the criterion is 20°). The lift lead shrank slightly (+3 → +2 frames). The 10 cm break during the pickup did not move (off_path 91% at t67, fix-6 86% at t68; book centre still ~22 cm up vs the reference 16 cm), so the orientation gain did not translate into position accuracy at the lift.
2. **The carry walk was lost: the policy lunges instead of stepping.** fix-6's walk_up 90% became fall_fwd 94% (median fall t130, p25–p75 128–132, very consistent). At t120 the pelvis is 0.31 m ahead of the ankles (fix-6 walk_up −0.03 m) and the feet travel only 0.34 m along the walk (fix-6 0.71 m), while the root reaches 1.12 m. Video (clip 2): one long lunge / kneel with the arm holding the book out toward the green reference, then a forward topple. The book follows the reference to ~16 cm until t120 (similar to fix-6) and is lost at the fall (drop t126, floor t135). fix-6's minority failure (fall_fwd 10%) is now the policy.
3. **Most likely cause (not tested):** with `op` 60 plus the rotation factor, the reward per carry step was 0.493 → 0.266 on the fix-6 policy (§3). Keeping the book close by stretching the body forward pays more right away than the slower step pattern, and the fall at ~t130 cuts only the last ~17 clip frames. Which of the two changes causes the lunge cannot be separated from this single bundled run.
4. **Lift metric drop is partly a grip-distance effect.** grasp 100% and off_desk 97% (the book leaves the desk in almost every episode), but `lift_success` uses evaluate.py's 0.20 m wrist radius; the book is held farther out (arm extended) in more episodes. Paper success (15.6%) falls because the episodes end at ~t130 before the 60 frames above 0.9 m accumulate.
5. **Perturbed start got worse:** lift 47% (fix-6 54%), knock 16% (fix-6 7%), and the walk transfers only as the same lunge (fall_fwd 66 of 96).
6. **What the numbers do not show:** whether rotation alone (with fix-6's `op` 30) keeps the fix-6 walk; whether `op` 60 alone produces the lunge. All results are one seed.

## 6. Videos

Rendered 2026-11-25 with `bash environment_iteration/scripts/video.sh fix-7` ([clips.json](clips.json)). Fresh rollouts, so env numbers do not match the evaluation. The 16-env clean rollout reproduces the evaluation: 15/16 fall_fwd (root 1.0–1.15 m, feet 0.25–0.37 m), falls at t124–134, 14/16 first break off_path at t63–78.

| clip | where | what to look for |
|---|---|---|
| 1_fall_fwd | `video/fix-7/e3000_stoch_s101/clips/1_fall_fwd__env0{1,2}__f*.mp4` | The new dominant ending (94%, t~130). Deep lunge with the book held out, then the forward topple. Does any foot leave the ground before the fall? |
| 2_carry_walk | `.../e3000_stoch_s101/clips/2_carry_walk__env01__f078-140.mp4` | The whole carry to the fall. The red book matches the green wireframe's orientation well (HUD 7–17°), but the body slides into a kneeling lunge instead of stepping. |
| 3_lift_off_path | `.../e3000_stoch_s101/clips/3_lift_off_path__env0{1,2}__f*.mp4` | The pickup break (t65; env 1 breaks late at t99 during the carry). Is the book still lifted too high / early vs the green pivot? |
| 4_stand_squat | not rendered | No stand_squat episode in the 16-env rollout (6% in the evaluation). |
| 5_perturbed_pickup | `video/fix-7/e3000_stoch_perturb_s101/clips/5_perturbed_pickup__env0{2,3}__f*.mp4` | knock (t44) and no_lift (t64) on the displaced book. |

Contact sheets: `sheets/*.png` in each folder. Full episodes: `full/envNN.mp4`.

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
