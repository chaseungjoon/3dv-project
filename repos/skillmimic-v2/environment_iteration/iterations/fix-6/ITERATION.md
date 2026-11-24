# fix-6

Parent: fix-5 | Created: 2026-11-23

Run in tmux from the repository root with `bash environment_iteration/scripts/train.sh fix-6`.
The run starts from scratch with seed 0 and resumes if interrupted.

## 1. Changes vs fix-5

Fix-6 implements Experiment A in `TalkFile_FIX5_IMPROVEMENT_ANALYSIS.md.md` §10.4 and §10.8. It is a reward-method experiment within the iteration loop. The single training change is `env.rewardWeights.op: 1 → 30`. It scales the existing direct book-position factor over the episode, including pickup and carry. Fix-6 uses fix-5's own-start task, ATS, episode horizon, observations, physics, and PPO settings. The evaluation keeps protocol v1 and its fixed reference timing.

In the fix-5 epoch-3000 clean stochastic cell (96 episodes), the median first excursion past 10 cm occurred at t63. Median peak book position error in t45–85 was 28.4 cm and median peak height above the reference was 20.9 cm. The brief's videos show the book rising ahead of the reference before the sideways carry. Here and below, `t` is the zero-based stored rollout step.

The direct factor is `exp(-op × d² / 3)` for 3D position error `d` in metres. At 10 cm it changes from 0.997 to 0.905. At 25 cm, the existing training termination radius, it changes from 0.979 to 0.535. I chose 30 as the first scale tested. On stored fix-5 t55–75 clean samples the reconstructed position factor had p10/median/p90 of 0.396/0.830/0.996 at 30. The corresponding p10 was 0.214 at 50 and 0.046 at 100. At 30 the factor stayed above 0.05 across the stored live t55–75 samples in both cells. Over t86–147, 46.3% of live clean samples had a reconstructed factor below 0.05, consistent with the large carry errors in the fix-5 evaluation. These estimates use saved position traces. The live check uses the reward's own tensors.

## 2. Hypothesis

The stronger position term should reduce early pickup overshoot without sacrificing lift. Compare epoch 3000 against fix-5 in the 96-episode clean stochastic cell. Look for a median per-episode maximum t45–85 error of at most 24 cm, a median first live 10 cm excursion at t66 or later (or fewer than half of episodes with a live excursion), and lift success of at least 90%. Check the number of live samples and falls alongside both timing measures. A later excursion coupled with earlier falls, little change in pickup error, or lift loss would count against the hypothesis. Report strict success and its Wilson interval, paper success, in-clip falls, and perturbed results alongside these pickup diagnostics.

## 3. Validation before training

- [x] Confirmed fix-5's recorded training fingerprint still matches (`92a60cd94ca1`). Compared the parsed YAML and run arguments to verify that `op` is the sole training change.
- [x] CPU portion of `bash environment_iteration/scripts/check.sh --gpu fix-6 --keep` passed shell and Python syntax, metric regression, and protocol and iteration loading.
- [x] Ran the saved-trace pickup probe on both fix-5 epoch-3000 cells with `--weights 1 30 50 100`.
- [x] Live GPU probe in the fix-6 environment with the frozen fix-5 epoch-3000 policy, 8 clean stochastic envs, seed 101. Across t45–85, changing only the weight in the reward inputs reproduced the live reward within `1.2e-7` absolute error. Median factors on the same samples were 0.648 for body imitation, 0.861 for the interaction graph, and 0.993 → 0.814 for the direct book-position term. The median total reward was 0.493 → 0.367. These paired measurements validate the reward path on one rollout.
- [x] Full scratch GPU pipeline passed with `bash environment_iteration/scripts/check.sh --gpu fix-6 --keep`. It interrupted at epoch 10, resumed to epoch 20, evaluated both cells, wrote its report, skipped training on a third invocation, rendered two clips, and recorded the scratch fingerprint.
- [x] Fixed a short-video crash exposed by the first pass. The walk classifier now marks the walk unobserved when the reference window is unavailable. A CPU regression covers the short trace and an early fall. All 192 stored fix-5 full-clip walk classifications match their previous values.

The live probe matched the reward's reference book position exactly to motion frame `t + 2`. At the same reward call the simulated book position in the HOI observation differed by up to 7.4 cm from the refreshed book root. The inherited code computes HOI observations before refreshing the simulator tensors. Fix-6 uses that same reward timing to isolate the weight change. The existing own-start capture filter also reads the HOI observation, so its error can differ from the position of the captured book state.

## 4. Results (epoch 3000)

Fix-5 baselines are from [its evaluation report](../fix-5/analysis/e3000/REPORT.md) and the saved-trace probe. Each cell has 96 episodes.

| Metric | fix-5 | fix-6 |
|---|---:|---:|
| Strict task success, clean (95% Wilson CI) | 0.0% (0–4%) | 1.0% (0–6%) |
| Strict task success, perturbed (95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) |
| Paper success, clean | 19.8% | 93.8% |
| Lift success, clean | 96.9% | 100.0% |
| Task progress, clean | 43.0% | 48.4% |
| Fell in clip, clean | 58.3% | 10.4% |
| Median first >10 cm while alive, clean | t63 | t67 (95 of 96 episodes) |
| Median peak 3D error, clean t45–85 | 28.4 cm | 17.2 cm |
| Median peak vertical lead, clean t45–85 | 20.9 cm | 9.4 cm |

Additional measures (agent, from the stored epoch-3000 traces; `metrics.body_outcome` for the walk classes, max over t 0–147 for the whole-clip error). fix-4 is included for context.

| Metric (clean stoch, 96 episodes) | fix-4 | fix-5 | fix-6 |
|---|---:|---:|---:|
| walk_up / stand_squat | 2% / 98% | 18% / 24% | **90% / 0%** |
| fall_fwd / fall_back | 0% / 0% | 17% / 42% | 10% / 0% |
| Median root progress along the walk | 0.15 m | 0.00 m | 0.77 m |
| Median whole-clip max book error | 1.33 m | 1.57 m | 0.25 m |
| Episodes with whole-clip max error ≤ 20 cm / ≤ 30 cm | 0% / 0% | 0% / 1% | 27% / 76% |
| Median drop frame (episodes with a drop) | 129 (99%) | 117 (77%) | 190 (92%) |
| Episodes within 10 cm at t 70 / 90 / 110 | 0% / 0% / 0% | 5% / 2% / 6% | 27% / 42% / 33% |
| Perturbed: lift / walk_up / fall_fwd | 72% / 6% / 2% | 56% / 12% / 12% | 54% / 73% / 24% |
| Perturbed: paper success / fell in clip | 11.5% / 2.1% | 11.5% / 51.0% | 35.4% / 27.1% |

Hypothesis (§2): all three criteria are met (peak t45–85 error 17.2 cm ≤ 24 cm; first live 10 cm excursion t67 ≥ t66; lift 100% ≥ 90%), with fewer in-clip falls (58% → 10%), not more. Strict task success is 1.0% (0–6%); the target is not reached.

Learning curve (det, 8 envs, trend only): lift appears at e750, is 100% from e1500 except e2500 (37.5%, held 13 frames), and is back at e2750–3000. The curve's fell-in-clip flips between 0% and 100% at e1250–1750. Single training seed.

## 5. Failure analysis

After evaluation, compare the saved traces using the commands below. The probe decomposes pickup error into along-path, cross-path, and vertical components and reports vertical speed error, first threshold crossing, and endpoint error among survivors.

```bash
.venv/bin/python environment_iteration/src/probes/book_tracking.py fix-5 fix-6
.venv/bin/python environment_iteration/src/probes/book_tracking.py fix-5 fix-6 --cell stoch_perturb
```

Probe output, epoch 3000 (p10 / p50 / p90, 96 episodes per cell):

| t45–85 | fix-5 clean | fix-6 clean | fix-5 perturbed | fix-6 perturbed |
|---|---|---|---|---|
| max 3D error (m) | 0.180 / 0.284 / 0.397 | 0.080 / 0.172 / 0.266 | 0.190 / 0.503 / 0.807 | 0.112 / 0.477 / 0.810 |
| peak vertical lead (m) | 0.097 / 0.209 / 0.280 | 0.025 / 0.094 / 0.149 | 0.000 / 0.064 / 0.199 | -0.002 / 0.005 / 0.121 |
| first >10 cm while alive (t) | 60 / 63 / 67.5 | 63.4 / 67 / 90.2 | 57 / 65 / 71 | 48 / 64 / 73 |
| final 3D error, survivors (m) | 0.295 / 1.267 / 1.736 (40) | 0.136 / 0.210 / 0.306 (86) | 0.383 / 1.597 / 1.782 (47) | 0.205 / 0.462 / 1.782 (70) |

What the numbers show:

1. **The carry walk is solved on the clean path.** 90% walk sideways with the book and stay up (root 0.77 m along the walk, ankles ~0.7 m, so real foot travel rather than a lean). fix-5's fall_back (42%) and fix-4's squat (98%) are gone. The book stays in hand through the clip (median drop t 190, after the clip). A stronger direct book-position reward fixed what fix-5's own-state starts only started: following the book along the path requires walking.
2. **The pickup is tighter but still breaks the strict 10 cm track.** 86% first break off_path at median t 68 (book centre ~27 cm up vs the reference's 19 cm, ~40° off its orientation) and 12% lead at t 63. The vertical overshoot halved (20.9 → 9.4 cm), but at t 70 the book is still 1.08 m vs 0.99 m and ahead of the reference (0.16 vs 0.06 m along the path). Only 27% are within 10 cm at t 70 and 16% at t 80.
3. **During the carry the error sits just above the line.** Median error 11–15 cm over t 90–130. 33–42% of episodes are back within 10 cm at t 90–110, but strict success needs every frame. The whole-clip max is 25 cm (median); 27% stay within 20 cm for the whole clip. rot_off (> 45°) occurs in 93% at median t 94, so orientation is a large part of the remaining error.
4. **Perturbed starts did not improve the grasp.** Lift 54% (fix-5 56%, fix-4 72%); first failures are no_lift 31% and knock 7% before the lift. Once lifted, the walk transfers (walk_up 73%, fell in clip 27%).
5. **What the numbers do not show:** whether the remaining t 63–80 error comes from the timing of the pivot-and-rotate or from a different grip. The 1 strict success (seed 103, env 29) was not reproduced in the fresh video rollouts. The `success` mode is from a single episode. All results come from one training seed.

## 6. Videos

Rendered 2026-11-24 with `bash environment_iteration/scripts/video.sh fix-6` (clips.json) and `... --clips environment_iteration/iterations/fix-6/clips_extra.json --tag extra` (clips_extra.json). These are fresh rollouts, so the env numbers do not match the evaluation. The fresh 16-env rollout on seed 101 reproduces the evaluation picture: 14/16 walk_up, 2/16 fall_fwd, 15/16 first break off_path at t 64–96.

| clip | where | what to look for |
|---|---|---|
| 1_walk_up | `video/fix-6/e3000_stoch_s101/clips/1_walk_up__env0{1,2}__f078-147.mp4` | The new carry walk. Check for real steps vs sliding. Is the 11–15 cm gap to the green wireframe height, lag, or orientation (HUD shows 25–47°)? |
| 2_lift_off_path | `.../e3000_stoch_s101/clips/2_lift_off_path__env0{0,1}__f*.mp4` | The dominant break (t 66 / 77). Does the book rise too high / early relative to the green pivot-and-rotate? |
| 3_lift_lead | `.../e3000_stoch_s101/clips/3_lift_lead__env10__f047-077.mp4` | The same overshoot, earlier (t 62)? |
| 4_fall_fwd | `.../e3000_stoch_s101/clips/4_fall_fwd__env0{0,9}__f*.mp4` | 10% of episodes: falls at t 135–144 after walking ~1.3 m with the root but ~0.5 m with the feet. A lean ahead of the feet? |
| 5_post_clip_drop | `video/fix-6/e3000_stoch_s103_extra/clips/5_post_clip_drop__env00__f142-197.mp4` | A typical ending after the reference stops (t 147): the hand opens and the book falls at ~182. The metric excludes this. (`e3000_stoch_s101/clips/5_post_clip_drop__env00` shows an in-clip drop of a fall_fwd episode instead; the selector did not filter by outcome.) |
| 6_success | not rendered | Neither fresh rollout (16 envs s101, 32 envs s103) contained a strict success. |
| 7_perturbed_pickup | `video/fix-6/e3000_stoch_perturb_s101/clips/7_perturbed_pickup__env0{1,2,3}__f*.mp4` | no_lift / knock on the displaced book. Does the hand go to the red book or to the green reference? |

Contact sheets: `sheets/*.png` in each folder. Full episodes: `full/envNN.mp4`.

## 7. Video notes (human)

## 8. Decision → next iteration
