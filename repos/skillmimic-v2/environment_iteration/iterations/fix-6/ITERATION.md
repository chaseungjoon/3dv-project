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
| Strict task success, clean (95% Wilson CI) | 0.0% (0–4%) | Pending |
| Strict task success, perturbed (95% Wilson CI) | 0.0% (0–4%) | Pending |
| Paper success, clean | 19.8% | Pending |
| Lift success, clean | 96.9% | Pending |
| Task progress, clean | 43.0% | Pending |
| Fell in clip, clean | 58.3% | Pending |
| Median first >10 cm while alive, clean | t63 | Pending |
| Median peak 3D error, clean t45–85 | 28.4 cm | Pending |
| Median peak vertical lead, clean t45–85 | 20.9 cm | Pending |

## 5. Failure analysis

After evaluation, compare the saved traces using the commands below. The probe decomposes pickup error into along-path, cross-path, and vertical components and reports vertical speed error, first threshold crossing, and endpoint error among survivors.

```bash
.venv/bin/python environment_iteration/src/probes/book_tracking.py fix-5 fix-6
.venv/bin/python environment_iteration/src/probes/book_tracking.py fix-5 fix-6 --cell stoch_perturb
```

## 6. Videos

Select pickup and carry clips after the epoch-3000 analysis.

## 7. Video notes (human)

## 8. Decision → next iteration
