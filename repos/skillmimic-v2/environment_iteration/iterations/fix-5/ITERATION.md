# fix-5

Parent: fix-4 | Created: 2026-11-20 | Summary: connection starts: 20% of training resets start from the policy's own post-pickup states (ref frames 75-95, from episodes started <= 60); fix-4 environment otherwise
Run: `bash environment_iteration/scripts/train.sh fix-5` (human, **~8.5 h**, 10.0 s/epoch measured, same cost as fix-4;
resumable). Everything below is filled in by the agent unless marked (human).

Direction: the human's plan `ENV_FIX_PLAN.md` (2026-11-20, copy in this folder): measure feet, body, hands and book on
one time axis first (E1), test the book load (E2) and the right-hand release (E3) only as far as E1 requires, then
make **one** change chosen by the plan's section 7 table. E1 and E2 were run on fix-4 e3000 (section 0); they select
the first row of that table ("only the real pickup path fails to walk -> include the policy's own pickup end states
in the start distribution"). E3 was not needed (section 0.3).

## 0. Diagnosis that chose the change (fix-4 e3000, no training)

Probe `src/probes/gait_grip.py` (new; foot / hand / book timeline, optional book-load assist). All rows: stoch,
32 envs x eval seeds 101/102/103 = 96 episodes per row (repeats of one training seed). Raw output with commands:
`probes/e1_e2_fix4_e3000.txt`. Window statistics are medians over reference frames 85-145 (the reference carry
walk: root travel 0.87 m along the walk, ankle swings L 85-92, R 96-107, L 114-124).

Event definitions (the plan's section 3.3, adapted to what the simulator exposes):
- foot contact = net contact force on `jLeftAnkle`/`jLeftBallFoot` (or right) > 20 N (body indices 59/60, 55/56 looked
  up by name at runtime); lift = both bodies of a foot below that for >= 3 frames after reference frame 80; step =
  lift-off..touchdown with >= 10 cm horizontal ankle displacement; slip = supporting ball-foot body faster than
  0.1 m/s horizontally.
- right release (*estimated*): right wrist > 0.25 m from the book centre for 5 frames once the book is up. The
  fingertip net force can't serve as contact evidence: the clenched fingers press on each other (0-200 N with the
  wrist 1 m from the book), and the simulator gives no contact pairs.
- left slip = book position in the left-wrist frame > 5 cm from its value 3 frames after the release; pelvis drop =
  root height < 0.78 m (0.86 m standing); progress = root >= 0.3 m along the reference walk direction; drop = no wrist
  within 0.3 m of the book.

### 0.1 E1: the frame-2 path vs reference starts

| start (reference state) | pelvis drop | R release | first foot lift | first step | root travel along walk | lifts in 85-145 | both feet down | slip of support foot | pelvis z | book err | drop |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **2 (evaluation path)** | **100% @63** | 100% @70 | **1%** | **0%** | **0.14 m** | 0 | 100% | 2.6% | 0.68 m | 0.53 m | 96% @135 |
| 20 (to clip end) | 100% @67 | 100% @72 | 12% | 11% | 0.06 m | 0 | 97% | 5% | 0.68 | 0.57 | 54% @140 |
| 40 | 100% @70 | 100% @73 | 23% | 23% | 0.24 m | 0 | 97% | 7% | 0.67 | 0.30 | 27% @141 |
| 50 | 100% @73 | 100% @74 | 67% | 67% | 0.27 m | 1 | 92% | 6% | 0.64 | 0.23 | 31% @140 |
| 60 | 100% @77 | 100% @73 | 67% | 67% | 0.28 m | 1 | 93% | 4% | 0.62 | 0.18 | 53% @140 |
| 65 | 100% @81 | 100% @74 | 96% | 96% | 0.30 m | 1 | 89% | 6% | 0.63 | 0.16 | 14% @142 |
| **70 (60 frames)** | 8% @121 | 100% @71 | **98% @86** | **97% @91** | **0.79 m** | 2 (0.33 m) | 63% | 41% | 0.83 | 0.10 | 0% |
| **85 (60 frames)** | 1% | 100% @86 | **100% @86** | **100% @92** | **0.78 m** | 3 (0.30 m) | 62% | 34% | 0.83 | 0.07 | 5% @128 |

Order of events on the frame-2 path (96/96 episodes): **pelvis drop (63) -> right release (70) -> left-hand slip (115)
-> drop (135)**. No foot lift-off between them.

Answers to the plan's E1 questions:
1. *Does it fail to step while the book is stable?* Yes. From 70 to ~115 the book sits still in the left hand
   (left-wrist frame within 5 cm, 76% of the window with the left wrist < 0.2 m) and neither foot leaves the floor.
2. *Does the book turn away right after a foot lifts?* There is no lift to precede it. The slip at ~115 comes while
   the arm reaches for the departed reference (fix-4 video `3_drop`).
3. *Do the feet move but land in place?* No. On the frame-2 path the feet don't move at all (0 lifts, 100% both
   feet down). Support-foot slip is rare (2.6% of support frames, median 0.006 m/s).
4. *Do book rotation / pelvis drop start right after the right release, before the walk?* The pelvis drop comes
   **before** the release (63 vs 70, 100%), during the lift.
5. *Is the problem gone from reference starts?* Yes, for starts with the book already up (70, 85): 2-3 steps of
   ~0.3 m, 0.78-0.79 m travel (reference 0.87), book within 0.07-0.10 m, pelvis 0.83 m. Every start that makes
   the policy do its own lift (2 .. 65) ends squatting (pelvis 0.62-0.68 m, stance 0.51-0.81 m), and travel stays
   at 0.06-0.30 m, at most one step.

H4 (foot contact / control): not supported. Slip is low on the failing path, and leg PD tracking error is *lower* on
the failing path (0.39 rad mean) than on the walking reference starts (0.47-0.53). The legs track their targets;
the targets just don't walk.

### 0.2 E2: book load (upward assist alpha x m x g at the book COM, world frame, 10-frame ramp)

Check that the force acts: book resting on the desk (assist from frame 0), net book contact force 4.91 / 2.45 / 0.00 N
for alpha 0 / 0.5 / 1.0, and the book rises 1-2 cm at alpha 1. The force reaches the book and scales as intended.
The absolute net force reads half of m g in this solver; the ratio is what matters. Book mass in the simulator
1.001 kg.

| frame-2 path | pelvis drop | first lift | first step | travel | lifts | book err | held (left) | drop |
|---|---|---|---|---:|---:|---:|---:|---|
| alpha 0 | 100% @63 | 3-4% | 0% | 0.14 m | 0 | 0.51-0.53 m | 0.75-0.77 | 95-98% @135 |
| alpha 0.5 from frame 75 | 100% @63 | 3% | 1% | 0.14 m | 0 | 0.51 | 0.85 | 78% @141 |
| alpha 1.0 from frame 75 | 100% @63 | 4% | 0% | 0.14 m | 0 | 0.49 | 0.87 | 53% @142 |
| alpha 0.5 from frame 45 | 100% @63 | 5% | 2% | 0.15 m | 0 | 0.50 | 0.84 | 75% @140 |
| alpha 1.0 from frame 45 | 100% @63 | 4% | 2% | 0.15 m | 0 | 0.48 | 0.85 | 54% @141 |

With the book weightless from frame 45 (before the lift and the squat), the squat still starts at 63 and the feet
stay down. Less load keeps the book in the hand longer (drop 96% -> 53%) but restores no walk. **H2 (the one-hand
load blocks the support transition) is not supported** by the tested assist levels. Caveat from the plan: this is a
zero-shot intervention on a policy trained without it.

### 0.3 Why not E3, and the decision

The plan runs E3 (keep the right hand) only if the right release causes the problem first, or if E2 can't separate
the causes. The pelvis drop comes before the release in 96/96 episodes, and the book stays stable for ~45 frames
after the release while the feet don't move. E3 was not run.

Plan section 7, row 1: *only the real pickup path fails to walk -> include the real pickup end states in the start
distribution ("connection" training); final evaluation from frame 2 without assistance.* That is fix-5.

One more fact shapes the design. The fix-4 training already starts mostly at frames 56-73 (ATS), and those starts
already produce the squat (rows 55-65 above). So training does visit "own lift -> squat" states, but only in the
last 0-35 frames of a 60-frame episode, which often end early when the book is > 0.25 m off. And the reward there is
~0.02 per step (training log of the scratch run below: 0.024 own vs 0.404 reference starts). fix-5 starts episodes
*from* those states with the full 60-frame horizon over the walk window, so the policy practises the post-pickup walk
from the state it really reaches.

## 1. Changes vs fix-4

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | Connection starts: 20% of training resets start from a buffered state of the policy itself (root, DOF pos / vel, book state, the 60-row history-encoder buffer), captured at reference frames 75-95 during training episodes that began at reference frame <= 60 (the policy did the pickup itself) while the book was within 0.20 m of its reference. The reference, phase, episode length and start frame of such an episode are those of the captured frame. Ring buffer of 1,024 states, at most 8 captures per step, so it follows the current policy. | `src/tasks/own_start.py` class `SkillMimicParahomeIterOwnStart` (subclass of fix-4's `SkillMimicParahomeIterATS`), registry entry, `env.yaml` keys `ownStartProb 0.2`, `ownStartFrames [75, 95]`, `ownStartMaxStart 60`, `ownStartObjDist 0.20`, `ownStartBuffer 1024`, `ownStartPerStep 8`, `ownStartMinFill 64`; `iteration.json` `task` | fix-4 @3000: no carry walk after the policy's own lift (root travel 0.26 m vs ~1.1 m, squat 0.68 m, 0 foot lift-offs in 85-145, drop 99% @129), while the same policy walks from reference states at >= 70 (section 0.1) | 30-epoch scratch run resumed from fix-4 e3000 (section 3) |

`diff iterations/fix-4/env.yaml iterations/fix-5/env.yaml` = the seven `ownStart*` keys; `iteration.json` differs
only in `task` (and name / parent / summary). Physics, assets, rewards, ATS, STF, termination, PPO and the run.py
flags are fix-4's.

Details that keep the change attributable:
- Own-start episodes are **excluded from the ATS statistics** (the per-start-frame reward and the start histogram),
  so ATS still measures reference starts only, as in fix-4.
- Own starts are not STF-noisy starts: `buffer_steps` (buffer-node reward smoothing) is 0 for them.
- Captured states come only from reference-start episodes (no own-start chains).
- Evaluation is unchanged (`--test` never captures or restores; rollout.py starts every env at frame 2, no assist).
- The buffer isn't saved in checkpoints. After a resume it refills before own starts resume: at least 64 states,
  8 captures per step, so a few steps once the policy lifts. A resumed run is therefore not bit-identical to an
  uninterrupted one (the rest of the training state is restored as before).
- Before the policy can lift (early training) nothing qualifies for capture, so fix-5 trains exactly like fix-4
  until the lift appears (fix-4: lift from e750).

## 2. Hypothesis

If the walk is missing only because training never starts from the state the policy's own pickup produces,
fix-5 at epoch 3000 (stoch, 96 clean frame-2 episodes, no assistance) should show:

- **root travel along the reference walk from frame 85 clearly above fix-4's 0.14 m** (median; reference 0.87 m,
  fix-4 from reference starts 70/85: 0.78 m), with foot lift-offs / steps in reference frames 85-145 (fix-4: 0 lifts,
  first step in 0% of episodes), measured with `src/probes/gait_grip.py fix-5 --start 2 --seeds 101,102,103`
  (fix-4's ITERATION.md trace number, root travel by frame 147 from the start, was 0.26 m);
- the drop later than fix-4's median frame 129 or rarer than 99%, and the book error at frame 120 below 0.69 m;
- lift success not below fix-4's 100% by more than noise, and falls in the clip still low (fix-4 0%). Own starts
  must not cost the pickup;
- task success > 0 would be a bonus. The strict criterion also breaks at 58-60 in fix-4 (lead / off_path during the
  lift), which this change does not target.

**Falsified** if travel stays <= ~0.3 m with no stepping from frame 2 while the training log shows own starts at
~20% of resets with a filled buffer. Then the policy does see its own post-pickup states but cannot learn the
walk from them. The near-zero reward there (0.024 per step vs 0.40) would then be the likely bottleneck: a reward /
signal question, not a start-distribution one. Also check the curve: if own starts trade the lift away (lift
success falls), the 20% share is too large.

What to read in `runs/fix-5_s0/train.log` after training: `grep '\[own\]'` (buffer fill, share of own starts,
captured frame range, own vs reference reward per step: the own reward should rise during training if the policy
learns to walk from those states).

## 3. Validation before training (agent, each < 30 min)

- [x] `bash environment_iteration/scripts/check.sh` (unit tests): ALL PASS. Fingerprints of finished runs fix-2/3/4
      list only the registry line and the new `src/tasks/own_start.py` (nothing to resume); fix-1 0 of 366 differ.
- [x] 30-epoch scratch run resumed from fix-4 e3000 (`EI_ITER_DIR=<scratch> launch.py train fix-5 --seed 0
      --epochs 3030 --num_envs 2048 --resume_from .../fix-4_s0_00003000.pth`, 5 min 18 s, exit 0). This exercises
      capture and restore with a policy that lifts (a from-scratch policy doesn't for ~750 epochs). One `[own]`
      line after 750 steps:
      `buffer 1024/1024 | own starts 6511 of 32927 resets (19.8%) | captures 5880 | buffer frames 75-95 (median 84)
      book err median 0.115 m | reward/step own 0.024 ref 0.404 | check: restore max err 7.63e-06, root after 1 step
      0.013 m from restored / 0.195 m from reference`.
      So: the share is as configured, the simulator holds the buffered state after the reset (max abs difference of
      root / DOF / book state 7.6e-6), and one step later the root is 1.3 cm from the restored position and 19.5 cm
      from where a reference start would have put it (physics continues from the own state). Cost: ~6,560
      env-steps/s total = **10.0 s/epoch** (fix-4 9.9) -> ~8.5 h for 3,000 epochs + eval.
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-5`: ALL PASS (interrupt at epoch 10 exit 130, resume to
      20, eval + analysis, re-run skips training, video 2 clips, fingerprint recorded). From scratch the buffer stays
      empty in 20 epochs (no lift yet), as expected; the restore path is covered by the resumed run above.

## 4. Results (from analysis/e3000/REPORT.md)

Epoch 3000, protocol v1, 96 clean stochastic episodes per cell (seeds 101-103 x 32). Rows marked * are computed
from the stored traces (`eval/stoch/e3000_s*.npz`), not by analyze.py; rows marked † come from
`src/probes/gait_grip.py fix-5 --start 2,85 --seeds 101,102,103` (fresh rollouts, 96 episodes per start; raw output
`probes/gait_fix5_e3000.txt`). Training finished 2026-11-20 21:17 (train.sh log `logs/train_20261120_130621.log`).

| metric | fix-4 | fix-5 |
|---|---:|---:|
| task success (stoch, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) |
| task success (perturbed, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) |
| paper success (stoch / perturbed) | 8.3% (4–16%) / 11.5% | **19.8% (13–29%)** / 11.5% |
| lift success (stoch / perturbed) | 100.0% / 71.9% (62–80%) | 96.9% (91–99%) / **56.2% (46–66%)** |
| held frames (stoch) | 70.2 | 55.4 |
| task progress | 40.0% | 43.0% |
| episodes within 10 cm at frame 60 (path table) | 18% | **88%** |
| first failure (mode @ median frame) | off_path 56% @60, lead 44% @58 | lead 59% @62, off_path 41% @66 |
| grasp event (median frame) | 100% @49 | 100% @52 |
| **fell in clip (stoch / perturbed)** | 0.0% / 2.1% | **58.3% (48–68%)** / 51.0% |
| fall at any time (300 frames) | 17.7% @262 | 96.9% @131 |
| drop / floor (median frame) | 99% @129 / 97% @137 | 77% @117 / 86.5% @139 |
| book error to reference at frame 70 / 90 / 120 (median) | 0.27 / 0.19 / 0.69 m | 0.24 / 0.29 / 1.05 m |
| * pelvis height at frame 100 (median) | 0.68 m | 0.72 m |
| * mean reward per step, frames 0-50 / 50-70 / 70-147 | 0.61 / 0.19 / 0.006 | 0.77 / **0.69** / **0.16** |
| * holding hand at frame 100 (wrist within 0.2 m of the book) | left 98%, right 0% | left 82%, right 0% |
| * body outcome in the walk (`metrics.body_outcome`, see below) | stand_squat 94, walk_up 2 | **walk_up 17**, fall_fwd 16, fall_back 40, stand_squat 23 |
| † from frame 2: root travel along the walk from 85 (median) / steps in 85-145 / first step | 0.14 m / 0 / 0% | 0.03 m / 0 / 17-18% |
| † from frame 2: root progress >= 0.3 m along the walk (share, median frame) | 1% | 44-49% @107 |
| † from frame 2: pelvis drop / right release / left slip (share @ median frame) | 100% @63 / 100% @70 / 100% @115 | 79-84% @85 / 97-100% @67 / 75-79% @94 |
| † from reference frame 85 (60 frames): travel / steps / book err / drop | 0.78 m / 3 / 0.07 m / 5% | 0.77 m / 3 / 0.08 m / 0% |

**Body outcome** (new, `src/metrics.py` `body_outcome`, added for this analysis; it does not change any failure mode
or metric). Walk direction = reference ankles from frame 85 to 145; root progress along it from frame 85 to the fall
or the clip end. `walk_up` = no fall in the clip and progress >= 0.30 m; `stand_squat` = no fall, less progress;
`fall_fwd` = fell after progress >= 0.30 m; `fall_back` = fell after moving >= 0.20 m against the walk direction.
fix-5 stoch per group (96 episodes):

| outcome | n (95% CI) | fall frame (median) | max root progress | ankles along walk | pelvis @100 | book err @120 | reward / step 85-147 |
|---|---|---:|---:|---:|---:|---:|---:|
| walk_up | 17 (11–27%) | - (221 after the clip) | 0.66 m | 0.56 m | 0.82 m | 0.20 m | 0.31 |
| fall_fwd | 16 (11–25%) | 126 | 0.88 m | 0.23 m | 0.81 m | 0.22 m | 0.18 |
| fall_back | 40 (32–52%) | 104 | 0.00 m | -0.14 m | 0.59 m | 1.84 m | 0.04 |
| stand_squat | 23 (17–33%) | - (170) | 0.00 m | -0.02 m | 0.73 m | 0.96 m | 0.05 |

The same classifier on the older iterations: fix-1 stoch fall_back 91/96, fix-4 stand_squat 94/96 (walk_up 2).

Own starts (training log `runs/fix-5_s0/train.log`, 129 `[own]` lines): the first captures came at step ~12,000
(~epoch 375 at 32 steps per epoch, once the policy lifted), the buffer was full by step ~18,000, and from then to
the end own starts were 19.6-20.2% of resets, captured at reference frames 75-95 (median 84), buffer book error 0.07-0.11 m.
Restore check every line: max error 7.6e-6, root 0.5-0.7 cm from the restored state after one step. **Reward per
step from own starts rose from 0.14 (step 12,000) to 0.38-0.45 (steps 48,000-96,000)**; reference-start reward
0.43 -> 0.67-0.69. The policy learned to collect reward from its own post-pickup states, as intended.

## 5. Failure analysis (agent)

**Hypothesis result: partly supported, with a new failure.** The change did what it was designed to do in training
(own-start reward 0.02 at the fix-4 checkpoint -> ~0.4), and on the frame-2 evaluation path the policy now leaves the
fix-4 squat in about half of the episodes: 33/96 move the root >= 0.3 m along the walk (fix-4 2/96), and 17/96
(18%, CI 11–27%) walk ~0.66 m sideways with the book and stay up to the clip end (book error ~0.2 m at frame 120).
But the hypothesis also required falls to stay low, and they did not: **58% fall in the clip (fix-4 0%)**. The
§2 median criteria are therefore not met: median travel from frame 85 is 0.03 m (falls end the measurement), the
book error at frame 120 is 1.05 m (fix-4 0.69), and the drop is rarer (77% vs 99%) but earlier (117 vs 129)
because the falls throw the book. By the falsification rule (travel <= 0.3 m *with no stepping*) it is not
falsified: there is walking now, but it is unstable.

**What the population looks like (four outcomes, videos `1_walk_up` .. `4_squat`):**

1. `walk_up` (18%): upright (pelvis 0.82 m), the book in one hand at the side, both ankles move ~0.55 m along the
   walk, reward 0.31 / step in the walk. The step detector finds < 3 consecutive airborne frames per foot, so these
   may be short shuffling or sliding steps rather than the reference's 0.3-0.5 m strides (to check in the video).
2. `fall_fwd` (17%): the root travels 0.88 m along the walk (as far as the reference) while the ankles move only
   ~0.23 m. The body leans toward the departing reference like a plank and topples at ~126 (sheet `3_fall_fwd`).
3. `fall_back` (42%, the largest group): the pelvis drops at ~82, the root moves away from the walk direction and
   the character collapses backwards next to the desk at ~104 (sheet `2_fall_back`), the book thrown onto the desk
   or the floor. This is fix-1's signature (91/96 fall_back by the same classifier).
4. `stand_squat` (24%): fix-4's wide squat in place, book held low near the desk edge.

The state at the start of the walk does not separate the groups: at reference frames 75-95 all four have pelvis
~0.79-0.80 m and book error 0.25-0.31 m, and the share of frames that would have qualified for capture (book
within 0.20 m) is not lower in the falling groups (fall_back 0.48, walk_up 0.10). So it is not that the falling
episodes reach states the buffer never contains. The divergence happens after frame ~85, inside the walk, and the
stochastic policy picks one of these four behaviours per episode. From the reference state at frame 85 the same
policy walks cleanly (0.77 m, 3 steps, book error 0.08 m, 0% drop, 0% pelvis drop), as fix-4 did.

**Interpretation (not proven).** The own-start states (captured with the book within 0.2 m) give the walk a reward
signal, and the policy uses it, but from the own post-pickup state it has not found a stable way to shift the
support for the first step: it either leans along the walk without stepping (fall_fwd), sits back (fall_back), or
does not try (squat). Where it does walk, the feet are rarely airborne for 3 frames (shuffling?). 60 frames from an own
start end at ~frame 145, so own-start episodes cover the whole walk; their reward per step stays at ~0.4 (reference
starts ~0.68), so they are still far from the reference or end early (the log does not say which). Whether the falls come from the own-start training
itself (the policy learned a lean that pays off in short own-start episodes but is unstable over the full path) or
from the walk being harder from the squat posture is not separated by these data.

**Lift and the strict criterion.** The pickup did not suffer in the clean cell (lift 97% vs 100%, within noise;
grasp 52 vs 49, closer to the reference 53-56), and it got more precise: 88% of episodes are within 10 cm at frame
60 (fix-4 18%), the first break moved from 58-60 to 62-66, and the reward per step in frames 50-70 rose from 0.19
to 0.69. The strict track still breaks during the pivot-and-rotate: the book is 14-20 cm up at frame ~62 where the
reference is 5-9 cm, ~30° from the reference orientation (REPORT clip candidates).

**Perturbed cell.** Lift 56% (CI 46–66%) vs fix-4's 72% (62–80%), CIs barely overlap: the pickup of a displaced book
got worse. Sheet `6_perturbed_pickup`: the hand pushes the displaced book across the desk (knock, env 2), lifts and
loses it during the rotation (env 1), or reaches toward the green reference pose while the red book lies elsewhere
(env 9: classified no_lift at 21 because the perturbed start is already ~10 cm off). One possible cause: 20% of
resets now go to own starts, leaving fewer pickup episodes (including the STF-noisy ones that resemble a displaced
book); not tested.

**Learning curve (det, 8 envs per point; trends only).** Lift 8/8 from e1000 on, no lift oscillation. Fell in clip
oscillates: 8/8 at e1250-e2000, 0/8 at e2250-e2500, 1/8 at e2750, 3/8 at e3000. Progress 41-50%, held 60-89 frames.
The fall rate at e3000 is not a settled state: e2250 / e2500 might fall far less on the 96-episode cells (not
evaluated; eval.sh evaluates e3000 only).

**What this does not show.** Whether walk_up episodes take real steps (video). Whether an earlier checkpoint (e2250 /
e2500, 0/8 falls in the curve) keeps the walk without the falls. Whether the falls come from the frame-2 path only or
also appear from own-start states (a policy_starts / gait_grip run from captured own states would need the buffer,
which is not saved). Whether two-hand carrying would be more stable.

## 6. Videos

clips.json -> `video/fix-5/e3000_stoch_s101/INDEX.md` (16 envs) and `video/fix-5/e3000_stoch_perturb_s101/INDEX.md`
(fresh rollouts, not replays). In this stoch rollout: walk_up 3, fall_fwd 5, fall_back 6, stand_squat 2 of 16, all
lifted. New selector for this iteration: `"select": {"outcome": ...}` in clips.json (render.py), using
`metrics.body_outcome`; the INDEX episode table has a new "walk outcome" column. I checked every contact sheet; each
shows what it was chosen for.

- `1_walk_up` (env 2, 4; frames 78-147): **the new behaviour.** Upright, book held at the side in one hand, the body
  moves along the walk while the green reference pulls ahead. Watch the feet: real steps or shuffling / sliding?
- `2_fall_back` (env 0, 1, 3; 30 frames before the fall): **the most common outcome.** The character sinks and
  falls backwards beside the desk, the book ends on the desk or floor. Is there a step attempt first? What triggers
  it: the right-hand release (~67), the pelvis drop (~82), or the first reference step (left foot 85-92)?
- `3_fall_fwd` (env 5, 8): walks / leans toward the reference, then topples forward stiffly with the book.
- `4_squat` (env 6): fix-4's squat in place, for comparison with clip 1 at frame ~80.
- `5_lift_break` (env 0, 1; frames 47-78): the lift, closer to the reference than fix-4 but ahead of it at ~62.
- `6_perturbed_pickup` (perturbed s101, env 1 / 2 / 9): book lifted and lost, book pushed, hand toward the reference.
- Full real-time episodes: `full/env00-15.mp4` in each folder (the outcome per env is in INDEX.md).

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
