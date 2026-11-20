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

| metric | fix-4 | fix-5 |
|---|---:|---:|
| task success (stoch, 96) | 0.0% (0–4%) | |
| task success (perturbed) | 0.0% (0–4%) | |
| paper success | 8.3% / 11.5% | |
| lift success | 100.0% / 71.9% | |
| task progress | 40.0% | |
| fell in clip | 0.0% / 2.1% | |
| dominant failure (mode @ frame) | off_path 56% @60, lead 44% @58 | |
| travel along the walk from 85 / lifts in 85-145 / first step (gait_grip.py, start 2, 96) | 0.14 m / 0 / 0% | |

## 5. Failure analysis (agent)

## 6. Videos

clips.json -> `video/fix-5/...` (see INDEX.md there). Recommended clips and what to look for:

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
