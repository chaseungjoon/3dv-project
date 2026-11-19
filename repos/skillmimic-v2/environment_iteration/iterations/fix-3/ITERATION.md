# fix-3

Parent: fix-2 | Created: 2026-11-18 | Summary: hand and book contact offset 0.005->0.02 m (per shape; fix-2 solver
otherwise): keeps the grasp discoverable, fix-2's holding
Run: `bash environment_iteration/scripts/train.sh fix-3` (human, **~14-17 h**, 20.2 s/epoch measured; resumable). Direction chosen by the agent
acting as the human reviewer (fix-2 ITERATION.md §7-8), at the user's request.

## 1. Changes vs fix-2

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | PhysX contact offset of every **hand** shape (40: wrists + finger segments) and the **book** shape 0.005 -> 0.02 m, set at actor creation; all other shapes keep fix-2's 0.005 (`sim.physx.contact_offset`) | env.yaml `handContactOffset`, `objectContactOffset`; task `SkillMimicParahomeIterContactOffset` (`src/tasks/contact_offset.py`, new) | fix-2 never grasps: no_lift 96% @65, lift 0%, held 0 (fix-1: lift 47%) | `src/probes/policy_env.py` (fix-1's policy zero-shot), `grip.py` static holds, cost run |

Everything else (solver substeps 4 / velocity iterations 8, hand asset, book, desk, reward, reference, termination,
PPO) is fix-2's. Diff: `diff iterations/fix-2/env.yaml iterations/fix-3/env.yaml` = the two keys;
`iteration.json` differs only in `task` (the new class = fix-2's class + the two keys; without them identical).

### Evidence (fix-2 ITERATION.md §8 has the full probe set)

fix-1's epoch-3000 policy (trained with contact offset 0.02, substeps 2, no velocity iterations) run **zero-shot**
in each environment, clean frame-2 starts, stoch, 3 seeds (101-103) x 32 envs = 96 episodes
(`src/probes/policy_env.py fix-1 --epochs 3000 --env <it> [--set ...] --seeds 101,102,103`):

| environment | lift | held frames | drop | fell in clip |
|---|---:|---:|---:|---:|
| fix-1 (its own) | 46% | 10.0 | 55% | 97% |
| fix-2 (offset 0.005 everywhere) | 3% (seed 101 only) | 1.4 | 0% | 84% |
| fix-2 + global offset 0.01 | 0% (seed 101 only) | 0.2 | 0% | 66% |
| fix-2 + global offset 0.02 | 41% | 25.5 | 21% | 59% |
| **fix-3 (hand + book 0.02, rest 0.005)** | **54%** (44 / 72 / 47) | **37.2** | **14%** | **58%** |
| fix-3 variant: hand 0.02, book 0.005 | 49% | 34.4 | 21% | 66% |
| fix-3 variant: hand 0.005, book 0.035 | 1% | 0.7 | 2% | 83% |

Static holds of the carry grip (perfect imitator, `grip.py --frames 75,90,100,110,120,130,140 --repeat 8`):
fix-1 0/56, fix-2 18/56, fix-2 + global 0.02 24/56, **fix-3 16/56** (within the probe's noise of fix-2).

Reading: the **hand shapes' 2 cm contact offset** is what made fix-1's grasp possible, and substeps 4 + velocity
iterations 8 are what make a held book stay held; fix-3 keeps both. It is not the hand-book pair distance (a 3.5 cm
book offset with a 0.5 cm hand offset does nothing), so the mechanism lies in the hand's other contacts (most likely
finger-finger contacts inside the hand, or finger-desk contacts under the book's edge); not resolved, and the
videos of fix-3 should show it. The book keeps 0.02 as well (the most-tested variant; hand-only is within noise). The demonstrated pickup is still not executable open-loop (`grip.py --replay
40:147 --ff 0.5` in fix-3: lost at frame 59), so the policy must again find its own grasp, as fix-1's did.

Why per shape: it changes only the hand and the book and leaves the feet / ground, body and desk contacts exactly
as in fix-2, so the change stays attributable to the grasp. It saves little time (global 0.02: 21.4 s/epoch, fix-3:
20.2 s/epoch, 40-epoch runs at 2,048 envs), so most of the extra cost over fix-2 (14.1 s/epoch measured the same
way) comes from the hand shapes' larger offset itself.

## 2. Hypothesis

With a hand-book contact the policy can discover (as in fix-1) and a solver that keeps a held book held (as in
fix-2), fix-3 at epoch 3000 (stoch, 96 episodes) should:
- lift again: lift success >= fix-1's 46.9% (fix-2: 0%), no_lift no longer the dominant mode;
- hold longer than fix-1 once lifted: held frames > 11.0 (fix-1), drop event < 55% (fix-1);
- fall later / less than fix-1 (fall in clip < 96%), because the probes show the collapse follows a failed pickup
  (fix-2 §8, probe 1).
Task success may still be 0%: fix-1 lost the 10 cm track at the lift onset (lag / off_path at frames 66-68), and
this change does not address the timing of the lift.

Falsified if lift success stays < 20% with no_lift dominant (as in fix-2): then the grasp is not discoverable at this
budget even with fix-1's contacts, and the problem is exploration of the pickup itself (next: pickup-phase start
states / termination), not contact simulation. If it lifts but held frames and drops are no better than fix-1's,
the solver gain does not survive learning.

## 3. Validation before training (agent, 2026-11-18)

- [x] `bash environment_iteration/scripts/check.sh`: ALL PASS (syntax, metric regression on 1,024 episodes,
      protocol, iterations; fix-1 fingerprint 366 files, 0 differ). fix-2's fingerprint now reports
      `src/tasks/__init__.py` changed and `src/tasks/contact_offset.py` added: expected (new class registered); fix-2's
      training is finished and its task class is unchanged.
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-3`: 20-epoch train, Ctrl-C at 10 (exit 130), resume
      to 20, eval + analysis, re-run skips training, video, fingerprint. ALL PASS (run with the global-offset draft;
      re-run 2026-11-18 14:19 on the final per-shape version: ALL PASS.
- [x] the change acts: log `[contact_offset] object: 1 shape(s) contact_offset 0.0050 -> 0.0200`, `hand: 40
      shape(s) 0.0050 -> 0.0200`; fix-1's policy transfer table above (lift 54% vs 3% in fix-2).
- [x] static holds 16/56 (fix-2 18/56); no NaN; the book rests at reset as before.
- [x] cost at 2,048 envs (40-epoch scratch run, `EI_ITER_DIR=<scratch> launch.py train fix-3 --epochs 40`): ~3,240
      env-steps/s = **20.2 s/epoch** (fix-2 measured the same way: 14.1 s; its full run then averaged ~12 s). Expect
      **~14-17 h** for 3,000 epochs (fix-2: ~10.3 h). Peak VRAM ~10.5 GB.
- [x] probe tools kept for reproduction: `src/probes/policy_env.py` (policy in another env), `src/probes/policy_starts.py`
      (mid-clip starts), `grip.py --replay ... --ff KI` (better-tracking imitator).

## 4. Results (from analysis/e3000/REPORT.md)

| metric | fix-1 | fix-2 | fix-3 |
|---|---:|---:|---:|
| task success (stoch, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) | 0.0% (0–4%) |
| task success (perturbed, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) | 0.0% (0–4%) |
| paper success | 0.0% | 0.0% | 0.0% |
| lift success | 46.9% | 0.0% | 0.0% |
| held frames | 11.0 | 0.0 | 0.9 |
| drop event | 55% @98 | 0% | 0% |
| task progress | 44.8% | 44.0% | 44.1% |
| fell in clip | 95.8% | 81.2% | 11.5% |
| dominant failure (mode @ frame) | lag 51% @66 | no_lift 96% @65 | lag 91% @66 |

## 5. Failure analysis (agent)

At epoch 3000, clean stochastic evaluation (96 episodes), neither strict task success nor lift succeeds; the lift and sustained-hold hypotheses did not materialize in the trained policy. The pre-training transfer probe was a different policy and does not predict this training outcome. The specific falsification criterion (lift <20% *and* no_lift dominant) does not apply literally: 91% are classified lag, not no_lift. This classifier calls a book centre 2 cm above its start with a nearby wrist "in hand", even when an edge remains on the desk; it does not prove a secure grasp. No book cleared the desk, and at frame 70 the median centre was 0.78 m versus 0.99 m in the reference. Compared with fix-2's stationary book, fix-3 moves it slightly, but does not pick it up. Median first track loss shifts only from frame 65 to 66.

Falling within the reference clip decreases from 81.2% to 11.5%; over the full 300-frame evaluation all episodes eventually fall, median frame 158 versus 129.5 for fix-2. This is an observed balance improvement, not evidence of successful carrying: the book remains at the desk. In the perturbed cell (96 episodes), 50% are lag, 30% no_lift and 19% knock; lift and task success remain 0%. The heuristic categories and causes should be checked visually.

The deterministic learning curve uses only eight environments per checkpoint and is indicative, not a final success estimate. Lift and paper success briefly appear at epoch 1000 (2/8) and disappear afterward; held-frame estimates rise at epochs 2500–2750 then fall to zero at 3000. This suggests unstable behavior, but neither the 8-episode points nor the late held-frame heuristic establishes a reliable pickup.

## 6. Videos

Fresh camera rollouts are not exact replays of evaluation episodes. Red is the simulated book; the green wireframe and dots show the reference. Contact sheets were inspected: the pickup sheets show the book remaining at the desk while the reference rises; the later-fall sheet shows the fall after the reference clip. Fine contact details need video review.

- `video/fix-3/e3000_stoch_s101/INDEX.md`: `1_lift_lag` shows pickup and first loss; inspect hand contact and pivot. `2_late_fall` shows later balance and fall (after frame 147 in the selected examples). `4_full_overview` provides approach through carry context.
- `video/fix-3/e3000_stoch_perturb_s101_v2/INDEX.md`: `3_perturbed_pickup` shows pickup with altered book position/orientation; inspect whether the hands reach the actual book.
- The first perturbed render was interrupted after capture (exit 143); the separate `v2` rerender completed successfully. Its fresh rollout can differ from evaluation.

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
