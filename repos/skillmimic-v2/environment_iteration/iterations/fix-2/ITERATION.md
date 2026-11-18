# fix-2

Parent: fix-1 | Created: 2026-11-17 | Summary: contact-simulation fidelity (substeps 2->4, PhysX velocity
iterations 0->8, contact offset 0.02->0.005 m)
Run: `bash environment_iteration/scripts/train.sh fix-2` (human, **~12 h**; resumable). Direction chosen by the agent
acting as the human reviewer (fix-1 ITERATION.md §7-8), at the user's request.

## 1. Changes vs fix-1

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | `sim.substeps` 2 -> 4 (physics 60 -> 120 Hz; control stays 30 Hz) | env.yaml `sim` | the book falls out of the demonstrated grip even for a perfect imitator | `src/probes/grip.py` static holds |
| 2 | `sim.physx.num_velocity_iterations` 0 -> 8 | env.yaml `sim.physx` | same | same |
| 3 | `sim.physx.contact_offset` 0.02 -> 0.005 m | env.yaml `sim.physx` | same (2 cm speculative contacts act before the fingers touch) | same |

Everything else (task class, hand asset, book, desk, reward, reference, PPO) is fix-1's. Diff:
`diff iterations/fix-1/env.yaml iterations/fix-2/env.yaml` = these three lines.

### Evidence (perfect-imitator probes, `src/probes/grip.py`)

A "perfect imitator" = root pinned to the reference root, every joint PD-driven to the reference pose, book placed
in the hand at its reference pose. "Held" = after settling (5 steps) the book stays within 2 cm of the wrist and
drops < 15 cm in 2 s. Carry frames 75, 90, 100, 110, 120, 130, 140 (reference frames).

| setting | held (env-frames) |
|---|---:|
| fix-1 (substeps 2, velocity iterations 0, contact offset 0.02) | **0/56** |
| substeps 2 + velocity iterations 8 or 16 (any contact offset) | 0/24 each |
| substeps 4 + velocity iterations 4 + contact offset 0.005 | 8/24 |
| substeps 4 + velocity iterations 8 + contact offset 0.02 | 12/24 |
| **substeps 4 + velocity iterations 8 + contact offset 0.005 (fix-2)** | **19/24, 15/56** (frames 110-140 mostly; 75-100 not) |
| substeps 4 + velocity iterations 16 + contact offset 0.005 | 8/24 |
| substeps 8 + velocity iterations 4 + contact offset 0.005 | 9/24 |
| fix-2 + book friction 2 | 20/56 (not taken: within noise, would blur attribution) |

Ruled out on the fix-1 solver (all 0 holds unless noted): book friction 2-8; torsional / rolling friction 0.05-1;
book mass 0.3 kg; fingers closed ±0.15-0.4 rad; finger capsule radius 7-12 mm (apparent 6-17/24 holds came from
neighbouring thick capsules jamming into each other, ~300 N in free air; 0/32 once intra-hand collisions are
filtered); a contact-consistent reference (`src/tools/contact_reference.py`: fingers moved 2-8 mm into the book;
0/42); chamfered book edges 8-20 mm (no change to the pickup); the contact-graph reward (disabled for ParaHome in
the paper itself, "r_cg = 1", so not part of the baseline).

Findings along the way that the next iterations need:
- The demonstration lifts and carries the book **one-handed** (left hand) from frame ~62; the right hand leaves at
  62-65. The left grip is a pinch near a long edge, the book's centre of mass ~7 cm away; without enough solver
  accuracy it rotates out of the pinch at ~6°/step.
- In the reference, the left index/middle/ring fingertips sit 13-15 mm *inside* the book and the thumb 3.6 mm
  outside (mocap noise); the joint values are per-joint exp maps (Isaac Gym uses them exactly so; FK in
  `src/tools/hand_fk.py` matches the stored joint positions to 0.37 cm).
- The **pickup is still not executable open-loop** with either solver: a perfect imitator never pivots the book up
  (lost at reference frame 67). A policy has to find its own grasp there (fix-1's did at epochs 1000-2000).
- fix-1's policy cannot lift at all in fix-2 (lift 44% -> 0%, same seed, 32 envs): its grasp relied on the loose
  2 cm contact offset. fix-2 has to learn a physically real grasp.

## 2. Hypothesis

With contacts that can actually hold the demonstrated grip, the policy can keep the book in hand during the
one-handed lift and carry. Expected at epoch 3000 vs fix-1 (stoch, 96 episodes):
- later first failure (fix-1: every episode breaks at frames 65-68) and higher task progress (fix-1: 44.8%);
- more held frames (fix-1: 11.0) and fewer drops (fix-1: 55% at frame ~98);
- lift success at least fix-1's (46.9%).
Task success may well stay near 0% at this budget: the pickup timing (frames 53-72) and the collapse when the
reference starts walking (frame ~85) are not addressed by this change.

Falsified if: lift success, held frames and the first-failure frame are not better than fix-1's (or worse, which
is possible because the loose contacts that fix-1 exploited are gone). Then the solver is not what limits learning
at 3k epochs, and fix-3 should target the pickup / the walk instead.

## 3. Validation before training (agent, 2026-11-17)

- [x] `bash environment_iteration/scripts/check.sh`: syntax, metric regression (1,024 episodes identical), protocol,
      iterations. ALL PASS. fix-1 fingerprint: 366 files shared with its training manifest, 0 differ (no upstream
      file changed).
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-2`: 20-epoch train, Ctrl-C at epoch 10 (exit 130),
      resume to 20, automatic eval + analysis, re-run skips training, video. ALL PASS.
- [x] probes above; the env builds, no NaN, the book rests at reset as in fix-1 (0.2 cm from the reference).
- [x] cost at 2,048 envs (40-epoch run): ~4,650 env-steps/s vs fix-1's 6,871 -> **~14.1 s/epoch, ~11.7 h for 3,000
      epochs** (fix-1: 9.5 s, 7.9 h); peak VRAM 9.4 GB.

## 4. Results (from analysis/e3000/REPORT.md)

Training: 3,000 epochs, seed 0, interrupted once at epoch 290 and resumed (`logs/train_20261117_235027.log`,
`logs/train_20261118_010815.log`); ~12.0 s/epoch after the resume (01:08-10:08 for 2,710 epochs), ~10.3 h wall
time in total. Evaluation + analysis ran automatically at the end (10:08-10:16).

| metric (epoch 3000, 96 episodes per cell) | fix-1 | fix-2 |
|---|---:|---:|
| task success (stoch) | 0.0% (95% CI 0-4%) | **0.0%** (0-4%) |
| task success (perturbed) | 0.0% | 0.0% |
| paper success | 0.0% | 0.0% |
| lift success | 46.9% | **0.0%** |
| lift success (perturbed) | 24.0% | 0.0% |
| max book rise (mean) | 22.9 cm | **0.3 cm** |
| task progress | 44.8% | 44.0% |
| held frames | 11.0 | **0.0** |
| grasp event | 93.8% @ 56.5 | 4.2% @ 60 |
| book left the desk (off_desk) | 76.0% @ 75 | 0.0% |
| drop / floor event | 55% @ 98 / 66% @ 115 | 0% / 0% |
| fell (any time) / in clip | 100% / 95.8% | 100% / 81.2% |
| fall frame (median) | 103 | **129.5** |
| body error (mean, m) | 0.215 | 0.339 |
| dominant first failure (mode @ frame) | lag 51% @66, off_path 25% @68, no_lift 16% @66 | **no_lift 96% @65** |
| dominant first failure (perturbed) | no_lift 46% @63 | no_lift 95% @65 |

Learning curve (det, 8 envs, clean starts; trends only), with the book's median path from the stored curve traces
(`eval/curve/e*.npz`):

| epoch | curve mode | held | book at frame 40 / 55 / 70 / 90 / 120 (median height, m) | book tilt @70 | fix-2 clip reward / step | fix-1 at same epoch |
|---:|---|---:|---|---:|---:|---|
| 500 | no_lift 8 | 0 | 0.74 / 0.74 / 0.74 / 0.74 / 0.74 | flat | 0.185 | no_lift, untouched |
| 1000 | lead 8 | 20.9 | 0.74 / 0.80 / 0.82 / 0.74 / 0.74 | flat | 0.235 | lift 100%, 1.14 m at 80 |
| 1250 | off_path 8 | 19.4 | 0.74 / 0.79 / 0.89 / 0.78 / 0.74 | 135° | 0.221 | |
| 2000 | lead 8 | 44.5 | 0.79 / 0.84 / 0.84 / 0.67 / 0.09 | ~90° (on edge) | 0.150 | lift 100%, 1.12 m at 80 |
| 2250 | lead 8 | 27.9 | 0.77 / 0.81 / 0.80 / 0.80 / 0.80 | ~90° (on edge) | 0.173 | |
| 2500 | lag 7, lead 1 | 25.2 | 0.75 / 0.80 / 0.80 / 0.80 / 0.80 | ~90° (on edge) | 0.199 | lift 0% |
| 2750 | no_lift 8 | 12.9 | 0.74 / 0.74 / 0.76 / 0.79 / 0.74 | flat | 0.285 | |
| 3000 | no_lift 8 | 0 | 0.74 / 0.74 / 0.74 / 0.74 / 0.74 | flat | 0.274 | lift 62% |

Reference book height at frames 40 / 55 / 70 / 90 / 120: 0.74 / 0.77 / 0.99 / 1.02 / 1.01 m. "Clip reward / step"
= mean per-step reward over frames 0-147 of the 8 det curve episodes (after a fall the reward is ~0).
Trainer mean reward (`runs/fix-2_s0/train.log`, RIS starts, not comparable to clip numbers): ~17-18 for
epochs 750-2250, rising to 20-24.6 for 2375-3000, the same epochs in which the curve loses the book.

## 5. Failure analysis (agent)

**Verdict on the hypothesis (§2): falsified at the budget.** Lift success (47% -> 0%), held frames (11 -> 0) and
the first-failure frame (66 -> 65) are all worse than fix-1's. By the criterion written before training, the
solver is not what limits learning at 3k epochs, at least not alone.

**What fix-2 does at epoch 3000.** It ignores the book. In 92/96 episodes the book never moves (median
displacement 0 cm, max rise 0.3 cm on average); the track breaks at frame 65, the first frame the reference book is
> 10 cm above the desk. The wrists come to ~0.17 m of the book during 50-65 (as close as fix-1's), but the contact
force on the book is ~13-23 N vs fix-1's 45-123 N (frames 0-65, mean over alive steps). Video: the hands rest flat
on / beside the book and then follow the reference hand path upward without it (clip 1). The perturbed cell is the
same (95% no_lift); there is nothing to adapt because no grasp is attempted.

**The character still falls, but later and without the book.** 100% fall (median 129.5 vs 103), 81% inside the
clip (vs 96%). The pelvis stays at ~0.82 m until ~110 and goes down at ~120-130, during the sideways carry walk.
Since nothing is in the hands, **the walk collapse is not caused by the book's weight** (fix-1 §5 left this open).
(Refined in §8 probe 1: started mid-clip with the book in hand, the same policy carries and walks without falling;
the collapse follows the failed pickup.)
In the clip-2 frames the character turns toward the walk while one arm stays reaching back over the desk, and
falls; whether that is a stepping failure or the arm pulling it back needs the video.

**The learning curve is unstable, and the intermediate policies did grasp.** From epoch 1000 to 2500 the curve
policy grabs the book with **both hands ~15 frames before the reference does** (grasp at frame 36-38 vs the
reference's ~53; "lead"), with fingertip forces of 200-700 N, pulls it towards the desk edge and **stands it on
its long edge** (tilt ~90°, centre 0.80 m) instead of lifting it (reference 1.20 m at frame 80). At e2000 it is
then dragged off the desk edge at frames 83-87 when the body turns to walk and lands on the floor at 94-100
(clips 5-6, e2000 rollout: 6/8 drops at 83-87). By e2750 this behaviour is gone and by e3000 the book is not
touched. So with the new contacts the policy did find *a* grasp, but not the demonstrated one-hand pinch lift, and
not one that lifts the book off the desk.

**The reward points the same way as in the full-budget study.** The clip reward per step is lower for the policies
that handle the book (0.15-0.24, e1000-e2500) than for the ones that ignore it (0.27-0.29, e2750-e3000), and the
trainer reward rises when the book is abandoned. That is the pattern of the task-avoidance reading
(project root `TASK_AVOIDANCE_FULL_BUDGET.md`): with a grasp that is now hard to find, ignoring the book and
imitating the body costs less reward than a wrong grasp that pulls the body off its reference. fix-1 could lift with
the loose 2 cm contact offset, so the lift paid for itself there.

What these numbers do **not** show:
- Whether e2750/e3000 is a reward optimum or forgetting / an exploration collapse. The clip reward is from 8 det
  episodes; the trainer reward mixes RIS starts. Separating them needs a reward decomposition (body vs object
  terms) per epoch, which the traces do not store.
- Whether more epochs would bring the two-hand grasp back and refine it (fix-1 also lost and regained lift).
  The loop budget is fixed at 3,000, so this does not change the verdict.
- Whether the demonstrated pickup is reachable at all for a policy in fix-2: the perfect imitator could not do it
  open-loop (§1), and the policy found only an edge-stand.
- Seed variance: one training seed.

Possible directions for fix-3 (for the human; evidence above):
1. **Pickup feasibility**: make the frames 50-67 pivot executable for the perfect imitator first (grip.py
   `--replay 30:147 --detail 46:72`), e.g. book placement / desk-edge geometry so the fingers can get under it. This
   targets the step every fix-2 policy fails.
2. **Walk collapse**: now shown to be independent of the book. Check whether the perfect imitator can do the
   sideways walk with the fix-2 solver (`--replay 79:147`); if yes, it is a learning problem (episode length 60,
   RIS sampling of the walk frames), if no, a ground / foot contact problem.
3. Reward-side changes (object reward weight, contact reward) would address the avoidance pattern but are a method
   change, outside this loop unless the team decides so (LOOP.md 6.3).

## 6. Videos

Rendered 2026-11-18 (`scripts/video.sh fix-2`, and `--clips iterations/fix-2/clips_e2000.json --epoch 2000`).
The rollouts reproduce the evaluation: at e3000, 7/8 stoch episodes no_lift at frame 65 (1 classifier "drop" from a
2 cm bump at frame 11), falls at 106-163; perturbed 8/8 no_lift. At e2000, 8/8 "lead" (grasp 35-38), 6/8 drops
at 83-87.

| folder | clip | what to look for |
|---|---|---|
| `video/fix-2/e3000_stoch_s101/` | 1_no_lift_approach (frames 40-80, envs 0-2) | fingers on top of / beside the book with ~20-30 N but no closing; hands rise along the green reference while the red book stays flat |
| | 2_empty_walk_collapse (25-30 frames before the fall, envs 0-2) | empty-handed sideways walk: stepping, foot sliding, one arm left reaching over the desk |
| | 4_full_episode_overview (30-147, env 0) | the whole clip in one piece |
| `video/fix-2/e3000_stoch_perturb_s101/` | 3_perturbed_miss (first failure ±, envs 0-1) | same as clip 1 with a rotated / moved book |
| `video/fix-2/e2000_stoch_s101_v2/` | 5_e2000_early_pivot (30-80, envs 0-2) | two-hand grab at ~36 (before the reference), dragging to the desk edge, standing the book on edge |
| | 6_e2000_drop (drop ±15, envs 0, 2) | book slid off the desk edge at ~84 as the body turns to walk |

`video/fix-2/e2000_stoch_s101/` is a first render of the same e2000 clips whose clip 6 was anchored at the first
failure (it duplicated clip 5); kept, superseded by `_v2`. Each folder: `INDEX.md` (all episodes and clips),
`sheets/*.png` (overview), `clips/*.mp4` (0.5x), `full/env*.mp4` (real time).

## 7. Video notes (human)

_Written by the agent acting as the human reviewer (2026-11-18, at the user's request), from the rendered clips
(sheets + full-size frames of `full/env*.mp4`) and the follow-up probes in §8._

- 1_no_lift_approach (e3000): at frame ~52 the right hand lies flat on the near edge of the book, fingers pointing
  across it; the left fingertips touch the top face from above. Nothing closes around an edge. HUD fingertip force
  17-31 N, the book does not move. At 60-66 the left hand rises along the green reference (inside the green
  wireframe) while the red book stays flat on the desk. It is imitating the hand path, not grasping.
- 2_empty_walk_collapse (e3000): from ~85 the body turns toward the sideways walk while one arm stays stretched
  over the desk towards the book; the character twists and goes down (env 1 at 106). The arm left on the desk looks
  like part of the collapse.
- 3_perturbed_miss: same as clip 1.
- 5_e2000_early_pivot (e2000, `_v2`): at frame ~38 both hands clamp the book (fingertip forces 290-690 N), well
  before the green reference moves. They drag it towards the body and tip it up on its long edge at the desk edge
  (frame ~72: book standing on edge, held by both hands, reference already at chest height).
- 6_e2000_drop: at ~80 the book is squeezed between the hands at the desk edge (0.81 m); at ~86 it slides out
  down the desk edge as the body turns to walk, on the floor at ~94.
- Overall: in e2000 the hands *can* get forces onto the book with fix-2 contacts, but only by squeezing with both
  hands; the demonstrated left-hand pinch lift never appears. In e3000 the grasp is gone.

## 8. Decision -> next iteration

_Decision by the agent acting as the human reviewer (2026-11-18)._ Before choosing, four probes (scratch scripts,
numbers below; all < 30 min):

1. **Is the carry / walk the problem?** No. Policies started mid-clip with the book in the reference pose
   (RIS-style start, 32 stoch envs, 60 frames = the training episode length): fix-2 e3000 keeps the book within
   0.25 m for all 60 frames from starts 55, 60, 65, 70 and 85 (median), with 0-6% falls; fix-1 e3000 the same. From
   start 50 (book still on the desk) the book is never lifted (> 0.25 m at frame ~74). So the policies can carry and
   walk; the full-clip collapse comes after the failed pickup, from a state the policy never sees in training
   (empty hands at the walk). This corrects the §5 reading "the walk fails independently of the book": the walk
   fails without the book, but it is not infeasible. **The pickup from rest is the only missing piece.**
2. **Can the demonstration's pickup be executed open-loop?** Still no. The "perfect imitator" of grip.py lags the
   reference by 2-6 cm during the pickup (joint PD stiffness 500 / damping 500 = 1 s time constant), so its fingers
   never touch the book. With velocity feed-forward + an integral term the lag shrinks (1-4 cm) and grasp forces
   appear (50-110 N at frames 48-56), but the book is still not lifted (lost at 67-69) with the original reference,
   with a contact-consistent reference (fingers 4 or 8 mm into the book) or with book friction 2. Mass 0.3-0.5 kg
   and friction 2 do not make the lift-phase grip (frames 75-100) holdable either (3-8/24 vs 4/24).
3. **What made fix-1 able to grasp?** fix-1's e3000 policy, run zero-shot in fix-2's solver with fix-1's contact
   offset 0.02 m (substeps 4, velocity iterations 8 kept), 3 seeds x 32 stoch envs:

   | environment | lift | held frames | drop | fell in clip |
   |---|---:|---:|---:|---:|
   | fix-1 (its own) | 46% | 10.0 | 55% | 97% |
   | fix-2 | 3% (1 seed) | 1.4 | 0% | 84% |
   | fix-2 solver + contact offset 0.01 | 0% (1 seed) | 0.2 | 0% | 66% |
   | **fix-2 solver + contact offset 0.02** | **41%** (53 / 47 / 22) | **25.5** | **21%** | **59%** |

   The fix-1 e15260 policy does not transfer to either solver (lift 0-3%); it is fitted to fix-1's dynamics.
4. **Does 0.02 cost the hold that fix-2 bought?** Static holds (perfect imitator, frames 75-140, 8 envs each):
   fix-2 18/56, fix-2 solver + 0.02 **24/56**, fix-1 0/56. So the hold improvement came from substeps and
   velocity iterations, not from the small contact offset.

-> **fix-3 = fix-2 with the hand-book contact offset back at 0.02 m** (substeps 4 and velocity iterations 8 kept).
One physical parameter, so the effect stays attributable. Expected: the grasp is discoverable again as in fix-1,
and once lifted the book is held longer than in fix-1. Implemented **per shape** (hand and book shapes 0.02, all
others fix-2's 0.005; new task class), so feet / ground, body and desk contacts stay exactly as in fix-2; it
transfers fix-1's policy at least as well as the global 0.02 (lift 54%, held 37.2, drop 14%) at about the same cost
(20.2 vs 21.4 s/epoch). Follow-up probes showed the hand's offset is the active part, not the book's (fix-3
ITERATION.md §1).

Not chosen (and why): reward weights / contact reward (a method change; the avoidance pattern in §5 may return if
the grasp is hard to find, which fix-3 tests); a contact-consistent reference (no effect on the open-loop pickup,
negative for holding in fix-2's probes); lighter book / more friction (no effect on lift-phase holds); a stricter
object termination (no evidence it helps discovery). If fix-3 lifts but still loses the track at the lift onset
(fix-1's lag at frames 65-68), the next candidates are the pickup timing (lift onset) and the drop during the
in-hand rotation.

