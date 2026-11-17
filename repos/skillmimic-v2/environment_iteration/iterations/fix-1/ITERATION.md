# fix-1

Parent: none (starting point) | Created: 2026-11-17 | Summary: FIX from the full-budget study, imported
Run: imported, not trained by this loop: `experiments/runs_paper_fix/place_book_fix_n2048_s0` (15,260 epochs,
seed 0, `run_paper_fix_full.sh`, git 44135d8), linked as `runs/fix-1_s0`. The loop evaluates its **epoch-3000
checkpoint** (= the loop budget; valid because the learning rate is constant and nothing in training depends on
`max_epochs`, see iteration.json). Epoch 15,260 is analyzed as a reference for what more training buys.
Code check: `src/fingerprint.py imported fix-1` -> the 366 files shared with that run's provenance manifest are
unchanged.

## 1. Changes vs upstream (PAPER = the authors' public configuration)

All from `experiments/configs/env/place_book_fix.yaml` (= `env.yaml` here) and task
`SkillMimicParahomeLocalHistRISBuffernodePhase` (project root `SKILLMIMIC_NOTES.md` 4):

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | desk = one fitted box (top 0.716 m) instead of the 10k-voxel hull | `staticBoxes` | book started 3.8 cm above its reference | earlier probes (SKILLMIMIC_NOTES 4) |
| 2 | book collision = its oriented bounding box, 1.0 kg (upstream 7.1 kg) | `dynamicUrdf: book_fix.urdf`, `ballDensity: 456` | coarse hull, unrealistic mass | make_book_collision.py |
| 3 | history-encoder input in the layout it was trained on | `histLayoutFix` | 61 sim vs 71 ref bodies layout bug | SKILLMIMIC_NOTES 4 |
| 4 | phase observation (frame / clip length) | task class | no time information once the history fills | – |
| 5 | training-only early termination when the book is > 0.25 m off | `objTerminationDist` | 60 frames of body reward after losing the book | – |

Bundled changes: their individual effects are unknown.

## 2. Hypothesis

(From the full-budget study.) The public configuration cannot learn to move the book at all (PAPER never moved
it, 1 B samples); FIX should at least lift it.

## 3. Validation

Imported run; no new training. Loop evaluation and analysis run on 2026-11-17 by the agent:
`scripts/eval.sh fix-1` and `scripts/eval.sh fix-1 --epoch 15260` (each ~6 min).

## 4. Results (`analysis/e3000/REPORT.md`, reference `analysis/e15260/REPORT.md`)

| metric (96 episodes per cell) | fix-1 @ 3000 (budget) | fix-1 @ 15260 (paper budget) |
|---|---:|---:|
| task success (stoch) | **0.0%** (95% CI 0-4%) | 0.0% (0-4%) |
| task success (perturbed) | 0.0% | 0.0% |
| paper success | 0.0% | 0.0% |
| lift success | 46.9% | 99.0% |
| lift success (perturbed) | 24.0% | 53.1% |
| task progress | 44.8% | 48.6% |
| held frames (evaluate.py definition) | 11.0 | 24.3 |
| fell in clip | 95.8% | 12.5% |
| dominant first failure | lag 51% / off_path 25% / no_lift 16% @ frame 66-68 | off_path 99% @ frame 72 |

Learning curve (det, 8 envs, so only a trend): lift 0% at 500, 100% at 1000-2000 (paper success 62% at 1000),
0% at 2500, 62% at 3000. Lift is learned and partly forgotten again; the same instability as in the full-budget
study (epochs 4,500-8,000).

## 5. Failure analysis

**The reference** (needed to read everything below): the book rests until frame ~53. Then the person pivots it
up on one edge and rotates it (41° at frame 60, ~90° at 72, ~180° at 90) while lifting it 46 cm by frame 80, and
from frame ~85 walks ~1.1 m sideways with it at up to ~0.8 m/s.

**fix-1 at the budget (epoch 3000)** has two walls, both in every episode:

1. **Lift onset, frames 53-72.** Every one of the 96 episodes loses the track at frames 65-68, the moment the
   reference rises fast (+19 cm between frames 60 and 70). 94% grasp the book (median frame 57, a pivot grip),
   but the book rises too slowly: at frame 70 it is at 0.80 m vs the reference's 0.99 m. Its orientation is
   > 45° off the reference by frame 72 in 100% of episodes. 21% never lift it at all (no_lift / knock).
   With the 10 cm per-frame criterion, being ~5 frames late during this fast rise is already a failure.
2. **Carry walk, frames ~85-105.** The pelvis starts sinking at ~85, when the reference starts walking sideways,
   and the character is down at median frame 103 (100% of episodes fall, 96% inside the clip). The 55% "drops"
   (median 98) are part of the same collapse (median 2 frames before the fall).

Perturbed start (book rotated ≤ 45°, moved ≤ 10 cm): 46% no_lift + 15% knock. The policy rarely adapts its
grasp to a displaced book (lift success 24% vs 47%).

**More training (epoch 15260)** fixes the timing of the lift (99% lift, track kept to frame ~72) and the balance
(12.5% fall in clip), but not the task: the book's orientation leaves the reference at ~77, it is dropped at ~87
(96%) and ends on the floor (~98), while the body covers only ~half of the reference's sideways walk (head
moves 0.57 m vs 1.1 m by frame 147). So even with 5x the samples, holding the book through the in-hand
rotation and the walk is not learned.

What these numbers do not tell: whether the grip slips (contact geometry, friction, box-shaped fingers) or the
policy lets go; whether the collapse is a balance / stepping problem or a consequence of the book's weight; why
the perturbed grasps miss. That is what the videos are for (§6).

## 6. Videos

`clips.json` (epoch 3000) and `clips_e15260.json` (reference), rendered 2026-11-17 (17 clips):
`video/fix-1/e3000_stoch_s101/`, `video/fix-1/e3000_stoch_perturb_s101/`, `video/fix-1/e15260_stoch_s101/`
(INDEX.md in each lists every clip; `sheets/*.png` = 5-frame overview per clip). The video rollouts reproduce the
evaluation: at 3000 all 8 episodes lose the track at frames 65-69 and fall at 95-111; at 15260 all 8 drop the
book at frames 84-90 while staying upright.

| clip | what to look for |
|---|---|
| 1_grasp_lift (frames 45-80) | grip location, pivot, slip in the fingers, arm speed vs the green wireframe |
| 2_never_lifted (first failure ±) | finger placement, pushing / tipping the book |
| 3_collapse (25 frames before the fall) | stepping, foot sliding, leaning on the desk, book pulling the body |
| 4_drop (around the drop) | hands open first, or the collapse takes the book |
| 5_perturbed_miss | hands going to the reference book (green) instead of the real one (red)? |
| 6_e15260_lift_rotate (55-90) | can the grip follow the reference rotation? |
| 7_e15260_carry_drop | arms pulled apart by following the reference while not walking? book twisting out? |

## 7. Video notes (human)

_Written by the agent acting as the human reviewer (2026-11-17, at the user's request), from the rendered clips
(sheets + full-size frames) and follow-up measurements._

- 1_grasp_lift (e3000): at frame ~60 the right hand pinches the near edge of the book; the green reference book is
  already a few cm higher. At ~70 the policy has stood the book **on its edge on the desk** (held by one edge)
  instead of lifting it; the reference book is at chest height.
- 3_collapse / 4_drop (e3000): the book stays on the desk; the character sinks and falls as the reference starts
  the sideways walk. The "drop" is the same moment as the collapse.
- 7_e15260_carry_drop: lift on time, but the book slides out of the hands at ~85-90 while the reference book is
  upright at the character's right side.
- Measured afterwards (left fingertips vs the reference book): **the demonstration switches to a one-hand carry.**
  The right hand leaves the book at frames 62-65; from then on the left hand alone pinches it (thumb on one face,
  fingers on the other, ~1.5 cm from a long edge, centre of mass ~7 cm away). The e15260 policy follows this
  (right hand away by 70-80) and loses the book at 85-90.

## 8. Decision -> next iteration

_Decision by the agent acting as the human reviewer._ Question first: can this environment physically execute
the demonstration at all? `src/probes/grip.py` (new) tests a **perfect imitator**: root pinned to the reference,
every joint PD-driven to the reference pose.

- Static carry grip (book in the left hand at the reference pose, frames 75-140): **held 0/56** in fix-1.
- Whole-clip replay: the book is never lifted (lost at reference frame 67, exactly where every e3000 policy fails).

Ruled out by the probes (details in fix-2 ITERATION.md §1): friction up to 8, torsional / rolling friction, a
0.3 kg book, finger closing ±0.4 rad, thicker finger capsules (looked like a fix only because thick capsules jam
into each other; 0/32 once intra-hand collisions are removed), chamfered book edges, a contact-consistent
reference (fingers moved 2-8 mm into the book), the contact-graph reward (the paper disables it for ParaHome).
What worked: **contact-simulation fidelity**. With 4 substeps, 8 PhysX velocity iterations and a 5 mm contact
offset the same perfect imitator holds the carry grip 15/56 (frames 110-140 mostly); every 2-substep setting stays
at 0.

-> **fix-2 = contact-simulation fidelity only** (substeps 2->4, velocity iterations 0->8, contact offset
0.02->0.005 m). Nothing else changes, so its effect is attributable. Not addressed (fix-3 candidates): the pickup
(a perfect imitator does not pivot the book up even with the new solver; a policy must find its own grasp there),
and the collapse at the start of the sideways walk.
