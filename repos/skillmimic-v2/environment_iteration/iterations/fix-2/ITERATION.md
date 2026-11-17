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

| metric | fix-1 | fix-2 |
|---|---:|---:|
| task success (stoch, 96) | 0.0% | |
| task success (perturbed) | 0.0% | |
| paper success | 0.0% | |
| lift success | 46.9% | |
| task progress | 44.8% | |
| held frames | 11.0 | |
| fell in clip | 95.8% | |
| dominant failure (mode @ frame) | lag 51% @66 | |

## 5. Failure analysis (agent)

## 6. Videos

## 7. Video notes (human)

## 8. Decision -> next iteration
