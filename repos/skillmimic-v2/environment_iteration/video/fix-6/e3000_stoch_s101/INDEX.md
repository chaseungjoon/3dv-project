# fix-6 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 66 | on time but 5 cm off the path; centre 21 cm up vs reference 13 cm, 35 deg from reference orientation | 56 | 140 | 148 | 144 | 0 | 1 | fall_fwd (1.31 / 0.57 m) | full/env00.mp4 |
| 1 | off_path | 77 | on time but 5 cm off the path; centre 46 cm up vs reference 44 cm, 38 deg from reference orientation | 55 | 197 | 206 | 210 | 0 | 1 | walk_up (0.82 / 0.78 m) | full/env01.mp4 |
| 2 | off_path | 69 | on time but 3 cm off the path; centre 29 cm up vs reference 22 cm, 32 deg from reference orientation | 57 | 180 | 187 | 229 | 0 | 1 | walk_up (0.77 / 0.73 m) | full/env02.mp4 |
| 3 | off_path | 67 | on time but 3 cm off the path; centre 25 cm up vs reference 16 cm, 42 deg from reference orientation | 53 | 192 | 200 | 208 | 0 | 1 | walk_up (0.70 / 0.65 m) | full/env03.mp4 |
| 4 | off_path | 64 | on time but 3 cm off the path; centre 19 cm up vs reference 9 cm, 40 deg from reference orientation | 54 | 190 | 197 | 207 | 0 | 1 | walk_up (0.80 / 0.72 m) | full/env04.mp4 |
| 5 | off_path | 67 | on time but 4 cm off the path; centre 24 cm up vs reference 16 cm, 34 deg from reference orientation | 56 | 187 | 194 | 213 | 0 | 1 | walk_up (0.81 / 0.78 m) | full/env05.mp4 |
| 6 | off_path | 96 | on time but 3 cm off the path; centre 22 cm up vs reference 23 cm, 41 deg from reference orientation | 56 | 204 | 212 | 220 | 0 | 1 | walk_up (0.76 / 0.72 m) | full/env06.mp4 |
| 7 | off_path | 64 | on time but 1 cm off the path; centre 19 cm up vs reference 9 cm, 31 deg from reference orientation | 54 | 192 | 200 | 210 | 0 | 1 | walk_up (0.71 / 0.62 m) | full/env07.mp4 |
| 8 | off_path | 64 | on time but 4 cm off the path; centre 18 cm up vs reference 9 cm, 31 deg from reference orientation | 55 | 189 | 196 | 208 | 0 | 1 | walk_up (0.69 / 0.63 m) | full/env08.mp4 |
| 9 | off_path | 78 | on time but 2 cm off the path; centre 43 cm up vs reference 45 cm, 30 deg from reference orientation | 57 | 131 | 138 | 135 | 0 | 1 | fall_fwd (1.25 / 0.48 m) | full/env09.mp4 |
| 10 | lead | 62 | nearest reference frame 68 (6 ahead, 4 cm); centre 17 cm up vs reference 7 cm, 37 deg from reference orientation | 54 | 188 | 196 | 198 | 0 | 1 | walk_up (0.83 / 0.78 m) | full/env10.mp4 |
| 11 | off_path | 68 | on time but 4 cm off the path; centre 25 cm up vs reference 19 cm, 35 deg from reference orientation | 57 | 195 | 202 | 213 | 0 | 1 | walk_up (0.69 / 0.61 m) | full/env11.mp4 |
| 12 | off_path | 64 | on time but 6 cm off the path; centre 16 cm up vs reference 9 cm, 28 deg from reference orientation | 56 | 196 | 203 | 207 | 0 | 1 | walk_up (0.76 / 0.69 m) | full/env12.mp4 |
| 13 | off_path | 66 | on time but 5 cm off the path; centre 20 cm up vs reference 13 cm, 30 deg from reference orientation | 56 | 189 | 196 | 220 | 0 | 1 | walk_up (0.78 / 0.75 m) | full/env13.mp4 |
| 14 | off_path | 65 | on time but 3 cm off the path; centre 21 cm up vs reference 11 cm, 35 deg from reference orientation | 53 | 190 | 198 | 209 | 0 | 1 | walk_up (0.79 / 0.73 m) | full/env14.mp4 |
| 15 | off_path | 96 | on time but 6 cm off the path; centre 18 cm up vs reference 23 cm, 42 deg from reference orientation | 57 | 193 | 215 | 208 | 0 | 1 | walk_up (0.77 / 0.74 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_walk_up | clips/1_walk_up__env01__f078-147.mp4 | 1 | off_path | frames 78-147 | 78-147 | on time but 5 cm off the path; centre 46 cm up vs reference 44 cm, 38 deg from reference orientation | New in fix-6: 90% walk sideways with the book and stay up (median root progress 0.77 m; fix-5 18%, fix-4 2%). The book is 11-15 cm from the green reference during the carry (33-42% within 10 cm at t 90-110). Real steps or sliding? Is the error mostly height, lag, or orientation (rot_off in 93% at ~94)? |
| 1_walk_up | clips/1_walk_up__env02__f078-147.mp4 | 2 | off_path | frames 78-147 | 78-147 | on time but 3 cm off the path; centre 29 cm up vs reference 22 cm, 32 deg from reference orientation | New in fix-6: 90% walk sideways with the book and stay up (median root progress 0.77 m; fix-5 18%, fix-4 2%). The book is 11-15 cm from the green reference during the carry (33-42% within 10 cm at t 90-110). Real steps or sliding? Is the error mostly height, lag, or orientation (rot_off in 93% at ~94)? |
| 2_lift_off_path | clips/2_lift_off_path__env00__f051-081.mp4 | 0 | off_path | first_bad@66 | 51-81 | on time but 5 cm off the path; centre 21 cm up vs reference 13 cm, 35 deg from reference orientation | 86% break the strict 10 cm track here (median t 68): book centre ~27 cm up vs the reference's 19 cm, ~40 deg off its orientation. At t 70 the book is at 1.08 m vs 0.99 m. Does it lift too high / too early, or skip the pivot-and-rotate? |
| 2_lift_off_path | clips/2_lift_off_path__env01__f062-092.mp4 | 1 | off_path | first_bad@77 | 62-92 | on time but 5 cm off the path; centre 46 cm up vs reference 44 cm, 38 deg from reference orientation | 86% break the strict 10 cm track here (median t 68): book centre ~27 cm up vs the reference's 19 cm, ~40 deg off its orientation. At t 70 the book is at 1.08 m vs 0.99 m. Does it lift too high / too early, or skip the pivot-and-rotate? |
| 3_lift_lead | clips/3_lift_lead__env10__f047-077.mp4 | 10 | lead | first_bad@62 | 47-77 | nearest reference frame 68 (6 ahead, 4 cm); centre 17 cm up vs reference 7 cm, 37 deg from reference orientation | 12%: breaks at t 63, ~5 frames ahead of the reference along its path. Compare with clip 2: same overshoot, just earlier? |
| 4_fall_fwd | clips/4_fall_fwd__env00__f114-149.mp4 | 0 | off_path | fall@144 | 114-149 | on time but 5 cm off the path; centre 21 cm up vs reference 13 cm, 35 deg from reference orientation | 10% fall in the clip after walking forward (fix-5's fall_back 42% is gone). Lean / topple, or a trip on a step? |
| 4_fall_fwd | clips/4_fall_fwd__env09__f105-140.mp4 | 9 | off_path | fall@135 | 105-140 | on time but 2 cm off the path; centre 43 cm up vs reference 45 cm, 30 deg from reference orientation | 10% fall in the clip after walking forward (fix-5's fall_back 42% is gone). Lean / topple, or a trip on a step? |
| 5_post_clip_drop | clips/5_post_clip_drop__env00__f110-155.mp4 | 0 | off_path | drop@140 | 110-155 | on time but 5 cm off the path; centre 21 cm up vs reference 13 cm, 35 deg from reference orientation | After the clip ends (t 147) the reference is over; 92% drop the book at ~190 and all fall at ~210. Outside the metric; check that it is just the end of the demonstration, not a problem that will show up earlier with a better lift. |
| 6_success | (none) | - | - | frame | - | no episode of this rollout matched; try more envs (--num_envs) or another seed | 1 of 96 evaluation episodes passed the strict criterion; may not appear in this fresh rollout. What does a passing lift look like? |
