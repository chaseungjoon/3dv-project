# fix-6 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 70 | on time but 8 cm off the path; centre 17 cm up vs reference 25 cm, 23 deg from reference orientation | 57 | 93 | 103 | 142 | 0 | 0 | fall_fwd (0.98 / 0.19 m) | full/env00.mp4 |
| 1 | no_lift | 38 | object moved 5 cm, rose 0 cm, 3 deg from reference orientation | 63 | 67 | 88 | 69 | 0 | 0 | fell_pre_walk (0.00 / 0.00 m) | full/env01.mp4 |
| 2 | knock | 45 | object moved 8 cm, rose -0 cm, 45 deg from reference orientation, never in hand | None | None | None | 142 | 0 | 0 | fall_fwd (0.83 / 0.07 m) | full/env02.mp4 |
| 3 | knock | 50 | object moved 7 cm, rose 2 cm, 29 deg from reference orientation, never in hand | 52 | None | None | 152 | 0 | 0 | walk_up (0.90 / 0.30 m) | full/env03.mp4 |
| 4 | lead | 60 | nearest reference frame 66 (6 ahead, 9 cm); centre 11 cm up vs reference 5 cm, 24 deg from reference orientation | 52 | 116 | 131 | 121 | 0 | 0 | fall_fwd (1.11 / 0.32 m) | full/env04.mp4 |
| 5 | off_path | 68 | on time but 5 cm off the path; centre 7 cm up vs reference 19 cm, 72 deg from reference orientation | 61 | None | None | 228 | 0 | 0 | walk_up (0.57 / 0.52 m) | full/env05.mp4 |
| 6 | off_path | 67 | on time but 2 cm off the path; centre 25 cm up vs reference 16 cm, 33 deg from reference orientation | 54 | 191 | 199 | 207 | 0 | 1 | walk_up (0.78 / 0.82 m) | full/env06.mp4 |
| 7 | off_path | 68 | on time but 6 cm off the path; centre 8 cm up vs reference 19 cm, 79 deg from reference orientation | 58 | None | None | 143 | 0 | 0 | fall_fwd (1.09 / 0.34 m) | full/env07.mp4 |
| 8 | off_path | 63 | on time but 4 cm off the path; centre 15 cm up vs reference 8 cm, 34 deg from reference orientation | 53 | 188 | 202 | 204 | 0 | 1 | walk_up (0.65 / 0.58 m) | full/env08.mp4 |
| 9 | no_lift | 30 | object moved 0 cm, rose -0 cm, 29 deg from reference orientation | None | None | None | 144 | 0 | 0 | fall_fwd (1.01 / 0.21 m) | full/env09.mp4 |
| 10 | lead | 62 | nearest reference frame 68 (6 ahead, 3 cm); centre 19 cm up vs reference 7 cm, 37 deg from reference orientation | 53 | 186 | 193 | 193 | 0 | 1 | walk_up (0.81 / 0.76 m) | full/env10.mp4 |
| 11 | knock | 49 | object moved 7 cm, rose -0 cm, 27 deg from reference orientation, never in hand | None | None | None | 134 | 0 | 0 | fall_fwd (0.89 / 0.13 m) | full/env11.mp4 |
| 12 | no_lift | 62 | object moved 0 cm, rose -0 cm, 65 deg from reference orientation | None | None | None | 140 | 0 | 0 | fall_fwd (0.89 / 0.12 m) | full/env12.mp4 |
| 13 | off_path | 64 | on time but 9 cm off the path; centre 12 cm up vs reference 9 cm, 25 deg from reference orientation | 56 | 132 | 149 | 138 | 0 | 1 | fall_fwd (1.17 / 0.73 m) | full/env13.mp4 |
| 14 | no_lift | 60 | object moved 0 cm, rose -0 cm, 41 deg from reference orientation | None | None | None | -1 | 0 | 0 | walk_up (0.55 / 0.47 m) | full/env14.mp4 |
| 15 | no_lift | 63 | object moved 1 cm, rose -0 cm, 61 deg from reference orientation | None | None | None | 142 | 0 | 0 | fall_fwd (0.99 / 0.17 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 7_perturbed_pickup | clips/7_perturbed_pickup__env01__f018-053.mp4 | 1 | no_lift | first_bad@38 | 18-53 | object moved 5 cm, rose 0 cm, 3 deg from reference orientation | Perturbed lift 54% (fix-5 56%, fix-4 72%); 31% no_lift, 7% knock. Does the hand go to the displaced book or to the green reference position? |
| 7_perturbed_pickup | clips/7_perturbed_pickup__env02__f025-060.mp4 | 2 | knock | first_bad@45 | 25-60 | object moved 8 cm, rose -0 cm, 45 deg from reference orientation, never in hand | Perturbed lift 54% (fix-5 56%, fix-4 72%); 31% no_lift, 7% knock. Does the hand go to the displaced book or to the green reference position? |
| 7_perturbed_pickup | clips/7_perturbed_pickup__env03__f030-065.mp4 | 3 | knock | first_bad@50 | 30-65 | object moved 7 cm, rose 2 cm, 29 deg from reference orientation, never in hand | Perturbed lift 54% (fix-5 56%, fix-4 72%); 31% no_lift, 7% knock. Does the hand go to the displaced book or to the green reference position? |
