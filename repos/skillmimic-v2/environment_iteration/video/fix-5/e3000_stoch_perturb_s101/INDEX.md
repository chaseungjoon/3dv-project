# fix-5 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 68 | on time but 2 cm off the path; centre 8 cm up vs reference 19 cm, 31 deg from reference orientation | 62 | None | None | -1 | 0 | 0 | stand_squat (0.02 / 0.00 m) | full/env00.mp4 |
| 1 | lag | 64 | nearest reference frame 56 (8 behind, 9 cm); centre 4 cm up vs reference 9 cm, 24 deg from reference orientation | 56 | None | None | 99 | 0 | 0 | fall_back (0.00 / -0.11 m) | full/env01.mp4 |
| 2 | knock | 51 | object moved 9 cm, rose 0 cm, 29 deg from reference orientation, never in hand | 60 | None | None | -1 | 0 | 0 | walk_up (0.67 / 0.58 m) | full/env02.mp4 |
| 3 | off_path | 58 | on time but 10 cm off the path; centre 3 cm up vs reference 4 cm, 42 deg from reference orientation | 53 | None | None | 119 | 0 | 0 | fall_fwd (0.55 / 0.04 m) | full/env03.mp4 |
| 4 | lead | 58 | nearest reference frame 66 (8 ahead, 7 cm); centre 13 cm up vs reference 4 cm, 33 deg from reference orientation | 48 | 168 | 179 | -1 | 0 | 1 | walk_up (0.61 / 0.50 m) | full/env04.mp4 |
| 5 | off_path | 63 | on time but 6 cm off the path; centre 16 cm up vs reference 8 cm, 35 deg from reference orientation | 56 | 96 | 121 | 97 | 0 | 1 | fall_back (0.00 / -0.11 m) | full/env05.mp4 |
| 6 | lead | 62 | nearest reference frame 68 (6 ahead, 2 cm); centre 18 cm up vs reference 7 cm, 26 deg from reference orientation | 51 | None | None | -1 | 0 | 1 | walk_up (0.65 / 0.56 m) | full/env06.mp4 |
| 7 | lead | 63 | nearest reference frame 68 (5 ahead, 4 cm); centre 18 cm up vs reference 8 cm, 37 deg from reference orientation | 53 | 127 | 146 | 152 | 0 | 1 | stand_squat (0.00 / -0.12 m) | full/env07.mp4 |
| 8 | lead | 62 | nearest reference frame 67 (5 ahead, 4 cm); centre 14 cm up vs reference 7 cm, 31 deg from reference orientation | 52 | 127 | 153 | 132 | 0 | 1 | fall_fwd (0.84 / 0.16 m) | full/env08.mp4 |
| 9 | no_lift | 21 | object moved 0 cm, rose -0 cm, 28 deg from reference orientation | None | None | None | -1 | 0 | 0 | walk_up (0.61 / 0.51 m) | full/env09.mp4 |
| 10 | lead | 58 | nearest reference frame 67 (9 ahead, 4 cm); centre 16 cm up vs reference 4 cm, 26 deg from reference orientation | 50 | 110 | 126 | 116 | 0 | 1 | fall_fwd (0.92 / 0.22 m) | full/env10.mp4 |
| 11 | off_path | 66 | on time but 10 cm off the path; centre 16 cm up vs reference 13 cm, 22 deg from reference orientation | 56 | None | None | -1 | 0 | 0 | stand_squat (0.02 / 0.02 m) | full/env11.mp4 |
| 12 | off_path | 69 | on time but 9 cm off the path; centre 24 cm up vs reference 22 cm, 25 deg from reference orientation | 54 | None | None | 165 | 0 | 1 | stand_squat (0.02 / -0.02 m) | full/env12.mp4 |
| 13 | off_path | 62 | on time but 8 cm off the path; centre 12 cm up vs reference 7 cm, 34 deg from reference orientation | 51 | 150 | 157 | 151 | 0 | 1 | stand_squat (0.00 / -0.03 m) | full/env13.mp4 |
| 14 | no_lift | 60 | object moved 0 cm, rose -0 cm, 41 deg from reference orientation | None | None | None | 144 | 0 | 0 | fall_fwd (0.77 / 0.13 m) | full/env14.mp4 |
| 15 | off_path | 67 | on time but 6 cm off the path; centre 5 cm up vs reference 16 cm, 47 deg from reference orientation | 60 | None | None | -1 | 0 | 1 | stand_squat (0.15 / 0.01 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 6_perturbed_pickup | clips/6_perturbed_pickup__env01__f044-079.mp4 | 1 | lag | first_bad@64 | 44-79 | nearest reference frame 56 (8 behind, 9 cm); centre 4 cm up vs reference 9 cm, 24 deg from reference orientation | Perturbed lift fell from 72% (fix-4) to 56%. Does the hand go to the displaced red book or to the green reference position? |
| 6_perturbed_pickup | clips/6_perturbed_pickup__env02__f031-066.mp4 | 2 | knock | first_bad@51 | 31-66 | object moved 9 cm, rose 0 cm, 29 deg from reference orientation, never in hand | Perturbed lift fell from 72% (fix-4) to 56%. Does the hand go to the displaced red book or to the green reference position? |
| 6_perturbed_pickup | clips/6_perturbed_pickup__env09__f001-036.mp4 | 9 | no_lift | first_bad@21 | 1-36 | object moved 0 cm, rose -0 cm, 28 deg from reference orientation | Perturbed lift fell from 72% (fix-4) to 56%. Does the hand go to the displaced red book or to the green reference position? |
