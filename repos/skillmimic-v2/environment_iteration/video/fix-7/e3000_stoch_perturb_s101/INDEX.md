# fix-7 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 67 | on time but 8 cm off the path; centre 21 cm up vs reference 16 cm, 24 deg from reference orientation | 53 | 113 | 120 | 132 | 0 | 1 | fall_fwd (1.10 / 0.30 m) | full/env00.mp4 |
| 1 | off_path | 65 | on time but 10 cm off the path; centre 8 cm up vs reference 11 cm, 14 deg from reference orientation | 55 | 121 | 128 | 128 | 0 | 1 | fall_fwd (1.11 / 0.31 m) | full/env01.mp4 |
| 2 | knock | 44 | object moved 8 cm, rose -0 cm, 35 deg from reference orientation, never in hand | None | None | None | 96 | 0 | 0 | fall_fwd (0.48 / 0.04 m) | full/env02.mp4 |
| 3 | no_lift | 64 | object moved 3 cm, rose -0 cm, 78 deg from reference orientation | None | None | None | -1 | 0 | 0 | stand_squat (0.06 / -0.00 m) | full/env03.mp4 |
| 4 | lag | 61 | nearest reference frame 47 (14 behind, 8 cm); centre -0 cm up vs reference 6 cm, 42 deg from reference orientation | 57 | None | None | -1 | 0 | 0 | stand_squat (0.09 / 0.00 m) | full/env04.mp4 |
| 5 | no_lift | 64 | object moved 2 cm, rose -0 cm, 42 deg from reference orientation | None | None | None | 94 | 0 | 0 | fall_inplace (0.00 / -0.09 m) | full/env05.mp4 |
| 6 | off_path | 102 | on time but 10 cm off the path; centre 24 cm up vs reference 24 cm, 8 deg from reference orientation | 55 | None | 141 | 134 | 0 | 1 | fall_fwd (1.16 / 0.35 m) | full/env06.mp4 |
| 7 | off_path | 75 | on time but 9 cm off the path; centre 33 cm up vs reference 40 cm, 68 deg from reference orientation | 58 | 82 | 103 | 132 | 0 | 1 | fall_fwd (0.97 / 0.20 m) | full/env07.mp4 |
| 8 | off_path | 65 | on time but 2 cm off the path; centre 21 cm up vs reference 11 cm, 24 deg from reference orientation | 52 | None | 135 | 135 | 0 | 1 | fall_fwd (1.13 / 0.33 m) | full/env08.mp4 |
| 9 | no_lift | 16 | object moved 0 cm, rose -0 cm, 28 deg from reference orientation | 59 | None | None | -1 | 0 | 0 | stand_squat (0.10 / 0.00 m) | full/env09.mp4 |
| 10 | off_path | 68 | on time but 4 cm off the path; centre 26 cm up vs reference 19 cm, 20 deg from reference orientation | 53 | 104 | 111 | 125 | 0 | 1 | fall_fwd (1.04 / 0.30 m) | full/env10.mp4 |
| 11 | no_lift | 61 | object moved 3 cm, rose -0 cm, 50 deg from reference orientation | None | None | None | -1 | 0 | 0 | stand_squat (0.10 / -0.00 m) | full/env11.mp4 |
| 12 | knock | 59 | object moved 17 cm, rose -0 cm, 61 deg from reference orientation, never in hand | None | None | None | 110 | 0 | 0 | fall_fwd (0.33 / 0.04 m) | full/env12.mp4 |
| 13 | off_path | 65 | on time but 9 cm off the path; centre 14 cm up vs reference 11 cm, 22 deg from reference orientation | 53 | 131 | 152 | 136 | 0 | 1 | fall_fwd (1.18 / 0.38 m) | full/env13.mp4 |
| 14 | no_lift | 60 | object moved 0 cm, rose -0 cm, 41 deg from reference orientation | None | None | None | 138 | 0 | 0 | fall_fwd (0.82 / 0.13 m) | full/env14.mp4 |
| 15 | no_lift | 64 | object moved 2 cm, rose -0 cm, 63 deg from reference orientation | None | None | None | -1 | 0 | 0 | stand_squat (0.08 / -0.00 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_perturbed_pickup | clips/5_perturbed_pickup__env02__f024-059.mp4 | 2 | knock | first_bad@44 | 24-59 | object moved 8 cm, rose -0 cm, 35 deg from reference orientation, never in hand | Perturbed lift 47% (fix-6 54%): no_lift 19%, knock 16% (fix-6 7%). Does the hand push the displaced book instead of grasping it? |
| 5_perturbed_pickup | clips/5_perturbed_pickup__env03__f044-079.mp4 | 3 | no_lift | first_bad@64 | 44-79 | object moved 3 cm, rose -0 cm, 78 deg from reference orientation | Perturbed lift 47% (fix-6 54%): no_lift 19%, knock 16% (fix-6 7%). Does the hand push the displaced book instead of grasping it? |
