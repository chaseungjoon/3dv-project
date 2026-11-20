# fix-4 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | off_path | 62 | on time but 10 cm off the path; centre 7 cm up vs reference 7 cm, 32 deg from reference orientation | 49 | 132 | 139 | -1 | 0 | 1 | full/env00.mp4 |
| 1 | off_path | 63 | on time but 11 cm off the path; centre 5 cm up vs reference 8 cm, 34 deg from reference orientation | 52 | 129 | 136 | -1 | 0 | 1 | full/env01.mp4 |
| 2 | knock | 45 | object moved 7 cm, rose -0 cm, 39 deg from reference orientation, never in hand | None | None | None | -1 | 0 | 0 | full/env02.mp4 |
| 3 | off_path | 56 | on time but 10 cm off the path; centre 3 cm up vs reference 3 cm, 29 deg from reference orientation | 51 | 95 | 102 | -1 | 0 | 1 | full/env03.mp4 |
| 4 | off_path | 59 | on time but 9 cm off the path; centre 6 cm up vs reference 5 cm, 19 deg from reference orientation | 48 | 133 | 141 | -1 | 0 | 1 | full/env04.mp4 |
| 5 | lead | 50 | nearest reference frame 55 (5 ahead, 9 cm); centre 3 cm up vs reference 0 cm, 33 deg from reference orientation | 49 | 120 | 129 | -1 | 0 | 1 | full/env05.mp4 |
| 6 | knock | 47 | object moved 13 cm, rose -0 cm, 22 deg from reference orientation, never in hand | None | None | None | -1 | 0 | 0 | full/env06.mp4 |
| 7 | lead | 56 | nearest reference frame 63 (7 ahead, 9 cm); centre 6 cm up vs reference 3 cm, 23 deg from reference orientation | 50 | 134 | 142 | -1 | 0 | 1 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 4_perturbed_pickup | clips/4_perturbed_pickup__env02__f025-060.mp4 | 2 | knock | first_bad@45 | 25-60 | object moved 7 cm, rose -0 cm, 39 deg from reference orientation, never in hand | With the book displaced/rotated, 28% never lift (no_lift / knock at 30-45, lag at ~54). Check whether the hand goes to the red book or to the green reference. |
| 4_perturbed_pickup | clips/4_perturbed_pickup__env06__f027-062.mp4 | 6 | knock | first_bad@47 | 27-62 | object moved 13 cm, rose -0 cm, 22 deg from reference orientation, never in hand | With the book displaced/rotated, 28% never lift (no_lift / knock at 30-45, lag at ~54). Check whether the hand goes to the red book or to the green reference. |
