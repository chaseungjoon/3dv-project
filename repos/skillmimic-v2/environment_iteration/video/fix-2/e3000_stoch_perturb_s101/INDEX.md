# fix-2 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | no_lift | 61 | object moved 1 cm, rose -0 cm, 43 deg from reference orientation | 70 | None | None | -1 | 0 | 0 | full/env00.mp4 |
| 1 | no_lift | 65 | object moved 0 cm, rose -0 cm, 50 deg from reference orientation | None | None | None | 103 | 0 | 0 | full/env01.mp4 |
| 2 | no_lift | 64 | object moved 0 cm, rose -0 cm, 81 deg from reference orientation | None | None | None | 100 | 0 | 0 | full/env02.mp4 |
| 3 | no_lift | 65 | object moved 1 cm, rose 0 cm, 71 deg from reference orientation | None | None | None | -1 | 0 | 0 | full/env03.mp4 |
| 4 | no_lift | 65 | object moved 0 cm, rose -0 cm, 45 deg from reference orientation | None | None | None | 138 | 0 | 0 | full/env04.mp4 |
| 5 | no_lift | 62 | object moved 0 cm, rose -0 cm, 39 deg from reference orientation | None | None | None | 146 | 0 | 0 | full/env05.mp4 |
| 6 | no_lift | 10 | object moved 1 cm, rose -0 cm, 19 deg from reference orientation | None | None | None | 145 | 0 | 0 | full/env06.mp4 |
| 7 | no_lift | 65 | object moved 0 cm, rose -0 cm, 43 deg from reference orientation | None | None | None | 175 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 3_perturbed_miss | clips/3_perturbed_miss__env00__f041-071.mp4 | 0 | no_lift | first_bad@61 | 41-71 | object moved 1 cm, rose -0 cm, 43 deg from reference orientation | Perturbed start (book yaw +-45 deg, xy <= 10 cm): 95% no_lift, 2% knock. Same question as clip 1: do the hands go to the red book or to where the green reference book is? |
| 3_perturbed_miss | clips/3_perturbed_miss__env01__f045-075.mp4 | 1 | no_lift | first_bad@65 | 45-75 | object moved 0 cm, rose -0 cm, 50 deg from reference orientation | Perturbed start (book yaw +-45 deg, xy <= 10 cm): 95% no_lift, 2% knock. Same question as clip 1: do the hands go to the red book or to where the green reference book is? |
