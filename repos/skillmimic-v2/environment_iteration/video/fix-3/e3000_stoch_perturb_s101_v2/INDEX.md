# fix-3 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | no_lift | 64 | object moved 3 cm, rose -0 cm, 47 deg from reference orientation | None | None | None | 153 | 0 | 0 | full/env00.mp4 |
| 1 | lag | 66 | nearest reference frame 53 (13 behind, 2 cm); centre 2 cm up vs reference 13 cm, 27 deg from reference orientation | 66 | None | None | 168 | 0 | 0 | full/env01.mp4 |
| 2 | off_path | 63 | on time but 9 cm off the path; centre 6 cm up vs reference 8 cm, 80 deg from reference orientation | 48 | None | None | 160 | 0 | 0 | full/env02.mp4 |
| 3 | no_lift | 65 | object moved 2 cm, rose -0 cm, 71 deg from reference orientation | None | None | None | 151 | 0 | 0 | full/env03.mp4 |
| 4 | lag | 65 | nearest reference frame 54 (11 behind, 4 cm); centre 2 cm up vs reference 11 cm, 24 deg from reference orientation | 64 | None | None | 161 | 0 | 0 | full/env04.mp4 |
| 5 | lag | 64 | nearest reference frame 55 (9 behind, 8 cm); centre 2 cm up vs reference 9 cm, 25 deg from reference orientation | 62 | None | None | 173 | 0 | 0 | full/env05.mp4 |
| 6 | no_lift | 65 | object moved 4 cm, rose 1 cm, 51 deg from reference orientation | None | None | None | 151 | 0 | 0 | full/env06.mp4 |
| 7 | lag | 65 | nearest reference frame 55 (10 behind, 4 cm); centre 3 cm up vs reference 11 cm, 26 deg from reference orientation | 61 | None | None | 144 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 3_perturbed_pickup | clips/3_perturbed_pickup__env00__f044-079.mp4 | 0 | no_lift | first_bad@64 | 44-79 | object moved 3 cm, rose -0 cm, 47 deg from reference orientation | With the book initially displaced or rotated, look at whether the hands reach the actual red book or follow the green reference, and whether contact pushes or lifts it. |
| 3_perturbed_pickup | clips/3_perturbed_pickup__env01__f046-081.mp4 | 1 | lag | first_bad@66 | 46-81 | nearest reference frame 53 (13 behind, 2 cm); centre 2 cm up vs reference 13 cm, 27 deg from reference orientation | With the book initially displaced or rotated, look at whether the hands reach the actual red book or follow the green reference, and whether contact pushes or lifts it. |
| 3_perturbed_pickup | clips/3_perturbed_pickup__env03__f045-080.mp4 | 3 | no_lift | first_bad@65 | 45-80 | object moved 2 cm, rose -0 cm, 71 deg from reference orientation | With the book initially displaced or rotated, look at whether the hands reach the actual red book or follow the green reference, and whether contact pushes or lifts it. |
