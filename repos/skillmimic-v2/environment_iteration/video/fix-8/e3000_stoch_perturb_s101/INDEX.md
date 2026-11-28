# fix-8 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 67 | on time but 9 cm off the path; centre 19 cm up vs reference 16 cm, 11 deg from reference orientation | 56 | 125 | 133 | 126 | 0 | 1 | fall_fwd (0.84 / 0.22 m) | full/env00.mp4 |
| 1 | off_path | 57 | on time but 10 cm off the path; centre 3 cm up vs reference 4 cm, 20 deg from reference orientation | 57 | None | None | 117 | 0 | 0 | fall_fwd (0.72 / 0.14 m) | full/env01.mp4 |
| 2 | no_lift | 34 | object moved 4 cm, rose -0 cm, 52 deg from reference orientation | None | None | None | 129 | 0 | 0 | fall_fwd (0.72 / 0.14 m) | full/env02.mp4 |
| 3 | off_path | 57 | on time but 10 cm off the path; centre 4 cm up vs reference 4 cm, 38 deg from reference orientation | 53 | None | None | 123 | 0 | 0 | fall_fwd (0.66 / 0.10 m) | full/env03.mp4 |
| 4 | off_path | 64 | on time but 10 cm off the path; centre 6 cm up vs reference 9 cm, 43 deg from reference orientation | 57 | 72 | None | 124 | 0 | 0 | fall_fwd (0.85 / 0.17 m) | full/env04.mp4 |
| 5 | no_lift | 59 | object moved 4 cm, rose -0 cm, 38 deg from reference orientation | None | None | None | 124 | 0 | 0 | fall_fwd (0.70 / 0.12 m) | full/env05.mp4 |
| 6 | off_path | 69 | on time but 9 cm off the path; centre 21 cm up vs reference 22 cm, 14 deg from reference orientation | 55 | 128 | 164 | 129 | 0 | 1 | fall_fwd (0.84 / 0.23 m) | full/env06.mp4 |
| 7 | no_lift | 64 | object moved 3 cm, rose -0 cm, 48 deg from reference orientation | None | None | None | 123 | 0 | 0 | fall_fwd (0.79 / 0.15 m) | full/env07.mp4 |
| 8 | off_path | 66 | on time but 9 cm off the path; centre 17 cm up vs reference 13 cm, 23 deg from reference orientation | 54 | 126 | 134 | 127 | 0 | 1 | fall_fwd (0.90 / 0.22 m) | full/env08.mp4 |
| 9 | no_lift | 9 | object moved 0 cm, rose -0 cm, 29 deg from reference orientation | None | None | None | 134 | 0 | 0 | fall_fwd (0.68 / 0.14 m) | full/env09.mp4 |
| 10 | off_path | 65 | on time but 8 cm off the path; centre 17 cm up vs reference 11 cm, 9 deg from reference orientation | 55 | 72 | 86 | 141 | 0 | 0 | fall_fwd (0.82 / 0.15 m) | full/env10.mp4 |
| 11 | knock | 34 | object moved 5 cm, rose -0 cm, 22 deg from reference orientation, never in hand | None | None | None | 132 | 0 | 0 | fall_fwd (0.73 / 0.13 m) | full/env11.mp4 |
| 12 | no_lift | 34 | object moved 2 cm, rose -0 cm, 39 deg from reference orientation | None | None | None | 147 | 0 | 0 | fall_fwd (0.72 / 0.16 m) | full/env12.mp4 |
| 13 | off_path | 60 | on time but 9 cm off the path; centre 6 cm up vs reference 5 cm, 26 deg from reference orientation | 53 | 123 | 145 | 124 | 0 | 1 | fall_fwd (0.81 / 0.20 m) | full/env13.mp4 |
| 14 | no_lift | 61 | object moved 1 cm, rose -0 cm, 41 deg from reference orientation | None | None | None | 123 | 0 | 0 | fall_fwd (0.72 / 0.14 m) | full/env14.mp4 |
| 15 | no_lift | 64 | object moved 3 cm, rose -0 cm, 63 deg from reference orientation | None | None | None | 159 | 0 | 0 | walk_up (0.36 / 0.03 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_perturbed_pickup | clips/5_perturbed_pickup__env02__f014-049.mp4 | 2 | no_lift | first_bad@34 | 14-49 | object moved 4 cm, rose -0 cm, 52 deg from reference orientation | Perturbed lift 52% (fix-6 54%): no_lift 24%, knock 9% (fix-6 31% / 7%), so about the same as fix-6. Does the hand go to the displaced book or to the green reference? |
| 5_perturbed_pickup | clips/5_perturbed_pickup__env05__f039-074.mp4 | 5 | no_lift | first_bad@59 | 39-74 | object moved 4 cm, rose -0 cm, 38 deg from reference orientation | Perturbed lift 52% (fix-6 54%): no_lift 24%, knock 9% (fix-6 31% / 7%), so about the same as fix-6. Does the hand go to the displaced book or to the green reference? |
