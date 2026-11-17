# fix-1 epoch 3000, cell stoch_perturb, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | lag | 50 | nearest reference frame 15 (35 behind, 9 cm); centre 2 cm up vs reference 0 cm, 23 deg from reference orientation | 50 | 109 | 140 | 110 | 0 | 0 | full/env00.mp4 |
| 1 | no_lift | 65 | object moved 4 cm, rose -0 cm, 46 deg from reference orientation | None | None | None | -1 | 0 | 0 | full/env01.mp4 |
| 2 | knock | 62 | object moved 6 cm, rose -0 cm, 67 deg from reference orientation, never in hand | None | None | None | 121 | 0 | 0 | full/env02.mp4 |
| 3 | knock | 54 | object moved 6 cm, rose 0 cm, 36 deg from reference orientation, never in hand | None | None | None | 113 | 0 | 0 | full/env03.mp4 |
| 4 | lag | 65 | nearest reference frame 10 (55 behind, 9 cm); centre 3 cm up vs reference 11 cm, 22 deg from reference orientation | 60 | 100 | 109 | 101 | 0 | 1 | full/env04.mp4 |
| 5 | no_lift | 63 | object moved 1 cm, rose -0 cm, 41 deg from reference orientation | None | None | None | 98 | 0 | 0 | full/env05.mp4 |
| 6 | no_lift | 30 | object moved 0 cm, rose -0 cm, 19 deg from reference orientation | None | None | None | 102 | 0 | 0 | full/env06.mp4 |
| 7 | off_path | 67 | on time but 7 cm off the path; centre 6 cm up vs reference 16 cm, 41 deg from reference orientation | 51 | None | None | 104 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_perturbed_miss | clips/5_perturbed_miss__env01__f045-075.mp4 | 1 | no_lift | first_bad@65 | 45-75 | object moved 4 cm, rose -0 cm, 46 deg from reference orientation | Perturbed start (book rotated up to 45 deg, moved up to 10 cm): 61% never lift it. Watch: do the hands go to where the reference book is (green) instead of the real book (red)? |
| 5_perturbed_miss | clips/5_perturbed_miss__env02__f042-072.mp4 | 2 | knock | first_bad@62 | 42-72 | object moved 6 cm, rose -0 cm, 67 deg from reference orientation, never in hand | Perturbed start (book rotated up to 45 deg, moved up to 10 cm): 61% never lift it. Watch: do the hands go to where the reference book is (green) instead of the real book (red)? |
| 5_perturbed_miss | clips/5_perturbed_miss__env03__f034-064.mp4 | 3 | knock | first_bad@54 | 34-64 | object moved 6 cm, rose 0 cm, 36 deg from reference orientation, never in hand | Perturbed start (book rotated up to 45 deg, moved up to 10 cm): 61% never lift it. Watch: do the hands go to where the reference book is (green) instead of the real book (red)? |
