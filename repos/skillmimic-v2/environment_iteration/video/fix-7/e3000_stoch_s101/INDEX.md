# fix-7 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | lag | 69 | nearest reference frame 64 (5 behind, 5 cm); centre 9 cm up vs reference 22 cm, 34 deg from reference orientation | 55 | 68 | None | 156 | 0 | 0 | walk_up (0.38 / 0.02 m) | full/env00.mp4 |
| 1 | off_path | 99 | on time but 10 cm off the path; centre 23 cm up vs reference 23 cm, 11 deg from reference orientation | 53 | 124 | 142 | 129 | 0 | 1 | fall_fwd (1.15 / 0.37 m) | full/env01.mp4 |
| 2 | off_path | 65 | on time but 3 cm off the path; centre 20 cm up vs reference 11 cm, 24 deg from reference orientation | 52 | None | 129 | 127 | 0 | 1 | fall_fwd (1.11 / 0.35 m) | full/env02.mp4 |
| 3 | off_path | 69 | on time but 2 cm off the path; centre 30 cm up vs reference 22 cm, 27 deg from reference orientation | 53 | None | 137 | 129 | 0 | 1 | fall_fwd (1.09 / 0.31 m) | full/env03.mp4 |
| 4 | off_path | 64 | on time but 7 cm off the path; centre 16 cm up vs reference 9 cm, 21 deg from reference orientation | 52 | None | 137 | 126 | 0 | 1 | fall_fwd (1.09 / 0.33 m) | full/env04.mp4 |
| 5 | off_path | 64 | on time but 4 cm off the path; centre 18 cm up vs reference 9 cm, 22 deg from reference orientation | 52 | None | 134 | 127 | 0 | 1 | fall_fwd (1.10 / 0.32 m) | full/env05.mp4 |
| 6 | off_path | 78 | on time but 6 cm off the path; centre 43 cm up vs reference 45 cm, 23 deg from reference orientation | 53 | 130 | 136 | 133 | 0 | 1 | fall_fwd (1.13 / 0.34 m) | full/env06.mp4 |
| 7 | off_path | 67 | on time but 2 cm off the path; centre 25 cm up vs reference 16 cm, 33 deg from reference orientation | 54 | 82 | 89 | 128 | 0 | 0 | fall_fwd (1.01 / 0.25 m) | full/env07.mp4 |
| 8 | off_path | 68 | on time but 10 cm off the path; centre 20 cm up vs reference 19 cm, 31 deg from reference orientation | 52 | 129 | 147 | 134 | 0 | 1 | fall_fwd (1.08 / 0.30 m) | full/env08.mp4 |
| 9 | off_path | 67 | on time but 6 cm off the path; centre 22 cm up vs reference 16 cm, 23 deg from reference orientation | 53 | 128 | 149 | 132 | 0 | 1 | fall_fwd (1.14 / 0.35 m) | full/env09.mp4 |
| 10 | lead | 63 | nearest reference frame 68 (5 ahead, 2 cm); centre 18 cm up vs reference 8 cm, 41 deg from reference orientation | 53 | 84 | 107 | 124 | 0 | 1 | fall_fwd (1.06 / 0.30 m) | full/env10.mp4 |
| 11 | off_path | 67 | on time but 5 cm off the path; centre 22 cm up vs reference 16 cm, 25 deg from reference orientation | 52 | 123 | 135 | 127 | 0 | 1 | fall_fwd (1.12 / 0.34 m) | full/env11.mp4 |
| 12 | off_path | 68 | on time but 6 cm off the path; centre 24 cm up vs reference 19 cm, 20 deg from reference orientation | 52 | 107 | 122 | 129 | 0 | 1 | fall_fwd (1.11 / 0.33 m) | full/env12.mp4 |
| 13 | off_path | 70 | on time but 9 cm off the path; centre 16 cm up vs reference 25 cm, 56 deg from reference orientation | 54 | 76 | 91 | 134 | 0 | 0 | fall_fwd (0.66 / 0.10 m) | full/env13.mp4 |
| 14 | off_path | 64 | on time but 2 cm off the path; centre 18 cm up vs reference 9 cm, 16 deg from reference orientation | 51 | None | 136 | 128 | 0 | 1 | fall_fwd (1.09 / 0.30 m) | full/env14.mp4 |
| 15 | off_path | 70 | on time but 4 cm off the path; centre 33 cm up vs reference 25 cm, 24 deg from reference orientation | 52 | 124 | 138 | 128 | 0 | 1 | fall_fwd (1.14 / 0.35 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_fall_fwd | clips/1_fall_fwd__env01__f089-134.mp4 | 1 | off_path | fall@129 | 89-134 | on time but 10 cm off the path; centre 23 cm up vs reference 23 cm, 11 deg from reference orientation | New dominant ending: 94% fall in the clip (fix-6 10%), 90% fall_fwd at median t 130. At t 120 the pelvis is 0.31 m ahead of the feet (fix-6 walk_up -0.03 m) and the feet moved only 0.34 m along the walk (fix-6 0.71 m). Does it lunge / lean after the book instead of stepping? |
| 1_fall_fwd | clips/1_fall_fwd__env02__f087-132.mp4 | 2 | off_path | fall@127 | 87-132 | on time but 3 cm off the path; centre 20 cm up vs reference 11 cm, 24 deg from reference orientation | New dominant ending: 94% fall in the clip (fix-6 10%), 90% fall_fwd at median t 130. At t 120 the pelvis is 0.31 m ahead of the feet (fix-6 walk_up -0.03 m) and the feet moved only 0.34 m along the walk (fix-6 0.71 m). Does it lunge / lean after the book instead of stepping? |
| 2_carry_walk | clips/2_carry_walk__env01__f078-140.mp4 | 1 | off_path | frames 78-140 | 78-140 | on time but 10 cm off the path; centre 23 cm up vs reference 23 cm, 11 deg from reference orientation | The whole carry up to the fall. Book orientation is closer to the green wireframe than fix-6 (carry median 21 vs 37 deg), position error is similar (16 vs 14 cm). How many real steps are taken, and when does the upper body start to lead? |
| 3_lift_off_path | clips/3_lift_off_path__env01__f084-114.mp4 | 1 | off_path | first_bad@99 | 84-114 | on time but 10 cm off the path; centre 23 cm up vs reference 23 cm, 11 deg from reference orientation | 91% break the strict 10 cm track here (median t 67, same as fix-6 t 68). Lift rotation error dropped (t55-80 median 20 vs 28 deg). Is the book still too high / ahead (centre ~22 cm up vs reference 16 cm)? |
| 3_lift_off_path | clips/3_lift_off_path__env02__f050-080.mp4 | 2 | off_path | first_bad@65 | 50-80 | on time but 3 cm off the path; centre 20 cm up vs reference 11 cm, 24 deg from reference orientation | 91% break the strict 10 cm track here (median t 67, same as fix-6 t 68). Lift rotation error dropped (t55-80 median 20 vs 28 deg). Is the book still too high / ahead (centre ~22 cm up vs reference 16 cm)? |
| 4_stand_squat | (none) | - | - | frame | - | no episode of this rollout matched; try more envs (--num_envs) or another seed | 6%: no walk and the book 1.5 m off at t 120. Book lost early, then standing? (may not appear in the 16-env rollout) |
