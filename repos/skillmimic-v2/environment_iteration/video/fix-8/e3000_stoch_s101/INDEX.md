# fix-8 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 63 | on time but 7 cm off the path; centre 14 cm up vs reference 8 cm, 20 deg from reference orientation | 54 | 123 | 131 | 124 | 0 | 1 | fall_fwd (0.84 / 0.22 m) | full/env00.mp4 |
| 1 | off_path | 64 | on time but 11 cm off the path; centre 10 cm up vs reference 9 cm, 20 deg from reference orientation | 54 | 124 | 131 | 125 | 0 | 1 | fall_fwd (0.96 / 0.23 m) | full/env01.mp4 |
| 2 | off_path | 64 | on time but 7 cm off the path; centre 16 cm up vs reference 9 cm, 17 deg from reference orientation | 54 | 123 | 152 | 124 | 0 | 1 | fall_fwd (0.95 / 0.24 m) | full/env02.mp4 |
| 3 | off_path | 62 | on time but 9 cm off the path; centre 12 cm up vs reference 7 cm, 27 deg from reference orientation | 53 | 121 | 132 | 122 | 0 | 1 | fall_fwd (0.86 / 0.22 m) | full/env03.mp4 |
| 4 | lead | 62 | nearest reference frame 67 (5 ahead, 8 cm); centre 13 cm up vs reference 7 cm, 25 deg from reference orientation | 52 | 123 | 148 | 124 | 0 | 1 | fall_fwd (0.88 / 0.23 m) | full/env04.mp4 |
| 5 | off_path | 63 | on time but 10 cm off the path; centre 9 cm up vs reference 8 cm, 21 deg from reference orientation | 55 | 124 | 131 | 125 | 0 | 1 | fall_fwd (0.84 / 0.21 m) | full/env05.mp4 |
| 6 | off_path | 65 | on time but 9 cm off the path; centre 14 cm up vs reference 11 cm, 20 deg from reference orientation | 53 | 125 | 142 | 126 | 0 | 1 | fall_fwd (0.80 / 0.19 m) | full/env06.mp4 |
| 7 | off_path | 66 | on time but 9 cm off the path; centre 17 cm up vs reference 13 cm, 16 deg from reference orientation | 52 | 124 | 131 | 125 | 0 | 1 | fall_fwd (0.88 / 0.21 m) | full/env07.mp4 |
| 8 | off_path | 64 | on time but 6 cm off the path; centre 16 cm up vs reference 9 cm, 25 deg from reference orientation | 53 | 123 | 141 | 124 | 0 | 1 | fall_fwd (0.84 / 0.20 m) | full/env08.mp4 |
| 9 | off_path | 67 | on time but 9 cm off the path; centre 19 cm up vs reference 16 cm, 14 deg from reference orientation | 56 | 124 | 132 | 125 | 0 | 1 | fall_fwd (0.87 / 0.24 m) | full/env09.mp4 |
| 10 | off_path | 62 | on time but 8 cm off the path; centre 12 cm up vs reference 7 cm, 20 deg from reference orientation | 53 | None | 140 | 122 | 0 | 1 | fall_fwd (0.98 / 0.28 m) | full/env10.mp4 |
| 11 | off_path | 66 | on time but 9 cm off the path; centre 18 cm up vs reference 13 cm, 20 deg from reference orientation | 54 | 126 | 166 | 127 | 0 | 1 | fall_fwd (0.79 / 0.17 m) | full/env11.mp4 |
| 12 | off_path | 63 | on time but 10 cm off the path; centre 11 cm up vs reference 8 cm, 24 deg from reference orientation | 54 | 124 | 133 | 126 | 0 | 1 | fall_fwd (0.91 / 0.25 m) | full/env12.mp4 |
| 13 | off_path | 64 | on time but 8 cm off the path; centre 14 cm up vs reference 9 cm, 8 deg from reference orientation | 53 | 125 | 145 | 126 | 0 | 0 | fall_fwd (0.87 / 0.23 m) | full/env13.mp4 |
| 14 | off_path | 65 | on time but 11 cm off the path; centre 9 cm up vs reference 11 cm, 22 deg from reference orientation | 54 | 126 | 134 | 127 | 0 | 1 | fall_fwd (0.84 / 0.19 m) | full/env14.mp4 |
| 15 | off_path | 65 | on time but 9 cm off the path; centre 13 cm up vs reference 11 cm, 22 deg from reference orientation | 55 | 124 | 132 | 125 | 0 | 1 | fall_fwd (0.89 / 0.22 m) | full/env15.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_fall_fwd | clips/1_fall_fwd__env00__f084-129.mp4 | 0 | off_path | fall@124 | 84-129 | on time but 7 cm off the path; centre 14 cm up vs reference 8 cm, 20 deg from reference orientation | New dominant ending: 99% fall in the clip (fix-6 10%), fall_fwd 96% at median t 126. At t 120 the pelvis is 0.38 m ahead of the ankles (fix-6 -0.02 m, fix-7 0.30 m) and the feet moved 0.22 m along the walk (fix-6 0.70 m). Is this the same lunge / kneel as fix-7, even though fix-8 has fix-6's reward weights? |
| 1_fall_fwd | clips/1_fall_fwd__env01__f085-130.mp4 | 1 | off_path | fall@125 | 85-130 | on time but 11 cm off the path; centre 10 cm up vs reference 9 cm, 20 deg from reference orientation | New dominant ending: 99% fall in the clip (fix-6 10%), fall_fwd 96% at median t 126. At t 120 the pelvis is 0.38 m ahead of the ankles (fix-6 -0.02 m, fix-7 0.30 m) and the feet moved 0.22 m along the walk (fix-6 0.70 m). Is this the same lunge / kneel as fix-7, even though fix-8 has fix-6's reward weights? |
| 2_carry_walk | clips/2_carry_walk__env00__f078-135.mp4 | 0 | off_path | frames 78-135 | 78-135 | on time but 7 cm off the path; centre 14 cm up vs reference 8 cm, 20 deg from reference orientation | The whole carry up to the fall. Carry position error 31 cm (fix-6 14 cm), orientation 44 deg (fix-6 37). Unlike fix-7, the book does not stay near the green wireframe. Count the real steps. When does the upper body start to lead? Is the book held out toward the reference or dragged behind? |
| 3_lift_off_path | clips/3_lift_off_path__env00__f048-078.mp4 | 0 | off_path | first_bad@63 | 48-78 | on time but 7 cm off the path; centre 14 cm up vs reference 8 cm, 20 deg from reference orientation | 90% break the 10 cm track here, at median t 64 (fix-6 t 67-68). The t45-85 peak error is 24.6 cm (fix-6 17.2) and the book is ahead along the path (median +6 cm vs fix-6 +3). Only 2% are within 10 cm at t 70 (fix-6 27%). Too fast / too far forward relative to the green pivot? |
| 3_lift_off_path | clips/3_lift_off_path__env01__f049-079.mp4 | 1 | off_path | first_bad@64 | 49-79 | on time but 11 cm off the path; centre 10 cm up vs reference 9 cm, 20 deg from reference orientation | 90% break the 10 cm track here, at median t 64 (fix-6 t 67-68). The t45-85 peak error is 24.6 cm (fix-6 17.2) and the book is ahead along the path (median +6 cm vs fix-6 +3). Only 2% are within 10 cm at t 70 (fix-6 27%). Too fast / too far forward relative to the green pivot? |
| 4_walk_up | (none) | - | - | frame | - | no episode of this rollout matched; try more envs (--num_envs) or another seed | 1 of 96 evaluation episodes still walks and stays up (fix-6's 90% mode). May not appear in the 16-env rollout. |
