# fix-1 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | lag | 65 | nearest reference frame 15 (50 behind, 7 cm); centre 3 cm up vs reference 11 cm, 39 deg from reference orientation | 64 | None | None | 111 | 0 | 0 | full/env00.mp4 |
| 1 | no_lift | 66 | object moved 2 cm, rose 2 cm, 38 deg from reference orientation | 71 | None | None | 106 | 0 | 0 | full/env01.mp4 |
| 2 | lag | 66 | nearest reference frame 15 (51 behind, 5 cm); centre 2 cm up vs reference 13 cm, 28 deg from reference orientation | 56 | 100 | 121 | 101 | 0 | 0 | full/env02.mp4 |
| 3 | lag | 68 | nearest reference frame 63 (5 behind, 4 cm); centre 7 cm up vs reference 19 cm, 13 deg from reference orientation | 49 | 96 | 134 | 98 | 0 | 1 | full/env03.mp4 |
| 4 | off_path | 65 | on time but 10 cm off the path; centre 6 cm up vs reference 11 cm, 16 deg from reference orientation | 46 | 96 | 116 | 99 | 0 | 0 | full/env04.mp4 |
| 5 | off_path | 69 | on time but 3 cm off the path; centre 10 cm up vs reference 22 cm, 19 deg from reference orientation | 58 | None | 128 | 101 | 0 | 1 | full/env05.mp4 |
| 6 | off_path | 68 | on time but 6 cm off the path; centre 7 cm up vs reference 19 cm, 20 deg from reference orientation | 51 | None | 122 | 95 | 0 | 1 | full/env06.mp4 |
| 7 | off_path | 67 | on time but 7 cm off the path; centre 5 cm up vs reference 16 cm, 27 deg from reference orientation | 51 | None | 117 | 102 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_grasp_lift | clips/1_grasp_lift__env00__f045-080.mp4 | 0 | lag | frames 45-80 | 45-80 | nearest reference frame 15 (50 behind, 7 cm); centre 3 cm up vs reference 11 cm, 39 deg from reference orientation | 76% of episodes: the task breaks at frame 66-68 while the book is in hand but behind the reference. The reference pivots the book up on one edge and rotates it (41 deg at 60 -> ~90 deg at 72) while lifting 19 cm in 10 frames. Watch: where the hands grip (edges? under?), whether the pivot happens, whether the book slips in the fingers, whether the arms are simply too slow vs the green wireframe. |
| 1_grasp_lift | clips/1_grasp_lift__env02__f045-080.mp4 | 2 | lag | frames 45-80 | 45-80 | nearest reference frame 15 (51 behind, 5 cm); centre 2 cm up vs reference 13 cm, 28 deg from reference orientation | 76% of episodes: the task breaks at frame 66-68 while the book is in hand but behind the reference. The reference pivots the book up on one edge and rotates it (41 deg at 60 -> ~90 deg at 72) while lifting 19 cm in 10 frames. Watch: where the hands grip (edges? under?), whether the pivot happens, whether the book slips in the fingers, whether the arms are simply too slow vs the green wireframe. |
| 1_grasp_lift | clips/1_grasp_lift__env03__f045-080.mp4 | 3 | lag | frames 45-80 | 45-80 | nearest reference frame 63 (5 behind, 4 cm); centre 7 cm up vs reference 19 cm, 13 deg from reference orientation | 76% of episodes: the task breaks at frame 66-68 while the book is in hand but behind the reference. The reference pivots the book up on one edge and rotates it (41 deg at 60 -> ~90 deg at 72) while lifting 19 cm in 10 frames. Watch: where the hands grip (edges? under?), whether the pivot happens, whether the book slips in the fingers, whether the arms are simply too slow vs the green wireframe. |
| 2_never_lifted | clips/2_never_lifted__env01__f051-076.mp4 | 1 | no_lift | first_bad@66 | 51-76 | object moved 2 cm, rose 2 cm, 38 deg from reference orientation | 21%: the book is touched but never comes up (or slides / tips) at frame ~65. Watch: finger placement, pushing the book across the desk, one hand vs two hands. |
| 3_collapse | clips/3_collapse__env00__f086-116.mp4 | 0 | lag | fall@111 | 86-116 | nearest reference frame 15 (50 behind, 7 cm); centre 3 cm up vs reference 11 cm, 39 deg from reference orientation | 100% fall, median frame 103; the pelvis starts sinking at ~85, exactly when the reference starts walking sideways (~0.8 m/s, ~1.1 m in total). Watch: does it try to step? feet sliding? leaning on the desk? does the book pull it over? |
| 3_collapse | clips/3_collapse__env01__f081-111.mp4 | 1 | no_lift | fall@106 | 81-111 | object moved 2 cm, rose 2 cm, 38 deg from reference orientation | 100% fall, median frame 103; the pelvis starts sinking at ~85, exactly when the reference starts walking sideways (~0.8 m/s, ~1.1 m in total). Watch: does it try to step? feet sliding? leaning on the desk? does the book pull it over? |
| 3_collapse | clips/3_collapse__env02__f076-106.mp4 | 2 | lag | fall@101 | 76-106 | nearest reference frame 15 (51 behind, 5 cm); centre 2 cm up vs reference 13 cm, 28 deg from reference orientation | 100% fall, median frame 103; the pelvis starts sinking at ~85, exactly when the reference starts walking sideways (~0.8 m/s, ~1.1 m in total). Watch: does it try to step? feet sliding? leaning on the desk? does the book pull it over? |
| 4_drop | clips/4_drop__env02__f088-110.mp4 | 2 | lag | drop@100 | 88-110 | nearest reference frame 15 (51 behind, 5 cm); centre 2 cm up vs reference 13 cm, 28 deg from reference orientation | 55% drop the book at median frame 98, 2 frames before the fall (median). Check whether the hands open first or the body collapse takes the book down. |
| 4_drop | clips/4_drop__env03__f084-106.mp4 | 3 | lag | drop@96 | 84-106 | nearest reference frame 63 (5 behind, 4 cm); centre 7 cm up vs reference 19 cm, 13 deg from reference orientation | 55% drop the book at median frame 98, 2 frames before the fall (median). Check whether the hands open first or the body collapse takes the book down. |
