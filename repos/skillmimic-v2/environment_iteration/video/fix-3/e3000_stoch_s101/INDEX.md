# fix-3 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | lag | 66 | nearest reference frame 55 (11 behind, 3 cm); centre 3 cm up vs reference 13 cm, 21 deg from reference orientation | 57 | None | None | 162 | 0 | 0 | full/env00.mp4 |
| 1 | lag | 66 | nearest reference frame 52 (14 behind, 2 cm); centre 2 cm up vs reference 13 cm, 27 deg from reference orientation | 66 | None | None | 165 | 0 | 0 | full/env01.mp4 |
| 2 | lag | 64 | nearest reference frame 54 (10 behind, 7 cm); centre 4 cm up vs reference 9 cm, 14 deg from reference orientation | 59 | None | None | 150 | 0 | 0 | full/env02.mp4 |
| 3 | lag | 63 | nearest reference frame 54 (9 behind, 8 cm); centre 3 cm up vs reference 8 cm, 20 deg from reference orientation | 61 | None | None | -1 | 0 | 0 | full/env03.mp4 |
| 4 | lag | 65 | nearest reference frame 53 (12 behind, 5 cm); centre 3 cm up vs reference 11 cm, 21 deg from reference orientation | 62 | None | None | 171 | 0 | 0 | full/env04.mp4 |
| 5 | no_lift | 65 | object moved 3 cm, rose -0 cm, 49 deg from reference orientation | None | None | None | 136 | 0 | 0 | full/env05.mp4 |
| 6 | lag | 65 | nearest reference frame 54 (11 behind, 4 cm); centre 2 cm up vs reference 11 cm, 28 deg from reference orientation | 65 | None | None | 176 | 0 | 0 | full/env06.mp4 |
| 7 | lag | 65 | nearest reference frame 54 (11 behind, 6 cm); centre 3 cm up vs reference 11 cm, 18 deg from reference orientation | 60 | None | None | 150 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_lift_lag | clips/1_lift_lag__env00__f046-088.mp4 | 0 | lag | first_bad@66 | 46-88 | nearest reference frame 55 (11 behind, 3 cm); centre 3 cm up vs reference 13 cm, 21 deg from reference orientation | Most clean episodes first break at frame 66, classified as lag. Watch how the hands contact the book, whether it pivots on the desk, and why it does not follow the rising green reference. The in-hand label is a geometric heuristic, not proof of a secure grip. |
| 1_lift_lag | clips/1_lift_lag__env01__f046-088.mp4 | 1 | lag | first_bad@66 | 46-88 | nearest reference frame 52 (14 behind, 2 cm); centre 2 cm up vs reference 13 cm, 27 deg from reference orientation | Most clean episodes first break at frame 66, classified as lag. Watch how the hands contact the book, whether it pivots on the desk, and why it does not follow the rising green reference. The in-hand label is a geometric heuristic, not proof of a secure grip. |
| 1_lift_lag | clips/1_lift_lag__env02__f044-086.mp4 | 2 | lag | first_bad@64 | 44-86 | nearest reference frame 54 (10 behind, 7 cm); centre 4 cm up vs reference 9 cm, 14 deg from reference orientation | Most clean episodes first break at frame 66, classified as lag. Watch how the hands contact the book, whether it pivots on the desk, and why it does not follow the rising green reference. The in-hand label is a geometric heuristic, not proof of a secure grip. |
| 2_late_fall | clips/2_late_fall__env00__f127-167.mp4 | 0 | lag | fall@162 | 127-167 | nearest reference frame 55 (11 behind, 3 cm); centre 3 cm up vs reference 13 cm, 21 deg from reference orientation | Falls occur later than in fix-2; inspect the empty-handed walk, balance and foot placement. Falls after frame 147 are outside the reference clip. |
| 2_late_fall | clips/2_late_fall__env01__f130-170.mp4 | 1 | lag | fall@165 | 130-170 | nearest reference frame 52 (14 behind, 2 cm); centre 2 cm up vs reference 13 cm, 27 deg from reference orientation | Falls occur later than in fix-2; inspect the empty-handed walk, balance and foot placement. Falls after frame 147 are outside the reference clip. |
| 4_full_overview | clips/4_full_overview__env00__f030-147.mp4 | 0 | lag | frames 30-147 | 30-147 | nearest reference frame 55 (11 behind, 3 cm); centre 3 cm up vs reference 13 cm, 21 deg from reference orientation | See approach, attempted pickup and subsequent walk together; compare the small book movement with the reference trajectory. |
