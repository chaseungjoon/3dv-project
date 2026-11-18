# fix-2 epoch 2000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | lead | 40 | nearest reference frame 63 (23 ahead, 9 cm); centre 4 cm up vs reference -0 cm, 34 deg from reference orientation | 37 | 84 | 95 | 112 | 0 | 0 | full/env00.mp4 |
| 1 | lead | 50 | nearest reference frame 64 (14 ahead, 4 cm); centre 8 cm up vs reference 0 cm, 50 deg from reference orientation | 36 | None | None | 117 | 0 | 0 | full/env01.mp4 |
| 2 | lead | 47 | nearest reference frame 64 (17 ahead, 4 cm); centre 9 cm up vs reference 0 cm, 56 deg from reference orientation | 38 | 84 | 93 | 103 | 0 | 0 | full/env02.mp4 |
| 3 | lead | 40 | nearest reference frame 63 (23 ahead, 8 cm); centre 5 cm up vs reference -0 cm, 37 deg from reference orientation | 35 | None | None | 126 | 0 | 0 | full/env03.mp4 |
| 4 | lead | 41 | nearest reference frame 62 (21 ahead, 8 cm); centre 4 cm up vs reference -0 cm, 30 deg from reference orientation | 37 | None | 109 | 115 | 0 | 0 | full/env04.mp4 |
| 5 | lead | 42 | nearest reference frame 63 (21 ahead, 8 cm); centre 4 cm up vs reference -0 cm, 37 deg from reference orientation | 11 | 86 | 95 | 119 | 0 | 0 | full/env05.mp4 |
| 6 | lead | 44 | nearest reference frame 64 (20 ahead, 6 cm); centre 7 cm up vs reference 0 cm, 52 deg from reference orientation | 36 | 85 | 94 | 116 | 0 | 0 | full/env06.mp4 |
| 7 | lead | 44 | nearest reference frame 64 (20 ahead, 6 cm); centre 7 cm up vs reference 0 cm, 45 deg from reference orientation | 36 | 86 | 100 | 105 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env00__f030-080.mp4 | 0 | lead | frames 30-80 | 30-80 | nearest reference frame 63 (23 ahead, 9 cm); centre 4 cm up vs reference -0 cm, 34 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env01__f030-080.mp4 | 1 | lead | frames 30-80 | 30-80 | nearest reference frame 64 (14 ahead, 4 cm); centre 8 cm up vs reference 0 cm, 50 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env02__f030-080.mp4 | 2 | lead | frames 30-80 | 30-80 | nearest reference frame 64 (17 ahead, 4 cm); centre 9 cm up vs reference 0 cm, 56 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 6_e2000_drop | clips/6_e2000_drop__env00__f069-099.mp4 | 0 | lead | drop@84 | 69-99 | nearest reference frame 63 (23 ahead, 9 cm); centre 4 cm up vs reference -0 cm, 34 deg from reference orientation | e2000: 6/8 rollout episodes drop the book at frames 83-87 (as the reference starts the sideways walk); it hits the floor at 94-100 and the character falls at 103-120. Watch: does the hand open / the book rotate out of the pinch, or is the edge-standing book pushed off the desk when the body turns to walk? |
| 6_e2000_drop | clips/6_e2000_drop__env02__f069-099.mp4 | 2 | lead | drop@84 | 69-99 | nearest reference frame 64 (17 ahead, 4 cm); centre 9 cm up vs reference 0 cm, 56 deg from reference orientation | e2000: 6/8 rollout episodes drop the book at frames 83-87 (as the reference starts the sideways walk); it hits the floor at 94-100 and the character falls at 103-120. Watch: does the hand open / the book rotate out of the pinch, or is the edge-standing book pushed off the desk when the body turns to walk? |
