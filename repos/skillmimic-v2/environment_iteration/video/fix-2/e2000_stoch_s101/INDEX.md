# fix-2 epoch 2000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | lead | 41 | nearest reference frame 63 (22 ahead, 7 cm); centre 5 cm up vs reference -0 cm, 35 deg from reference orientation | 37 | 83 | 97 | 113 | 0 | 0 | full/env00.mp4 |
| 1 | lead | 49 | nearest reference frame 64 (15 ahead, 5 cm); centre 8 cm up vs reference 0 cm, 54 deg from reference orientation | 36 | None | None | 119 | 0 | 0 | full/env01.mp4 |
| 2 | lead | 47 | nearest reference frame 64 (17 ahead, 4 cm); centre 9 cm up vs reference 0 cm, 55 deg from reference orientation | 38 | 83 | 98 | 103 | 0 | 0 | full/env02.mp4 |
| 3 | lead | 41 | nearest reference frame 63 (22 ahead, 7 cm); centre 5 cm up vs reference -0 cm, 41 deg from reference orientation | 36 | None | None | 114 | 0 | 0 | full/env03.mp4 |
| 4 | lead | 45 | nearest reference frame 63 (18 ahead, 7 cm); centre 6 cm up vs reference 0 cm, 42 deg from reference orientation | 38 | 84 | 100 | 116 | 0 | 0 | full/env04.mp4 |
| 5 | lead | 42 | nearest reference frame 63 (21 ahead, 8 cm); centre 4 cm up vs reference -0 cm, 39 deg from reference orientation | 11 | 83 | 94 | 120 | 0 | 0 | full/env05.mp4 |
| 6 | lead | 45 | nearest reference frame 64 (19 ahead, 6 cm); centre 7 cm up vs reference 0 cm, 53 deg from reference orientation | 36 | 87 | 97 | 115 | 0 | 0 | full/env06.mp4 |
| 7 | lead | 44 | nearest reference frame 63 (19 ahead, 6 cm); centre 7 cm up vs reference 0 cm, 45 deg from reference orientation | 36 | 86 | 94 | 106 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env00__f030-080.mp4 | 0 | lead | frames 30-80 | 30-80 | nearest reference frame 63 (22 ahead, 7 cm); centre 5 cm up vs reference -0 cm, 35 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env01__f030-080.mp4 | 1 | lead | frames 30-80 | 30-80 | nearest reference frame 64 (15 ahead, 5 cm); centre 8 cm up vs reference 0 cm, 54 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 5_e2000_early_pivot | clips/5_e2000_early_pivot__env02__f030-080.mp4 | 2 | lead | frames 30-80 | 30-80 | nearest reference frame 64 (17 ahead, 4 cm); centre 9 cm up vs reference 0 cm, 55 deg from reference orientation | Curve e1000-e2500 (det, 8 envs): the book centre rises 6-12 cm from frame ~40, i.e. before the reference starts lifting at ~53 ('lead'), and its tilt goes to ~90 deg: it is stood on its long edge on the desk (centre 0.80 m) instead of lifted (reference 1.20 m at frame 80). Watch: which hand(s) pivot it, where the fingers are, and why it is not lifted off the desk (fingers under / around the book? one hand pushing it up against the other?). |
| 6_e2000_book_lost | clips/6_e2000_book_lost__env00__f026-066.mp4 | 0 | lead | first_bad@41 | 26-66 | nearest reference frame 63 (22 ahead, 7 cm); centre 5 cm up vs reference -0 cm, 35 deg from reference orientation | e2000 det: the edge-standing book is carried / pushed off the desk at ~90-100 and ends on the floor. Watch: is the book held at all when the reference starts walking, or pushed along the desk by the hand? |
| 6_e2000_book_lost | clips/6_e2000_book_lost__env01__f034-074.mp4 | 1 | lead | first_bad@49 | 34-74 | nearest reference frame 64 (15 ahead, 5 cm); centre 8 cm up vs reference 0 cm, 54 deg from reference orientation | e2000 det: the edge-standing book is carried / pushed off the desk at ~90-100 and ends on the floor. Watch: is the book held at all when the reference starts walking, or pushed along the desk by the hand? |
