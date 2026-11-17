# fix-1 epoch 15260, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | off_path | 74 | on time but 6 cm off the path; centre 28 cm up vs reference 38 cm, 35 deg from reference orientation | 53 | 84 | 95 | -1 | 0 | 1 | full/env00.mp4 |
| 1 | off_path | 73 | on time but 5 cm off the path; centre 28 cm up vs reference 35 cm, 25 deg from reference orientation | 54 | 90 | 111 | -1 | 0 | 1 | full/env01.mp4 |
| 2 | off_path | 72 | on time but 7 cm off the path; centre 26 cm up vs reference 31 cm, 29 deg from reference orientation | 56 | 86 | 96 | 140 | 0 | 1 | full/env02.mp4 |
| 3 | off_path | 72 | on time but 8 cm off the path; centre 26 cm up vs reference 31 cm, 26 deg from reference orientation | 55 | 85 | 94 | -1 | 0 | 1 | full/env03.mp4 |
| 4 | off_path | 76 | on time but 7 cm off the path; centre 32 cm up vs reference 42 cm, 35 deg from reference orientation | 53 | 86 | 95 | -1 | 0 | 1 | full/env04.mp4 |
| 5 | off_path | 71 | on time but 10 cm off the path; centre 27 cm up vs reference 28 cm, 29 deg from reference orientation | 53 | 85 | 93 | -1 | 0 | 1 | full/env05.mp4 |
| 6 | off_path | 72 | on time but 6 cm off the path; centre 24 cm up vs reference 31 cm, 25 deg from reference orientation | 54 | 87 | 96 | -1 | 0 | 1 | full/env06.mp4 |
| 7 | off_path | 72 | on time but 6 cm off the path; centre 26 cm up vs reference 31 cm, 29 deg from reference orientation | 56 | 90 | 100 | -1 | 0 | 1 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 6_e15260_lift_rotate | clips/6_e15260_lift_rotate__env00__f055-090.mp4 | 0 | off_path | frames 55-90 | 55-90 | on time but 6 cm off the path; centre 28 cm up vs reference 38 cm, 35 deg from reference orientation | Lift is on time; compare the book's rotation with the green wireframe (reference turns the book ~180 deg by frame 90). Does the grip allow that rotation? |
| 6_e15260_lift_rotate | clips/6_e15260_lift_rotate__env01__f055-090.mp4 | 1 | off_path | frames 55-90 | 55-90 | on time but 5 cm off the path; centre 28 cm up vs reference 35 cm, 25 deg from reference orientation | Lift is on time; compare the book's rotation with the green wireframe (reference turns the book ~180 deg by frame 90). Does the grip allow that rotation? |
| 7_e15260_carry_drop | clips/7_e15260_carry_drop__env00__f069-099.mp4 | 0 | off_path | drop@84 | 69-99 | on time but 6 cm off the path; centre 28 cm up vs reference 38 cm, 35 deg from reference orientation | 96% drop at median frame 87 without falling. Watch: hands following the reference arms while the body does not walk (arms pulled apart?), or the book twisting out. |
| 7_e15260_carry_drop | clips/7_e15260_carry_drop__env01__f075-105.mp4 | 1 | off_path | drop@90 | 75-105 | on time but 5 cm off the path; centre 28 cm up vs reference 35 cm, 25 deg from reference orientation | 96% drop at median frame 87 without falling. Watch: hands following the reference arms while the body does not walk (arms pulled apart?), or the book twisting out. |
| 7_e15260_carry_drop | clips/7_e15260_carry_drop__env02__f071-101.mp4 | 2 | off_path | drop@86 | 71-101 | on time but 7 cm off the path; centre 26 cm up vs reference 31 cm, 29 deg from reference orientation | 96% drop at median frame 87 without falling. Watch: hands following the reference arms while the body does not walk (arms pulled apart?), or the book twisting out. |
