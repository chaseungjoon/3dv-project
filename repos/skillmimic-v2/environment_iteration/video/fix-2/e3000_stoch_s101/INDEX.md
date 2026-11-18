# fix-2 epoch 3000, cell stoch, seed 101

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | no_lift | 65 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | None | None | None | 134 | 0 | 0 | full/env00.mp4 |
| 1 | no_lift | 65 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | None | None | None | 106 | 0 | 0 | full/env01.mp4 |
| 2 | no_lift | 65 | object moved 0 cm, rose -0 cm, 52 deg from reference orientation | None | None | None | 112 | 0 | 0 | full/env02.mp4 |
| 3 | no_lift | 65 | object moved 1 cm, rose -0 cm, 51 deg from reference orientation | None | None | None | 131 | 0 | 0 | full/env03.mp4 |
| 4 | no_lift | 65 | object moved 0 cm, rose -0 cm, 51 deg from reference orientation | None | None | None | 151 | 0 | 0 | full/env04.mp4 |
| 5 | drop | 65 | last in hand at frame 17, 52 deg from reference orientation | 11 | None | None | 128 | 0 | 0 | full/env05.mp4 |
| 6 | no_lift | 65 | object moved 1 cm, rose -0 cm, 50 deg from reference orientation | None | None | None | 140 | 0 | 0 | full/env06.mp4 |
| 7 | no_lift | 65 | object moved 0 cm, rose -0 cm, 52 deg from reference orientation | None | None | None | 163 | 0 | 0 | full/env07.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 1_no_lift_approach | clips/1_no_lift_approach__env00__f040-080.mp4 | 0 | no_lift | frames 40-80 | 40-80 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | 96% of episodes: the book never moves (median displacement 0 cm, height 0.74 m all clip) and the track breaks at frame 65 when the reference rises. Wrists come to ~17 cm of the book at frames 50-65, but the contact force on the book is ~4x lower than fix-1's. Watch: do the fingers touch the book at all, or do the hands hover / follow the green reference dots past the red book? Does the right hand leave at 62-65 as in the demo while the left hand never closes? |
| 1_no_lift_approach | clips/1_no_lift_approach__env01__f040-080.mp4 | 1 | no_lift | frames 40-80 | 40-80 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | 96% of episodes: the book never moves (median displacement 0 cm, height 0.74 m all clip) and the track breaks at frame 65 when the reference rises. Wrists come to ~17 cm of the book at frames 50-65, but the contact force on the book is ~4x lower than fix-1's. Watch: do the fingers touch the book at all, or do the hands hover / follow the green reference dots past the red book? Does the right hand leave at 62-65 as in the demo while the left hand never closes? |
| 1_no_lift_approach | clips/1_no_lift_approach__env02__f040-080.mp4 | 2 | no_lift | frames 40-80 | 40-80 | object moved 0 cm, rose -0 cm, 52 deg from reference orientation | 96% of episodes: the book never moves (median displacement 0 cm, height 0.74 m all clip) and the track breaks at frame 65 when the reference rises. Wrists come to ~17 cm of the book at frames 50-65, but the contact force on the book is ~4x lower than fix-1's. Watch: do the fingers touch the book at all, or do the hands hover / follow the green reference dots past the red book? Does the right hand leave at 62-65 as in the demo while the left hand never closes? |
| 2_empty_walk_collapse | clips/2_empty_walk_collapse__env00__f104-139.mp4 | 0 | no_lift | fall@134 | 104-139 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | 100% fall (median frame 130, 81% inside the clip) although the book stays on the desk; fix-1 fell at ~103 with the book. The pelvis stays up until ~115 and sinks during the sideways walk. Watch: is it a stepping / balance failure of the empty-handed walk (feet sliding, crossing legs, leaning toward where the reference book is)? This tells whether the walk fails independently of the book. |
| 2_empty_walk_collapse | clips/2_empty_walk_collapse__env01__f076-111.mp4 | 1 | no_lift | fall@106 | 76-111 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | 100% fall (median frame 130, 81% inside the clip) although the book stays on the desk; fix-1 fell at ~103 with the book. The pelvis stays up until ~115 and sinks during the sideways walk. Watch: is it a stepping / balance failure of the empty-handed walk (feet sliding, crossing legs, leaning toward where the reference book is)? This tells whether the walk fails independently of the book. |
| 2_empty_walk_collapse | clips/2_empty_walk_collapse__env02__f082-117.mp4 | 2 | no_lift | fall@112 | 82-117 | object moved 0 cm, rose -0 cm, 52 deg from reference orientation | 100% fall (median frame 130, 81% inside the clip) although the book stays on the desk; fix-1 fell at ~103 with the book. The pelvis stays up until ~115 and sinks during the sideways walk. Watch: is it a stepping / balance failure of the empty-handed walk (feet sliding, crossing legs, leaning toward where the reference book is)? This tells whether the walk fails independently of the book. |
| 4_full_episode_overview | clips/4_full_episode_overview__env00__f030-147.mp4 | 0 | no_lift | frames 30-147 | 30-147 | object moved 0 cm, rose -0 cm, 53 deg from reference orientation | One whole clip window for context: approach (30-53), the moment the reference lifts (53-79) and the carry walk (79-147) with the book left on the desk. |
