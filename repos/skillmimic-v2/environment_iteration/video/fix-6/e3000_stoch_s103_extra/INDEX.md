# fix-6 epoch 3000, cell stoch, seed 103

Fresh rollout with cameras (src/render.py); events re-detected in this rollout.
Green wireframe = reference book, green dots = reference key bodies, red box = simulated book.

## Episodes of this rollout

| env | mode | broke at | detail | grasp | drop | floor | fall | task | lift | walk outcome (root / feet along walk) | full video |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | off_path | 67 | on time but 3 cm off the path; centre 24 cm up vs reference 16 cm, 37 deg from reference orientation | 55 | 182 | 190 | 214 | 0 | 1 | walk_up (0.86 / 0.83 m) | full/env00.mp4 |
| 1 | off_path | 69 | on time but 3 cm off the path; centre 29 cm up vs reference 22 cm, 45 deg from reference orientation | 55 | 193 | 202 | 210 | 0 | 1 | walk_up (0.75 / 0.71 m) | full/env01.mp4 |
| 2 | off_path | 93 | on time but 6 cm off the path; centre 22 cm up vs reference 24 cm, 24 deg from reference orientation | 58 | None | 213 | 214 | 0 | 1 | walk_up (0.82 / 0.71 m) | full/env02.mp4 |
| 3 | off_path | 91 | on time but 9 cm off the path; centre 26 cm up vs reference 27 cm, 28 deg from reference orientation | 55 | 188 | 195 | 214 | 0 | 1 | walk_up (0.82 / 0.80 m) | full/env03.mp4 |
| 4 | off_path | 79 | on time but 4 cm off the path; centre 48 cm up vs reference 45 cm, 21 deg from reference orientation | 55 | 192 | 200 | 212 | 0 | 1 | walk_up (0.80 / 0.78 m) | full/env04.mp4 |
| 5 | off_path | 96 | on time but 10 cm off the path; centre 21 cm up vs reference 23 cm, 49 deg from reference orientation | 58 | 189 | 197 | 206 | 0 | 1 | walk_up (0.77 / 0.75 m) | full/env05.mp4 |
| 6 | off_path | 68 | on time but 5 cm off the path; centre 26 cm up vs reference 19 cm, 34 deg from reference orientation | 56 | 192 | 199 | 206 | 0 | 1 | walk_up (0.83 / 0.78 m) | full/env06.mp4 |
| 7 | off_path | 65 | on time but 3 cm off the path; centre 22 cm up vs reference 11 cm, 41 deg from reference orientation | 54 | 190 | 198 | 212 | 0 | 1 | walk_up (0.71 / 0.63 m) | full/env07.mp4 |
| 8 | off_path | 67 | on time but 3 cm off the path; centre 23 cm up vs reference 16 cm, 41 deg from reference orientation | 54 | None | 148 | 133 | 0 | 1 | fall_fwd (1.20 / 0.47 m) | full/env08.mp4 |
| 9 | off_path | 76 | on time but 3 cm off the path; centre 42 cm up vs reference 42 cm, 28 deg from reference orientation | 58 | 178 | 186 | 228 | 0 | 1 | walk_up (0.75 / 0.63 m) | full/env09.mp4 |
| 10 | off_path | 67 | on time but 2 cm off the path; centre 25 cm up vs reference 16 cm, 39 deg from reference orientation | 54 | 197 | 204 | 209 | 0 | 1 | walk_up (0.79 / 0.73 m) | full/env10.mp4 |
| 11 | off_path | 65 | on time but 1 cm off the path; centre 21 cm up vs reference 11 cm, 33 deg from reference orientation | 53 | 193 | 200 | 205 | 0 | 1 | walk_up (0.78 / 0.74 m) | full/env11.mp4 |
| 12 | off_path | 67 | on time but 5 cm off the path; centre 23 cm up vs reference 16 cm, 32 deg from reference orientation | 58 | 200 | 207 | 216 | 0 | 1 | walk_up (0.81 / 0.77 m) | full/env12.mp4 |
| 13 | lead | 63 | nearest reference frame 68 (5 ahead, 4 cm); centre 17 cm up vs reference 8 cm, 33 deg from reference orientation | 54 | 187 | 195 | 204 | 0 | 1 | walk_up (0.72 / 0.67 m) | full/env13.mp4 |
| 14 | off_path | 88 | on time but 7 cm off the path; centre 30 cm up vs reference 31 cm, 18 deg from reference orientation | 58 | 195 | 203 | 219 | 0 | 1 | walk_up (0.73 / 0.66 m) | full/env14.mp4 |
| 15 | off_path | 70 | on time but 4 cm off the path; centre 31 cm up vs reference 25 cm, 34 deg from reference orientation | 57 | 185 | 192 | 209 | 0 | 1 | walk_up (0.80 / 0.76 m) | full/env15.mp4 |
| 16 | off_path | 68 | on time but 4 cm off the path; centre 25 cm up vs reference 19 cm, 36 deg from reference orientation | 55 | 185 | 199 | 212 | 0 | 1 | walk_up (0.75 / 0.70 m) | full/env16.mp4 |
| 17 | off_path | 64 | on time but 4 cm off the path; centre 18 cm up vs reference 9 cm, 36 deg from reference orientation | 53 | 194 | 202 | 210 | 0 | 1 | walk_up (0.83 / 0.77 m) | full/env17.mp4 |
| 18 | off_path | 65 | on time but 2 cm off the path; centre 21 cm up vs reference 11 cm, 43 deg from reference orientation | 52 | 189 | 197 | 207 | 0 | 1 | walk_up (0.75 / 0.70 m) | full/env18.mp4 |
| 19 | off_path | 67 | on time but 3 cm off the path; centre 24 cm up vs reference 16 cm, 42 deg from reference orientation | 54 | 133 | 151 | 138 | 0 | 1 | fall_fwd (1.29 / 0.54 m) | full/env19.mp4 |
| 20 | off_path | 91 | on time but 8 cm off the path; centre 24 cm up vs reference 27 cm, 30 deg from reference orientation | 58 | 195 | 204 | 208 | 0 | 1 | walk_up (0.83 / 0.80 m) | full/env20.mp4 |
| 21 | off_path | 92 | on time but 8 cm off the path; centre 19 cm up vs reference 25 cm, 28 deg from reference orientation | 56 | 185 | 200 | 215 | 0 | 1 | walk_up (0.73 / 0.66 m) | full/env21.mp4 |
| 22 | off_path | 65 | on time but 3 cm off the path; centre 19 cm up vs reference 11 cm, 34 deg from reference orientation | 55 | 197 | 204 | 222 | 0 | 1 | walk_up (0.72 / 0.64 m) | full/env22.mp4 |
| 23 | off_path | 68 | on time but 2 cm off the path; centre 27 cm up vs reference 19 cm, 38 deg from reference orientation | 54 | 192 | 199 | 210 | 0 | 1 | walk_up (0.76 / 0.72 m) | full/env23.mp4 |
| 24 | off_path | 67 | on time but 4 cm off the path; centre 23 cm up vs reference 16 cm, 37 deg from reference orientation | 56 | 187 | 195 | 211 | 0 | 1 | walk_up (0.82 / 0.77 m) | full/env24.mp4 |
| 25 | off_path | 70 | on time but 2 cm off the path; centre 32 cm up vs reference 25 cm, 38 deg from reference orientation | 55 | 185 | 205 | 227 | 0 | 1 | walk_up (0.78 / 0.69 m) | full/env25.mp4 |
| 26 | off_path | 70 | on time but 6 cm off the path; centre 30 cm up vs reference 25 cm, 22 deg from reference orientation | 57 | 177 | 184 | 202 | 0 | 1 | walk_up (0.69 / 0.66 m) | full/env26.mp4 |
| 27 | off_path | 74 | on time but 4 cm off the path; centre 41 cm up vs reference 38 cm, 38 deg from reference orientation | 55 | 190 | 198 | 210 | 0 | 1 | walk_up (0.77 / 0.74 m) | full/env27.mp4 |
| 28 | off_path | 65 | on time but 4 cm off the path; centre 20 cm up vs reference 11 cm, 37 deg from reference orientation | 54 | 190 | 198 | 207 | 0 | 1 | walk_up (0.78 / 0.72 m) | full/env28.mp4 |
| 29 | lag | 145 | nearest reference frame 132 (13 behind, 2 cm); centre 22 cm up vs reference 21 cm, 29 deg from reference orientation | 58 | 186 | 193 | 216 | 0 | 1 | walk_up (0.79 / 0.76 m) | full/env29.mp4 |
| 30 | off_path | 66 | on time but 5 cm off the path; centre 21 cm up vs reference 13 cm, 32 deg from reference orientation | 56 | 195 | 202 | 214 | 0 | 1 | walk_up (0.71 / 0.65 m) | full/env30.mp4 |
| 31 | off_path | 94 | on time but 8 cm off the path; centre 18 cm up vs reference 24 cm, 37 deg from reference orientation | 57 | 202 | 211 | 214 | 0 | 1 | walk_up (0.71 / 0.67 m) | full/env31.mp4 |

## Clips

| clip | file | env | mode | anchor | window | detail | what to look for |
|---|---|---:|---|---|---|---|---|
| 5_post_clip_drop | clips/5_post_clip_drop__env00__f142-197.mp4 | 0 | off_path | drop@182 | 142-197 | on time but 3 cm off the path; centre 24 cm up vs reference 16 cm, 37 deg from reference orientation | Typical fix-6 ending: walks up through the clip, then drops the book at ~190 and falls at ~210, after the reference ends (t 147). Outside the metric; check it is just the end of the demonstration. |
| 6_success | (none) | - | - | frame | - | no episode of this rollout matched; try more envs (--num_envs) or another seed | 1 of 96 evaluation episodes (seed 103) passed the strict criterion. What does a passing lift look like compared with clip 2 (off_path)? |
