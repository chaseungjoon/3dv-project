# Environment-iteration loop

Goal: reproduce the SkillMimic-V2 baseline on ParaHome `place_book` (paper: 100% success) by fixing the
**environment**, one iteration at a time, until the policy reaches **≥ 70% task success** under the fixed protocol
below.

## 1. Why this loop

- The full-budget study (`TRAINING_BUDGET_RESULTS.md`, project root) trained PAPER and FIX for the paper's
  1 B samples (15,260 epochs). Both stayed at 0% task success, and the epoch-3000 results already showed the same
  picture. **The training budget is not the bottleneck; the environment is.**
- So the budget is now fixed at **3,000 epochs** and every iteration changes the environment only, guided by
  where and how the previous policy fails.
- `FIX` from that study is the starting point and is imported here as **fix-1**.

## 2. The loop

```
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │                                                                             │
        ▼                                                                             │
 1. TRAIN fix-n            human     scripts/train.sh fix-n   (~8 h, resumable; auto-runs step 2a)
 2. ANALYZE                agent     a) scripts/eval.sh fix-n  (auto after training, ~6 min)
                                     b) read analysis/e3000/REPORT.md, write ITERATION.md §4-5
                                     c) choose failure moments -> iterations/fix-n/clips.json
 3. VIDEO                  agent     scripts/video.sh fix-n    (~2 min) -> video/fix-n/.../INDEX.md
                           human     watch the clips, write ITERATION.md §7 (what you saw) + direction
 4. FIX -> fix-(n+1)       agent     scripts/new_iteration.sh fix-(n+1) "...", change the environment
                                     as the human directed, validate (scripts/check.sh [--gpu]),
                                     write ITERATION.md §1-3, update STATUS.md
        │                                                                             │
        └──────────────────────────── human starts step 1 again ──────────────────────┘
```

Stop when the target is reached (STATUS.md says `TARGET REACHED`), then confirm it with extra training seeds
(protocol `train.seed` 1 and 2, a separate decision).

Who does what:

| | human | agent (LLM) |
|---|---|---|
| training (> 30 min) | runs `train.sh` in tmux | never runs it; hands over the exact command |
| evaluation / analysis | – | runs `eval.sh` / `analyze.sh`, interprets, documents |
| videos | watches, writes notes, decides the direction of the next fix | picks the moments, renders them |
| environment changes | decides **what** to change | implements, validates, documents |

## 3. Fixed protocol (`protocol.json`, version 1)

Identical for every iteration; changing it invalidates every comparison (see AGENTS.md 4).

| item | value |
|---|---|
| clip | `place_book` (150 frames, book lifted from the desk and carried ~1.7 m sideways) |
| training | 3,000 epochs x 2,048 envs x 32 steps = 196.6 M samples, seed 0 (~8 h on the RTX 5070 with fix-1's solver, ~12 h from fix-2 on) |
| learning curve | det, 8 envs, every 250 epochs |
| final evaluation | epoch 3,000; stoch, 32 envs x eval seeds 101/102/103 = 96 episodes; same with the paper's eps-perturbed start (book yaw ±45°, xy ≤ 10 cm) |
| start state | clean reference frame 2 for every env (`state_init_random_prob = 0` in evaluation; see the note in protocol.json) |
| **target** | **task success ≥ 70%** in the plain stoch cell |

Metrics (same definitions as `experiments/tools/evaluate.py`; `src/tests/test_metrics.py` proves it on 1,024
stored episodes):

- **task success**: the book stays within 10 cm of the reference book at every clip frame and the character does
  not fall in the clip. This is the strict per-frame criterion all earlier results used.
- paper success: the upstream metric (> 60 frames with the book above 0.9 m near a wrist, root up).
- lift success, task progress (fraction of frames before the first 10 cm break), held frames, fell in clip.

Frame convention: **frame t = control step t after the reset at reference frame 2** (t = 0..147). The video HUD
also shows the reference frame number (t + 3).

## 4. Failure analysis (`src/metrics.py`, `src/analyze.py`)

For every evaluation episode the analysis finds the frame the task first breaks (`first_bad`) and classifies why:

| mode | meaning |
|---|---|
| fall | the character fell while the book was still on track |
| no_reach | no wrist came within 0.30 m of the book |
| no_lift | hands reached the book but it stayed at its start pose while the reference lifted it |
| knock | the book moved (pushed, slid, tipped) without being in hand |
| drop | the book was in hand, left the hands, then left the track |
| lag / lead | in hand but behind / ahead of the reference along its path |
| off_path | in hand, on time, but displaced from the path (wrong height / place) |

"In hand" = book centre ≥ 2 cm above its start with a wrist within 0.30 m (a pivot grip with one edge on the
desk counts, because the demonstration pivots the book up that way). Events per episode: `grasp`, `off_desk`
(lowest corner ≥ 2 cm up), `rot_off` (> 45° from the reference orientation), `drop`, `floor`, `fall`,
`track_lost`. The report gives their frequencies and median frames, the median book path against the reference,
the learning curve, and representative episodes per mode (clip candidates).

The reference itself (useful when reading any result): the book rests until frame ~53, is pivoted up on one
edge and rotated (tilt 41° at frame 60, ~90° at 72, ~180° at 90) while rising 46 cm by frame 80, and from frame
~85 the person walks ~1.1 m sideways with it.

## 5. Videos (`src/render.py`)

GPU PhysX is not bit-reproducible, so evaluation episodes cannot be replayed. A video run re-rolls the same
policy / cell / seed with cameras, re-detects the events in that rollout and cuts the requested moments. Clips are
therefore requested by **mode, event or frame window** in `iterations/fix-n/clips.json`, never by evaluation env
index. Each video: wide fixed view + close-up following the book, the reference book as a green wireframe, the
reference key bodies as green dots, and a HUD with live numbers, status and an event timeline. Clips are slow
motion (0.5x); full episodes are real time.

## 5b. Physics probes (`src/probes/grip.py`)

Before an 8-12 h training, an environment change is checked with a **perfect imitator**: the humanoid's root is
pinned to the reference and every joint is PD-driven to the reference pose. Static holds (book placed in the hand
at a reference frame) and a whole-clip replay show whether the environment can physically execute the
demonstration at all. fix-1's environment could not (0/56 holds); fix-2's solver settings make the carry grip
holdable (15/56). The pickup (frames ~50-67) is not executable open-loop in either, so a policy must find its own
grasp there.

## 6. Iteration rules

1. One iteration = one coherent environment change set, motivated by the previous iteration's failures (written
   in its ITERATION.md §1 with the failure it targets). Prefer few changes per iteration so effects stay
   attributable; when several are bundled, say so.
2. An iteration never edits upstream files under `skillmimic/` or an earlier iteration's files. New behaviour is
   an env.yaml key, a new asset file, or a task subclass in `src/tasks/` (AGENTS.md 5).
3. The protocol is fixed. Only the environment (scene, object, physics, termination, observation the env
   provides, reference handling) changes. PPO hyperparameters and reward weights are not part of this loop unless
   the human explicitly decides so (then note it as a deviation in ITERATION.md and STATUS.md).
4. Every result, including bad ones, stays: iterations are never deleted or overwritten.

## 7. Map

```
environment_iteration/
  LOOP.md  STATUS.md  AGENTS.md  README.md  protocol.json
  iterations/fix-<n>/
    iteration.json   definition (task class, asset, env config, run.py flags, parent)
    env.yaml         this iteration's environment config
    ITERATION.md     changes, hypothesis, validation, results, analysis, videos, human notes, decision
    clips.json       which moments to render
    runs/            checkpoints + trainer logs (git-ignored; fix-1 links to experiments/runs_paper_fix/)
    eval/            evaluation results (.json committed, .npz traces git-ignored)
    analysis/e<ep>/  REPORT.md, summary.json, episodes.csv, plots
    logs/            train.sh logs
  video/fix-<n>/e<ep>_<cell>_s<seed>/   INDEX.md, sheets/*.png (committed); full/, clips/ mp4 (git-ignored)
  scripts/   train.sh eval.sh analyze.sh video.sh new_iteration.sh status.sh check.sh
  src/       iterlib launch rollout evaluate metrics analyze render status new_iteration fingerprint
             tasks/ (iteration task classes)  tests/ (metric regression, pipeline check)
```
