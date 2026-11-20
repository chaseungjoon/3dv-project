# AGENTS.md: rules for LLM agents in the environment-iteration loop

Read `LOOP.md` (what the loop is) and `STATUS.md` (where it is now) before doing anything. All paths below are
relative to `repos/skillmimic-v2/`. Python is `.venv/bin/python` (Isaac Gym, torch sm_120 build).

## 1. Hard rules

1. **Never run anything that takes longer than 30 minutes.** Training (`scripts/train.sh`, `launch.py train`,
   `skillmimic/run.py` without `--test`, `run_paper_fix_full.sh`, `run_all.sh`, ...) is run by the human. Write
   the code, validate it with the short checks in §6, and end your turn with the exact command for the human,
   e.g. `bash environment_iteration/scripts/train.sh fix-3` (in tmux). If unsure about a duration, measure a
   small piece first and extrapolate. Never start a long job "in the background" to get around this.
2. **One GPU job at a time.** Before any GPU command, check that no training is running:
   `bash environment_iteration/scripts/status.sh` (it says `GPU busy`), or `nvidia-smi`. The scripts refuse GPU work
   while `train.sh` holds the lock; do not override that with `EI_FORCE_GPU=1`.
3. **Do not change what an existing iteration depends on.** Never edit `skillmimic/` (upstream) in a way that changes
   behaviour, an earlier iteration's `env.yaml` / `iteration.json`, a task class an earlier iteration uses, or an
   existing asset file. New behaviour = new env.yaml key with the old default, new asset file, or new class in
   `src/tasks/` (§5). Check with `.venv/bin/python environment_iteration/src/fingerprint.py diff <it>`.
4. **Never edit files a running training depends on** (its fingerprint makes the resume refuse, which costs the
   human hours). If a training is running, only write docs or prepare the next iteration's own files.
5. **Do not change `protocol.json`** (budget, evaluation cells, target). If the human decides to change it: bump
   `version`, re-evaluate every iteration (`eval.sh <it> --force`), and record it in STATUS.md's decision log.
6. **Never delete or overwrite iterations, runs, evaluations or videos.** Never `rm` checkpoints. A failed idea
   stays as a documented iteration.
7. Results are reported as measured, including 0%. Say which epoch / cell / n a number comes from; with 96
   episodes, give the Wilson CI (REPORT.md has it). Do not call an 8-episode curve point a result.
8. The human directs what to change. You may propose changes (with the evidence), but implement the direction
   the human chose; if you think it is wrong, say so with data, then do what they decided.

## 2. Your jobs

1. create / modify source code of the loop (`src/`, `scripts/`) and of the iteration (`src/tasks/`, assets, env.yaml)
2. fix the environment according to the human's directions -> new iteration `fix-<n+1>`
3. test the code (`scripts/check.sh`)
4. validate (only runs under 30 min: `scripts/check.sh --gpu <it>`, short probes, `scripts/eval.sh`)
5. write results into the documentation and analyze them (ITERATION.md, STATUS.md)
6. generate videos of chosen frames (`clips.json` -> `scripts/video.sh`)

## 3. Step-by-step

### After the human finished training fix-n (`train.sh` ran eval + analysis itself)

1. `bash environment_iteration/scripts/status.sh`, confirm fix-n is `analyzed`. If it says `trained, needs eval.sh`,
   run `bash environment_iteration/scripts/eval.sh fix-n` (~6 min).
2. Read `iterations/fix-n/analysis/e3000/REPORT.md` (+ `object_path.png`, `failures.png`) and the parent's report.
   Compare: which failure mode / event shrank, moved later or earlier, what is new. Check the learning curve for
   instability (lift appearing and disappearing).
3. Fill `ITERATION.md` §4 (results table vs parent) and §5 (interpretation: what the numbers show, what they do not).
4. Choose 3-6 moments for video and write `iterations/fix-n/clips.json` (format: `src/render.py` docstring). Prefer
   the dominant first-failure mode at its median frame, the first late event (drop / fall), the perturbed cell's
   dominant mode, and anything that changed vs the parent. Each clip gets a `why` = what the human should look for.
5. `bash environment_iteration/scripts/video.sh fix-n` (~2 min). Look at `video/fix-n/.../sheets/*.png` yourself
   (Read tool) to check the clips show what you intended; fix clips.json and re-render if not.
6. Fill ITERATION.md §6 (which clips, where, what to look for), update STATUS.md (phase: "waiting for human video
   notes"), and tell the human where the clips are.

### After the human watched the videos and gave a direction

1. `bash environment_iteration/scripts/new_iteration.sh fix-(n+1) "<one-line change>"`.
2. Implement the change (§5). Write ITERATION.md §1 (each change -> the failure it targets) and §2 (hypothesis,
   and what result would falsify it).
3. Validate (§6). Record what you ran and the outcome in §3.
4. Update STATUS.md (current iteration, phase "ready to train", next action) and hand over:
   `bash environment_iteration/scripts/train.sh fix-(n+1)` (tmux, ~8 h, resumable by re-running).

## 4. Reading results correctly

- Every number comes from the fixed protocol: epoch 3000, clean frame-2 starts, 96 stoch episodes per cell.
  Numbers from `experiments/tools/evaluate.py` (earlier documents) include ~7% noisy-start episodes and are
  not directly comparable (protocol.json note).
- Evaluation is not bit-reproducible (GPU PhysX); 96-episode rates move by a few points between re-runs. The
  8-env curve points can be off by ~30 points; use them only for trends.
- `task_success` is strict (10 cm at every frame). Look at task progress, held frames, the median path table and
  the event frames to see partial progress.
- The failure classifier is a heuristic; when a mode looks surprising, look at the video before believing it.
  Changing `src/metrics.py` failure logic is allowed (it does not change the evaluate.py metrics; keep
  `tests/test_metrics.py` passing); re-run `scripts/analyze.sh` for every iteration afterwards (CPU, it
  recomputes events from the stored traces).

## 5. How to implement an environment change

Where things live:

| what | where |
|---|---|
| env config of the iteration | `iterations/fix-n/env.yaml` (copied from the parent; edit freely) |
| run.py flags, task class, asset | `iterations/fix-n/iteration.json` (`task`, `asset`, `common_args`, `train_args`, `extra_paths`) |
| new task behaviour | new module in `src/tasks/` + `REGISTRY` entry (see `src/tasks/__init__.py`), subclass the parent's class |
| new object / scene asset | new file under `skillmimic/data/assets/...` with an iteration suffix (e.g. `book_fix3.urdf`); never edit an existing one |
| object geometry used by the failure analysis | `iteration.json` `"object_box"` (8-vertex box .obj), default the fix-1 book box |
| files the task reads that are not covered by the fingerprint | add them to `iteration.json` `extra_paths` |

Upstream code reads env cfg keys with `cfg['env'].get(...)`; existing `[3dv]` keys you can reuse:
`staticBoxes`, `dynamicUrdf`, `dynamicVhacd`, `histLayoutFix`, `objTerminationDist`, `ballDensity` (object
density), `plane` friction, `sim.physx.*`. Where the upstream class has no hook, subclass in `src/tasks/` and
override the method; keep the parent's behaviour when your new key is absent.

Useful upstream facts: `skillmimic/env/tasks/skillmimic_parahome_localhist.py` (fix-1 task
`SkillMimicParahomeLocalHistRISBuffernodePhase`: phase obs, object early termination),
`humanoid_object_task.py` (scene / object loading, `_build_target`), `skillmimic_parahome.py` (reward
`compute_humanoid_reward`, reset). Earlier diagnosis notes: project root `SKILLMIMIC_NOTES.md`.

## 6. Validation (all < 30 min)

| check | command | time |
|---|---|---|
| unit / regression | `bash environment_iteration/scripts/check.sh` | ~15 s |
| end to end on GPU | `bash environment_iteration/scripts/check.sh --gpu fix-n` (scratch folder: 20-epoch train, Ctrl-C, resume, auto eval + analysis, video) | ~5-8 min |
| the env builds and the change acts | a short probe: `src/rollout.py fix-n --epoch <e> --num_envs 8` on a parent checkpoint when obs/action sizes are unchanged, or a custom script in your scratch dir | minutes |
| evaluation of a finished run | `bash environment_iteration/scripts/eval.sh fix-n` | ~6 min |
| can the env physically do the demo? | `src/probes/grip.py fix-n --frames 90,110,130` (perfect imitator holds the object at reference frames; `--variants` at actor creation, `--set 'sim.substeps=4;...'`, `--asset`, `--motion`, `--book_urdf`), `--replay 30:147 [--detail 46:72]` (perfect imitator over the clip) | ~1 min per call |
| feet / hands / book timeline of a policy | `src/probes/gait_grip.py fix-n --start 2,85 --seeds 101,102,103` (foot contact, lift-offs, steps, slip, travel along the walk, pelvis, release / slip / drop order; `--assist a --assist_from f`: upward book-load assist a*m*g, diagnostic only) | ~20 s per start |
| videos | `bash environment_iteration/scripts/video.sh fix-n` | ~1-3 min |

A parent checkpoint only loads into fix-n when the observation and action sizes are unchanged (an asset or
observation change breaks this; then validate with `check.sh --gpu`, which trains from scratch).

## 7. Documentation duties

- `iterations/fix-n/ITERATION.md`: all sections except §7 (human notes). Keep numbers traceable to
  `analysis/e3000/REPORT.md`.
- `STATUS.md`: the hand-written part (current iteration, phase, next action, decision log). The results table
  between the AUTO-STATUS markers is regenerated by `src/status.py` (every script calls it); never hand-edit it.
- Auto-generated files (`analysis/*/REPORT.md`, `video/*/INDEX.md`) are never edited by hand.
- Language: these loop documents are in English; the human may write notes in Korean.

## 8. Pitfalls already found

- **Test the physics before training.** A change that should make the demonstration feasible must first improve
  `src/probes/grip.py` (held counts over >= 24 env-frames; single frames and 2-env means are noisy). fix-1's solver
  could not hold the demonstrated grip at all (0/56).
- **Check side effects of geometry changes**: thicker finger capsules "held" the book only because neighbouring
  capsules jammed into each other (~300 N in free air). Measure free-air contact forces of the hand after any hand
  geometry change.
- **Shape / body / DOF properties must be set at actor creation** (`_build_env` / `_build_target`); changes after
  `prepare_sim` silently have no effect on the GPU pipeline (env_probe.py PROBE_SHAPE gave identical numbers for
  every value). `src/tasks/contact.py` shows the pattern.
- **Observation size is chosen by a substring of the humanoid asset file name**
  (`skillmimic/env/tasks/humanoid_task.py` `_setup_character_props`): a new hand asset must keep
  `mocap_parahome_boxhand_histphase` in its name. Isaac Gym also rejects asset names with an extra dot
  (`..._r0.009.xml` -> "Unrecognized extension").
- **Joint values are per-joint exp maps** (3 values per joint), both in the motion file and in Isaac Gym
  (`src/tools/hand_fk.py`, verified to 0.0 deg). Not sequential x-y-z hinge angles.
- The paper disables the contact-graph reward for ParaHome (`r_cg = 1`); the reward weights in env.yaml are the
  paper's. Changing rewards is a method change, not an environment fix.
- `sim.substeps` costs training time roughly linearly in the physics share (fix-2: 4 substeps -> 1.5x per epoch).
  Measure s/epoch with a 40-epoch scratch run at 2,048 envs (`EI_ITER_DIR=<scratch> .../launch.py train ...`) and
  report it before handing over a training.
- zsh does not split `$VAR` into words: in loops over optional arguments use arrays (`X=(--set "$s"); cmd "${X[@]}"`),
  otherwise the probe receives one wrong argument and prints nothing.

- Upstream applies `state_init_random_prob` (STF noisy init, 0.1 in env.yaml) also at test time. The loop's
  evaluation forces 0 (`src/rollout.py`). Training keeps 0.1, as the paper.
- Isaac Gym root / object tensors are env-local; `set_camera_location` takes env-local coordinates; the camera view
  matrix works in global coordinates (local + `get_env_origin`).
- Camera sensors need `BaseTask(enable_camera_sensors=True)` (render.py patches the default) and must be created
  inside `_build_env`, before `prepare_sim`.
- Background jobs started from a non-interactive shell ignore SIGINT; to test Ctrl-C behaviour, launch with
  SIGINT reset to default in a new process group (see `src/tests/pipeline_check.py`).
- An Isaac Gym process can build only one simulation reliably: one environment per process (why eval.sh calls
  evaluate.py twice).
- `experiments/tools/evaluate.py` and the paper-budget scripts belong to earlier studies; the loop does not use
  them (except `ckpt.py` / `prune.py` helpers) and does not modify them.
