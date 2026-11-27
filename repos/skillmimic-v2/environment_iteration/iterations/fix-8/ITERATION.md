# fix-8

Parent: fix-6 | Created: 2026-11-28 | Summary: state sync (U1): reward, object termination and own-start capture use the post-step simulator state and its reference frame; fix-6 environment otherwise
Run: `bash environment_iteration/scripts/train.sh fix-8` (human, ~8 h; resumable). Everything below is filled in
by the agent unless marked (human).

## 1. Changes vs fix-6

Fix-8 implements U0 (instrumentation) and U1 (state sync) of the human's brief
`TalkFile_3DV_Fix6_환경_업데이트_분석_및_실행계획.md` (2026-11-28, sections 4, 5, 11, 13: "narrow the next update to U0 +
U1"). Parent is **fix-6** (best so far), not fix-7: fix-7's bundled reward change is not carried over. Reward weights
(`op: 30`, `or: 0`), physics, ATS, own-start settings, observation, episode length and PPO are fix-6's. One coherent
change set, opt-in through one env.yaml key.

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| U0 | probe of what each part of a control step reads (state index, reference frame, phase, history row) + the loaded config (termination height, asset, book mass / inertia / shape props, solver) | `src/probes/state_timing.py` (new, read-only) | the brief's §3.2 / §5.1 questions: declared vs effective values, reward timing | run on fix-6 and fix-8 (§3) |
| U1 | `stateSync: True`: the reward, both early terminations and the own-start capture are computed on the refreshed post-step state s_{k+1} and its reference frame sframe+k+1 (previously the pre-refresh state s_k and frame sframe+k); capture filter, stored state and stored frame come from the same snapshot | task `SkillMimicParahomeIterStateSync` (`src/tasks/state_sync.py`, subclass of fix-6's `SkillMimicParahomeIterOwnStart`), `env.yaml` `stateSync` | fix-6 pickup imprecision (off_path 86% @68, t45-85 peak error 17.2 cm) under a reward that credits each action one step late; own starts restored one frame behind their reference | `tests/test_state_sync.py`, state_timing probe, training-mode resume probe (§3) |

What the legacy order does (measured, §3): at control step k+1 the reward is `sim(ref[sframe+k], s_k)`. The pairing
(state after n actions vs reference frame sframe+n) is right, but s_k does not depend on the action a_k just taken:
every reward arrives one step late. The first reward after a reset reads the previous episode's rigid bodies
(root 86 cm off, reward 3e-8). The fall termination reads s_{k+1} and the book termination s_k. The own-start capture
filters on s_k and stores s_{k+1} labelled frame sframe+k, so own starts were restored one frame behind their
reference. With `stateSync` the reward of step k+1 is `sim(ref[sframe+k+1], s_{k+1})`: same pairing, no delay.

Kept fixed on purpose (details in the `state_sync.py` docstring):
- policy input: obs_buf, history rows and the phase value are untouched (checkpoints of fix-6 load and act
  identically).
- number of rewards per episode and the episode length. The last state of an episode (index E) has no row in
  `hoi_data_batch`. Its reference comes from the motion file (frame sframe+E). At the clip end it holds the last
  frame (146), which shifts the reference by one frame for that one state.
- velocity-smoothness term: dof velocities of s_1 are still zeroed (as before), and `_hist_obs` gets the reset
  state's dof velocities at a reset (legacy had the same values there via its stale s_0 row).
- the fix-1 book termination keeps its `progress_buf > 1` guard, so it first checks s_3 (legacy s_2), which is also
  where the fall check starts.
- evaluation (protocol v1): outside `post_physics_step`, `_curr_ref_obs` keeps its legacy value. `src/rollout.py`
  traces (ref_obj, ref_key_pos) therefore read the same frames as for every earlier iteration. Only the trace's
  `reward` field follows the new timing.
- effective termination height: env.yaml says `terminationHeight: 0.25`, but upstream never reads it.
  `HumanoidWholeBody._build_termination_heights` hard-codes 0.30 m, measured as the loaded value for fix-6 and fix-8.
  Fix-8 keeps 0.30 m (brief §3.2 item 2) and adds a comment in env.yaml.

Not in fix-8 (separate candidates per the brief): reset first-observation / history restore (U2), start-distribution
changes (U3), extra observations (U4), contact / solver / geometry (U5), initial book pose noise (U6).

## 2. Hypothesis

Immediate credit for the action that moved the book should tighten the pickup that fix-6 already does. Reward
weights and physics are unchanged, so any difference comes from the timing. Compare epoch 3000 against fix-6 in the
96-episode clean stochastic cell. Supported if all of these hold (the brief's §12.3 criteria):

- median per-episode peak t45-85 book error clearly below fix-6's 17.2 cm (≤ 15 cm), with the median first live
  10 cm excursion at t67 or later.
- lift ≥ 95% and fell in clip ≤ 15%, i.e. fix-6's carry walk is kept (walk_up stays near 90%; a root-only lunge as
  in fix-7 does not count as progress).
- episodes with whole-clip max error ≤ 20 cm and no fall above fix-6's 27.1%.

Falsified if the peak pickup error stays ≥ 17 cm, or if lift or the walk is lost (lift < 95% or fell in clip > 15%).
Strict task success (Wilson CI), paper success and the perturbed cell are reported alongside. Single training seed:
a difference of a few points in a 96-episode rate is within re-evaluation noise.

## 3. Validation before training (agent, each < 30 min)

- [x] `bash environment_iteration/scripts/check.sh`: all CPU checks pass, including the new
  `tests/test_state_sync.py` (5 tests). They cover: the reference row and frame per state index, the tail row inside
  the 60-frame cap and at the clip end, the reward seeing the synced reference while `_curr_ref_obs` is legacy after
  the step, the capture frame / error, delegation when `stateSync` is off, and skipping the pre-refresh build.
  `fingerprint.py diff` for fix-5/6/7 shows only `src/tasks/__init__.py` changed and `state_sync.py` added. This is
  the same additive registry change fix-7 made when it added `obj_rot.py`. None of those runs is resuming.
- [x] U0 probe, fix-6 e3000 policy, 8 clean stoch envs, seed 101, 150 steps
  (`probes/state_timing_fix-6_e3000/state_timing.json` in fix-6 and fix-8).

  | read by | fix-6 env (legacy) | fix-8 env (`stateSync`) |
  |---|---|---|
  | reward input state at stored step t | s_t (previous refresh: 0.00 cm from it) | s_{t+1} (0.00 cm from the refreshed book) |
  | reward book vs this step's refreshed book | median 0.77 cm, t45-85 median 3.5 / max 5.7 cm | 0.00 cm |
  | reward reference frame | t+2 | t+3 (= rollout `ref_frame` label) |
  | rollout trace reference (`_curr_ref_obs`) | t+2 | t+2 (unchanged) |
  | phase observation x clip length | t+2 | t+2 (unchanged) |
  | first reward after reset (t=0) | 3e-8 (root of the previous episode, 86 cm off) | 0.971 |
  | reward at t = 1, 2 | 0.971, 0.953 | 0.953, 0.915 (fix-6's values one step earlier) |
  | history row appended vs refreshed state | 0 | 0 |

  Effective values (both envs): termination height 0.300 m (env.yaml 0.25), asset
  `mocap_parahome_boxhand_histphase.xml`, obs size 1031, book 1.001 kg, inertia diag (4.28, 5.18, 8.03)e-3 kg m²,
  one shape, friction 1.0, contact offset 0.02, rest offset 0. Humanoid: 61 bodies, 43.4 kg, all shapes friction
  1.0 / contact offset 0.02. Sim: substeps 2, TGS, 10 / 0 iterations.
- [x] Training-mode probe (scratch, not kept; log excerpt `probes/train_resume_fix6_e3000.log`): fix-6's e3000
  full state resumed in the fix-8 env, 2,048 envs, epochs 3000 → 3061. No errors. The buffer filled in < 1 rate
  interval, own starts were 19.8-19.9% of resets, and captured frames were 75-95 (median 84, book error median
  5 cm). Restore max error was 7.6e-6. Reference-row check was 0 (every reward reference equals the motion-file row
  of its recorded frame). Own-capture check was 0 (stored error = distance recomputed from the stored book and
  stored frame). The legacy reward-input lag that the sync removes was book 2.1 cm mean / 14.8-27.5 cm max, root
  1.3 cm mean, over all training steps. Mean episode reward 30.8-32.9 (fix-6 at e3000: 31.1). Throughput: 6,200
  fps total ≈ 10.6 s/epoch, vs ~6,700 at the end of fix-6 and ~6,300 for fix-7, so **~9-10 h for 3,000 epochs**.
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-8` passed (scratch `.check/20261128_014611`). The
  pipeline was interrupted at epoch 10, resumed to 20, ran eval + analysis, skipped training on re-run, rendered 2
  clips and recorded the fingerprint. CPU part: all pass.

Findings for the human (not changed in fix-8, decision needed):
1. **Evaluation pairing (protocol v1).** The rollout trace stores the book after t+1 actions against reference frame
   t+2 (HUD label t+3). That is one frame behind the pairing the reward uses, in every iteration. The strict 10 cm
   metric is therefore measured against the previous reference frame. Correcting it would be protocol v2 and needs a
   re-evaluation of every iteration (AGENTS.md 1.5, brief §5.3).
2. **Phase observation.** For the state s_{t+1} the phase says frame t+2, again one behind. At a reset the phase
   equals the state's frame, so the first two observations of an episode carry the same phase. Unchanged in fix-8
   (input format kept). A candidate for a separate change.
3. **`terminationHeight` is dead config** (effective 0.30 m). Making the code follow the YAML (0.25 m) would change
   the fall definition and is a separate behaviour change.

## 4. Results (from analysis/e3000/REPORT.md)

| metric | fix-6 | fix-8 |
|---|---:|---:|
| task success (stoch, 96) | | |
| task success (perturbed) | | |
| paper success | | |
| lift success | | |
| task progress | | |
| fell in clip | | |
| dominant failure (mode @ frame) | | |

## 5. Failure analysis (agent)

## 6. Videos

clips.json -> `video/fix-8/...` (see INDEX.md there). Recommended clips and what to look for:

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
