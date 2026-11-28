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

## 4. Results (epoch 3000)

From [analysis/e3000/REPORT.md](analysis/e3000/REPORT.md) and the fix-6 / fix-7 reports. 96 episodes per cell.

| metric | fix-6 | fix-7 | fix-8 |
|---|---:|---:|---:|
| task success, clean (95% Wilson CI) | 1.0% (0–6%) | 0.0% (0–4%) | **0.0% (0–4%)** |
| task success, perturbed (95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) | 0.0% (0–4%) |
| paper success, clean | 93.8% | 15.6% | **6.2%** |
| lift success, clean | 100.0% | 77.1% | 90.6% |
| task progress, clean | 48.4% | 47.7% | 43.6% |
| fell in clip, clean | 10.4% | 93.8% | **99.0%** |
| dominant first failure, clean | off_path 86% @ t68 | off_path 91% @ t67 | off_path 90% @ **t64** |
| late events, clean (median frame) | drop 92% @190, fall 100% @210 | drop 61% @126, fall 96% @130 | drop 96% @125, fall 100% @126 |

§2 criteria (clean stoch; `book_tracking.py fix-6 fix-8`, p10 / p50 / p90):

| t45–85 | fix-6 clean | fix-8 clean | fix-6 perturbed | fix-8 perturbed |
|---|---|---|---|---|
| max 3D error (m) | 0.080 / 0.172 / 0.266 | 0.172 / **0.246** / 0.335 | 0.112 / 0.477 / 0.810 | 0.172 / 0.448 / 0.833 |
| peak vertical lead (m) | 0.025 / 0.094 / 0.149 | 0.009 / 0.086 / 0.141 | -0.002 / 0.005 / 0.121 | -0.002 / 0.007 / 0.119 |
| signed along-path error (m) | 0.003 / 0.031 / 0.176 | 0.008 / 0.062 / 0.241 | -0.292 / 0.005 / 0.135 | -0.176 / 0.031 / 0.187 |
| first >10 cm while alive (t) | 63.4 / 67 / 90.2 | 62 / **64** / 67.5 | 48 / 64 / 73 | 47.8 / 63 / 68 |
| final 3D error, survivors (m) | 0.136 / 0.210 / 0.306 (86) | 1.013 (1) | 0.205 / 0.462 / 1.782 (70) | 1.021 (1) |

| §2 criterion | target | fix-8 | met? |
|---|---|---:|:---:|
| median peak t45–85 error | ≤ 15 cm | 24.6 cm (fix-6 17.2) | no (worse) |
| median first live 10 cm excursion | ≥ t67 | t64 | no (earlier) |
| lift | ≥ 95% | 90.6% | no |
| fell in clip | ≤ 15% | 99.0% | no |
| whole-clip max ≤ 20 cm, no fall | > 27.1% | 0% | no |

Additional measures (agent, from the stored epoch-3000 traces with `metrics.body_outcome` / `metrics.rot_err_deg`;
carry / lift errors = median over episodes of the per-episode median over live frames, t85–147 / t55–80; "pelvis
ahead" = root minus mean ankle position along the reference walk direction at t120; reward = trace `reward`, whose
timing differs between fix-6 (legacy) and fix-8 (synced), so compare it only roughly):

| metric (clean stoch, 96 episodes) | fix-6 | fix-7 | fix-8 |
|---|---:|---:|---:|
| walk_up / fall_fwd | 90% / 10% | 0% / 94% | **1% / 96%** |
| median root / feet progress along the walk | 0.77 / 0.70 m | 1.12 / 0.34 m | 0.88 / 0.22 m |
| median pelvis ahead of the ankles at t120 | -0.02 m | 0.30 m | 0.38 m |
| median in-clip fall frame (episodes) | 134 (10) | 130 (90) | 126 (95), p25–p75 124–128 |
| median carry position error, t85–147 | 14.4 cm | 15.2 cm | **30.7 cm** |
| median carry / lift rotation error | 37° / 29° | 18° / 23° | 44° / 26° |
| within 10 cm at t 70 / 90 / 110 / 130 | 27 / 42 / 33 / 16% | 23 / 19 / 17 / 0% | 2 / 0 / 0 / 0% |
| whole-clip max ≤ 20 / 30 cm, no fall | 27 / 76% | 0 / 0% | 0 / 0% |
| mean reward per step t50–70 / t85–119 | 0.757 / 0.502 | 0.669 / 0.273 | 0.678 / 0.151 |
| perturbed: lift / walk_up / fell in clip | 54% / 73% / 27% | 47% / 2% / 70% | 52% / 1% / 99% |
| perturbed: no_lift / knock (first failure) | 31% / 7% | 19% / 16% | 24% / 9% |

Learning curve (det, 8 envs, trend only), `body_outcome` per curve checkpoint:

| epoch | 250–750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250–3000 |
|---|---|---|---|---|---|---|---|
| fix-6 | stand 8 | walk_up 8 | fall_fwd 8 | fall_fwd 8 | fall_fwd 8 | **walk_up 8** | **walk_up 8** |
| fix-7 | stand 8 | stand 8 | fall_fwd 8 | walk_up 7, fall_fwd 1 | fall_fwd 8 | fall_fwd 8 | fall_fwd 8 |
| fix-8 | stand 8 | fall_fwd 5, walk_up 3 | fall_fwd 8 | fall_fwd 4, stand 3, walk_up 1 | fall_fwd 8 | fall_fwd 8 | fall_fwd 8 |

Lift appears at e1000 (87.5%) and stays at 75–100%. Paper success peaks at e1500 (87.5%, 8 envs) and is ~0 from e1750.
Single training seed.

## 5. Failure analysis (agent)

Hypothesis (§2): **falsified.** Every criterion moved the wrong way: the pickup is looser (peak error 24.6 vs
17.2 cm, first break t64 vs t67), and fix-6's carry walk is gone (walk_up 90% → 1%, fell in clip 10% → 99%).
Fix-8 is the worst of fix-6/7/8 on every clean measure except lift and rotation.

What the numbers show:

1. **Fix-8 has fix-7's lunge, without fix-7's reward change.** The ending is the same forward fall: fall_fwd 96%
   at t126 (p25–p75 124–128), pelvis 0.38 m ahead of the ankles at t120, feet 0.22 m along the walk while the root
   reaches 0.88 m. Contact sheet `2_carry_walk`: the knees bend into a kneel, the right arm reaches the book out
   toward the green wireframe, then the body topples. 16/16 fresh rollouts on seed 101 are fall_fwd at t122–127.
   Fix-7's §5 attributed the lunge to `op` 60 + rotation reward. Fix-8 keeps fix-6's weights (`op` 30, no rotation)
   and still lunges, so that attribution does not hold on its own.
2. **The lunge is a shared intermediate stage of training.** All three runs are fall_fwd on the 8-env curve at
   e1250–1750. Fix-6 left it for walk_up at e2000 and stayed there. Fix-7 and fix-8 did not leave it within 3,000
   epochs. One seed per configuration cannot say whether fix-6's escape is caused by its configuration or by luck in
   that run. The difference between fix-6 and fix-8 is a one-step shift in reward timing, which should matter little
   to PPO (γ = 0.99, GAE). That makes run-to-run variance a serious candidate for the whole 90% → 1% walk change.
3. **The pickup also got worse, but less dramatically.** The first break is earlier (t64 vs t67). The book is ahead
   along the path (median +6 cm vs +3) and laterally off (p10 −11 cm vs −4), while the vertical lead is the same
   (8.6 vs 9.4 cm). Only 2% are within 10 cm at t70 (fix-6 27%). The `op` 30 position factor over t55–75 is lower
   (median 0.88 vs 0.96). Immediate credit did not tighten the lift.
4. **Carry error doubled before the fall.** Median 30.7 cm over t85–147 vs fix-6's 14.4 cm. The reward per carry step
   t85–119 is 0.15 (fix-6 0.50, fix-7 0.27), so the lunge here is not a reward-efficient carry, unlike fix-7, which
   kept the book near the reference (15 cm) while lunging. The policy is stuck in a poorer local optimum, not
   exploiting the reward.
5. **The perturbed cell is unchanged at the pickup** (lift 52% vs 54%, no_lift 24% vs 31%, knock 9% vs 7%). It
   inherits the lunge afterwards (fell in clip 99%).
6. **What the numbers do not show:** whether fix-6's configuration reliably produces the walk (no second seed
   exists for any iteration). Also unknown: whether the state sync itself is neutral, harmful or helpful once the
   walk is found. The protocol findings in §3 (evaluation pairs the book with the previous reference frame; phase
   lags one frame) apply equally to fix-6 and fix-8, so they do not explain the difference.

Suggestion for the human's decision (§8; evidence above, not implemented): before attributing further reward or
environment changes, measure run-to-run variance. Retrain fix-6 unchanged with `train.seed` 1, ideally also fix-8
seed 1. If fix-6 seed 1 also lunges, fix-6's walk was a lucky escape, and later iterations need a change that
targets the e1250–1750 lunge stage directly (e.g. a balance / centre-of-mass or foot-progress term, or starts that
put the policy into the walk). If fix-6 seed 1 walks, the state sync is the likely cause and fix-9 should drop it.
A second seed is a protocol-level decision (LOOP §2, protocol.json `train.seed`).

## 6. Videos

Rendered 2026-11-28 with `bash environment_iteration/scripts/video.sh fix-8` ([clips.json](clips.json)). These are
fresh rollouts, so env numbers do not match the evaluation. The 16-env clean rollout reproduces the evaluation:
16/16 fall_fwd (root 0.79–0.98 m, feet 0.17–0.28 m), falls at t122–127, 15/16 first break off_path at t62–67, 1 lead
at t62.

| clip | where | what to look for |
|---|---|---|
| 1_fall_fwd | `video/fix-8/e3000_stoch_s101/clips/1_fall_fwd__env0{0,1}__f08*-1*.mp4` | The dominant ending (96%, t~126). Same kneel / lunge as fix-7? Does any foot leave the ground before the fall? |
| 2_carry_walk | `.../e3000_stoch_s101/clips/2_carry_walk__env00__f078-135.mp4` | The whole carry. The right arm holds the book out ahead of the body and the knees fold (sheet: book error 0.27 → 0.45 m at t96–116). How many real steps happen? When does the upper body start to lead? |
| 3_lift_off_path | `.../e3000_stoch_s101/clips/3_lift_off_path__env0{0,1}__f04*-07*.mp4` | The pickup break (t63–64). Is the book lifted earlier or further forward than the green pivot? Compare with fix-6's `2_lift_off_path` (t66 / 77). |
| 4_walk_up | not rendered | No walk_up episode in the 16-env rollout (1% in the evaluation). |
| 5_perturbed_pickup | `video/fix-8/e3000_stoch_perturb_s101/clips/5_perturbed_pickup__env0{2,5}__f*.mp4` | no_lift on the displaced book (env 2 at t34, env 5 at t59). Does the hand go to the red book or to the green reference? |

For a direct comparison of the walk, watch next to fix-6's `video/fix-6/e3000_stoch_s101/clips/1_walk_up__env01__f078-147.mp4`.
Contact sheets: `sheets/*.png` in each folder. Full episodes: `full/envNN.mp4`.

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
