# fix-4

Parent: fix-1 | Created: 2026-11-19 | Summary: ATS actually enabled: --reweight now reaches the time sampler (upstream never passed it, so starts were uniform); fix-1 environment otherwise
Run: `bash environment_iteration/scripts/train.sh fix-4` (human, **~8.5 h**, 9.9 s/epoch measured, same cost as fix-1; resumable). Everything below is filled in
by the agent unless marked (human).

## 1. Changes vs fix-1

Direction: project root of this repo `FIX1_ANALYSIS.md` (human-supplied analysis of fix-1, added 2026-11-19),
section 3 ③ "ATS 연결 누락 의심" and section 6.3 ("학습 신호 문제라면: 물리 설정을 고정하고 ... 샘플링 연결을 각각 별도
실험으로"). Parent is **fix-1** (not fix-3): physics, assets, rewards, PPO and the run.py flags are fix-1's, so the
only difference is the start-frame sampler. `diff iterations/fix-1/env.yaml iterations/fix-4/env.yaml` = the one
`atsSampler` key; iteration.json differs only in `task`.

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | Adaptive trajectory sampling (ATS) actually used for training start frames: env key `atsSampler: True` hands run.py `--reweight` (already in fix-1's train_args) to `ParaMotionDataHandler.reweight`, and a working reweight sampler is bound to the handler | `src/tasks/ats.py` class `SkillMimicParahomeIterATS` (subclass of fix-1's `SkillMimicParahomeLocalHistRISBuffernodePhase`), registry entry, `env.yaml` `atsSampler`, `iteration.json` `task` | fix-1 @3000: every episode breaks at the lift onset (frames 65-68, lag 51% / off_path 25% / no_lift 16%) and collapses at the carry-walk transition (fall 100% @103). Uniform starts spend ~1/147 of the resets on each frame regardless of how badly it is imitated | runtime `[ats]` log lines in a 60-epoch scratch run (section 3) |

### The bug (two layers, both verified 2026-11-19)

1. **Flag never reaches the sampler.** `skillmimic/env/tasks/skillmimic_parahome.py` `_load_motion` builds
   `ParaMotionDataHandler(..., reweight_alpha=self.reweight_alpha)` without `reweight=`, so the handler keeps the
   default `reweight=False`. The task itself has `self.reweight = True` (from `--reweight`), so it records the
   per-start-frame reward (`record_motion_time_reward`) and recomputes `time_sample_rate` every 750 control steps
   (5 x 150 frames), but `ParaMotionDataHandler.sample_time` checks the handler's own flag and always takes the
   uniform branch (`np.random.randint(2, L-2+1)`). Every earlier run in this project (PAPER, FIX, fix-1 to fix-3)
   trained with uniform start frames, although the logs print "Reweighted time sampling rate".
2. **The reweight branch itself crashes.** Enabling the flag alone fails on the first reset:
   `possible_times = [np.arange(s, e + 1) for s, e in zip(start, end)]` with the int `start = 2`
   -> `TypeError: 'int' object is not iterable` (first fix-4 scratch run). The branch has never executed.
   `_ats_sample_time` replaces it (bound to the handler instance only when `atsSampler` is on): per motion it draws
   `2 + choice(L-3, p=time_sample_rate[m])`, the same support as the uniform branch (frames 2 .. L-2, index i =
   frame i + 2), and keeps upstream's `truncate_time` / `play_dataset` handling.

Sampling rule (upstream `_reweight_time_sampling_rate`, unchanged): `p = (1 - alpha) / (L - 3) + alpha *
softmax(-5 * r)`, `r` = running mean reward of episodes started at that frame, `alpha = --reweight_alpha = 1.0`
(fix-1's train_args). With per-step rewards of ~0-0.3 the hardest frame is started from up to ~4.5x as often as the
easiest (`exp(5 * 0.3)`); the measured early-training range was 0.0006-0.023 vs uniform 0.0068.

Not changed: the noisy-state init (STF, `state_init_random_prob 0.1`) still re-samples the start frame for 10% of
resets via `noisy_resample_time`, as in fix-1. Evaluation is unaffected: `src/rollout.py` starts every env at frame 2
(`--state_init 2`) and test runs do not pass `--reweight`. ATS state (`time_sample_rate`, `motion_time_seqreward`)
was already saved in checkpoints for resume (`get_train_state`), so a resumed run continues with the learned rates.

What fix-4 does **not** do: FIX1_ANALYSIS.md section 5 puts a physics diagnosis first (A/B: does the e15260 policy's
own grasp hold when the target freezes vs during the carry?). That diagnosis needs new instrumentation and was not
run; fix-4 tests the cheapest, independent item of section 6 (sampling, physics fixed) so the effect stays
attributable. The contact / solver line of fixes is fix-2/fix-3's branch.

Logging added for the analysis (FIX1_ANALYSIS.md: "ATS는 먼저 런타임의 활성화 값과 실제 시작 프레임 분포를 확인한다"): every rate
update prints `[ats] step N motion 0: sampler reweight=True p max .. min .. entropy .. top frames [..]`; every 20th
update also prints the histogram of the start frames actually used since the previous histogram (15-frame bins).
`grep '\[ats\]' iterations/fix-4/logs/*` after training shows where ATS concentrated.

## 2. Hypothesis

fix-1's two bottlenecks are short windows of the clip (lift onset ~frames 60-72, carry-walk transition ~85-105).
If they are imitated worst, ATS starts more training episodes there (and the 60-frame episodes that start just
before them get more practice of the transition), so at epoch 3000 (stoch, 96 episodes) fix-4 should show:

- lift success clearly above fix-1's 47% (outside its CI), and the median first failure later than frame 66;
- fewer in-clip falls than fix-1's 96%, or the fall later than frame 103;
- task success > 0 would be a bonus; the 70% target is not expected from this change alone.

Falsified if lift success <= ~47% and the first failure / fall frames stay at 65-68 / ~103 (the same picture as
fix-1). Then the training signal from start-frame sampling is not the bottleneck at 3,000 epochs. Also check in the
log where ATS concentrated: if the top frames are not around 55-105 (e.g. early approach frames 7-23, as at step
750-1500 of the scratch run), the reward does not single out the pickup / carry frames, and the next change should
be a different signal, not more sampling.

## 3. Validation before training (agent, each < 30 min)

- [x] `bash environment_iteration/scripts/check.sh` (unit tests): ALL PASS. Fingerprint: fix-1 0 of 366 shared files
      differ (fix-1's behaviour untouched); fix-2 / fix-3 report only the registry line + the new `src/tasks/ats.py`
      (finished runs, nothing to resume).
- [x] Enabling only the flag (first scratch run): `TypeError: 'int' object is not iterable` in upstream
      `sample_time` on the first reset -> layer 2 of the bug (section 1), fixed by `_ats_sample_time`.
- [x] 60-epoch scratch run at 2,048 envs (`EI_ITER_DIR=<scratch> launch.py train fix-4 --seed 0 --epochs 60
      --num_envs 2048`, 10 min 20 s, exit 0, no traceback):
      - `[ats] atsSampler on: sampler reweight=True alpha=1.0` at start;
      - rate updates at steps 750 and 1500: `p max 0.0219 min 0.0006 (uniform 0.0068) entropy 0.913 top frames
        [8, 12, 23, 17, 20]`, then `p max 0.0232 min 0.0009, entropy 0.923, top [8, 7, 22, 13, 18]`;
      - start frames during the first 750 steps (sampler still uniform; n = 78,787 resets incl. early
        terminations): ~0.09 per 15-frame bin, 0.14 in frames 75-89 (STF noisy re-sampling + early termination);
      - cost: ~6,600 env-steps/s total = **9.9 s/epoch** (fix-1: 9.5 s/epoch) -> ~8.3 h for 3,000 epochs + eval.
      Early in training ATS favours the approach frames (7-23), not the pickup; whether it moves to 55-105 as the
      policy learns the approach is the first thing to read in the training log.
- [x] `bash environment_iteration/scripts/check.sh --gpu fix-4`: ALL PASS (interrupt at epoch 10 exit 130, resume to
      20, eval + analysis, re-run skips training, video 2 clips, fingerprint recorded). 20 epochs = 640 steps < 750,
      so no rate update happens inside this check; the 60-epoch run above covers the updates. Resume restores
      `time_sample_rate` through upstream's `get_train_state` / `set_train_state`, which the bound sampler reads.

## 4. Results (from analysis/e3000/REPORT.md)

Epoch 3000, protocol v1, 96 clean stochastic episodes per cell (seeds 101-103 x 32). Extra columns marked * are
computed from the stored traces (`eval/stoch/e3000_s*.npz`), not by analyze.py.

| metric | fix-1 | fix-4 |
|---|---:|---:|
| task success (stoch, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) |
| task success (perturbed, 96; 95% Wilson CI) | 0.0% (0–4%) | 0.0% (0–4%) |
| paper success (stoch / perturbed) | 0.0% / 0.0% | 8.3% / 11.5% |
| lift success (stoch / perturbed) | 46.9% / 24.0% | **100.0% / 71.9%** |
| held frames | 11.0 | 70.2 |
| task progress | 44.8% | 40.0% |
| fell in clip (stoch / perturbed) | 95.8% / 96.9% | **0.0% / 2.1%** |
| fall at any time (300 frames) | 100% @103 | 17.7% @262 |
| dominant first failure (mode @ median frame) | lag 51% @66, off_path 25% @68 | off_path 56% @60, lead 44% @58 |
| grasp event (median frame) | 93.8% @56.5 | 100% @49 |
| drop / floor (median frame) | 55% @98 / 66% @115 | 99% @129 / 97% @137 |
| book error to reference at frame 70 / 90 / 120 (median) | 0.20 / 0.72 / 2.24 m | 0.27 / 0.19 / 0.69 m |
| * pelvis height at frame 100 (median; 0.86 m standing at reset) | 0.46 m (falling) | 0.68 m (squatting) |
| * root horizontal travel by frame 147 (median; reference ~1.1 m) | 0.81 m | 0.26 m |
| * mean reward per step, frames 0-50 / 50-70 / 70-147 | 0.76 / 0.69 / 0.05 | 0.61 / 0.19 / 0.006 |
| * holding hand at frame 100 (wrist within 0.2 m of the book) | left 43% | left 98%, right 0% |

ATS (training log `runs/fix-4_s0/train.log`, 138 `[ats]` lines): after the first ~1,500 steps it moved from the
approach frames (7-23) to the lift onset and stayed there for the rest of training. Top frames at steps 30,000,
58,500, 88,500 and 96,000 were all in 56-73 (e.g. `[64, 65, 63, 62, 66]`, p max 0.026 vs uniform 0.0068, entropy
0.94). Start-frame histograms actually used (15-frame bins, ~0.7 M resets each): frames 60-74 got 0.15-0.19 of the
starts in the middle of training (uniform 0.10), frames 0-29 dropped to 0.04 each, and late in training the mass
moved to 90-134 (0.15-0.16 per bin). So ATS was active and concentrated on the frames the hypothesis named.

Mid-clip start probe (`src/probes/policy_starts.py fix-4 --start 55,70,85,100,115 --epochs 2250,3000`, 32 envs,
reference state incl. the book in hand, 60-frame horizon, seed 101): from starts 70, 85, 100 and 115 the e3000
policy keeps the book within 0.25 m of the reference until the horizon or clip end (60 / 60 / 50 / 35 frames) and
0% fall. It breaks the 10 cm criterion after 4-34 frames. From start 55 (before the lift) the book stays
within 0.25 m for 60 frames but leaves 10 cm after 14. e2250 gives the same picture.

## 5. Failure analysis (agent)

**Hypothesis result: confirmed for the lift, mixed for the walk.** The lift criterion is met by a wide margin: lift
100% vs fix-1's 47% (96 episodes each; fix-1's 95% CI upper bound is ~57%), and in-clip falls go from 96% to 0%.
The criterion "median first failure later than frame 66" is not met. The track breaks earlier (58-60) because the
policy grasps at ~49, before the reference (~53-56), and lifts the book slightly ahead of or beside the reference
path (5-7 cm high vs 4-5 cm, 15-30° off its orientation). The 10 cm criterion then fails 5 frames before fix-1 did.
Task success stays 0%. Correcting uniform start sampling was the largest single change in this loop so far:
the lift and the balance problem of fix-1 both disappear with fix-1's physics unchanged.

**Where it fails now: the carry walk is not executed after its own lift.** Traces + videos (stoch s101, sheets
`2_carry_no_walk`, `3_drop`):

1. The book is taken and carried in the **left hand** (98% at frame 100; right wrist > 0.6 m from the book after
   ~80). This is the demonstrated one-hand carry grip, the one fix-2's probe found to be unholdable statically with
   fix-1's solver. The policy holds it dynamically for ~70 frames anyway (held frames 70.2).
2. From frame ~60 the character drops into a **wide-stance squat** (pelvis 0.86 -> 0.68 m), the left arm
   extended sideways at shoulder height. It hardly steps (root travel 0.26 m vs ~1.1 m in the reference); the
   green reference walks away.
3. The book error therefore grows after frame 90 (0.19 m @90, 0.33 @100, 0.69 @120). The arm stretches after the
   reference book, and the book slips off the fingertips at median frame 129 and reaches the floor at 137 (99% /
   97%). The character itself does not fall in the clip.

**Why the walk is missing (interpretation, not proven).** The probe shows the policy *can* carry and walk when it
starts from the reference state at frames 70-115: no falls, book within the training termination distance to clip
end. From frame 2 it reaches frame ~85 in a different state (squat, early one-hand grasp, book ~0.2 m off) and from
there does not walk. Two facts suggest the training signal is missing in this state. First, the per-step reward from
frame 70 on is ~0.006 (fix-1: 0.05). The product-form reward is near zero everywhere along the off-reference squat
path, so there is no gradient to start walking. Second, in training any episode whose book is > 0.25 m off is
terminated (`objTerminationDist`). The frame-2 evaluation error exceeds that from ~frame 100, a state the policy
has never been trained in, which is why the free run degrades into reaching and dropping. ATS cannot fix this:
it starts episodes from reference states, not from the policy's own lift state.

**Learning curve (det, 8 envs per point; trends only).** Lift appears at e750 and stays at 8/8 from then on (no
lift oscillation, unlike fix-2/3). The walk does oscillate. Traces show root travel 0.65-1.08 m at e1000-e2500, but
those policies fell in the clip at e1250-e2500 (8/8). At e2750-e3000 root travel drops to 0.23-0.30 m, the
pelvis to 0.67-0.76 m, and falls to 0/8. The final policy appears to trade the walk for not falling. Paper success
flips between 0% and 100% on these 8-env points (e750-1250, e2750) and is 8.3% on the 96-episode cell.

**Perturbed cell.** Lift 72% (fix-1 24%). The non-lifting 28% are no_lift (10%, @34), knock (10%, @44) and lag
(10%, @54). The sheet `4_perturbed_pickup` shows the hand pressing on the red book near the desk edge while the
reference wireframe is elsewhere; the book is pushed rather than grasped.

**What this does not show.** Whether a two-hand carry (both wrists near the book) would walk; whether the squat is
a balance compensation for the 1 kg book held at arm's length or a learned "safe" pose. FIX1_ANALYSIS §5 A/B was
not run. The 0.25 m termination interpretation is untested (no run with a different termination distance).

## 6. Videos

clips.json -> `video/fix-4/e3000_stoch_s101/INDEX.md` and `video/fix-4/e3000_stoch_perturb_s101/INDEX.md`
(fresh rollouts, not replays; 8 envs each; in this stoch rollout 8/8 lift, drops at 90-131, no falls). I checked all
contact sheets; each shows what it was chosen for.

- `1_early_lift` (env 0, 1, 2; frames ~45-80): grasp at 46-51, lift with both hands over the desk, book ahead / beside
  the green wireframe from ~58. Look at the timing vs the reference and when the right hand lets go.
- `2_carry_no_walk` (env 0, 1, 2; frames 75-135): **the main clip**. Wide-stance squat, left arm out sideways with the
  book, the reference walks off. Does it attempt any step? Is the squat a balance reaction to the arm load?
- `3_drop` (env 0 drop @90, env 1 drop @126): arm reaching toward the departed reference, book sliding off the
  fingertips. env 0 is an early-drop outlier; env 1 is typical.
- `4_perturbed_pickup` (perturbed s101, env 2 / 6): hand pushes the displaced book at the desk edge, knock @45-47.
- `5_full_overview` (env 0, frames 30-147): whole sequence in slow motion. Full real-time episodes:
  `full/env00-07.mp4` in each folder.

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
