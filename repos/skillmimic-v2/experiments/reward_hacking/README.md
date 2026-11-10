# Reward-hacking tests

Claim to test: **"the policy is reward hacking"**. During training the reward kept rising while the real success
swung (book lift 100% → 0% → 38%, `BASELINE_SKILLMIMIC_2.md`). That disagreement is necessary for reward hacking
but does not prove it. Four explanations fit it, and the tests below separate them:

| | Explanation | What it would mean |
|---|---|---|
| A | **Reward hacking** | The reward itself gives a high score to failed behaviour (a loophole) |
| B | **Different exam** | Training episodes start at random frames and last 60 frames, the evaluation starts at frame 2 and runs the whole clip. The policy aces the first and fails the second (goal misgeneralization) |
| C | **Moving exam** | ATS keeps changing which start frames are practised, so the average training reward is over a different mix each epoch |
| D | **Noise** | One training seed, 8 evaluation envs (12.5% steps): some of the swing may be luck |

## How to run

All commands from `repos/skillmimic-v2`. One GPU job at a time (scripts refuse to start if another runs).
Interrupted? Run the same command again: finished checkpoints are skipped.

```bash
bash experiments/reward_hacking/run_tests.sh        # tests 0 1 3 4 5 in a row, about 1.5-2 h, no screen needed
bash experiments/reward_hacking/test_6.sh           # at the screen: watch 6 checkpoints, write notes
bash experiments/reward_hacking/test_7.sh seed1     # long (8-9 h), in tmux
bash experiments/reward_hacking/test_7.sh objw      # long (8-9 h), in tmux
```

| Test | Script | GPU / time | Question | Supports |
|---|---|---|---|---|
| 0 | `test_0.sh` | none, ~10 s | Is the rising training reward just longer episodes? What did ATS practise, and what did training score for frame-2 starts? | C, B |
| 1 | `test_1.sh` | ~30-40 min | On the **same** frame-2 rollout, does a failing checkpoint earn as much reward as a succeeding one? | A if yes, B/C if no |
| 2 | (part of `test_1.sh`) | - | Which factor is earned: body, object, interaction graph? | A's mechanism |
| 3 | `test_3.sh` | 1 env, ~1 min | Do fake cheats (object never touched, let go early, late by k frames, ...) keep most of the demo's reward? | A if yes |
| 4 | `test_4.sh` | ~30-40 min | Does a checkpoint that fails from frame 2 succeed when started mid-clip? | B if yes |
| 5 | `test_5.sh` | ~20-30 min | With 64 envs and 95% intervals, is the 100 → 0 → 38% swing real? | D if no |
| 6 | `test_6.sh` | viewer, ~10 min | What does the policy actually do? | all (human check) |
| 7 | `test_7.sh seed1` / `objw` | 8-9 h each | Does the swing repeat with another seed? Does a stronger object reward make reward and success move together? | D / A |

There is no `test_2.sh`: test 2 is the factor breakdown recorded by test 1.

## Outputs (`experiments/results/reward_hacking/`)

| Path | From | Content |
|---|---|---|
| `t0/<run>_{tb,ats,joined}.csv`, `<run>_ats.npz`, `correlations.csv` | test 0 | episode return / length / per-frame; ATS reward and sampling per start frame; joined with the 8-env curve |
| `eval/{det,stoch}/<run>/*.json,npz` | tests 1, 7 | evaluation + per-step reward and factors |
| `t1/curve.csv`, `t1/correlations.csv` | tests 1, 7 | reward vs success per checkpoint and per episode |
| `t3/<clip>_cheats.{csv,npz}` | test 3 | reward of each cheat vs the demonstration |
| `eval/det_start<F>/<run>/`, `t4/by_start.csv` | test 4 | success and reward by start frame |
| `eval/{det,stoch}_n64/<run>/`, `t5/n64.csv` | test 5 | 64-env rates with 95% intervals next to the 8-env values |
| `t6_observations.md` | test 6 | your notes from watching |

Console logs: `experiments/logs/reward_hacking/` (git-ignored).

## What the code adds

- `experiments/tools/rh_reward.py`: the training reward (`compute_humanoid_reward`) split into its factors.
  The evaluator checks it against the upstream function on the first steps of every rollout.
- `experiments/tools/evaluate.py --log_reward`: records the reward and factors per step (off by default, so the
  baseline evaluation is unchanged). JSON columns: `rew_mean`, `rew_sum`, `train_return`, `train_len`, `train_mean`
  (the episode under the training rules: at most 60 steps, ended by a fall or the object > 0.25 m off),
  `r_*_mean`.
- `train_log.py` (test 0), `cheater_audit.py` (test 3), `summarize.py` (tables for tests 1, 4, 5).
- Method `fix_objw` (`experiments/tools/methods.py`, `configs/env/*_fix_objw.yaml`): `fix` with object-position
  weight 1 → 20, used by test 7.
- `write_results.py` now takes only seed-0 runs for the baseline document, so a test-7 seed-1 run does not replace
  the baseline.

Why no contact-reward repair: the upstream contact-graph term is commented out, and every contact label in the
ParaHome reference clips is 0, so there is nothing to imitate.

Overrides (environment variables): `RUNS`, `BOOK_RUN`, `CUP_RUN`, `RH_OUT`, `T1_EPOCHS`, `T5_EPOCHS`,
`BOOK_STARTS`, `CUP_STARTS`, `BOOK_EPOCHS`, `CUP_EPOCHS` (test 4), `CLIP`, `EPOCHS`, `NUM_ENVS`, `TRAIN_RETRIES` (test 7).
