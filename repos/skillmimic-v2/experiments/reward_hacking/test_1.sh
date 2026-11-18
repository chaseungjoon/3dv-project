#!/usr/bin/env bash
# Test 1 (+ test 2): the training reward measured on the evaluation rollouts. GPU, about 30-40 minutes.
#   bash experiments/reward_hacking/test_1.sh
# For every 100th checkpoint of the cup and book runs, rerun the standard evaluation (start at frame 2, whole clip)
# with the training reward and its factors logged on every step (evaluate.py --log_reward):
#   deterministic 8 envs + stochastic 32 envs.
# Question: on the SAME rollout, does a failing checkpoint earn as much reward as a succeeding one?
#   yes -> the reward itself scores the failure well (reward hacking).
#   no  -> the reward tracks success, and the train/eval mismatch comes from how training samples episodes.
# Test 2 is the factor breakdown in the same files (r_body, r_obj, r_ig): which part of the reward is being earned.
# Interrupted? Run it again: finished checkpoints are skipped.
# Output: experiments/results/reward_hacking/eval/{det,stoch}/<run>/ and t1/{curve,correlations}.csv
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_1
rh_gpu_free
for run in $RUNS; do
  # shellcheck disable=SC2086
  EP=$(rh_epochs "$run" ${T1_EPOCHS:-$(seq 100 100 3000)})
  smv2_log "--- $run: epochs $EP ---"
  rh_eval "$run" --epochs "$EP" --mode det --num_envs 8   || exit $?
  rh_eval "$run" --epochs "$EP" --mode stoch --num_envs 32 || exit $?
done
"$PY" "$RH_DIR/summarize.py" --root "$RH_OUT" t1
smv2_log "=== test_1 done: $RH_OUT/t1/ ==="
