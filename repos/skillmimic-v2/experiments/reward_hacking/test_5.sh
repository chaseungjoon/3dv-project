#!/usr/bin/env bash
# Test 5: noise test. GPU, about 20-30 minutes.
#   bash experiments/reward_hacking/test_5.sh
# The baseline curve used 8 deterministic envs (success moves in 12.5% steps, one training seed). Re-evaluate every
# 250th checkpoint with 64 envs, deterministic and stochastic, and report 95% intervals (t5/n64.csv).
# Question: is the book's 100% -> 0% -> 38% swing real, or within evaluation noise?
# Output: experiments/results/reward_hacking/eval/{det_n64,stoch_n64}/<run>/ and t5/n64.csv
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_5
rh_gpu_free
for run in $RUNS; do
  extra=""; [[ "$run" == drink_cup* ]] && extra="2200 2710"   # cup: best checkpoints of the 8-env curve
  # shellcheck disable=SC2086
  EP=$(rh_epochs "$run" ${T5_EPOCHS:-$(seq 250 250 3000) $extra})
  smv2_log "--- $run: epochs $EP ---"
  rh_eval "$run" --epochs "$EP" --mode det --num_envs 64 --tag n64   || exit $?
  rh_eval "$run" --epochs "$EP" --mode stoch --num_envs 64 --tag n64 || exit $?
done
"$PY" "$RH_DIR/summarize.py" --root "$RH_OUT" t5
smv2_log "=== test_5 done: $RH_OUT/t5/ ==="
