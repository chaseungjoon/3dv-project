#!/usr/bin/env bash
# Test 0: what the training reward is made of. CPU only, about 10 seconds. No GPU, safe to run during other jobs.
#   bash experiments/reward_hacking/test_0.sh
# Splits the logged training reward into episode length x per-frame reward, reads the ATS state (training reward
# per start frame, start-frame sampling) from the checkpoints, and correlates each with the existing evaluation curve.
# Output: experiments/results/reward_hacking/t0/   (README.md, test 0)
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_0
# shellcheck disable=SC2086
"$PY" "$RH_DIR/train_log.py" --runs $RUNS place_book_ours_n2048_s0 drink_cup_ours_n2048_s0 --every 50 --runs_dir "$RUNS_DIR" --out_dir "$RH_OUT/t0" 2>&1 | grep -v -E 'Warning|torch.load'
smv2_log "=== test_0 done: $RH_OUT/t0/ ==="
