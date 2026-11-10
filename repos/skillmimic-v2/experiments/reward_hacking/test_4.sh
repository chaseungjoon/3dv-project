#!/usr/bin/env bash
# Test 4: start-frame test. GPU, about 30-40 minutes (one environment build per start frame).
#   bash experiments/reward_hacking/test_4.sh
# Evaluates selected checkpoints starting from several reference frames instead of only frame 2
# (deterministic 8 envs, reward logged). Training episodes start at random frames, the evaluation always at frame 2.
# Question: does a checkpoint that fails from frame 2 succeed when started mid-clip (e.g. with the book already in hand)?
#   yes -> "different exam" (train/test mismatch at the transitions), not a reward loophole.
# For starts after the lift, lift_success is not meaningful (the object is already up): read task_progress,
# task_success (from that start to the clip end), obj_err_mean.
# Output: experiments/results/reward_hacking/eval/det_start<F>/<run>/ and t4/by_start.csv
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_4
rh_gpu_free
BOOK_STARTS="${BOOK_STARTS:-2 30 50 60 80 100 120}"     # book: 150 frames, reference lift at 57
CUP_STARTS="${CUP_STARTS:-2 30 45 60 95 125 150}"       # cup: 180 frames, lift at 33, put-down 95-125
BOOK_EPOCHS="${BOOK_EPOCHS:-750 1000 1500 2000 2500 3000}"
CUP_EPOCHS="${CUP_EPOCHS:-2000 2200 2500 2710 3000}"
for run in $RUNS; do
  case "$run" in
    place_book*) starts=$BOOK_STARTS; eps=$BOOK_EPOCHS ;;
    drink_cup*)  starts=$CUP_STARTS;  eps=$CUP_EPOCHS ;;
    *) smv2_log "unknown clip for $run, skipped"; continue ;;
  esac
  # shellcheck disable=SC2086
  EP=$(rh_epochs "$run" $eps)
  for s in $starts; do
    smv2_log "--- $run: start frame $s, epochs $EP ---"
    rh_eval "$run" --epochs "$EP" --mode det --num_envs 8 --start_frame "$s" --tag "start$s" || exit $?
  done
done
"$PY" "$RH_DIR/summarize.py" --root "$RH_OUT" t4
smv2_log "=== test_4 done: $RH_OUT/t4/ ==="
