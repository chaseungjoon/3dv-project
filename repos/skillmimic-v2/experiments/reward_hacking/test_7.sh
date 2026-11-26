#!/usr/bin/env bash
# Test 7: repair tests. LONG: one book training run each, about 8-9 hours + 20 minutes of evaluation.
#   bash experiments/reward_hacking/test_7.sh seed1     # same everything, training seed 1 (is the swing reproducible?)
#   bash experiments/reward_hacking/test_7.sh objw      # same everything, object-position reward weight 1 -> 20
#   bash experiments/reward_hacking/test_7.sh all       # seed1, then objw (about 18 hours)
#   CLIP=drink_cup bash experiments/reward_hacking/test_7.sh objw   # the cup instead of the book
# Run inside tmux. Interrupted (Ctrl-C, crash, power)? Run the same command again: training continues from the last
# checkpoint (every 10 epochs) and finished evaluations are skipped. A crash (e.g. GPU memory) is retried twice.
#
# seed1: the baseline is a single seed. If seed 1 shows the same "reward rises while success swings", the effect is
#        a property of the setup, not luck of one run (explanation D in README.md).
# objw : the object-position term has weight 1 (the body terms 20), so a 25 cm object error costs only 2% of the
#        reward through that term. Weight 20 makes it cost like a body error (34%). If the reward was being gamed by
#        neglecting the object, success should follow the reward more closely in this run (test 1 on it shows that).
#        Note the interaction-graph term already penalises object error at weight 20; this repair doubles that pressure.
#        (The contact-graph reward cannot be used as a repair: every reference contact label in ParaHome is 0.)
# After training, the run gets the test-0 and test-1 analyses (curve every 100 epochs, reward logged).
# Output: experiments/runs/<clip>_<fix|fix_objw>_n2048_s<seed>/, experiments/results/reward_hacking/eval/{det,stoch}/<run>/
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
VARIANT="${1:-}"
CLIP="${CLIP:-place_book}"
EPOCHS="${EPOCHS:-3000}"
TRAIN_RETRIES="${TRAIN_RETRIES:-2}"
case "$VARIANT" in
  seed1) jobs=("fix 1") ;;
  objw)  jobs=("fix_objw 0") ;;
  all)   jobs=("fix 1" "fix_objw 0") ;;
  *) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
rh_start "test_7_${VARIANT}_${CLIP}"
rh_gpu_free
for job in "${jobs[@]}"; do
  read -r method seed <<< "$job"
  run="${CLIP}_${method}_n${NUM_ENVS}_s${seed}"
  smv2_log "=== $run: training to epoch $EPOCHS ==="
  for try in $(seq 0 "$TRAIN_RETRIES"); do
    (( try > 0 )) && { smv2_log "training died (exit $code), retry $try/$TRAIN_RETRIES from the last checkpoint in 1 min"; sleep 60; }
    bash experiments/scripts/train.sh "$CLIP" "$method" "$EPOCHS" "$seed"
    code=$?
    (( code == 0 || code == 130 )) && break
  done
  (( code == 0 )) || { smv2_log "$run: training not finished (exit $code). Run the same command again to continue."; exit "$code"; }
  smv2_log "=== $run: evaluation (test-1 curve, every 100 epochs) ==="
  EP=$(rh_epochs "$run" $(seq 100 100 "$EPOCHS"))
  rh_eval "$run" --epochs "$EP" --mode det --num_envs 8   || exit $?
  rh_eval "$run" --epochs "$EP" --mode stoch --num_envs 32 || exit $?
  "$PY" "$RH_DIR/train_log.py" --runs "$run" --every 50 --runs_dir "$RUNS_DIR" --out_dir "$RH_OUT/t0" 2>&1 | grep -v -E 'Warning|torch.load'
done
"$PY" "$RH_DIR/summarize.py" --root "$RH_OUT" t1
smv2_log "=== test_7 $VARIANT done ==="
