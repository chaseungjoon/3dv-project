#!/usr/bin/env bash
# Test 6: watch the policies. Needs a screen (Isaac Gym viewer window), about 10 s of motion per checkpoint.
#   bash experiments/reward_hacking/test_6.sh                     # book 1000 2000 3000, then cup 2100 2200 3000
#   bash experiments/reward_hacking/test_6.sh place_book 1000 2000 # one clip, chosen epochs
# Each checkpoint plays once from frame 2 (1 env, deterministic). Close nothing: the next checkpoint starts by itself.
# Write what you see into experiments/results/reward_hacking/t6_observations.md (template created below);
# that file is what the analysis reads. What to look for is listed in the template.
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_6
rh_gpu_free
NOTES="$RH_OUT/t6_observations.md"
if [[ ! -f "$NOTES" ]]; then
  cat > "$NOTES" <<'MD'
# Test 6: what the policies do (filled in by hand while watching test_6.sh)

For each checkpoint, one or two lines per question. "Don't know" is a fine answer.

1. Does it pick the object up? With which hand(s)? Does the grip look like the demonstration?
2. After picking it up, does it carry the object along the demonstrated path, or stop / drop / leave it somewhere?
3. Does the body keep copying the demonstration (walking, turning) even when the object is no longer in the hand?
4. Anything odd: object pushed with the arm or body instead of the hand, object stuck to the hand or launched,
   hand passing through the object or furniture, sliding feet, falling on purpose, standing still.
5. When does it go wrong (roughly which part of the clip: reach / pick up / lift / carry / put down)?

## place_book epoch 1000 (lift 100%, falls 88% in the 8-env curve)

## place_book epoch 2000 (lift 0%)

## place_book epoch 3000 (lift 38%)

## drink_cup epoch 2100 (lift 0%, one checkpoint before grasping appears)

## drink_cup epoch 2200 (lift 88%)

## drink_cup epoch 3000 (lift 75%)
MD
  smv2_log "created $NOTES (fill it in while watching)"
fi
view() {  # view <run> <epochs comma list>
  smv2_log "--- viewer: $1 epochs $2 ---"
  "$PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/$1" --epochs "$2" --mode det --viewer \
    --out_dir "$RH_EVAL" --tag viewer --force 2>&1 | grep --line-buffered -E '^\[eval|Error|Traceback'
}
if [[ $# -ge 2 ]]; then
  run="${1}_fix_n2048_s0"; shift
  view "$run" "$(rh_epochs "$run" "$@")"
else
  view "$BOOK_RUN" "$(rh_epochs "$BOOK_RUN" 1000 2000 3000)"
  view "$CUP_RUN" "$(rh_epochs "$CUP_RUN" 2100 2200 3000)"
fi
smv2_log "=== test_6 done. Notes: $NOTES ==="
