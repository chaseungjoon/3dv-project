#!/usr/bin/env bash
# Test 3: cheater audit. How much training reward do fake "cheating" behaviours earn? GPU only to load the
# reference (1 env), under a minute in total.
#   bash experiments/reward_hacking/test_3.sh
# Cheats are built from the reference clip without physics: body perfect but object never touched, object let go
# at frame X, the whole motion or only the object k frames late, standing still, object or body shifted by d.
# Each is scored with the exact training reward and with the training episode rules (60 frames, object > 25 cm ends it).
# A cheat that fails the task yet keeps most of the demonstration's reward = a loophole in the reward.
# Output: experiments/results/reward_hacking/t3/<clip>_cheats.{csv,npz}
source "$(dirname "${BASH_SOURCE[0]}")/_rh.sh"
rh_start test_3
rh_gpu_free
for clip in place_book drink_cup; do
  "$PY" "$RH_DIR/cheater_audit.py" --clip "$clip" --out_dir "$RH_OUT/t3" 2>&1 \
    | grep --line-buffered -E '^\[t3|^cheat|^[a-z][a-z_0-9]+ +[01] |Error|Traceback|assert' || exit 1
done
smv2_log "=== test_3 done: $RH_OUT/t3/ ==="
