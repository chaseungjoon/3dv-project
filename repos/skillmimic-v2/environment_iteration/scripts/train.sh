#!/usr/bin/env bash
# Train one iteration with the fixed loop budget, then evaluate + analyze it. HUMAN-RUN (~8 h on the RTX 5070).
#
#   bash environment_iteration/scripts/train.sh fix-2          # run it inside tmux
#
# Interrupted (Ctrl-C, crash, reboot)? Run the SAME command again: it continues from the newest checkpoint
# (epoch, optimizer, normalizers, ATS weights, RNG; saved every 10 epochs). A finished training is skipped, so
# re-running after the end only redoes missing evaluation / analysis.
#
# What it does:
#   0. one instance (lock), waits until no other process uses the GPU, disk check, Isaac Gym import check
#   1. fingerprint (src/fingerprint.py): the first launch hashes every file that can change this training; a resume
#      REFUSES if any of them changed (EI_ALLOW_CHANGED=1 continues anyway, recorded in the log)
#   2. trains to protocol.json train.epochs (3000) x num_envs (2048), seed 0. A crashed segment (exit != 0, != 130)
#      is retried from the newest checkpoint up to train.retries times, 60 s apart
#      checkpoint cleaner (every 5 min + at the end): keeps the curve epochs (every 250), the final epoch, the
#      newest 3 and the non-numbered files; ~0.7 GB per iteration instead of ~13 GB. EI_KEEP_ALL=1 keeps everything
#   3. scripts/eval.sh (standard evaluation, ~6 min) -> src/analyze.py -> src/status.py
# Log: iterations/<it>/logs/train_<time>.log (+ the full trainer output in runs/<run>/train.log)
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
smv2_require_setup
[[ $# -eq 1 ]] || { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
ei_iteration "$1"
IT="$ITERATION"
eval "$("$PY" "$EI_SRC/iterlib.py" proto)" || exit 2
"$PY" -c "
import sys; sys.path.insert(0, '$EI_SRC'); import iterlib as L
it = L.load('$IT')
if it.get('imported'): sys.exit('$IT is an imported run (' + it['imported'].get('from_run', '') + '); nothing to train')
" || exit 2
mkdir -p "$RUN_DIR" "$LOGS_DIR"

exec 9> "$LOCK"
flock -n 9 || { ei_log "Another training holds $LOCK ($(cat "$LOCK_INFO" 2>/dev/null)). Not starting a second one."; exit 1; }
LOG="$LOGS_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i -a "$LOG" 9>&-) 2>&1      # tee -i: Ctrl-C reaches the trainer, not the logger
"$PY" -c "import json,sys; json.dump(dict(iteration='$IT', pid=$$, started='$(date -Iseconds)', log='$LOG'), open('$LOCK_INFO','w'))"
INTERRUPTED=0
trap 'INTERRUPTED=1' INT
CLEANER=""
stop_cleaner() { [[ -n "$CLEANER" ]] || return 0; pkill -P "$CLEANER" 2>/dev/null; kill "$CLEANER" 2>/dev/null; wait "$CLEANER" 2>/dev/null; CLEANER=""; }
trap 'stop_cleaner; rm -f "$LOCK_INFO"' EXIT

ei_log "=== $IT: train $EPOCHS epochs x $NUM_ENVS envs, seed $SEED ($((EPOCHS * NUM_ENVS * 32)) samples) ==="
ei_log "run dir: $(realpath --relative-to="$SMV2_REPO" "$RUN_DIR")   log: $(realpath --relative-to="$SMV2_REPO" "$LOG")"
(( TEST_MODE )) && ei_log "TEST MODE (EI_ITER_DIR=$EI_ITER_DIR)"

"$PY" -c "import isaacgym, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 \
  || { ei_log "Isaac Gym / CUDA import check failed with $PY"; exit 2; }
gpu_others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v -x -e '' -e "$$" || true; }
if [[ -n "$(gpu_others)" ]]; then
  ei_log "Another process uses the GPU (PID $(gpu_others | tr '\n' ' ')). Waiting until it is gone (checked every minute)."
  while [[ -n "$(gpu_others)" ]]; do sleep 60 9>&-; (( INTERRUPTED )) && { ei_log "Interrupted while waiting."; exit 130; }; done
  sleep 30 9>&-
fi

latest() { "$PY" -c "import sys; sys.path.insert(0, '$EI_SRC'); import iterlib as L; print(L.latest_epoch('$RUN_DIR'))"; }
E0=$(latest)
if (( E0 < EPOCHS )); then
  free_mb=$(df -Pm "$RUN_DIR" | awk 'NR==2 {print $4}')
  need_mb=$(( 10240 + 45 * (EPOCHS - E0) / 10 ))
  [[ "${EI_KEEP_ALL:-0}" == 1 ]] || need_mb=$(( 10240 + 45 * 20 ))
  (( free_mb >= need_mb )) || { ei_log "Not enough disk: ${free_mb} MiB free, ${need_mb} MiB needed."; exit 2; }
  if ! "$PY" "$EI_SRC/fingerprint.py" check "$IT"; then
    if [[ "${EI_ALLOW_CHANGED:-0}" == 1 ]]; then
      ei_log "EI_ALLOW_CHANGED=1: continuing although files changed (they are listed above)"
    else
      ei_log "Files this training depends on changed since its first launch (list above). Resuming would mix two"
      ei_log "experiments. Restore them, or knowingly continue: EI_ALLOW_CHANGED=1 bash $0 $IT"
      exit 3
    fi
  fi
fi

prune() { [[ "${EI_KEEP_ALL:-0}" == 1 ]] || "$PY" experiments/tools/paper_budget/prune.py "$RUN_DIR" "$KEEP_EPOCHS" \
            --newest "$KEEP_NEWEST" || ei_log "checkpoint cleaner failed (training not affected)"; }

CODE=0
for try in $(seq 0 "$RETRIES"); do
  E=$(latest)
  (( E >= EPOCHS )) && { ei_log "training complete (epoch $E)"; CODE=0; break; }
  if (( try > 0 )); then
    ei_log "training died (exit $CODE). Retrying from the newest checkpoint in 60 s ($try/$RETRIES)"
    sleep 60 9>&-; (( INTERRUPTED )) && break
  fi
  RESUME=()
  if (( E > 0 )); then
    IFS=$'\t' read -r RP RE < <("$PY" experiments/tools/ckpt.py latest "$RUN_DIR")
    [[ -n "$RP" ]] && { RESUME=(--resume_from "$RP"); ei_log "resuming from epoch $RE ($(basename "$RP"))"; }
  fi
  "$PY" - "$RUN_DIR/run_meta.json" <<PYEOF
import json, os, sys, datetime
p = sys.argv[1]
m = json.load(open(p)) if os.path.isfile(p) else {}
m.update(iteration="$IT", run_name="$RUN_NAME", epochs=$EPOCHS, seed=$SEED, num_envs=$NUM_ENVS)
m.setdefault("segments", []).append(dict(started_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                         resume_epoch=$E))
json.dump(m, open(p, "w"), indent=1)
PYEOF
  prune
  if [[ "${EI_KEEP_ALL:-0}" != 1 ]]; then ( while sleep "$PRUNE_EVERY"; do prune; done ) 9>&- & CLEANER=$!; fi
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -l 30 >> "$RUN_DIR/gpu.csv" 2>/dev/null 9>&- &
  GPUL=$!
  "$PY" "$EI_SRC/launch.py" train "$IT" --seed "$SEED" --epochs "$EPOCHS" --num_envs "$NUM_ENVS" "${RESUME[@]}" 2>&1 \
    | tee -a "$RUN_DIR/train.log" | smv2_filter \
    | grep --line-buffered -E '^\[launch|epoch_num:[0-9]*[05]0 |MAX EPOCHS|resumed full training state|Error|error|Traceback|Killed|out of memory'
  CODE=${PIPESTATUS[0]}
  kill "$GPUL" 2>/dev/null
  stop_cleaner
  prune
  "$PY" - "$RUN_DIR/run_meta.json" "$CODE" <<'PYEOF'
import datetime, json, sys
p, code = sys.argv[1], int(sys.argv[2])
m = json.load(open(p)); s = m["segments"][-1]; now = datetime.datetime.now().astimezone()
s.update(finished_at=now.isoformat(timespec="seconds"), exit_code=code,
         wall_s=round((now - datetime.datetime.fromisoformat(s["started_at"])).total_seconds()))
m["wall_s_total"] = sum(x.get("wall_s", 0) for x in m["segments"])
json.dump(m, open(p, "w"), indent=1)
PYEOF
  (( CODE == 130 || INTERRUPTED )) && break
  (( CODE == 0 )) && (( $(latest) >= EPOCHS )) && break
done

if (( CODE == 130 || INTERRUPTED )); then
  ei_log "Interrupted at epoch $(latest)/$EPOCHS. Run the same command again to continue: bash $0 $IT"
  exit 130
fi
if (( $(latest) < EPOCHS )); then
  ei_log "$IT did not reach epoch $EPOCHS (exit $CODE). See $(realpath --relative-to="$SMV2_REPO" "$RUN_DIR")/train.log, then re-run."
  exit 1
fi
ei_log "=== $IT trained ($(latest) epochs). Evaluating ==="
rm -f "$LOCK_INFO"                       # the GPU guard of eval.sh must not see our own lock
flock -u 9
EI_FORCE_GPU=1 bash "$EI/scripts/eval.sh" "$IT" || { ei_log "evaluation failed; re-run: bash $EI/scripts/eval.sh $IT"; exit 1; }
ei_log "=== $IT done. Report: $(realpath --relative-to="$SMV2_REPO" "$ANALYSIS_DIR")/e$EPOCHS/REPORT.md ==="
ei_log "Next (agent): analyze, write clips.json, render videos (AGENTS.md 3)."
