#!/usr/bin/env bash
# Shared settings for the reward-hacking tests (README.md in this folder). Sourced by test_*.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_common.sh"   # cd to the repo, PY, RUNS_DIR, smv2_log, ...
smv2_require_setup

RH_DIR=experiments/reward_hacking
RH_OUT="${RH_OUT:-experiments/results/reward_hacking}"   # all test outputs (json/npz/csv, small)
RH_EVAL=$RH_OUT/eval
RH_LOGS=experiments/logs/reward_hacking             # full console logs (git-ignored)
BOOK_RUN="${BOOK_RUN:-place_book_fix_n2048_s0}"
CUP_RUN="${CUP_RUN:-drink_cup_fix_n2048_s0}"
RUNS="${RUNS:-$BOOK_RUN $CUP_RUN}"
mkdir -p "$RH_OUT" "$RH_LOGS"

# rh_start <test name>: copy everything this script prints into experiments/logs/reward_hacking/<name>_<time>.log
rh_start() {
  local log="$RH_LOGS/$1_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$log") 2>&1
  smv2_log "=== $1 start (log $log) ==="
}

# One GPU job at a time (12 GB): refuse to start while another training/evaluation of this repo runs.
rh_gpu_free() {
  local others
  others=$(pgrep -f -- "$SMV2_REPO/.venv/bin/python|run_baseline.sh|skillmimic/run.py" | grep -v -x "$$" || true)
  if [[ -n "$others" ]]; then
    smv2_log "another SkillMimic job is running (PID $(echo "$others" | tr '\n' ' ')). Wait for it or stop it, then rerun."
    exit 1
  fi
}

# rh_eval <run> <evaluate.py args...>: evaluate.py with the training reward logged, into $RH_EVAL.
# Already evaluated checkpoints are skipped, so an interrupted test continues where it stopped.
rh_eval() {
  local run="$1"; shift
  [[ -d "$RUNS_DIR/$run/nn" ]] || { smv2_log "no run $RUNS_DIR/$run, skipped"; return 0; }
  "$PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/$run" --out_dir "$RH_EVAL" --log_reward "$@" 2>&1 \
    | grep --line-buffered -E '^\[eval|Error|Traceback|assert'
  local c=${PIPESTATUS[0]}
  (( c == 0 )) || { smv2_log "evaluation failed (exit $c): $run $*"; return "$c"; }
}

# Epochs that have a numbered checkpoint, from a list (seq output). Keeps only those that exist in the run.
rh_epochs() {
  local run="$1"; shift
  local out=() e
  for e in "$@"; do [[ -f "$RUNS_DIR/$run/nn/${run}_$(printf %08d "$e").pth" ]] && out+=("$e"); done
  (IFS=,; echo "${out[*]}")
}
