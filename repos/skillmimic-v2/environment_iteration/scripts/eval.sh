#!/usr/bin/env bash
# Standard evaluation + analysis of one iteration (protocol.json "eval"). GPU, ~6 min. Agent may run it.
#
#   bash environment_iteration/scripts/eval.sh fix-2 [--epoch E] [--force]
#
#   1. learning curve: det, 8 envs, at every curve epoch that has a checkpoint (250, 500, ..., 3000)
#   2. final: epoch 3000 (or --epoch E): stoch 32 envs x seeds 101/102/103, plain and eps-perturbed start
#   3. src/analyze.py -> iterations/<it>/analysis/e<epoch>/ (REPORT.md, summary.json, plots)
#   4. src/status.py  -> STATUS.md results table
# Finished rollouts are skipped (--force redoes them). Refuses to run while a training holds the GPU.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
smv2_require_setup
ei_iteration "${1:-}"; shift
EP=() FORCE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch) EP=(--epoch "$2"); shift 2 ;;
    --force) FORCE=(--force); shift ;;
    *) echo "unknown argument $1 (usage: eval.sh fix-N [--epoch E] [--force])" >&2; exit 2 ;;
  esac
done
ei_gpu_free_or_exit
run() { "$PY" "$EI_SRC/evaluate.py" "$ITERATION" "$@" 2>&1 | grep --line-buffered -E '^\[eval|Error|Traceback|error:'; return "${PIPESTATUS[0]}"; }
ei_log "=== $ITERATION: learning curve ==="
run --part curve "${FORCE[@]}" || { ei_log "curve evaluation failed"; exit 1; }
ei_log "=== $ITERATION: final evaluation ==="
run --part final "${EP[@]}" "${FORCE[@]}" || { ei_log "final evaluation failed"; exit 1; }
"$PY" "$EI_SRC/analyze.py" "$ITERATION" "${EP[@]}" || exit 1
"$PY" "$EI_SRC/status.py"
