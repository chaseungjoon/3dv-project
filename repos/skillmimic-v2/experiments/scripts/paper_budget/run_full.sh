#!/usr/bin/env bash
# PAPER (method ours) vs FIX (method fix) on place_book at the paper budget: 3 training seeds each, 15,260 epochs
# (2048 env x 32 steps x 15,260 = 1,000,079,360 samples per run). Plan: PAPER_FIX_FULL_BUDGET.md (project root).
#
#   bash run_paper_fix_full.sh [--s0 | --s0s1 | --plan]    # from repos/skillmimic-v2 (thin wrapper); run it inside tmux
#
# Scope (which runs of the fixed order this launch trains; everything is saved in the same place either way):
#   --s0     PAPER s0, FIX s0                          (2 runs, ~3.5 days)  pilot, 1 seed per condition
#   --s0s1   + FIX s1, PAPER s1                        (4 runs, ~7 days)    2 seeds per condition
#   --plan   + PAPER s2, FIX s2                        (6 runs, ~10.5 days) the pre-registered design
#   (none)   same as --plan
# The script stops by itself when its scope is done. A larger scope later continues from there (finished runs and
# evaluations are skipped), e.g. `--s0` now, `--s0s1` next week. Interrupted? Run the same command again.
#
# One command does everything, and the SAME command continues after any interruption (Ctrl-C, crash, reboot):
#   0. lock (one instance), wait until no other process uses the GPU, disk-space check, Isaac Gym import check
#   1. provenance: first launch hashes every file that can change the numbers (code, configs, assets, motion,
#      history encoder, rl_games). Later launches re-hash and REFUSE to continue if anything changed (plan section 3).
#   2. for each run, in the pre-registered order PAPER s0 -> FIX s0 -> FIX s1 -> PAPER s1 -> PAPER s2 -> FIX s2:
#        train to 15,260 epochs (train.sh resumes from the newest full checkpoint, saved every 10 epochs, with
#        optimizer / normalizers / ATS / RNG state; a crashed segment is retried TRAIN_RETRIES times, 1 min apart)
#        then evaluate that run (plan sections 8-10):
#          det curve (8 env, seed 0) at 500,750,1000,1500,...,15000,15260
#          final 15,260: stoch 32 env and stoch + object perturbation 32 env, eval seed 0 and eval seeds 101,102,103
#        finished training / finished evaluations are skipped, so a re-run only does what is missing
#      Checkpoint cleaner: while a run trains, a background loop (every 5 min, and once after training) deletes numbered
#      checkpoints that will never be used. Kept: the 32 eval epochs above (incl. 15,260), the newest 3 (resume +
#      spares), the latest <run>.pth and the upstream best-reward files. ~1.6 GB per run instead of ~65 GB.
#      Deletions are logged to <run_dir>/pruned.log. PB_KEEP_ALL=1 disables it (keeps every 10th-epoch checkpoint).
#   3. checkpoint verification (epoch 15,260, frame 1,000,079,360, SHA-256) and the summary (summarize.py)
#
# Output
#   experiments/runs_paper_public/place_book_ours_n2048_s{0,1,2}/   PAPER checkpoints (nn/), train.log, gpu.csv, run_meta.json
#   experiments/runs_paper_fix/place_book_fix_n2048_s{0,1,2}/       FIX   ...
#   experiments/results_paper_public/eval/<set>/<run>/*.json|npz   PAPER evaluations
#   experiments/results_paper_fix/eval/<set>/<run>/*.json|npz      FIX   evaluations
#   experiments/results_paper_budget/                              summary.md, *.csv, curves.png, checkpoints.csv,
#                                                                  manifest/ (hashes, git, invocations), logs/
#
# Nothing here is meant to be changed between launches. PB_* variables exist only for the pipeline test and are
# refused unless PB_OUT points somewhere other than experiments/ (so a test can never write into the real results):
#   PB_OUT (output prefix), PB_EPOCHS, PB_ORDER ("ours:0 fix:0 ..."), PB_CURVE, PB_EVAL_SEEDS, PB_PRUNE_EVERY (s),
#   PB_PRUNE_NEWEST
# Allowed on the real run:
#   PB_KEEP_ALL=1       keep every checkpoint (no cleaner; ~65 GB per run)
#   PB_ALLOW_CHANGED=1  continue although the provenance check found changed files (recorded in the manifest)
SCOPE=plan
case "$#:${1:-}" in
  0:)          ;;
  1:--s0)      SCOPE=s0 ;;
  1:--s0s1)    SCOPE=s0s1 ;;
  1:--plan)    SCOPE=plan ;;
  1:-h|1:--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "usage: bash run_paper_fix_full.sh [--s0 | --s0s1 | --plan]   (one flag at most; none = --plan)" >&2; exit 2 ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"   # cd to repos/skillmimic-v2, PY, smv2_* helpers
smv2_require_setup

# ---------------- fixed design (PAPER_FIX_FULL_BUDGET.md section 2) ----------------
CLIP=place_book
export NUM_ENVS=2048                       # fixed; not taken from the caller's environment
EPOCHS="${PB_EPOCHS:-15260}"
ORDER="${PB_ORDER:-ours:0 fix:0 fix:1 ours:1 ours:2 fix:2}"
CURVE="${PB_CURVE:-500,750,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,15000,15260}"
EVAL_SEEDS="${PB_EVAL_SEEDS:-101 102 103}"
TRAIN_RETRIES="${TRAIN_RETRIES:-2}"
PRUNE_EVERY="${PB_PRUNE_EVERY:-300}"
PRUNE_NEWEST="${PB_PRUNE_NEWEST:-3}"
OUT="${PB_OUT:-experiments}"
OUT="${OUT%/}"
if [[ "$OUT" == experiments || "$OUT" == "$SMV2_REPO/experiments" ]]; then
  for v in PB_EPOCHS PB_ORDER PB_CURVE PB_EVAL_SEEDS PB_PRUNE_EVERY PB_PRUNE_NEWEST; do
    [[ -n "${!v:-}" ]] && { echo "$v is a test-only override; it needs PB_OUT outside experiments/ (refusing to touch the real results)" >&2; exit 2; }
  done
fi
PAPER_RUNS="$OUT/runs_paper_public";  PAPER_RES="$OUT/results_paper_public"
FIX_RUNS="$OUT/runs_paper_fix";       FIX_RES="$OUT/results_paper_fix"
SUMMARY="$OUT/results_paper_budget"
# runs of this launch = first 2 / 4 / 6 entries of the fixed order (matched PAPER/FIX pairs)
case "$SCOPE" in s0) K=2 ;; s0s1) K=4 ;; plan) K=$(wc -w <<< "$ORDER") ;; esac
RUN_ORDER=$(tr ' ' '\n' <<< "$ORDER" | grep -v '^$' | head -n "$K" | tr '\n' ' ')
RUN_ORDER="${RUN_ORDER% }"
mkdir -p "$PAPER_RUNS" "$FIX_RUNS" "$PAPER_RES" "$FIX_RES" "$SUMMARY/logs"

runs_dir() { [[ $1 == ours ]] && echo "$PAPER_RUNS" || echo "$FIX_RUNS"; }
res_dir()  { [[ $1 == ours ]] && echo "$PAPER_RES"  || echo "$FIX_RES"; }
cond()     { [[ $1 == ours ]] && echo PAPER || echo FIX; }
run_name() { echo "${CLIP}_${1}_n${NUM_ENVS}_s${2}"; }

# ---------------- one instance, full log ----------------
exec 9> "$SUMMARY/.run_full.lock"
flock -n 9 || { smv2_log "run_full.sh is already running for $OUT (lock $SUMMARY/.run_full.lock). Not starting a second one."; exit 1; }
LOG="$SUMMARY/logs/run_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i -a "$LOG" 9>&-) 2>&1      # tee -i: Ctrl-C reaches the training, not the logger; 9>&-: logger never holds the lock
INTERRUPTED=0
trap 'INTERRUPTED=1' INT

# config for summarize.py (rewritten every launch; the design values above are constant)
"$PY" - "$SUMMARY/config.json" <<PYEOF
import json, sys
order = [dict(method=o.split(":")[0], seed=int(o.split(":")[1])) for o in "$ORDER".split()]
for o in order: o["cond"] = "PAPER" if o["method"] == "ours" else "FIX"
json.dump(dict(clip="$CLIP", num_envs=$NUM_ENVS, epochs=$EPOCHS, order=order, scope="$SCOPE", scope_runs="$RUN_ORDER".split(),
               seeds=sorted({o["seed"] for o in order}), eval_seeds=[int(s) for s in "$EVAL_SEEDS".split()],
               curve_epochs=[int(e) for e in "$CURVE".split(",")], summary_dir="$SUMMARY",
               conditions=dict(PAPER=dict(method="ours", runs_dir="$PAPER_RUNS", results_dir="$PAPER_RES"),
                               FIX=dict(method="fix", runs_dir="$FIX_RUNS", results_dir="$FIX_RES"))),
          open(sys.argv[1], "w"), indent=1)
PYEOF
summarize() { "$PY" experiments/tools/paper_budget/summarize.py --config "$SUMMARY/config.json" || smv2_log "summary failed (results are untouched)"; }

latest_epoch() {  # newest numbered checkpoint by file name (no torch load)
  local d="$1/nn" f e=0
  [[ -d "$d" ]] || { echo 0; return; }
  for f in "$d"/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].pth; do
    [[ -e "$f" ]] || continue
    f="${f%.pth}"; f="${f##*_}"; (( 10#$f > e )) && e=$((10#$f))
  done
  echo "$e"
}

smv2_log "=== PAPER vs FIX full budget: $CLIP, $EPOCHS epochs/run ($((EPOCHS * NUM_ENVS * 32)) samples) ==="
smv2_log "scope --$SCOPE: this launch trains and evaluates [$RUN_ORDER] of the order [$ORDER], then stops"
smv2_log "log: $LOG"
for o in $ORDER; do
  m=${o%%:*} s=${o##*:} r=$(run_name "${o%%:*}" "${o##*:}")
  smv2_log "  $(cond "$m") s$s  $r: epoch $(latest_epoch "$(runs_dir "$m")/$r")/$EPOCHS"
done

# ---------------- preflight ----------------
"$PY" -c "import isaacgym, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 \
  || { smv2_log "Isaac Gym / CUDA import check failed with $PY"; exit 2; }

gpu_others() {  # compute processes on the GPU that are not ours
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v -x -e '' -e "$$" || true
}
if [[ -n "$(gpu_others)" ]]; then
  smv2_log "Another process uses the GPU (PID $(gpu_others | tr '\n' ' ')). Waiting until it is gone (checked every minute)."
  while [[ -n "$(gpu_others)" ]]; do
    sleep 60
    (( INTERRUPTED )) && { smv2_log "Interrupted while waiting."; exit 130; }
  done
  sleep 30
fi

CKPT_MB=45   # measured: 43 MB per full checkpoint, one every 10 epochs
KEEP="$CURVE,$EPOCHS"
need_mb=10240
for o in $RUN_ORDER; do
  r=$(run_name "${o%%:*}" "${o##*:}"); e=$(latest_epoch "$(runs_dir "${o%%:*}")/$r")
  (( e >= EPOCHS )) && continue
  if [[ "${PB_KEEP_ALL:-0}" == 1 ]]; then
    need_mb=$(( need_mb + (EPOCHS - e) / 10 * CKPT_MB ))
  else  # eval checkpoints still to come + newest/best/latest files
    n=$(tr ',' '\n' <<< "$KEEP" | sort -un | awk -v e="$e" '$1 > e' | wc -l)
    need_mb=$(( need_mb + (n + PRUNE_NEWEST + 3) * CKPT_MB ))
  fi
done
free_mb=$(df -Pm "$SUMMARY" | awk 'NR==2 {print $4}')
smv2_log "disk: ${free_mb} MiB free, ${need_mb} MiB still needed (checkpoints + 10 GiB margin)"
(( free_mb >= need_mb )) || { smv2_log "Not enough disk space. Free up $(( (need_mb - free_mb) / 1024 + 1 )) GiB and re-run."; exit 2; }

# ---------------- provenance ----------------
if ! PY="$PY" "$PY" experiments/tools/paper_budget/provenance.py check "$SUMMARY"; then
  if [[ "${PB_ALLOW_CHANGED:-0}" == 1 ]]; then
    smv2_log "PB_ALLOW_CHANGED=1: continuing with changed files (recorded in $SUMMARY/manifest/invocations.jsonl)"
  else
    smv2_log "Files that affect training/evaluation changed since the first launch (list above). The plan forbids this."
    smv2_log "Restore them, or knowingly continue with PB_ALLOW_CHANGED=1 bash run_paper_fix_full.sh"
    exit 3
  fi
fi

# ---------------- checkpoint cleaner ----------------
CLEANER=""
prune() { [[ "${PB_KEEP_ALL:-0}" == 1 ]] || "$PY" experiments/tools/paper_budget/prune.py "$1" "$KEEP" --newest "$PRUNE_NEWEST" \
            || smv2_log "checkpoint cleaner failed for $1 (training is not affected)"; }
start_cleaner() {
  [[ "${PB_KEEP_ALL:-0}" == 1 ]] && return
  ( while sleep "$PRUNE_EVERY"; do prune "$1"; done ) 9>&- &   # 9>&-: the cleaner (and its sleep) never holds the lock
  CLEANER=$!
}
stop_cleaner() {
  [[ -n "$CLEANER" ]] || return 0
  pkill -P "$CLEANER" 2>/dev/null; kill "$CLEANER" 2>/dev/null; wait "$CLEANER" 2>/dev/null; CLEANER=""
}
trap 'stop_cleaner' EXIT
[[ "${PB_KEEP_ALL:-0}" == 1 ]] && smv2_log "PB_KEEP_ALL=1: checkpoint cleaner off, every 10th-epoch checkpoint is kept"

# ---------------- train + evaluate ----------------
EVAL_FAILED=""
ev() {  # ev <method> <run> <evaluate.py args...>
  local m=$1 r=$2; shift 2
  "$PY" experiments/tools/evaluate.py --run_dir "$(runs_dir "$m")/$r" --out_dir "$(res_dir "$m")/eval" "$@" 2>&1 \
    | grep --line-buffered -E '^\[eval|Error|Traceback'
  local c=${PIPESTATUS[0]}
  (( c == 0 )) || { smv2_log "evaluation failed (exit $c): $r $*"; EVAL_FAILED="$EVAL_FAILED $r"; }
  return "$c"
}
evaluate_run() {  # all pre-registered evaluations of one finished run; done ones are skipped by evaluate.py
  local m=$1 r=$2 es
  ev "$m" "$r" --epochs "$CURVE" --mode det                                    || return
  ev "$m" "$r" --epochs "$EPOCHS" --mode stoch --num_envs 32                   || return
  ev "$m" "$r" --epochs "$EPOCHS" --mode stoch --num_envs 32 --perturb         || return
  for es in $EVAL_SEEDS; do
    (( INTERRUPTED )) && return 130
    ev "$m" "$r" --epochs "$EPOCHS" --mode stoch --num_envs 32 --seed "$es" --tag "s$es"           || return
    ev "$m" "$r" --epochs "$EPOCHS" --mode stoch --num_envs 32 --perturb --seed "$es" --tag "s$es" || return
  done
}

i=0; N=$(wc -w <<< "$RUN_ORDER")
for o in $RUN_ORDER; do
  i=$((i + 1)); m=${o%%:*}; s=${o##*:}; r=$(run_name "$m" "$s")
  smv2_log "=== [$i/$N] $(cond "$m") seed $s ($r): train to $EPOCHS ==="
  CODE=0
  mkdir -p "$(runs_dir "$m")/$r"
  prune "$(runs_dir "$m")/$r"
  start_cleaner "$(runs_dir "$m")/$r"
  for try in $(seq 0 "$TRAIN_RETRIES"); do
    if (( try > 0 )); then
      smv2_log "training died (exit $CODE). Retrying from the newest checkpoint in 60 s ($try/$TRAIN_RETRIES)"
      sleep 60
      (( INTERRUPTED )) && break
    fi
    PY="$PY" RUNS_DIR="$(runs_dir "$m")" RESULTS_DIR="$(res_dir "$m")" NUM_ENVS="$NUM_ENVS" \
      bash experiments/scripts/train.sh "$CLIP" "$m" "$EPOCHS" "$s"
    CODE=$?
    (( CODE == 0 || CODE == 130 || INTERRUPTED )) && break
  done
  stop_cleaner
  prune "$(runs_dir "$m")/$r"
  if (( CODE == 130 || INTERRUPTED )); then
    smv2_log "Interrupted. Run the same command again to continue from the newest checkpoint of $r."
    summarize; exit 130
  fi
  if (( CODE != 0 )) || (( $(latest_epoch "$(runs_dir "$m")/$r") < EPOCHS )); then
    smv2_log "$r did not reach epoch $EPOCHS (exit $CODE) after $((TRAIN_RETRIES + 1)) attempts. Stopping so the run order stays as planned."
    smv2_log "See $(runs_dir "$m")/$r/train.log; fix the cause (not the code/config), then run the same command again."
    summarize; exit 1
  fi
  smv2_log "=== [$i/$N] $r: evaluate ==="
  evaluate_run "$m" "$r"
  (( INTERRUPTED )) && { smv2_log "Interrupted during evaluation. Run the same command again."; summarize; exit 130; }
  summarize
done

# second chance for evaluations that failed above
if [[ -n "$EVAL_FAILED" ]]; then
  smv2_log "=== retrying failed evaluations:$EVAL_FAILED ==="
  EVAL_FAILED=""
  for o in $RUN_ORDER; do evaluate_run "${o%%:*}" "$(run_name "${o%%:*}" "${o##*:}")"; done
fi

# ---------------- verification + summary ----------------
smv2_log "=== final checkpoint verification (epoch, frame, SHA-256) ==="
DIRS=(); for o in $RUN_ORDER; do DIRS+=("$(runs_dir "${o%%:*}")/$(run_name "${o%%:*}" "${o##*:}")"); done
"$PY" experiments/tools/paper_budget/provenance.py ckpts "$SUMMARY" "$EPOCHS" "${DIRS[@]}"
VERIFY=$?
summarize
if [[ -n "$EVAL_FAILED" || $VERIFY -ne 0 ]]; then
  smv2_log "=== finished with problems (evaluations failed:${EVAL_FAILED:- none}, checkpoint check exit $VERIFY). Run the same command again. ==="
  exit 1
fi
smv2_log "=== scope --$SCOPE done: all $N runs [$RUN_ORDER] trained and evaluated. Read $SUMMARY/summary.md ==="
case "$SCOPE" in
  s0)   smv2_log "To add the s1 pair later: bash run_paper_fix_full.sh --s0s1   (or --plan / no flag for all 6)" ;;
  s0s1) smv2_log "To add the s2 pair later: bash run_paper_fix_full.sh --plan   (or no flag)" ;;
esac
