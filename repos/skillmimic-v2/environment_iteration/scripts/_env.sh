# Sourced by every environment_iteration script. cd to repos/skillmimic-v2, PY, smv2_* helpers, EI paths.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/experiments/scripts/_common.sh"
EI="$SMV2_REPO/environment_iteration"
EI_SRC="$EI/src"
LOCK="$EI/.train.lock"            # held by scripts/train.sh for the whole training (one GPU job at a time)
LOCK_INFO="$EI/.train.lock.info"  # who holds it (read by status.py and the GPU guards below)

ei_log() { printf '[ei %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Validate an iteration name and load its paths as shell variables (RUN_DIR, EVAL_DIR, LOGS_DIR, ...).
ei_iteration() {
  local it="${1:-}"
  [[ "$it" =~ ^fix-[0-9]+$ ]] || { echo "iteration name required: fix-<n> (see: $PY $EI_SRC/iterlib.py list)" >&2; exit 2; }
  local vars
  vars=$("$PY" "$EI_SRC/iterlib.py" paths "$it") || exit 2
  eval "$vars"
  ITERATION="$it"
}

# Refuse GPU work while a training holds the lock (12 GB cannot hold training + evaluation). EI_FORCE_GPU=1 skips.
ei_gpu_free_or_exit() {
  [[ "${EI_FORCE_GPU:-0}" == 1 ]] && return 0
  if [[ -f "$LOCK_INFO" ]]; then
    local pid; pid=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('pid',''))" "$LOCK_INFO" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      ei_log "A training is running (pid $pid, $(cat "$LOCK_INFO")). Not starting GPU work. (EI_FORCE_GPU=1 to override)"
      exit 4
    fi
  fi
}
