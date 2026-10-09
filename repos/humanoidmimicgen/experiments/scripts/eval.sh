#!/usr/bin/env bash
# 저장된 체크포인트를 MuJoCo + WBC로 closed-loop 평가 (공식 evaluate_policy_example.py).
#   사용법: bash experiments/scripts/eval.sh [episodes] [steps...]   (기본 100 episode, 모든 체크포인트)
# 성공 = 1250 step(25초) 안에 task predicate가 한 번이라도 참 (공식 정의). 성공하면 바로 다음 episode로 넘어간다
# (--terminate-on-success: 성공률은 같고 시간만 줄인다). seed 0..N-1 = 공식 보고 방식.
# 결과: experiments/results/eval/<run>/eval_<step>_<N>ep.json
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hmg_require_setup
N="${1:-100}"; shift || true
SEED="${SEED:-1000}"; RUN="$(hmg_run_name)"; D="$RUNS_DIR/$RUN"
STEPS=("$@"); [[ ${#STEPS[@]} -eq 0 ]] && STEPS=($(ls "$D/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort))
[[ ${#STEPS[@]} -gt 0 ]] || { echo "체크포인트 없음: $D/checkpoints" >&2; exit 1; }
OUT="$RESULTS_DIR/eval/$RUN"; mkdir -p "$OUT"
for s in "${STEPS[@]}"; do
  J="$OUT/eval_${s}_${N}ep.json"
  [[ -f "$J" ]] && { hmg_log "건너뜀 (이미 있음): $J"; continue; }
  hmg_log "평가: $RUN step $s, $N episodes"
  T0=$(date +%s)
  "$PY" scripts/evaluate_policy_example.py --checkpoint-dir "$D/checkpoints/$s" --task "$TASK" --num-episodes "$N" --seed 0 \
    --terminate-on-success --output "$J" 2>&1 | tee "$OUT/eval_${s}_${N}ep.log" | grep --line-buffered -E "episode [0-9]+/|success_rate|Error|Traceback"
  hmg_log "step $s: $(( $(date +%s) - T0 )) s"
done
