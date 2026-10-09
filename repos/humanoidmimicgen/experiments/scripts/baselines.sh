#!/usr/bin/env bash
# 학습 없는 기준 정책 4개를 공식 평가 루프로 (성공 판정이 얼마나 쉬운지 = 바닥값). 약 20분, GPU 거의 안 씀.
#   hold: 가만히 서 있기 / mean: 학습 데이터 평균 행동을 계속 / replay: 무작위 시범의 행동을 열린 루프로 재생 / noise: 무작위 행동
#   사용법: bash experiments/scripts/baselines.sh [episodes]   (기본 100)
# 결과: experiments/results/baselines/<policy>_<N>ep.json
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hmg_require_setup
N="${1:-100}"
for p in hold mean replay noise; do
  J="$RESULTS_DIR/baselines/${p}_${N}ep.json"
  [[ -f "$J" ]] && { hmg_log "건너뜀 (이미 있음): $J"; continue; }
  "$PY" experiments/tools/eval_baselines.py --policy "$p" --task "$TASK" --num-episodes "$N" --data-root "$DATA_DIR/projected" \
    --output "$J" 2>&1 | grep --line-buffered -E "success_rate|Error|Traceback"
done
