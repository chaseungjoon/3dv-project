#!/usr/bin/env bash
# experiments/runs/ 아래 모든 run을 평가한다 (이미 평가한 체크포인트는 건너뜀).
#   사용법: bash experiments/scripts/eval_all.sh
# 학습이 돌고 있는 중에는 실행하지 말 것 (GPU 메모리 경쟁, 진행 중 run의 체크포인트가 바뀜).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

shopt -s nullglob
RUNS=()
for d in "$RUNS_DIR"/*/; do
  [[ -d "$d/nn" ]] && RUNS+=("${d%/}")
done
[[ ${#RUNS[@]} -gt 0 ]] || { echo "평가할 run이 없습니다: $RUNS_DIR" >&2; exit 1; }
bash experiments/scripts/eval_run.sh "${RUNS[@]}"
