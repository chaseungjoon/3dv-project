#!/usr/bin/env bash
# 한 run의 체크포인트 평가.  사용법: bash experiments/scripts/eval_run.sh <run_name> [all|final]
#   all  : 250 epoch마다 저장된 체크포인트 전부, 결정적 8 env (학습 곡선)
#   final: 마지막 체크포인트를 결정적 8 env + 확률적 32 env + 물체 위치 교란(eps-NSR) 32 env로
# 이미 평가한 체크포인트는 건너뛴다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN="${1:?run 이름이 필요합니다 (experiments/runs/ 아래 폴더)}"
WHICH="${2:-all}"
D="$RUNS_DIR/$RUN"
[[ -d "$D/nn" ]] || { echo "run 없음: $D" >&2; exit 1; }
ev() { "$PY" experiments/tools/evaluate.py --run_dir "$D" "$@" 2>&1 | grep --line-buffered -E '^\[eval|Error|Traceback'; }
if [[ "$WHICH" == all ]]; then
  ev --which all --mode det
else
  ev --which final --mode det
  ev --which final --mode stoch
  ev --which final --mode stoch --perturb
fi
