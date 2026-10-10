#!/usr/bin/env bash
# 한 run의 체크포인트 평가.  사용법: bash experiments/scripts/eval_run.sh <run_name> [curve <epoch,epoch,...>|all|final]
#   curve: 지정한 epoch의 번호 붙은 체크포인트, 결정적 8 env (학습 곡선)
#   all  : 번호 붙은 체크포인트 전부 (10 epoch마다라 많다), 결정적 8 env
#   final: 마지막 번호 체크포인트를 결정적 8 env + 확률적 32 env + 물체 위치 교란(eps-NSR) 32 env로
# 이미 평가한 체크포인트는 건너뛴다. 결과: experiments/results/eval/<mode>/<run>/
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RUN="${1:?run 이름이 필요합니다 (experiments/runs/ 아래 폴더)}"
WHICH="${2:-final}"
D="$RUNS_DIR/$RUN"
[[ -d "$D/nn" ]] || { echo "run 없음: $D" >&2; exit 1; }
ev() {
  "$PY" experiments/tools/evaluate.py --run_dir "$D" --out_dir "$RESULTS_DIR/eval" "$@" 2>&1 \
    | grep --line-buffered -E '^\[eval|Error|Traceback'
  local c=${PIPESTATUS[0]}
  (( c == 0 )) || { echo "평가 실패 (exit $c): $RUN $*" >&2; return "$c"; }
}
case "$WHICH" in
  curve) ev --epochs "${3:?epoch 목록이 필요합니다 (예: 250,500,750)}" --mode det ;;
  all)   ev --which all --mode det ;;
  final) ev --which final --mode det && ev --which final --mode stoch && ev --which final --mode stoch --perturb ;;
  *)     echo "모르는 모드: $WHICH (curve|all|final)" >&2; exit 1 ;;
esac
