#!/usr/bin/env bash
# 학습 run 하나를 평가한다 (학습 아님, GPU 수 분).
#   사용법: bash experiments/scripts/eval_run.sh <run_dir> [<run_dir> ...]
#   1) 결정적 평가(env 1개, 공식 프로토콜): 저장된 모든 체크포인트 -> 학습 곡선
#   2) 확률적 평가(env 32개): 저장된 모든 체크포인트 -> 행동 잡음 강건성, 더 매끄러운 곡선
# 이미 평가한 체크포인트는 건너뛴다 (다시 하려면 FORCE=1).
# 모든 run은 같은 평가 설정(configs/env/eval_default.yaml)으로 채점된다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[[ $# -gt 0 ]] || { echo "run 폴더를 주세요 (예: experiments/runs/default_s0_...)" >&2; exit 2; }
EXTRA=()
[[ "${FORCE:-0}" == "1" ]] && EXTRA+=(--force)

for RUN in "$@"; do
  t09_log "평가: $RUN (결정적, 모든 체크포인트)"
  "$PY" experiments/tools/evaluate.py --run_dir "$RUN" --which all --mode det \
     --motion_file "$SCENE" --robot_type "$ROBOT" "${EXTRA[@]}" || exit 1
  t09_log "평가: $RUN (확률적 32 env, 모든 체크포인트)"
  "$PY" experiments/tools/evaluate.py --run_dir "$RUN" --which all --mode stoch \
     --motion_file "$SCENE" --robot_type "$ROBOT" "${EXTRA[@]}" || exit 1
done
