#!/usr/bin/env bash
# WristMimic run 평가 (T09 평가기 그대로: 시퀀스 전체, 실패해도 reset 없음, 채점 설정 eval_default.yaml 고정).
#   사용법: bash experiments/scripts/eval_wm.sh <run_name> [det|stoch|both]   (기본 both)
#   det  : 결정적 행동, env 8개 (물리 잡음 반복), 저장된 모든 체크포인트 -> 학습 곡선
#   stoch: 학습 때와 같은 행동 잡음, env 32개, 모든 체크포인트
# 결과: experiments/results/eval/<mode>/<run>/<ckpt>.json (+ npz)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ph_require_t09
RUN="${1:?run 이름이 필요합니다 (experiments/runs/ 아래)}"
MODES="${2:-both}"; [[ "$MODES" == both ]] && MODES="det stoch"
KEY=$(sed -E 's/^wm_([a-z]+_s[0-9]+)_.*/\1/' <<<"$RUN")
read -r SCENE ROBOT < <(ph_scene "$KEY") || exit 1
for m in $MODES; do
  ph_log "평가: $RUN ($m, 모든 체크포인트)"
  ( cd "$T09_REPO" && "$T09_PY" experiments/tools/evaluate.py --run_dir "$RUNS_DIR/$RUN" --which all --mode "$m" \
      --motion_file "$SCENE" --robot_type "$ROBOT" --out_dir "$RESULTS_DIR/eval" ) 2>&1 | grep --line-buffered -E '^\[eval|Error|Traceback'
done
