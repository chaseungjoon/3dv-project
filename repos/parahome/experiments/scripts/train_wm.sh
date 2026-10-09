#!/usr/bin/env bash
# WristMimic(T09 baseline 방법) 학습 1회를 ParaHome 장면에.   사용법: bash experiments/scripts/train_wm.sh <scene> <epochs> [seed]
#   scene : cup_s110 | book_s11 | cup_s89 | book_s149 | kettle_s110
#   설정은 T09 baseline과 **완전히 같다**: ../t09-wristmimic/experiments/configs/env/default.yaml (12GB PhysX 버퍼),
#   1024 env, minibatch 16384, 항상 프레임 0에서 시작, 500 epoch마다 체크포인트. 바뀌는 것은 장면과 그 사람의 XML뿐.
# 환경변수: VARIANT (기본 default; diag_hybrid = T09 진단 설정, CFG_DIR 자동 지정)
# 같은 명령을 다시 실행하면 가장 최근 체크포인트(500 epoch 단위)부터 이어서 학습한다.
# 결과: experiments/runs/wm_<scene>_<variant>_s<seed>/  (T09 train.sh 형식: nn/, summaries/, train.log, gpu.csv, run_meta.json)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ph_require_t09
KEY="${1:?scene이 필요합니다 (cup_s110|book_s11|cup_s89|book_s149|kettle_s110)}"
EPOCHS="${2:?epoch 수가 필요합니다 (500의 배수, 예: 3000)}"
SEED="${3:-0}"
VARIANT="${VARIANT:-default}"
read -r SCENE ROBOT < <(ph_scene "$KEY") || exit 1
CFG_DIR="experiments/configs/env"
[[ "$VARIANT" == diag_hybrid ]] && CFG_DIR="experiments/configs/diag"
RUN_NAME="wm_${KEY}_${VARIANT}_s${SEED}"
RESUME=""
LATEST=$(ls -t "$RUNS_DIR/$RUN_NAME/nn/"*.pth 2>/dev/null | head -1)   # 500 epoch마다 저장 + 끝날 때 nn/<run>.pth
[[ -f "$LATEST" ]] && { RESUME="$LATEST"; ph_log "$RUN_NAME: $LATEST 에서 이어서 학습"; }
ph_gpu_check
ph_log "학습: $RUN_NAME  scene=$SCENE robot=$ROBOT epochs=$EPOCHS"
RUNS_DIR="$RUNS_DIR" RESULTS_DIR="$RESULTS_DIR" SCENE="$SCENE" ROBOT="$ROBOT" CFG_DIR="$CFG_DIR" \
  RUN_NAME="$RUN_NAME" RESUME_FROM="$RESUME" \
  bash "$T09_REPO/experiments/scripts/train.sh" "$VARIANT" "$SEED" "$EPOCHS" \
  | grep --line-buffered -E '^\[t09|epoch_num: *[0-9]*00[^0-9]|Error|error|Traceback|Killed|out of memory'
