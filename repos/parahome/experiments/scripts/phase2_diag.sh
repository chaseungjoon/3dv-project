#!/usr/bin/env bash
# Phase 2 (선택): T09 진단 설정(랜덤 시작 Hybrid, rolloutLength 60, PSI 끔)을 같은 장면에.
#   T09 kettle에서는 이 설정으로 "들고 걷기"는 배웠지만 "쥐고 들기"(프레임 45~60)는 그대로 막혔다.
#   컵/책에서도 같은 순간에 막히면 -> 쥐고 드는 동작 자체가 이 방법/예산의 병목 (장면 문제가 아님).
#   장면 하나 약 8 s/epoch -> 2000 epoch 약 4.5시간.
# 환경변수: SCENES (기본 "cup_s110"), EPOCHS (기본 2000)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SCENES="${SCENES:-cup_s110}"
EPOCHS="${EPOCHS:-2000}"
[[ -f "$T09_REPO/experiments/configs/diag/diag_hybrid.yaml" ]] || { echo "T09 진단 설정 없음 (../t09-wristmimic/experiments/scripts/diag_hybrid.sh 가 만든다)" >&2; exit 2; }
for s in $SCENES; do
  VARIANT=diag_hybrid bash experiments/scripts/train_wm.sh "$s" "$EPOCHS" 0 || exit 1
  bash experiments/scripts/eval_wm.sh "wm_${s}_diag_hybrid_s0" both
done
bash experiments/scripts/report.sh
