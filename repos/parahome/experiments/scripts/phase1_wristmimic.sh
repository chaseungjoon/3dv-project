#!/usr/bin/env bash
# Phase 1: T09 방법(WristMimic)을 그대로 Drink Cup / Place Book에. "T09 0%는 주전자 장면 탓인가?"
#   T09 baseline과 같은 설정 (1024 env, 12GB 버퍼, 항상 프레임 0에서 시작, 같은 PPO), 장면과 XML만 다르다.
#   장면 하나 약 6.2 s/epoch -> 3000 epoch 약 5.2시간. 비교 대상 = T09 kettle baseline의 같은 epoch 체크포인트.
# 환경변수: SCENES (기본 "cup_s110 book_s11"), EPOCHS (기본 3000, 500의 배수), SEED
# 끊겨도 같은 명령을 다시 실행하면 이어서 한다 (_latest.pth).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SCENES="${SCENES:-cup_s110 book_s11}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
for s in $SCENES; do
  bash experiments/scripts/train_wm.sh "$s" "$EPOCHS" "$SEED" || { ph_log "학습 실패: $s"; exit 1; }
  bash experiments/scripts/eval_wm.sh "wm_${s}_default_s${SEED}" both
  bash experiments/scripts/report.sh "$EPOCHS"
done
ph_log "Phase 1 끝. experiments/results/report/REPORT.md"
