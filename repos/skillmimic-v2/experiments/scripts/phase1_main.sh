#!/usr/bin/env bash
# Phase 1 (주 실험): SkillMimic-V2(ours)가 12GB 5070에서 Place Book / Drink Cup / Place Kettle을 배우는가.
#   논문과 같은 2048 env, 같은 PPO 설정. 예산 EPOCHS (기본 3000 = 1.97억 샘플 = T09 baseline과 같은 샘플 수,
#   논문 1.0B의 20%). run 하나 약 9~10시간 (10.6~12 s/epoch). 끝나면 250 epoch마다 저장된 체크포인트를 모두 평가.
# 환경변수: CLIPS (기본 "drink_cup place_book place_kettle"), EPOCHS, SEED, NUM_ENVS
# 중간에 끊겨도 같은 명령을 다시 실행하면 마지막 저장(250 epoch 단위)부터 이어서 한다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CLIPS="${CLIPS:-drink_cup place_book place_kettle}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
for c in $CLIPS; do
  bash experiments/scripts/train.sh "$c" ours "$EPOCHS" "$SEED" || { smv2_log "학습 실패: $c"; exit 1; }
  RUN="${c}_ours_n${NUM_ENVS}_s${SEED}"
  bash experiments/scripts/eval_run.sh "$RUN" all
  bash experiments/scripts/eval_run.sh "$RUN" final
  bash experiments/scripts/report.sh
done
smv2_log "Phase 1 끝. experiments/results/report/REPORT.md"
