#!/usr/bin/env bash
# Phase 2: 같은 예산의 SkillMimic(sm, v1) baseline. 논문에서는 이 clip들에서 0%.
#   "참조 동작을 그대로 따라가는 방식은 쥐고 드는 순간에서 실패한다"(T09와 같은 실패)가 5070에서도 재현되는가.
#   run 하나 약 5.5~6시간 (6.5~7.3 s/epoch).
# 환경변수: CLIPS (기본 "drink_cup place_kettle"), EPOCHS (기본 3000), SEED, NUM_ENVS
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CLIPS="${CLIPS:-drink_cup place_kettle}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
for c in $CLIPS; do
  bash experiments/scripts/train.sh "$c" sm "$EPOCHS" "$SEED" || { smv2_log "학습 실패: $c"; exit 1; }
  RUN="${c}_sm_n${NUM_ENVS}_s${SEED}"
  bash experiments/scripts/eval_run.sh "$RUN" all
  bash experiments/scripts/eval_run.sh "$RUN" final
  bash experiments/scripts/report.sh
done
smv2_log "Phase 2 끝. experiments/results/report/REPORT.md"
