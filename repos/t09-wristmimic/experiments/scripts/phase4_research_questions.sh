#!/usr/bin/env bash
# Phase 4: 연구 질문 변형 (seed 0, 예산 T).
#   RQ1 회전 임계값: rot2_x2, rot2_x0.5   (Phase 3의 pos2_*와 짝)
#   RQ2 손목 보상 가중치: gwp_30, gwp_140
#   RQ2 접촉 창 시점: win_early5, win_late5
#   사용법: BUDGET=<T> bash experiments/scripts/phase4_research_questions.sh
#   하나만 고를 때: JOBS="rot2_x2:0 rot2_x0.5:0" BUDGET=<T> bash ...
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t09_require_budget
# shellcheck disable=SC2086
bash experiments/scripts/queue.sh "$BUDGET" ${JOBS:-rot2_x2:0 rot2_x0.5:0 gwp_30:0 gwp_140:0 win_early5:0 win_late5:0}
