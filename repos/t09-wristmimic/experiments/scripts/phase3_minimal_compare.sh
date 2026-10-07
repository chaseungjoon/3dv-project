#!/usr/bin/env bash
# Phase 3: 가이드의 최소 비교 + 첫 실험 (seed 0, 예산 T).
#   reset_off  : 손목 reset 끔            wrist_off : 손목 제약 전부 끔
#   pos2_x2    : stage 2 위치 임계값 2배  pos2_x0.5 : 0.5배
#   사용법: BUDGET=<T> bash experiments/scripts/phase3_minimal_compare.sh
#   순서는 중요도 순. 시간이 부족하면 앞의 두 개만: JOBS="reset_off:0 pos2_x2:0"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t09_require_budget
# shellcheck disable=SC2086
bash experiments/scripts/queue.sh "$BUDGET" ${JOBS:-reset_off:0 pos2_x2:0 wrist_off:0 pos2_x0.5:0}
