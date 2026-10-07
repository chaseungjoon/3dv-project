#!/usr/bin/env bash
# Phase 5: 주장에 쓸 변형에 seed 1, 2를 추가 (가설 판정은 seed 3개 평균으로 한다).
#   사용법: BUDGET=<T> bash experiments/scripts/phase5_extra_seeds.sh <variant> [<variant> ...]
#   예)     BUDGET=3000 bash experiments/scripts/phase5_extra_seeds.sh pos2_x2 rot2_x2
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t09_require_budget
[[ $# -gt 0 ]] || { echo "variant를 하나 이상 주세요" >&2; exit 2; }
JOBS=()
for v in "$@"; do JOBS+=("$v:1" "$v:2"); done
bash experiments/scripts/queue.sh "$BUDGET" "${JOBS[@]}"
