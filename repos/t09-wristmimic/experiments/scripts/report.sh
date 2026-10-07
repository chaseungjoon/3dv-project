#!/usr/bin/env bash
# 평가 결과(json)와 학습 로그를 모아 표, 그림, REPORT.md를 만든다 (GPU 안 씀, 수 초).
#   사용법: bash experiments/scripts/report.sh [BUDGET]
#   BUDGET(학습 예산 T, epoch)을 주면 모든 비교를 그 epoch의 체크포인트로 한다.
#   안 주면 각 run의 마지막 체크포인트를 쓰고, Phase 1 곡선에서 T 후보를 제안한다.
# 결과: experiments/results/report/ (REPORT.md, *.csv, *.md 표, fig_*.png)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ARGS=()
[[ -n "${1:-${BUDGET:-}}" ]] && ARGS+=(--budget "${1:-$BUDGET}")
"$PY" experiments/tools/aggregate.py "${ARGS[@]}"
