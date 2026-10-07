#!/usr/bin/env bash
# 저장된 예측과 closed-loop 결과로 지표, 표, 그림, REPORT.md를 만든다 (GPU 안 씀, 1분 이내).
# 결과: experiments/results/report/ (REPORT.md, offline.csv, sim.csv, fig_*.png)
#   사용법: bash experiments/scripts/report.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_log "지표 계산 (experiments/results/metrics/)"
"$PY" experiments/tools/metrics.py 2>&1 | t10_filter
t10_log "리포트 (experiments/results/report/REPORT.md)"
"$PY" experiments/tools/aggregate.py 2>&1 | t10_filter
