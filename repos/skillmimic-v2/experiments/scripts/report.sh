#!/usr/bin/env bash
# 결과 표/그림 다시 만들기 (GPU 안 씀, 수 초).  -> experiments/results/report/REPORT.md
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" experiments/tools/report.py "$@" 2>&1 | grep -v -i warn
