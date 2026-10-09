#!/usr/bin/env bash
# 표/그림 다시 만들기 (GPU 안 씀). -> experiments/results/report/REPORT.md
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" experiments/tools/report.py "$@"
