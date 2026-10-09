#!/usr/bin/env bash
# 표/그림 다시 만들기 (GPU 안 씀).  사용법: bash experiments/scripts/report.sh [budget_epoch]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARGS=(); [[ -n "${1:-}" ]] && ARGS=(--budget "$1")
"$PY" experiments/tools/report.py "${ARGS[@]}"
