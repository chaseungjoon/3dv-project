#!/usr/bin/env bash
# 데이터 audit (CPU, 약 1분): Place Book / Drink Cup / Kettle 구간의 운동학 난이도
#   -> experiments/results/audit/AUDIT.md, clips.csv
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PY" experiments/tools/audit_clips.py "$@"
