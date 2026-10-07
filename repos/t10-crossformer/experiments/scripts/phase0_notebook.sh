#!/usr/bin/env bash
# Phase 0: 공식 Colab 예제(inference_pretrained.ipynb)를 로컬에서 재현 (오프라인 추론 확인, 약 1분).
# 결과: experiments/results/notebook/ (notebook_step2_bridge.png, notebook_repro.json)
#   사용법: bash experiments/scripts/phase0_notebook.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
"$PY" experiments/tools/notebook_repro.py 2>&1 | t10_filter
