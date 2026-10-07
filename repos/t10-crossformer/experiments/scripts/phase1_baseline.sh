#!/usr/bin/env bash
# Phase 1: 공식 사용법 그대로의 frozen CrossFormer를 held-out episode 전체에 오프라인 평가.
#   baseline_lang  공식 notebook의 언어 조건 (zero goal image + 문장 embedding)
#   baseline_goal  학습과 같은 goal-image 조건 (마지막 frame을 goal로, 언어 0)
# 이어서 지표(단위 probe 포함)와 리포트를 만든다. GPU 약 10분 (bridge+fractal 기준 추정).
#   사용법: bash experiments/scripts/phase1_baseline.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
t10_require_setup
for v in baseline_lang baseline_goal; do run_offline "$v"; done
bash experiments/scripts/report.sh
