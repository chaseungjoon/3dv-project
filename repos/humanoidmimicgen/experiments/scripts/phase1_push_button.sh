#!/usr/bin/env bash
# Phase 1: Push Button. 공식 예제 그대로 Diffusion Policy 20K update 학습 -> 5K/10K/15K/20K 체크포인트를 100 episode씩 평가,
#   + 학습 없는 기준 정책 4개 (baselines.sh, 성공 판정의 바닥값). 약 2.5~3시간.
# 환경변수: TASK (기본 02_push_button), STEPS (기본 20000), EPISODES (기본 100), SEED (기본 1000)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
STEPS="${STEPS:-20000}"; EPISODES="${EPISODES:-100}"
bash experiments/scripts/train.sh "$STEPS" || exit 1
bash experiments/scripts/eval.sh "$EPISODES"
bash experiments/scripts/baselines.sh "$EPISODES"
bash experiments/scripts/report.sh
hmg_log "Phase 1 끝. experiments/results/report/REPORT.md"
