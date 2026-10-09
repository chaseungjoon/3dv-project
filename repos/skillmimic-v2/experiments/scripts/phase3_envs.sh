#!/usr/bin/env bash
# Phase 3 (하드웨어 가설 직접 검증): env 수를 T09처럼 1024로 줄이면 결과가 나빠지는가.
#   T09는 2048에서 OOM이라 1024로 학습했다. 여기서는 같은 방법(ours), 같은 clip, **같은 샘플 수**
#   (1024 env x 2배 epoch)로 2048 run(Phase 1)과 비교한다. minibatch 16384는 T09처럼 그대로 둔다
#   (epoch당 gradient step이 절반 -> 같은 샘플에서 gradient step도 같다).
# 환경변수: CLIP (기본 drink_cup), EPOCHS (기본 6000 = Phase 1의 3000 x 2), SEED
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CLIP="${CLIP:-drink_cup}"
EPOCHS="${EPOCHS:-6000}"
SEED="${SEED:-0}"
export NUM_ENVS=1024
bash experiments/scripts/train.sh "$CLIP" ours "$EPOCHS" "$SEED" || exit 1
RUN="${CLIP}_ours_n1024_s${SEED}"
bash experiments/scripts/eval_run.sh "$RUN" all
bash experiments/scripts/eval_run.sh "$RUN" final
bash experiments/scripts/report.sh
smv2_log "Phase 3 끝. experiments/results/report/REPORT.md"
