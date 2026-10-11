#!/usr/bin/env bash
# SkillMimic-V2 baseline 한 clip: 학습 -> 평가. run_cup.sh / run_book.sh / run_kettle.sh 가 부른다.
#   사용법: bash experiments/scripts/run_baseline.sh <drink_cup|place_book|place_kettle> [method]
#   method 기본값: drink_cup은 fix (고친 환경, 프로젝트 루트 SKILLMIMIC_NOTES.md 4절), 나머지는 ours (업스트림 그대로). _common.sh smv2_method
# 환경변수: EPOCHS (기본 3000 = 1.97억 샘플, SKILLMIMIC_NOTES.md 1.2), SEED (0), NUM_ENVS (2048), EVAL_EVERY (250), TRAIN_RETRIES (2)
#
# 끊겨도 (Ctrl-C, 터미널 종료, 정전) 같은 명령을 다시 실행하면 된다.
#   학습: 마지막 체크포인트(10 epoch마다 저장)부터 epoch, optimizer, ATS 상태까지 그대로 이어서 한다.
#   평가: 이미 끝난 체크포인트는 건너뛴다. 학습이 끝난 run이면 학습은 건너뛰고 남은 평가만 한다.
# 결과 문서는 만들지 않는다. 세 run이 끝나면 저장소 루트에서 bash write_results.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
smv2_require_setup

CLIP="${1:?clip이 필요합니다 (drink_cup|place_book|place_kettle)}"
EPOCHS="${EPOCHS:-3000}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
METHOD="${2:-$(smv2_method "$CLIP")}"
RUN="${CLIP}_${METHOD}_n${NUM_ENVS}_s${SEED}"

smv2_log "=== $RUN: 학습 (목표 $EPOCHS epoch) ==="
# 학습이 죽으면 (예: PhysX GPU 메모리 할당 실패, 첫 컵 run에서 epoch 624에 한 번) 1분 쉬고 마지막 체크포인트부터
# TRAIN_RETRIES번(기본 2)까지 다시 이어서 한다. Ctrl-C(130)는 다시 하지 않는다.
TRAIN_RETRIES="${TRAIN_RETRIES:-2}"
for try in $(seq 0 "$TRAIN_RETRIES"); do
  (( try > 0 )) && { smv2_log "학습이 죽었다 (exit $CODE). 1분 뒤 마지막 체크포인트부터 다시 ($try/$TRAIN_RETRIES)"; sleep 60; }
  bash experiments/scripts/train.sh "$CLIP" "$METHOD" "$EPOCHS" "$SEED"
  CODE=$?
  (( CODE == 0 || CODE == 130 )) && break
done
if (( CODE != 0 )); then
  smv2_log "학습이 끝나지 않았다 (exit $CODE). 같은 명령을 다시 실행하면 마지막 체크포인트부터 이어서 한다."
  exit "$CODE"
fi

smv2_log "=== $RUN: 평가 (학습 곡선 ${EVAL_EVERY} epoch마다 + 최종) ==="
CURVE=$(seq -s, "$EVAL_EVERY" "$EVAL_EVERY" "$EPOCHS")
(( EPOCHS % EVAL_EVERY == 0 )) || CURVE="$CURVE,$EPOCHS"
bash experiments/scripts/eval_run.sh "$RUN" curve "$CURVE" || exit $?
bash experiments/scripts/eval_run.sh "$RUN" final || exit $?
smv2_log "=== $RUN 끝. 세 run이 모두 끝나면: bash write_results.sh ==="
