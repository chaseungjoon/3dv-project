#!/usr/bin/env bash
# 여러 학습을 순서대로 실행 (GPU 1장이라 동시 실행하지 않는다).
#   사용법: bash experiments/scripts/queue.sh <epochs> <variant:seed> [<variant:seed> ...]
#   예)     bash experiments/scripts/queue.sh 3000 default:1 default:2
# 각 학습이 끝나면 AUTO_EVAL=1(기본)이면 바로 그 run을 평가한다 (eval_run.sh, 수 분).
# 한 run이 실패해도 다음 run으로 넘어간다. 마지막에 성공/실패 목록을 출력한다.
# 큐를 중간에 멈추려면 Ctrl+C (진행 중 run의 체크포인트는 남는다; README 6절 '이어 학습').
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EPOCHS="${1:?epoch 수가 필요합니다}"
shift
[[ $# -gt 0 ]] || { echo "variant:seed 를 하나 이상 주세요" >&2; exit 2; }
AUTO_EVAL="${AUTO_EVAL:-1}"

declare -a DONE FAILED
trap 'echo; t09_log "큐 중단됨. 완료: ${DONE[*]:-없음} / 실패: ${FAILED[*]:-없음}"; exit 130' INT

for job in "$@"; do
  VARIANT="${job%%:*}"
  SEED="${job##*:}"
  NAME="${VARIANT}_s${SEED}_$(date +%m%d-%H%M%S)"
  t09_log "===== [$job] 학습 시작 ($EPOCHS epochs) -> $RUNS_DIR/$NAME ====="
  if RUN_NAME="$NAME" bash experiments/scripts/train.sh "$VARIANT" "$SEED" "$EPOCHS"; then
    DONE+=("$NAME")
    if [[ "$AUTO_EVAL" == "1" ]]; then
      bash experiments/scripts/eval_run.sh "$RUNS_DIR/$NAME" || t09_log "평가 실패: $NAME (나중에 eval_run.sh로 다시)"
    fi
  else
    FAILED+=("$NAME")
    t09_log "학습 실패: $NAME (train.log 확인)"
  fi
done

t09_log "큐 완료. 완료: ${DONE[*]:-없음} / 실패: ${FAILED[*]:-없음}"
[[ ${#FAILED[@]} -eq 0 ]]
