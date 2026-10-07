#!/usr/bin/env bash
# (임시) 중단된 학습을 가장 최근 체크포인트에서 이어 학습한다.
#   사용법: bash experiments/scripts/resume_latest.sh            # 가장 최근에 만든 run
#           bash experiments/scripts/resume_latest.sh <run_dir>  # 특정 run
# variant/seed/목표 epoch는 run_meta.json에서 읽는다. 끝나면 그 run을 평가한다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN_DIR="${1:-$(ls -dt "$RUNS_DIR"/*/ 2>/dev/null | grep -v '/_[^/]*/$' | head -1)}"
RUN_DIR="${RUN_DIR%/}"
[[ -f "$RUN_DIR/run_meta.json" ]] || { echo "run을 찾을 수 없음: $RUN_DIR" >&2; exit 1; }
NAME="$(basename "$RUN_DIR")"

read -r VARIANT SEED EPOCHS < <("$PY" -c "import json;m=json.load(open('$RUN_DIR/run_meta.json'));print(m['variant'],m['seed'],m['epochs'])")

# _latest.pth가 우선 (매 epoch 저장). 없으면 번호가 가장 큰 체크포인트.
CKPT="$RUN_DIR/nn/${NAME}_latest.pth"
[[ -f "$CKPT" ]] || CKPT="$(ls "$RUN_DIR"/nn/"${NAME}"_[0-9]*.pth 2>/dev/null | sort | tail -1)"
[[ -f "$CKPT" ]] || { echo "체크포인트 없음: $RUN_DIR/nn" >&2; exit 1; }

t09_log "이어 학습: $NAME (variant=$VARIANT seed=$SEED 목표 epoch=$EPOCHS) <- $CKPT"
RUN_NAME="$NAME" RESUME_FROM="$CKPT" bash experiments/scripts/train.sh "$VARIANT" "$SEED" "$EPOCHS" || exit $?
bash experiments/scripts/eval_run.sh "$RUN_DIR"
