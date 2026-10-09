#!/usr/bin/env bash
# 한 번만 실행. 다시 실행해도 안전하다 (있는 것은 건너뜀).
#  1) uv 환경 (Isaac Gym + 직접 빌드한 torch 2.4.1 sm_120, ../../isaacgym-env)
#  2) 원본 ParaHome 장면 s6/s10/s22 링크 (업스트림 코드가 책상/식탁 위치를 여기서 읽는다)
#  3) history encoder 학습 (ParaHome 8개 clip, 3000 epoch, 약 6분)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ ! -x "$PY" ]]; then
  smv2_log "uv sync"
  uv sync || exit 1
fi

for s in s6 s10 s22; do
  if [[ ! -f "$PARAHOME_SEQ/$s/object_transformations.pkl" ]]; then
    echo "원본 ParaHome 없음: $PARAHOME_SEQ/$s  ->  bash ../parahome/experiments/scripts/setup.sh 먼저" >&2; exit 2
  fi
  ln -sfn "$(realpath --relative-to=skillmimic/data/motions/ParaHome "$PARAHOME_SEQ/$s")" "skillmimic/data/motions/ParaHome/$s"
done
smv2_log "ParaHome 링크: $(ls -d skillmimic/data/motions/ParaHome/s{6,10,22} | tr '\n' ' ')"

HIST=experiments/checkpoints/hist_encoder/parahome_hist60.ckpt
if [[ ! -f "$HIST" ]]; then
  mkdir -p experiments/logs
  smv2_log "history encoder 학습 (약 6분) -> $HIST"
  "$PY" experiments/tools/train_hist_encoder.py --out "$HIST" 2>&1 | tee experiments/logs/hist_encoder.log | grep '\[hist\]' || exit 1
fi
smv2_log "준비 완료"
