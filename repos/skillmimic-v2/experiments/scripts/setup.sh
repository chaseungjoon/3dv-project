#!/usr/bin/env bash
# 한 번만 실행. 다시 실행해도 안전하다 (있는 것은 건너뜀).
#  1) uv 환경 (Isaac Gym + 직접 빌드한 torch 2.4.1 sm_120, ../../isaacgym-env)
#  2) 원본 ParaHome 장면 s6/s10/s22 (업스트림 코드가 책상/식탁 위치를 여기서 읽는다). experiments/data/parahome_seq에 없으면
#     ParaHome seq.zip(Google Drive, 약 2.7 GB)을 받아 세 장면만 풀고 지운다 (87 MB). 그리고 skillmimic/data/motions/ParaHome/ 에 링크
#  3) history encoder 학습 (ParaHome 8개 clip, 3000 epoch, 약 6분)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ ! -x "$PY" ]]; then
  smv2_log "uv sync"
  uv sync || exit 1
fi

PARAHOME_SEQ_GDRIVE=10MYSSM2H7f6g2n9nnXta48qmAhZ7r4yd   # ParaHome README의 seq.zip
if ! (for s in s6 s10 s22; do [[ -f "$PARAHOME_SEQ/$s/object_transformations.pkl" ]] || exit 1; done); then
  TMP=$(mktemp -d)
  smv2_log "ParaHome seq.zip 다운로드 (약 2.7 GB, 세 장면만 남김) -> $PARAHOME_SEQ"
  uvx gdown "$PARAHOME_SEQ_GDRIVE" -O "$TMP/seq.zip" || { rm -rf "$TMP"; exit 1; }
  unzip -q "$TMP/seq.zip" 'seq/s6/*' 'seq/s10/*' 'seq/s22/*' -d "$TMP" || { rm -rf "$TMP"; exit 1; }
  mkdir -p "$PARAHOME_SEQ" && mv "$TMP"/seq/s6 "$TMP"/seq/s10 "$TMP"/seq/s22 "$PARAHOME_SEQ"/ && rm -rf "$TMP"
fi
for s in s6 s10 s22; do
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
