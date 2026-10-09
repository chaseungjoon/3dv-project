#!/usr/bin/env bash
# 한 번만 실행 (다시 실행해도 안전). README의 설치 순서 그대로, pip 대신 uv.
#  1) Python 3.10 venv + humanoidmimicgen[wbc-replay]
#  2) 학습용 LeRobot (고정 commit), torch 2.10 cu128 (RTX 50 sm_120 지원), numpy 1.26
#  3) WBC 하체 정책 (stand.onnx, walk.onnx)
#  4) 사람 시범 1개 (재생 점검용, 0.8 MB) + 학습 데이터 TASK의 8개 shard (push button 약 6.8 GB, 1~2시간)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ ! -x "$PY" ]]; then
  uv venv --python 3.10 .venv || exit 1
  uv pip install --python "$PY" -e ".[wbc-replay]" || exit 1
  uv pip install --python "$PY" "lerobot @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a" || exit 1
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0" "torchvision==0.25.0" || exit 1
  uv pip install --python "$PY" "numpy==1.26.4" "opencv-python-headless==4.11.0.86" || exit 1
  uv pip install --python "$PY" matplotlib || exit 1   # 리포트 그림용 (추가)
fi
"$PY" -m humanoidmimicgen.download_wbc_policies | tail -2
if [[ ! -f "experiments/data/source_demo_replay/datasets/$TASK/demo.hdf5" ]]; then
  .venv/bin/hf download linkenv/humanoidmimicgen-g1-source-demo-replay --repo-type dataset \
    --include "datasets/$TASK/*" --local-dir experiments/data/source_demo_replay || exit 1
fi
mkdir -p experiments/logs
hmg_log "학습 데이터 준비: $TASK -> $DATA_DIR (처음이면 1~2시간)"
"$PY" experiments/tools/prepare_data.py --task "$TASK" --data-dir "$DATA_DIR" 2>&1 | tee "experiments/logs/prepare_$TASK.log" | grep --line-buffered '^\[prep\]\|Error' || exit 1
hmg_log "준비 완료"
