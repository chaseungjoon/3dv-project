#!/usr/bin/env bash
# 학습 1회.  사용법: bash experiments/scripts/train.sh <clip> <method> <epochs> [seed]
#   clip   : place_book | drink_cup | place_kettle
#   method : ours (SkillMimic-V2)
#   epochs : **총** epoch 수. 1 epoch = NUM_ENVS x 32 샘플 (2048 env이면 65,536)
# 환경변수: NUM_ENVS (기본 2048), RUN_NAME (기본 <clip>_<method>_n<envs>_s<seed>)
#
# 이어 학습: 같은 run 폴더에 체크포인트가 있으면 **가장 최근에 저장된 것부터 그대로 이어서** 학습한다
#   (epoch 번호, optimizer, 정규화 통계, ATS 샘플링 가중치, 난수 상태까지 복원. experiments/tools/ckpt.py).
#   이미 epochs 이상 학습했으면 건너뛴다. epochs를 늘려서 다시 부르면 거기서부터 더 학습한다.
# 결과 폴더 experiments/runs/<run_name>/
#   nn/<run_name>_000XXXXX.pth  10 epoch마다, 지우거나 덮어쓰지 않음,  nn/<run_name>.pth  최신
#   nn/<run_name>_e<N>_r<보상>.pth  보상 최고 2개 (업스트림 동작)
#   summaries/ TensorBoard,  train.log 표준 출력 전체 (이어 쓰기),  gpu.csv 30초마다 GPU,  run_meta.json 구간별 기록
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
smv2_require_setup

CLIP="${1:?clip이 필요합니다 (place_book|drink_cup|place_kettle)}"
METHOD="${2:?method가 필요합니다 (ours)}"
EPOCHS="${3:?epoch 수가 필요합니다 (예: 3000)}"
SEED="${4:-0}"
ARGS=$("$PY" experiments/tools/methods.py train_args "$CLIP" "$METHOD") || exit 1

RUN_NAME="${RUN_NAME:-${CLIP}_${METHOD}_n${NUM_ENVS}_s${SEED}}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"

RESUME_ARGS=()
RESUME_PATH="" RESUME_EPOCH=0
if [[ -d "$RUN_DIR/nn" ]]; then
  IFS=$'\t' read -r RESUME_PATH RESUME_EPOCH < <("$PY" experiments/tools/ckpt.py latest "$RUN_DIR")
  RESUME_EPOCH="${RESUME_EPOCH:-0}"
fi
if [[ -n "$RESUME_PATH" ]]; then
  if (( RESUME_EPOCH >= EPOCHS )); then
    smv2_log "$RUN_NAME: 이미 epoch $RESUME_EPOCH >= $EPOCHS. 학습 건너뜀"; exit 0
  fi
  smv2_log "$RUN_NAME: epoch $RESUME_EPOCH 에서 이어서 학습 ($(basename "$RESUME_PATH"))"
  RESUME_ARGS=(--resume_from "$RESUME_PATH")
fi

"$PY" - "$RUN_DIR/run_meta.json" <<PYEOF
import json, os, subprocess, sys
def sh(c):
    try: return subprocess.check_output(c, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return ''
p = sys.argv[1]
m = json.load(open(p)) if os.path.isfile(p) else {}
m.update(dict(run_name="$RUN_NAME", clip="$CLIP", method="$METHOD", epochs=int("$EPOCHS"), seed=int("$SEED"),
              num_envs=int("$NUM_ENVS"), samples_per_epoch=int("$NUM_ENVS") * 32, args="$ARGS",
              git=sh("git rev-parse --short HEAD"), gpu=sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")))
m.setdefault("segments", []).append(dict(started_at=sh("date -Iseconds"), resume_from="$RESUME_PATH",
                                         resume_epoch=int("$RESUME_EPOCH"), target_epochs=int("$EPOCHS")))
json.dump(m, open(p, "w"), indent=1)
PYEOF

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 30 >> "$RUN_DIR/gpu.csv" 2>/dev/null &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null' EXIT
# Ctrl-C는 학습 프로세스에도 같이 간다. 셸은 살아남아 이 구간의 기록(run_meta.json)을 마무리한다.
trap 'smv2_log "중단 신호. 학습 프로세스 종료를 기다린다 (다시 실행하면 마지막 저장부터 이어서 학습)"' INT

smv2_gpu_check
smv2_log "학습 시작: $RUN_NAME (clip=$CLIP method=$METHOD epochs=$EPOCHS seed=$SEED envs=$NUM_ENVS)"
# shellcheck disable=SC2086
SMV2_RUN_NAME="$RUN_NAME" "$PY" skillmimic/run.py $ARGS --experiment "$RUN_NAME" \
  --num_envs "$NUM_ENVS" --max_epochs "$EPOCHS" --seed "$SEED" \
  --output_path "$RUNS_DIR/" --headless "${RESUME_ARGS[@]}" 2>&1 \
  | tee -a "$RUN_DIR/train.log" | smv2_filter | grep --line-buffered -E 'epoch_num:[0-9]*[05]0 |MAX EPOCHS|resumed full training state|Error|error|Traceback|Killed|out of memory'
CODE=${PIPESTATUS[0]}
trap - INT

"$PY" - "$RUN_DIR" "$CODE" <<'PYEOF'
import datetime, glob, json, os, sys
d, code = sys.argv[1], int(sys.argv[2])
p = os.path.join(d, "run_meta.json"); m = json.load(open(p))
now = datetime.datetime.now().astimezone()
seg = m["segments"][-1]
seg.update(finished_at=now.isoformat(timespec="seconds"), exit_code=code,
           wall_s=round((now - datetime.datetime.fromisoformat(seg["started_at"])).total_seconds()))
m["wall_s_total"] = sum(s.get("wall_s", 0) for s in m["segments"])
mem = []
if os.path.isfile(os.path.join(d, "gpu.csv")):
    for line in open(os.path.join(d, "gpu.csv")):
        try: mem.append(float(line.split(",")[1]))
        except Exception: pass
m["peak_gpu_mem_mib"] = max(mem) if mem else None
m["checkpoints"] = len(glob.glob(os.path.join(d, "nn", "*.pth")))
json.dump(m, open(p, "w"), indent=1)
PYEOF
smv2_log "학습 끝: $RUN_NAME (exit $CODE)"
exit "$CODE"
