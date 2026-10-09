#!/usr/bin/env bash
# Diffusion Policy 학습 (공식 예제 그대로: 64-step horizon, 50-step chunk, batch 16, 5K마다 저장).
#   사용법: bash experiments/scripts/train.sh [steps]      (기본 20000 = 공식 기본값)
# 환경변수: TASK (기본 02_push_button), SEED (기본 1000 = 공식 예시), BATCH (기본 16), NUM_WORKERS (기본 4)
# 같은 명령을 다시 실행하면 마지막 완전한 체크포인트부터 이어서 한다 (--resume).
# 결과: experiments/runs/<task>_dp_s<seed>/ (checkpoints/005000 ... , train.log, gpu.csv, run_meta.json)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
hmg_require_setup
STEPS="${1:-20000}"
SEED="${SEED:-1000}"; BATCH="${BATCH:-16}"; NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
RUN="$(hmg_run_name)"; D="$RUNS_DIR/$RUN"
RESUME=(); [[ -d "$D/checkpoints" ]] && RESUME=(--resume) && hmg_log "$RUN: 이어서 학습"
mkdir -p "$RUNS_DIR"
[[ -d "$D" && ${#RESUME[@]} -eq 0 ]] && { echo "$D 가 있지만 체크포인트가 없다. 지우고 다시: rm -rf $D" >&2; exit 1; }
hmg_gpu_check
LOGDIR="$RUNS_DIR/_logs/$RUN"; mkdir -p "$LOGDIR"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 30 >> "$LOGDIR/gpu.csv" 2>/dev/null &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
cat > "$LOGDIR/run_meta.json" <<JSON
{"run": "$RUN", "task": "$TASK", "steps": $STEPS, "seed": $SEED, "batch_size": $BATCH, "save_freq": $SAVE_FREQ,
 "gpu": "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)", "started_at": "$(date -Iseconds)"}
JSON
hmg_log "학습: $RUN (steps=$STEPS batch=$BATCH)"
START=$(date +%s)
"$PY" scripts/train_policy_example.py --task "$TASK" --output-dir "$D" --data-dir "$DATA_DIR" \
  --steps "$STEPS" --batch-size "$BATCH" --save-freq "$SAVE_FREQ" --num-workers "$NUM_WORKERS" --seed "$SEED" "${RESUME[@]}" 2>&1 \
  | tee -a "$LOGDIR/train.log" | grep --line-buffered -E "step:[0-9.]+K? |Checkpoint|Error|Traceback|out of memory|HMG_"
CODE=${PIPESTATUS[0]}
"$PY" - "$LOGDIR" "$CODE" "$START" <<'PYEOF'
import json, os, sys, time
d, code, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = os.path.join(d, "run_meta.json"); m = json.load(open(p))
mem = []
for l in open(os.path.join(d, "gpu.csv")) if os.path.isfile(os.path.join(d, "gpu.csv")) else []:
    try: mem.append(float(l.split(",")[1]))
    except Exception: pass
m.setdefault("segments", []).append(dict(exit_code=code, wall_s=int(time.time()) - start))
m.update(peak_gpu_mem_mib=max(mem) if mem else None, wall_s_total=sum(s["wall_s"] for s in m["segments"]))
json.dump(m, open(p, "w"), indent=1)
PYEOF
hmg_log "학습 끝: $RUN (exit $CODE)"
exit "$CODE"
