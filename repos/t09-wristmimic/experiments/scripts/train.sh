#!/usr/bin/env bash
# 학습 1회 실행.  사용법: bash experiments/scripts/train.sh <variant> <seed> <epochs>
#   variant: experiments/configs/env/<variant>.yaml (variants.yaml 참고)
#   epochs : 학습 epoch 수 (500의 배수 권장 -> 마지막 체크포인트가 정확히 그 epoch)
# 환경변수: NUM_ENVS, MINIBATCH, SCENE, ROBOT, RUN_NAME(이어 학습 시 같은 이름), RESUME_FROM(체크포인트)
#
# 결과 폴더 experiments/runs/<run_name>/
#   nn/          체크포인트 (<run_name>_000XXXXX.pth 500 epoch마다, _latest.pth 매 epoch)
#   summaries/   TensorBoard 로그 (wm/* = 진행률, 보상 성분, 시간)
#   source/      설정 snapshot, git diff, 재현 명령 (run.py가 저장)
#   train.log    표준 출력 전체
#   gpu.csv      30초마다 GPU 메모리/사용률
#   run_meta.json  variant, seed, 예산, 시작/종료 시각, 소요 시간, 최대 VRAM
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

VARIANT="${1:?variant 이름이 필요합니다 (예: default)}"
SEED="${2:?seed가 필요합니다 (예: 0)}"
EPOCHS="${3:?epoch 수가 필요합니다 (예: 3000)}"
shift 3

CFG_ENV="$CFG_DIR/$VARIANT.yaml"
[[ -f "$CFG_ENV" ]] || { echo "설정 없음: $CFG_ENV (make_configs.py 실행?)" >&2; exit 1; }
"$PY" experiments/tools/make_configs.py --check >/dev/null || { echo "configs/env가 variants.yaml과 다릅니다. make_configs.py를 다시 실행하세요." >&2; exit 1; }

RUN_NAME="${RUN_NAME:-${VARIANT}_s${SEED}_$(date +%m%d-%H%M%S)}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume_from "$RESUME_FROM")
fi

# ---- run_meta.json (시작) ----
"$PY" - "$RUN_DIR/run_meta.json" <<EOF
import json, subprocess, sys, os, hashlib
def sh(c):
    try: return subprocess.check_output(c, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return ''
path = sys.argv[1]
meta = json.load(open(path)) if os.path.isfile(path) else {}
meta.update(dict(
    run_name="$RUN_NAME", variant="$VARIANT", seed=int("$SEED"), epochs=int("$EPOCHS"),
    num_envs=int("$NUM_ENVS"), minibatch=int("$MINIBATCH"), scene="$SCENE", robot="$ROBOT",
    cfg_env="$CFG_ENV", cfg_env_sha1=hashlib.sha1(open("$CFG_ENV","rb").read()).hexdigest()[:12],
    cfg_train="$CFG_TRAIN", resume_from="${RESUME_FROM:-}",
    git=sh("git rev-parse --short HEAD") + ("-dirty" if sh("git status --porcelain --untracked-files=no") else ""),
    gpu=sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
    started_at=sh("date -Iseconds"), finished_at=None, exit_code=None))
meta.setdefault("segments", []).append({"started_at": meta["started_at"], "resume_from": meta["resume_from"]})
json.dump(meta, open(path, "w"), indent=1)
EOF

# ---- GPU 로거 ----
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -l 30 >> "$RUN_DIR/gpu.csv" 2>/dev/null &
GPU_LOGGER=$!

finish() {
  local code=$1
  kill "$GPU_LOGGER" 2>/dev/null
  "$PY" - "$RUN_DIR" "$code" <<'EOF'
import json, sys, os, datetime, glob
run_dir, code = sys.argv[1], int(sys.argv[2])
path = os.path.join(run_dir, "run_meta.json")
meta = json.load(open(path))
now = datetime.datetime.now().astimezone()
meta["finished_at"] = now.isoformat(timespec="seconds")
meta["exit_code"] = code
seg = meta["segments"][-1]
seg["finished_at"] = meta["finished_at"]; seg["exit_code"] = code
seg["wall_s"] = round((now - datetime.datetime.fromisoformat(seg["started_at"])).total_seconds())
meta["wall_s_total"] = sum(s.get("wall_s", 0) for s in meta["segments"])
mem = []
for line in open(os.path.join(run_dir, "gpu.csv")) if os.path.isfile(os.path.join(run_dir, "gpu.csv")) else []:
    try: mem.append(float(line.split(",")[1]))
    except Exception: pass
meta["peak_gpu_mem_mib"] = max(mem) if mem else None
meta["checkpoints"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "nn", "*.pth")))
json.dump(meta, open(path, "w"), indent=1)
print(f"[t09] run_meta.json 갱신: exit={code} wall={seg['wall_s']}s peak_gpu={meta['peak_gpu_mem_mib']} MiB")
EOF
}
trap 'finish 130; exit 130' INT TERM

t09_log "학습 시작: $RUN_NAME  (variant=$VARIANT seed=$SEED epochs=$EPOCHS envs=$NUM_ENVS)"
t09_log "폴더: $RUN_DIR   모니터링: tensorboard --logdir $RUNS_DIR"

INTERMIMIC_EXACT_RUN_NAME=1 "$PY" intermimic/run.py \
  --task InterMimic_MULTI_OBJ \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --output "$RUNS_DIR" \
  --experiment "$RUN_NAME" \
  --num_envs "$NUM_ENVS" \
  --minibatch_size "$MINIBATCH" \
  --max_iterations "$EPOCHS" \
  --seed "$SEED" \
  --motion_file "$SCENE" \
  --robot_type "$ROBOT" \
  --headless \
  --num_position_iterations 20 \
  --num_velocity_iterations 0 \
  "${RESUME_ARGS[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
CODE=${PIPESTATUS[0]}

trap - INT TERM
finish "$CODE"
t09_log "학습 종료: $RUN_NAME (exit $CODE)"

exit "$CODE"
