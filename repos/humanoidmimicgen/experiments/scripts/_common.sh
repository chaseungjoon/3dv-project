# HumanoidMimicGen 실험 공통 설정. 다른 스크립트가 source 한다. 값은 환경변수로 덮어쓸 수 있다.
set -o pipefail

HMG_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HMG_REPO" || exit 1

PY="${PY:-$HMG_REPO/.venv/bin/python}"
TASK="${TASK:-02_push_button}"                       # 9개 중 하나 (scripts/train_policy_example.py --help)
DATA_DIR="${DATA_DIR:-$HMG_REPO/experiments/data/policy_data}"   # 학습 데이터 캐시 (push button 약 6.8 GB)
RUNS_DIR="${RUNS_DIR:-$HMG_REPO/experiments/runs}"
RESULTS_DIR="${RESULTS_DIR:-$HMG_REPO/experiments/results}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" WANDB_MODE=disabled PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

hmg_log() { printf '[hmg %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

hmg_require_setup() {
  local miss=0
  [[ -x "$PY" ]] || { echo "가상환경 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  [[ -f humanoidmimicgen/wbc/external_dependencies/sim2mujoco/resources/robots/g1/policy/walk.onnx ]] || { echo "WBC 정책 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  (( miss == 0 )) || exit 2
}

hmg_gpu_check() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$used" && "$used" -gt 3000 ]]; then
    hmg_log "주의: GPU 메모리를 이미 ${used} MiB 쓰고 있다. 다른 학습(Isaac Gym 등)과 동시에 돌리지 않는다."
  fi
}

# 학습 run 폴더 이름
hmg_run_name() { echo "${TASK}_dp_s${SEED:-1000}"; }
