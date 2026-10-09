# SkillMimic-V2 실험 공통 설정. 다른 스크립트가 source 한다. 값은 환경변수로 덮어쓸 수 있다.
set -o pipefail

SMV2_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SMV2_REPO" || exit 1

PY="${PY:-$SMV2_REPO/.venv/bin/python}"
RUNS_DIR="${RUNS_DIR:-experiments/runs}"
RESULTS_DIR="${RESULTS_DIR:-experiments/results}"
NUM_ENVS="${NUM_ENVS:-2048}"          # 논문과 같은 값. 12GB에서 약 9.8GB (PROTOCOL.md 1)
PARAHOME_SEQ="${PARAHOME_SEQ:-$SMV2_REPO/../parahome/data/seq}"   # 원본 ParaHome (책상/식탁 위치를 읽는다)
export PYTHONUNBUFFERED=1
# torch 캐시 단편화를 줄여 최대 VRAM 약 0.3 GB 절약 (2048 env: 11.1 -> 10.8 GB, 수치 결과에는 영향 없음)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

smv2_log() { printf '[smv2 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Isaac Gym 경고 등 반복 출력 제거 (전체 로그는 각 run의 train.log)
smv2_filter() {
  grep --line-buffered -v -E 'Warning|warn\(|UserWarning|^\s*$|actor_mlp|critic_mlp|^value\.|^mu\.|^sigma$|RunningMeanStd|build mlp|Not connected to PVD|Physics Engine|GPU Pipeline|Importing module|Setting GYM_USD|VHACD' || true
}

smv2_require_setup() {
  local miss=0
  [[ -x "$PY" ]] || { echo "가상환경 없음: cd $SMV2_REPO && uv sync" >&2; miss=1; }
  [[ -f experiments/checkpoints/hist_encoder/parahome_hist60.ckpt ]] || { echo "history encoder 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  [[ -f skillmimic/data/motions/ParaHome/s10/object_transformations.pkl ]] || { echo "ParaHome 원본 링크 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  (( miss == 0 )) || exit 2
}

# GPU를 쓰는 다른 학습이 돌고 있으면 경고 (12GB를 나눠 쓰면 OOM)
smv2_gpu_check() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$used" && "$used" -gt 3000 ]]; then
    smv2_log "주의: GPU 메모리를 이미 ${used} MiB 쓰고 있다. 다른 학습이 돌고 있으면 OOM이 날 수 있다."
  fi
}
