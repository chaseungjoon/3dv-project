# T10 공통 설정. 다른 스크립트가 source 한다. 값은 환경변수로 덮어쓸 수 있다.
#   예) DATASETS="bridge_dataset" bash experiments/scripts/phase1_baseline.sh
set -o pipefail

T10_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$T10_REPO" || exit 1

PY="${PY:-$T10_REPO/.venv/bin/python}"
# 주 embodiment 2개 (PROTOCOL.md 1절). taco_play(Franka)는 선택: DATASETS="... taco_play"
DATASETS="${DATASETS:-bridge_dataset fractal20220817_data}"
EVAL_BATCH="${EVAL_BATCH:-32}"   # 12GB에서 128은 OOM, 32가 안전

export PYTHONPATH="$T10_REPO${PYTHONPATH:+:$PYTHONPATH}"   # crossformer 패키지는 설치하지 않고 경로로 쓴다
export XLA_PYTHON_CLIENT_PREALLOCATE=false                  # jax가 VRAM 75%를 미리 잡지 않게 (SAPIEN과 공유)
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$T10_REPO/experiments/checkpoints/tfhub}"   # Universal Sentence Encoder
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-$T10_REPO/experiments/third_party/ManiSkill2_real2sim/data}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONUNBUFFERED=1
export EVAL_BATCH

t10_log() { printf '[t10 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# 표준 오류의 TF/XLA 경고 (sm_120 PTX JIT, cuDNN factory 중복 등록 등)를 걸러 화면을 읽을 수 있게 한다.
# 전체 로그가 필요하면 T10_VERBOSE=1
t10_filter() {
  if [[ "${T10_VERBOSE:-0}" == 1 ]]; then cat; else
    grep --line-buffered -v -E 'I0000|W0000|E0000|cuda_(dnn|fft|blas)|computation_placer|oneDNN|cpu_feature_guard|AVX512|tf_record_dataset_op|gpu_device.cc|UserWarning|logger.warn|svulkan2|absl::InitializeLog' || true
  fi
}

t10_require_setup() {
  local miss=0
  [[ -x "$PY" ]] || { echo "가상환경 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  [[ -f experiments/checkpoints/crossformer/config.json ]] || { echo "체크포인트 없음: bash experiments/scripts/setup.sh" >&2; miss=1; }
  for d in $DATASETS; do
    ls experiments/data/"$d"/*/dataset_info.json >/dev/null 2>&1 || { echo "데이터 없음 ($d): bash experiments/scripts/setup.sh" >&2; miss=1; }
  done
  (( miss == 0 )) || exit 2
}

t10_require_sim() {
  [[ -d "$MS2_REAL2SIM_ASSET_DIR/real_inpainting" ]] || { echo "SimplerEnv asset 없음: bash experiments/scripts/setup.sh" >&2; exit 2; }
}

# 한 variant를 오프라인 평가: run_offline <variant> [추가 인자...]
run_offline() {
  local v="$1"; shift
  t10_log "offline: $v ($DATASETS)"
  # shellcheck disable=SC2086
  "$PY" experiments/tools/evaluate.py --variant "$v" --datasets $DATASETS "$@" 2>&1 | t10_filter
}
