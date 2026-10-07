# T09 공통 설정. 다른 스크립트가 source 한다. 값은 환경변수로 덮어쓸 수 있다.
#   예) NUM_ENVS=512 bash experiments/scripts/train.sh default 0 3000
set -o pipefail

T09_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$T09_REPO" || exit 1

PY="${PY:-$T09_REPO/.venv/bin/python}"
SCENE="${SCENE:-InterAct/Parahome/s110_0_kettle_table2desk}"   # 주 장면 (PROTOCOL.md 1절)
ROBOT="${ROBOT:-sim_human/s110_ROM.xml}"                         # 데이터 골격과 일치하는 손 모델
NUM_ENVS="${NUM_ENVS:-1024}"                                     # 12GB 상한 (2048은 OOM)
MINIBATCH="${MINIBATCH:-16384}"
CFG_TRAIN="${CFG_TRAIN:-intermimic/data/cfg/train/rlg/parahome.yaml}"
CFG_DIR="${CFG_DIR:-experiments/configs/env}"
RUNS_DIR="${RUNS_DIR:-experiments/runs}"
RESULTS_DIR="${RESULTS_DIR:-experiments/results}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # 추가 지표는 TensorBoard(wm/*)로도 기록된다
export PYTHONUNBUFFERED=1

t09_log() { printf '[t09 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

t09_require_budget() {
  if [[ -z "${BUDGET:-}" ]]; then
    echo "BUDGET(학습 예산 T, epoch 수)이 필요합니다. Phase 1 결과로 정한 값을 넣으세요." >&2
    echo "  예) BUDGET=3000 bash $0" >&2
    exit 2
  fi
  if (( BUDGET % 500 != 0 )); then
    echo "BUDGET은 500의 배수여야 합니다 (체크포인트가 500 epoch마다 저장됨): $BUDGET" >&2
    exit 2
  fi
}
