# ParaHome 실험 공통 설정. 다른 스크립트가 source 한다. 값은 환경변수로 덮어쓸 수 있다.
set -o pipefail

PH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PH_REPO" || exit 1

PY="${PY:-$PH_REPO/.venv/bin/python}"                       # CPU 분석용 (audit)
T09_REPO="${T09_REPO:-$(cd "$PH_REPO/../t09-wristmimic" && pwd)}"   # WristMimic 코드 + Isaac Gym 환경
T09_PY="$T09_REPO/.venv/bin/python"
RUNS_DIR="${RUNS_DIR:-$PH_REPO/experiments/runs}"           # 절대 경로 (T09 스크립트가 T09 저장소에서 돈다)
RESULTS_DIR="${RESULTS_DIR:-$PH_REPO/experiments/results}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled

ph_log() { printf '[parahome %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# WristMimic 장면 (../t09-wristmimic/InterAct/Parahome) 과 그 피험자 골격 XML (데이터 뼈 길이와 0.1 mm 이내, T09 PROTOCOL 1.1과 같은 규칙)
#   key           SCENE                                         ROBOT
ph_scene() {
  case "$1" in
    cup_s110)    echo "InterAct/Parahome/s110_16_cup_drink sim_human/s110_ROM.xml" ;;          # Drink Cup, T09 주 장면과 같은 사람/XML
    book_s11)    echo "InterAct/Parahome/s11_book_desk2bookshelf sim_human/s11_ROM.xml" ;;     # Place Book (책상 -> 책장)
    cup_s89)     echo "InterAct/Parahome/s89_2_drink_cup sim_human/s89_ROM.xml" ;;             # Drink Cup 2번째 사례 (왼손, 369 frame)
    book_s149)   echo "InterAct/Parahome/s149_15_book_bookshelf2table sim_human/s149_ROM.xml" ;; # Place Book 2번째 사례 (책상 -> 책장, 120 frame)
    kettle_s110) echo "InterAct/Parahome/s110_0_kettle_table2desk sim_human/s110_ROM.xml" ;;   # T09 baseline 장면 (대조군)
    *) echo "알 수 없는 장면: $1 (cup_s110|book_s11|cup_s89|book_s149|kettle_s110)" >&2; return 1 ;;
  esac
}

ph_require_t09() {
  [[ -x "$T09_PY" ]] || { echo "T09 환경 없음: $T09_PY  (../t09-wristmimic 에서 uv sync, isaacgym-env/README.md)" >&2; exit 2; }
  [[ -f "$T09_REPO/experiments/configs/env/default.yaml" ]] || { echo "T09 실험 설정 없음" >&2; exit 2; }
}

ph_gpu_check() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$used" && "$used" -gt 3000 ]]; then
    ph_log "주의: GPU 메모리를 이미 ${used} MiB 쓰고 있다. 다른 학습이 돌고 있으면 OOM이 날 수 있다."
  fi
}
