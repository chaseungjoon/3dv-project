#!/usr/bin/env bash
# 파이프라인 점검 (약 10분): audit + 두 장면을 20 epoch씩 학습해서 장면이 로드되고 평가/리포트가 도는지 확인.
# 숫자는 의미 없다. 결과는 experiments/runs/_check, experiments/results/_check 에만 쓴다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ph_require_t09
export RUNS_DIR="$PH_REPO/experiments/runs/_check"
CHK="$PH_REPO/experiments/results/_check"
rm -rf "$RUNS_DIR" "$CHK"
fail() { echo "파이프라인 점검 실패: $*" >&2; exit 1; }

ph_log "1/4 audit (CPU)"
bash experiments/scripts/audit.sh || fail audit
for s in cup_s110 book_s11; do
  ph_log "2/4 $s 20 epoch 학습 (1024 env)"
  bash experiments/scripts/train_wm.sh "$s" 20 0 || fail "train $s"
  CK="$RUNS_DIR/wm_${s}_default_s0/nn/wm_${s}_default_s0_latest.pth"
  [[ -f "$CK" ]] || fail "체크포인트 없음 $CK"
  read -r SCENE ROBOT < <(ph_scene "$s")
  ph_log "3/4 $s 평가 (det 8 env)"
  ( cd "$T09_REPO" && "$T09_PY" experiments/tools/evaluate.py --checkpoints "$CK" --mode det \
      --motion_file "$SCENE" --robot_type "$ROBOT" --out_dir "$CHK/eval" ) 2>&1 | grep -E '^\[eval|Error|Traceback' || fail "eval $s"
done
ph_log "4/4 리포트"
"$PY" experiments/tools/report.py --eval_dir "$CHK/eval" --out "$CHK/report" || fail report
python3 - "$RUNS_DIR" <<'PYEOF'
import json, glob, re, sys
for d in sorted(glob.glob(sys.argv[1] + "/wm_*")):
    m = json.load(open(d + "/run_meta.json")); log = open(d + "/train.log").read()
    tot = [float(x) for x in re.findall(r"total time: ([\d.]+)", log)][3:]
    spe = sum(tot) / max(1, len(tot))
    print(f"[parahome] {d.split('/')[-1]}: {spe:.2f} s/epoch -> 3000 epoch {3000 * spe / 3600:.1f} h, 최대 GPU {m.get('peak_gpu_mem_mib')} MiB")
PYEOF
ph_log "파이프라인 점검 통과 ($CHK/report/REPORT.md)"
