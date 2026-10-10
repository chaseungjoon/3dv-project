#!/usr/bin/env bash
# 파이프라인 점검 (약 15분, GPU). 학습 숫자 자체는 의미 없다. 확인하는 것:
#   1) 학습을 중간에 Ctrl-C로 끊고 (프로세스 그룹에 SIGINT) 같은 명령으로 다시 실행하면
#      마지막 체크포인트부터 epoch, optimizer, frame, ATS 상태, 난수 상태가 그대로 이어지는가
#   2) 앞 구간의 번호 체크포인트가 덮어써지지 않는가
#   3) 평가 (곡선 + 최종 det/stoch/교란)와 write_results.py가 끝까지 도는가
# 결과는 experiments/runs/_check, experiments/results/_check 에만 쓰고, 통과하면 지운다 (KEEP_CHECK=1이면 남김).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
smv2_require_setup
export RUNS_DIR=experiments/runs/_check RESULTS_DIR=experiments/results/_check
RUN=drink_cup_ours_n${NUM_ENVS}_s0
D="$RUNS_DIR/$RUN"
rm -rf "$RUNS_DIR" "$RESULTS_DIR"
fail() { echo "파이프라인 점검 실패: $*" >&2; exit 1; }

smv2_log "1/5 학습 30 epoch 시작, epoch 24 근처에서 Ctrl-C (프로세스 그룹에 SIGINT)"
# 셸 스크립트의 & 작업은 SIGINT가 무시(SIG_IGN)된 채 시작하고 exec로 상속된다. 터미널의 Ctrl-C와 같게 하려고
# 기본 동작으로 되돌린 뒤 새 프로세스 그룹에서 train.sh를 띄운다.
"$PY" -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.setsid(); os.execvp("bash", ["bash"] + sys.argv[1:])' \
  experiments/scripts/train.sh drink_cup ours 30 > /dev/null 2>&1 &
PGID=$!
for _ in $(seq 1 180); do
  grep -q 'epoch_num:24 ' "$D/train.log" 2>/dev/null && break
  kill -0 "$PGID" 2>/dev/null || fail "학습이 epoch 24 전에 끝남 ($D/train.log)"
  sleep 5
done
grep -q 'epoch_num:24 ' "$D/train.log" || { kill -INT -- -"$PGID"; fail "15분 안에 epoch 24에 못 감"; }
kill -INT -- -"$PGID"
wait "$PGID"
smv2_log "   중단됨 (exit $?). 저장된 체크포인트: $(ls "$D/nn" | tr '\n' ' ')"
"$PY" - "$D" > "$RUNS_DIR/before.txt" <<'PYEOF' || fail "중단 전 체크포인트 기록"
import glob, hashlib, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "nn", "*_000000[0-9]0.pth"))):
    print(os.path.basename(p), hashlib.sha1(open(p, "rb").read()).hexdigest())
PYEOF

smv2_log "2/5 같은 명령으로 다시 실행 -> 이어서 30 epoch까지"
bash experiments/scripts/train.sh drink_cup ours 30 || fail "이어 학습"

smv2_log "3/5 이어 학습 검사"
"$PY" - "$D" "$RUNS_DIR/before.txt" <<'PYEOF' || fail "이어 학습 검사"
import hashlib, json, os, re, sys
import torch
d, before = sys.argv[1], sys.argv[2]
run = os.path.basename(d)
ok = True
def check(cond, msg):
    global ok
    print(("   OK   " if cond else "   FAIL ") + msg); ok &= bool(cond)

meta = json.load(open(os.path.join(d, "run_meta.json")))
seg = meta["segments"]
check(len(seg) == 2 and seg[-1].get("resume_epoch", 0) >= 20, f"2번째 구간이 epoch {seg[-1].get('resume_epoch')} 에서 시작 ({os.path.basename(seg[-1].get('resume_from', ''))})")
r0 = seg[-1].get("resume_epoch", 0)
log = open(os.path.join(d, "train.log"), errors="ignore").read()
part2 = log[log.index("resumed full training state"):] if "resumed full training state" in log else ""
eps = [int(x) for x in re.findall(r"epoch_num:(\d+) ", part2)]
check(bool(eps) and eps[0] == r0 + 1 and eps == list(range(r0 + 1, eps[-1] + 1)) and eps[-1] == 31,
      f"이어 학습 epoch 번호가 {r0 + 1}부터 연속, 31에서 끝 (본 것: {eps[:2]}...{eps[-1:]})")
for line in open(before):
    name, h = line.split()
    p = os.path.join(d, "nn", name)
    check(os.path.isfile(p) and hashlib.sha1(open(p, "rb").read()).hexdigest() == h, f"{name} 덮어쓰지 않음")

def st(name):
    return torch.load(os.path.join(d, "nn", name), map_location="cpu")
def opt_step(s):
    v = next(iter(s["optimizer"]["state"].values()))["step"]
    return float(v)
a, b = st(f"{run}_00000010.pth"), st(f"{run}_00000030.pth")
check(b["epoch"] == 30 and b["frame"] == 3 * a["frame"], f"epoch 30, frame {b['frame']} = 3 x {a['frame']} (frame 카운터 이어짐)")
check(abs(opt_step(b) - 3 * opt_step(a)) < 1e-6, f"Adam step {opt_step(b):.0f} = 3 x {opt_step(a):.0f} (optimizer 상태 이어짐)")
es_a, es_b = a.get("env_state"), b.get("env_state")
check(es_a is not None and es_b is not None, "env_state(ATS) 저장됨")
if es_a and es_b:
    pa, pb = es_a["progress_buf_total"], es_b["progress_buf_total"]
    check(abs(pb - 3 * pa) <= 3, f"ATS step 카운터 {pb} ~ 3 x {pa} (이어짐)")
    check(any(float(v.abs().sum()) > 0 for v in es_b["motion_time_seqreward"].values()), "ATS frame별 보상 기록 있음")
check("rng_state" in b, "난수 상태 저장됨")
check(not any(f.endswith(".tmp") for f in os.listdir(os.path.join(d, "nn"))), "미완성 저장(.tmp) 없음")
sys.exit(0 if ok else 1)
PYEOF

smv2_log "4/5 평가 (곡선 10,20,30 + 최종 det/stoch/교란)"
bash experiments/scripts/eval_run.sh "$RUN" curve 10,20,30 || fail "곡선 평가"
bash experiments/scripts/eval_run.sh "$RUN" final || fail "최종 평가"
n=$(find "$RESULTS_DIR/eval" -name '*.json' | wc -l)
(( n == 5 )) || fail "평가 결과 $n개 (5개여야 함: det 3 + stoch 1 + 교란 1)"

smv2_log "5/5 결과 문서"
"$PY" experiments/tools/write_results.py --runs_dir "$RUNS_DIR" --eval_dir "$RESULTS_DIR/eval" \
  --out "$RESULTS_DIR/BASELINE_check.md" --fig_dir "$RESULTS_DIR/figures" --table_dir "$RESULTS_DIR/tables" \
  | grep results || fail "write_results"
grep -q '완료 (30 epoch, 평가 완료)' "$RESULTS_DIR/BASELINE_check.md" || fail "결과 문서에 완료 상태 없음"

if [[ "${KEEP_CHECK:-0}" == 1 ]]; then
  smv2_log "파이프라인 점검 통과 (결과 남김: $RUNS_DIR, $RESULTS_DIR)"
else
  rm -rf "$RUNS_DIR" "$RESULTS_DIR"
  smv2_log "파이프라인 점검 통과 (점검 결과는 지웠다. 남기려면 KEEP_CHECK=1)"
fi
