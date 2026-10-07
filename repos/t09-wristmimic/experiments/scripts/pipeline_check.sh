#!/usr/bin/env bash
# 파이프라인 점검 (짧은 검증, 약 5분). 긴 학습 전에 한 번 돌린다.
#   사용법: bash experiments/scripts/pipeline_check.sh
#   1) 설정 파일 검사 (손목 키 외 변경 없음)
#   2) 기본 설정 20 epoch 학습 -> epoch당 시간, env-step/s, 최대 VRAM 실측
#   3) 결정적 평가(env 8개)를 별도 프로세스로 2번 -> 두 결과가 허용 오차 안인지 (재현성)
#   4) 원본 run.py --test --test_no_reset(env 1개) 결과가 우리 평가 분포 안에 있는지
#   5) 확률적 평가(4 env), 리포트 생성
# 산출물은 experiments/runs/_checks, experiments/results/_checks 에만 쓴다 (본 실험과 섞이지 않음).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CHECK_EPOCHS="${CHECK_EPOCHS:-20}"
CHECK_RUNS="experiments/runs/_checks"
CHECK_OUT="experiments/results/_checks"
NAME="pipeline_$(date +%m%d-%H%M%S)"
RUN="$CHECK_RUNS/$NAME"
fail() { t09_log "실패: $*"; exit 1; }

t09_log "1/5 설정 파일 검사"
"$PY" experiments/tools/make_configs.py --check || fail "configs"

t09_log "2/5 짧은 학습 ($CHECK_EPOCHS epochs, $NUM_ENVS envs)"
RUNS_DIR="$CHECK_RUNS" RUN_NAME="$NAME" bash experiments/scripts/train.sh default 0 "$CHECK_EPOCHS" || fail "train"
CKPT="$RUN/nn/${NAME}_latest.pth"
[[ -f "$CKPT" ]] || fail "체크포인트 없음: $CKPT"

t09_log "3/5 결정적 평가 2회 (별도 프로세스, env 8개씩)"
for i in 1 2; do
  "$PY" experiments/tools/evaluate.py --checkpoints "$CKPT" --mode det --out_dir "$CHECK_OUT" --tag "rep$i" --force \
     --motion_file "$SCENE" --robot_type "$ROBOT" || fail "eval rep$i"
done

t09_log "4/5 원본 run.py --test (env 1개) 와 비교"
OFFICIAL_LOG="$RUN/official_test.log"
timeout 900 "$PY" intermimic/run.py --task InterMimic_MULTI_OBJ \
  --cfg_env "$CFG_DIR/eval_default.yaml" --cfg_train "$CFG_TRAIN" \
  --test --test_no_reset --checkpoint "$CKPT" --num_envs 1 \
  --motion_file "$SCENE" --robot_type "$ROBOT" \
  --num_position_iterations 20 --num_velocity_iterations 0 --headless 2>&1 \
  | tee "$OFFICIAL_LOG" | grep -m1 '\[Eval\] 1 episodes'

"$PY" - "$CHECK_OUT" "$NAME" "$OFFICIAL_LOG" "$RUN" <<'PYEOF' || fail "비교"
# GPU PhysX는 비트 단위로 재현되지 않는다 (PROTOCOL.md 4.3). 그래서 '같은 값'이 아니라
# '같은 분포'인지 본다: 두 번의 8-env 평가 평균이 가깝고, 원본 1-env 결과가 그 분포 안에 있는지.
import json, re, sys, os, glob
out, name, log, run = sys.argv[1:]
r = [json.load(open(glob.glob(f"{out}/det_rep{i}/{name}/*.json")[0])) for i in (1, 2)]
ok = True
for k, tol in (('official_success', 0.26), ('progress', 0.05), ('obj_pos_err_local_mean', 0.02), ('obj_pos_err_mean', 0.02)):
    a, b = r[0]['aggregate'][k], r[1]['aggregate'][k]
    good = abs(a - b) <= tol
    ok &= good
    print(f"[check] 재현성 {k:24s}: {a:.4f} vs {b:.4f} (허용 {tol}) {'OK' if good else 'FAIL'}")
eps = r[0]['episodes'] + r[1]['episodes']
lo = min(e['obj_pos_err_local_mean'] for e in eps); hi = max(e['obj_pos_err_local_mean'] for e in eps)
print(f"[check] 물리 잡음에 의한 퍼짐 (16 rollout): obj_pos_err(local) {lo:.4f} ~ {hi:.4f}, 성공률 {sum(e['official_success'] for e in eps)}/16")
m = re.search(r"\[Eval\] 1 episodes \| Success: (\d+) \| Fail: (\d+).*?obj_pos_err: ([\d.]+)m \| obj_rot_err: ([\d.]+)rad", open(log).read())
if not m:
    sys.exit("[check] 원본 테스트 출력에서 [Eval] 줄을 못 찾음")
off_succ, off_pos = int(m.group(1)), float(m.group(3))
margin = max(0.01, 0.5 * (hi - lo))
in_range = lo - margin <= off_pos <= hi + margin
print(f"[check] 원본 run.py (1 env): success={off_succ} obj_pos_err(local)={off_pos:.4f} -> 우리 분포 안: {'OK' if in_range else 'FAIL'}")
ok &= in_range
flags = all(e['official_success'] == e['official_success_env_flag'] for e in eps)
unexpl = sum(e['first_cause_unexplained'] for e in eps)
print(f"[check] recorder 실패 플래그 == env terminate 플래그: {'OK' if flags else 'MISMATCH'}")
print(f"[check] 원인 미설명 실패 0건: {'OK' if not unexpl else unexpl}")
ok &= flags and not unexpl
meta = json.load(open(os.path.join(run, 'run_meta.json')))
logtxt = open(os.path.join(run, 'train.log')).read()
fps = [float(x) for x in re.findall(r"fps step: ([\d.]+)", logtxt)]
tot = [float(x) for x in re.findall(r"total time: ([\d.]+)", logtxt)]
if tot:
    steady = tot[3:] or tot
    spe = sum(steady) / len(steady)
    print(f"[check] epoch당 시간 {spe:.2f}s (처음 3 epoch 제외), env-step/s {sum(fps[3:] or fps)/len(fps[3:] or fps):.0f}, 최대 GPU 메모리 {meta.get('peak_gpu_mem_mib')} MiB")
    for T in (1000, 2000, 3000, 4000, 6000):
        print(f"[check]   {T:5d} epoch 예상 {T*spe/3600:5.1f} h")
json.dump({'ok': bool(ok), 'official_obj_pos_err_local': off_pos, 'ours_range': [lo, hi]}, open(os.path.join(run, 'pipeline_check.json'), 'w'))
sys.exit(0 if ok else 1)
PYEOF

t09_log "5/5 확률적 평가(4 env) + 리포트"
"$PY" experiments/tools/evaluate.py --checkpoints "$CKPT" --mode stoch --num_envs 4 --out_dir "$CHECK_OUT" --force \
   --motion_file "$SCENE" --robot_type "$ROBOT" || fail "stoch eval"
"$PY" experiments/tools/aggregate.py --eval_dir "$CHECK_OUT" --runs_dir "$CHECK_RUNS" --out "$CHECK_OUT/report" \
   --det_subdir det_rep1 || fail "aggregate"
t09_log "파이프라인 점검 통과. 리포트: $CHECK_OUT/report/REPORT.md"
