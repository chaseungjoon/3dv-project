# HumanoidMimicGen (G1, Push Button): 빠른 시작

**HumanoidMimicGen** (Lin et al., arXiv 2605.27724, `NVlabs/humanoidmimicgen@d82844d`)의 G1 휴머노이드 벤치마크에서
**Push Button**(산업용 패널로 걸어가서 빨간 버튼 누르기)을 공식 예제 그대로 학습/평가한다.

- 시뮬레이터: **MuJoCo 3.2.6 (CPU) + RoboSuite**, 하체는 GR00T WBC (ONNX, CPU). GPU는 Diffusion Policy 학습/추론에만 쓴다.
  → Isaac Gym, 직접 빌드한 torch가 **필요 없다** (공식 torch 2.10 cu128이 sm_120을 지원).
- 데이터: 공개 학습 세트의 Push Button 1,012 demo (사람 시범 1개에서 HMG로 생성, 8 shard).
- 정책: LeRobot Diffusion Policy, 64-step horizon / 50-step chunk, batch 16, 20K update (공식 예제 기본값).
- 평가: 공식 `evaluate_policy_example.py` (seed 0..N-1, 1250 step = 25초 안에 버튼이 1 cm 이상 눌리면 성공).
- 논문 참고값 (Push Button, 1,000 demo, 100 rollout): **DP 0.55**, Flow Matching 1.00, GR00T VLA 0.92 (8x H100), 사람 100 demo로 학습한 VLA 0.82.

Place Book / Drink Cup은 이 벤치마크에 없다 → `../skillmimic-v2/QUICKSTART.md`, `../parahome/QUICKSTART.md`.

> **모든 명령은 `~/code/3dv-project/repos/humanoidmimicgen` 에서 실행한다.** GPU 작업은 한 번에 하나만 (Isaac Gym 학습과 동시에 돌리지 않는다).

```bash
cd ~/code/3dv-project/repos/humanoidmimicgen
```

## 1. 준비 (한 번만, 이 컴퓨터에서는 이미 완료)

```bash
bash experiments/scripts/setup.sh
```

venv(Python 3.10) + LeRobot 고정 commit + torch 2.10 cu128, WBC 정책, 사람 시범 1개, Push Button 학습 데이터 (원본 2.2 GB, 변환 포함 약 25분).
업스트림 데이터 준비 코드의 버그 하나를 고쳤다 (`experiments/docs/NOTES.md` 1절).

## 2. 파이프라인 점검 (약 3분, 2026-10-09 통과)

```bash
bash experiments/scripts/pipeline_check.sh
```

무작위 행동, 사람 시범의 WBC 재생 (성공해야 함), 200 update 학습, 2 episode 평가, 기준 정책, 리포트. 마지막 줄 `파이프라인 점검 통과`.

## 3. 본 실험

```bash
bash experiments/scripts/phase1_push_button.sh
```

| 단계 | 내용 | 시간 (측정 기반 추정) |
|---|---|---|
| 학습 | 20K update, 5K마다 저장 (체크포인트 하나 3 GB) | 약 1시간 (5.5 update/s, 최대 VRAM 약 6 GB) |
| 평가 | 5K/10K/15K/20K 각 100 episode | 약 1~1.5시간 (실패 episode가 많을수록 길다) |
| 기준 정책 | hold / mean / replay / noise 각 100 episode | 약 20분 |

끊기면 같은 명령을 다시 실행한다 (학습은 마지막 5K 체크포인트부터 `--resume`, 이미 끝난 평가는 건너뜀).

## 4. 결과 보기

```bash
bash experiments/scripts/report.sh      # -> experiments/results/report/REPORT.md, fig_success.png, success.csv
```

| 파일 | 내용 |
|---|---|
| `experiments/results/report/REPORT.md` | 학습 시간/VRAM, 체크포인트별 성공률 (95% CI), 기준 정책 바닥값, 논문 값 |
| `experiments/results/eval/<run>/eval_<step>_100ep.json` (+ `.log`) | episode별 성공과 길이 |
| `experiments/results/baselines/*.json` | 학습 없는 기준 정책 결과 |
| `experiments/runs/` (git 제외) | 체크포인트, `_logs/<run>/train.log`, `gpu.csv`, `run_meta.json` |

## 5. 결과를 읽을 때 주의 (점검에서 발견)

- **성공 판정이 쉽다.** 200 update만 학습한 정책(loss 0.97, 사실상 미학습)이 2/2 성공했고, 데이터 평균 행동을 계속 내는 것만으로 1/5 성공했다.
  그래서 학습 정책의 성공률은 반드시 기준 정책(`baselines.sh`) 바닥값과 같이 본다. 가만히 서 있기(hold)와 시범 재생(replay)은 0/5.
- 시범 1개의 행동을 열린 루프로 재생하면 실패한다 → 장면 배치(제어함 위치가 0.5 m 범위에서 무작위)에 맞춰 반응해야 한다.
- 논문 DP 0.55는 논문 쪽 DP 설정(밝히지 않음)이고, 우리는 저장소의 예제 설정이다. 같은 숫자를 기대하지 않는다.

## 6. 개별 명령

```bash
bash experiments/scripts/train.sh 20000                 # 학습만
bash experiments/scripts/eval.sh 100                    # 모든 체크포인트 100 episode
bash experiments/scripts/eval.sh 20 020000              # 20K만 20 episode (빠른 확인)
bash experiments/scripts/baselines.sh 100               # 기준 정책만
TASK=03_box_lift bash experiments/scripts/setup.sh      # (선택) 다른 과제: 데이터 준비 후 같은 스크립트에 TASK=...
```

## 7. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `HTTP Error 404` (데이터 준비) | 업스트림 버그 (`meta/stats.json` 요구). 이 저장소에서는 고쳤다. 다시 `setup.sh` |
| `LeRobot must be 0.4.4 installed from ...` | `setup.sh`가 고정 commit으로 설치한다. 다른 lerobot을 설치하지 않는다 |
| `playback_dataset.py --action-source recorded` 가 `v2.1 format` 오류 | 학습용 LeRobot 0.4.4와 재생용 0.3.3이 충돌 (업스트림 README 순서대로 설치해도 같음). 우리 실험에는 필요 없다 (NOTES.md 2절) |
| EGL 오류 | `export MUJOCO_GL=egl` (스크립트가 자동 설정) |
