# SkillMimic-V2 baseline (컵 / 책 / 주전자): 빠른 시작

T09 baseline: **SkillMimic-V2** (Yu et al., SIGGRAPH 2025, arXiv 2505.02094)를 RTX 5070 12GB에서 ParaHome clip 세 개에 학습/평가한다.
계획과 근거: `../../PROPOSAL_T09.md`. 규칙과 지표: `experiments/docs/PROTOCOL.md`. 결과: `../../BASELINE_SKILLMIMIC.md`.

| 스크립트 | clip | 내용 | 시간 |
|---|---|---|---|
| `run_cup.sh` | `drink_cup` | 식탁의 컵을 들어 입으로 가져가 마시기 (180 frame) | 약 9시간 + 평가 10분 |
| `run_book.sh` | `place_book` | 책상의 책을 들어 옮겨 놓기 (150 frame) | 약 9시간 + 평가 10분 |
| `run_kettle.sh` | `place_kettle` | 주전자를 들어 옮겨 놓기 (100 frame). T09와 같은 물체 | 약 10시간 + 평가 10분 |
| `write_results.sh` | | 세 run의 결과를 `../../BASELINE_SKILLMIMIC.md`로 정리 (GPU 안 씀) | 수 초 |

> **모든 명령은 `~/code/3dv-project/repos/skillmimic-v2` 에서 실행한다.** 학습은 `tmux` 안에서 권장. GPU 작업은 **한 번에 하나만** (다른 저장소 포함).

## 1. 준비 (한 번만, 이 컴퓨터에서는 이미 완료)

```bash
bash experiments/scripts/setup.sh                 # uv 환경, ParaHome 장면 s6/s10/s22 (가구 위치, experiments/data/), history encoder (약 6분)
bash experiments/scripts/pipeline_check.sh        # 약 15분. 끊고 다시 실행했을 때 그대로 이어지는지까지 확인 (2026-10-10 통과)
```

## 2. 본 실험

```bash
cd ~/code/3dv-project/repos/skillmimic-v2
bash run_cup.sh
bash run_book.sh
bash run_kettle.sh
bash write_results.sh      # 세 run이 끝난 뒤. 중간에 실행해도 된다 (진행 상태를 적는다)
```

각 `run_*.sh`는 **학습 3000 epoch (2048 env, 1.97억 샘플) → 평가 (250 epoch마다의 학습 곡선 + 최종 체크포인트 det/stoch/교란)** 를 한다.
결과 문서는 만들지 않는다. 그건 `write_results.sh`가 한다.

### 끊겼을 때

**같은 명령을 다시 실행하면 된다.** Ctrl-C, 터미널 종료, 정전 모두 같다.

- 학습: 마지막 체크포인트부터 **그대로** 이어서 한다. epoch 번호, optimizer(Adam) 상태, 입력/value 정규화 통계, ATS 샘플링 가중치, 난수 상태까지 복원된다.
  체크포인트는 10 epoch(약 2분)마다 저장되므로 잃는 것은 최대 10 epoch다.
- 평가: 이미 평가한 체크포인트는 건너뛴다. 학습이 끝난 run이면 학습은 건너뛰고 남은 평가만 한다.
- 저장은 임시 파일에 쓴 뒤 이름을 바꾸므로, 저장 도중에 끊겨도 깨진 체크포인트가 남지 않는다.

### 옵션 (환경변수)

| 변수 | 기본 | 뜻 |
|---|---|---|
| `EPOCHS` | 3000 | 총 epoch. 끝난 run에 더 큰 값을 주면 거기서부터 더 학습한다 (`EPOCHS=6000 bash run_cup.sh`) |
| `NUM_ENVS` | 2048 | 메모리 부족일 때만 1536 (run 이름이 바뀌고 보고서에 남는다) |
| `SEED` | 0 | seed 추가 실험용 |
| `EVAL_EVERY` | 250 | 학습 곡선 평가 간격 (epoch) |

### 학습 중 보기

```bash
tensorboard --logdir experiments/runs                        # 보상 곡선
tail -f experiments/runs/drink_cup_ours_n2048_s0/train.log   # 전체 출력
bash write_results.sh                                        # 진행 상태 (GPU 안 씀)
```

중간 체크포인트를 평가하고 싶으면 Ctrl-C로 멈추고 평가한 뒤 다시 `run_*.sh` (학습은 그대로 이어진다):

```bash
bash experiments/scripts/eval_run.sh drink_cup_ours_n2048_s0 curve 1000
```

## 3. 결과 위치

| 파일 | 내용 |
|---|---|
| `../../BASELINE_SKILLMIMIC.md` | 진행 상태, 요약, 판정 (PROTOCOL 5절 규칙), 학습 run 표, 최종 상세, 학습 곡선 |
| `experiments/results/figures/baseline_curves.png` | 세 clip의 보상, 성공률, 물체 오차 vs epoch |
| `experiments/results/tables/{runs,final,curve}.csv` | 원자료 표 |
| `experiments/results/eval/<det\|stoch\|stoch_perturb>/<run>/*.json, *.npz` | 체크포인트별 env별 지표, frame별 궤적 |
| `experiments/runs/<run>/` (git 제외) | `nn/` 체크포인트 (10 epoch마다 `_000XXXXX.pth`, 지우지 않음, run당 약 13 GB), `train.log`, `gpu.csv`, `run_meta.json` (끊긴 구간 기록), `summaries/` |

run 이름: `drink_cup_ours_n2048_s0`, `place_book_ours_n2048_s0`, `place_kettle_ours_n2048_s0`.

## 4. 개별 명령

```bash
bash experiments/scripts/train.sh drink_cup ours 3000                 # 학습만 (자동 이어 학습)
bash experiments/scripts/eval_run.sh drink_cup_ours_n2048_s0 final    # 마지막 체크포인트: det + stoch + 교란
bash experiments/scripts/eval_run.sh drink_cup_ours_n2048_s0 curve 250,500,750
.venv/bin/python experiments/tools/ckpt.py latest experiments/runs/drink_cup_ours_n2048_s0   # 다음에 이어 학습할 체크포인트
# 눈으로 보기 (화면 필요, env 1개)
.venv/bin/python experiments/tools/evaluate.py --run_dir experiments/runs/drink_cup_ours_n2048_s0 --viewer --tag viewer --force
```

## 5. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA out of memory`, `PxgCudaDeviceMemoryAllocator fail` | 2048 env는 clip에 따라 9.5~10.8 GB를 쓴다 (`CODE_NOTES.md` 3절). 브라우저 등 GPU를 쓰는 프로그램을 끄고 다시. 그래도 안 되면 `NUM_ENVS=1536` |
| `history encoder 없음`, `ParaHome 원본 링크 없음` | `bash experiments/scripts/setup.sh` (장면이 없으면 ParaHome seq.zip을 받아 s6/s10/s22만 남긴다) |
| `[ckpt] cannot load ...` | 깨진 체크포인트는 건너뛰고 그 전 것에서 이어간다. 정상 동작 |
| `shape mismatch ... 1025 ... 1028` | asset과 task가 안 맞음. 직접 `run.py`를 부르지 말고 `train.sh`를 쓴다 (`CODE_NOTES.md` 2절) |
| 시작 후 1~2분 멈춘 듯 보임 | 2048 env 생성 + VHACD. 정상 |
