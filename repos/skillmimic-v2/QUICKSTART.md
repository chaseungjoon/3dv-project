# SkillMimic-V2 (ParaHome: Place Book / Drink Cup): 빠른 시작

**SkillMimic-V2** (Yu et al., SIGGRAPH 2025, arXiv 2505.02094)를 RTX 5070 12GB에서 ParaHome 가정 조작 clip에 학습/평가한다.
목적은 T09 viability 점검이다: **같은 GPU에서 다른 방법이 "물체를 잡고 드는" 과제를 배우는가?** (질문과 판정 기준: `experiments/docs/PROTOCOL.md`)

| clip | 내용 | 비고 |
|---|---|---|
| `drink_cup` | 식탁의 컵을 들어 입으로 가져가 마시기 (180 frame, 6초) | 요청 Task |
| `place_book` | 책상의 책을 들어 옮겨 놓기 (150 frame, 5초) | 요청 Task |
| `place_kettle` | 주전자를 들어 옮겨 놓기 (100 frame) | T09와 같은 물체. T09와의 다리 |

Push Button은 ParaHome/SkillMimic-V2에 없다 → `../humanoidmimicgen/QUICKSTART.md`.

> **모든 명령은 `~/code/3dv-project/repos/skillmimic-v2` 에서 실행한다.** 학습은 `tmux` 안에서 권장. GPU 작업은 **한 번에 하나만** (다른 저장소 포함).

```bash
cd ~/code/3dv-project/repos/skillmimic-v2
```

## 1. 준비 (한 번만, 이 컴퓨터에서는 이미 완료)

```bash
bash ../parahome/experiments/scripts/setup.sh     # 원본 ParaHome 데이터 (이미 받음)
bash experiments/scripts/setup.sh                 # uv 환경, ParaHome 장면 링크, history encoder (약 6분)
```

## 2. 파이프라인 점검 (약 15분, 2026-10-09 통과)

```bash
bash experiments/scripts/pipeline_check.sh
```

마지막 줄이 `파이프라인 점검 통과` 이면 된다. 학습 20 epoch + 자동 이어 학습 + SM 학습 + 평가 4종 + 리포트를 한 번씩 돈다.

## 3. 본 실험 (순서대로 복사해서 실행)

| 단계 | 명령 | 시간 | 무엇을 얻나 |
|---|---|---|---|
| **Phase 1** | `bash experiments/scripts/phase1_main.sh` | 약 28시간 (cup 9, book 9, kettle 10) | **SkillMimic-V2가 5070에서 Drink Cup / Place Book / Place Kettle을 배우는가** (2048 env, 3000 epoch) |
| Phase 2 | `bash experiments/scripts/phase2_baseline.sh` | 약 12시간 (cup 5.5, kettle 6.2) | 같은 예산의 SkillMimic(v1). 논문에서는 0% → 여기서도 0%인가 |
| Phase 3 | `bash experiments/scripts/phase3_envs.sh` | 약 10시간 | env를 T09처럼 1024로 줄이면 (같은 샘플 수) 결과가 나빠지는가 |

시간이 부족하면 **Phase 1을 clip 하나씩** 나눠 돌린다 (우선순위 순):

```bash
CLIPS="drink_cup"    bash experiments/scripts/phase1_main.sh    # 약 9시간
CLIPS="place_book"   bash experiments/scripts/phase1_main.sh    # 약 9시간
CLIPS="place_kettle" bash experiments/scripts/phase1_main.sh    # 약 10시간 (T09 다리)
```

- 각 Phase는 학습이 끝나면 평가(250 epoch마다 저장된 체크포인트 전부, 약 5분)와 리포트를 자동으로 한다.
- **중간에 끊겨도 같은 명령을 다시 실행하면 마지막 저장(250 epoch 단위)부터 이어서 한다.** 끝난 run은 건너뛴다.
- 예산을 바꾸려면 `EPOCHS=2000 bash experiments/scripts/phase1_main.sh` (모든 비교는 같은 epoch에서).
- 학습 중 모니터링: `tensorboard --logdir experiments/runs` 또는 `tail -f experiments/runs/<run>/train.log`.

## 4. 결과 보기

```bash
bash experiments/scripts/report.sh            # 언제든 다시 만들 수 있다 (GPU 안 씀)
```

| 파일 | 내용 |
|---|---|
| `experiments/results/report/REPORT.md` | 학습 속도/VRAM, 최종 성공률 (det/stoch/교란), 학습 곡선 표 |
| `experiments/results/report/fig_curve_<clip>.png` | 보상, 성공률, 물체 오차 vs epoch |
| `experiments/results/report/{runs,final,curve}.csv` | 원자료 |
| `experiments/results/eval/<mode>/<run>/*.json, *.npz` | 체크포인트별 env별 지표, frame별 궤적 |
| `experiments/runs/<run>/` (git 제외) | 체크포인트, `train.log`, `gpu.csv`, `run_meta.json` |

## 5. 개별 명령

```bash
bash experiments/scripts/train.sh drink_cup ours 3000        # 학습 1회 (clip, method, epochs, [seed])
bash experiments/scripts/eval_run.sh drink_cup_ours_n2048_s0 all     # 체크포인트 전부 det 평가
bash experiments/scripts/eval_run.sh drink_cup_ours_n2048_s0 final   # 마지막: det + stoch + 교란
# 눈으로 보기 (화면 필요, env 1개)
.venv/bin/python experiments/tools/evaluate.py --run_dir experiments/runs/drink_cup_ours_n2048_s0 --viewer --tag viewer --force
```

## 6. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA out of memory`, `PxgCudaDeviceMemoryAllocator fail` | 2048 env는 clip에 따라 9.2~11.2 GB를 쓴다 (`CODE_NOTES.md` 3절). 브라우저 등 GPU를 쓰는 프로그램을 끄고 다시. 그래도 안 되면 `NUM_ENVS=1536` (보고서에 명시) |
| `history encoder 없음`, `ParaHome 원본 링크 없음` | `bash experiments/scripts/setup.sh` |
| `shape mismatch ... 1025 ... 1028` | asset과 task가 안 맞음. 직접 `run.py`를 부르지 말고 `train.sh`를 쓴다 (`CODE_NOTES.md` 2절) |
| 시작 후 1~2분 멈춘 듯 보임 | 2048 env 생성 + VHACD. 정상 |

자세한 내용: 규칙과 지표 `experiments/docs/PROTOCOL.md`, 코드 분석과 원본 대비 변경 `experiments/docs/CODE_NOTES.md`.
