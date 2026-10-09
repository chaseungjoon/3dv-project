# ParaHome (Drink Cup / Place Book with the T09 method): 빠른 시작

**ParaHome** (Kim et al., CVPR 2025, `snuvclab/ParaHome@535dada`)은 T09 장면(주전자)의 원본 데이터셋이다.
이 저장소에서 하는 일은 두 가지다.

1. **데이터 audit (CPU)**: 원본 207 시퀀스에서 Drink Cup / Place Book / Kettle 구간을 모두 찾아 "쥐고 들기"가 얼마나 어려운지 잰다.
2. **T09 방법을 다른 물체에 (GPU)**: T09 baseline과 **완전히 같은** WristMimic 코드, 12GB 설정, 평가기로 컵 마시기와 책 옮기기를 학습한다.
   장면만 바꿔서 "T09의 0%가 주전자 탓인가?"를 본다. 질문과 판정 기준은 `experiments/docs/PROTOCOL.md`.

| 장면 key | 내용 | 비고 |
|---|---|---|
| `cup_s110` | 컵을 들어 입으로 가져가 마시기 (200 frame) | T09 주 장면과 같은 사람/XML |
| `book_s11` | 책상의 책을 두 손으로 들고 걸어가 책장에 놓기 (150 frame) | |
| `cup_s89`, `book_s149` | 두 번째 사례 (선택) | |

Push Button은 ParaHome에 없다 → `../humanoidmimicgen/QUICKSTART.md`.
WristMimic 코드와 Isaac Gym 환경은 `../t09-wristmimic`의 것을 그대로 쓴다 (따로 설치할 것 없음). 결과는 이 저장소 `experiments/` 아래에 쌓인다.

> **모든 명령은 `~/code/3dv-project/repos/parahome` 에서 실행한다.** GPU 작업은 한 번에 하나만.

```bash
cd ~/code/3dv-project/repos/parahome
```

## 1. 준비 (한 번만, 이 컴퓨터에서는 이미 완료)

```bash
bash experiments/scripts/setup.sh     # audit용 uv 환경 + ParaHome 데이터 (Google Drive, 압축 2.7 GB -> 7.7 GB)
```

## 2. 데이터 audit (CPU, 약 1분, 이미 실행함)

```bash
bash experiments/scripts/audit.sh     # -> experiments/results/audit/AUDIT.md, clips.csv
```

## 3. 파이프라인 점검 (약 10분)

```bash
bash experiments/scripts/pipeline_check.sh
```

컵/책 장면을 20 epoch씩 학습하고 평가, 리포트까지 돈다. 마지막 줄 `파이프라인 점검 통과`. epoch당 시간과 최대 VRAM을 출력한다.

## 4. 본 실험

| 단계 | 명령 | 시간 | 무엇을 얻나 |
|---|---|---|---|
| **Phase 1** | `bash experiments/scripts/phase1_wristmimic.sh` | 약 11시간 (장면당 약 5.2시간 + 평가 약 15분) | T09 설정 그대로 컵/책 3000 epoch. T09 주전자 결과와 같은 epoch끼리 비교 |
| Phase 2 (선택) | `bash experiments/scripts/phase2_diag.sh` | 약 4.5시간 | T09 진단 설정(랜덤 시작)을 컵에. 주전자에서처럼 "들기"에서만 막히는가 |
| (선택) | `SCENES="cup_s89 book_s149" bash experiments/scripts/phase1_wristmimic.sh` | 약 11시간 | 두 번째 사례 |

장면 하나씩 나눠 돌리려면:

```bash
SCENES="cup_s110" bash experiments/scripts/phase1_wristmimic.sh     # 약 5.5시간
SCENES="book_s11" bash experiments/scripts/phase1_wristmimic.sh     # 약 5.5시간
```

끊기면 같은 명령을 다시 실행한다 (`_latest.pth`부터 이어서 학습, 이미 평가한 체크포인트는 건너뜀).

## 5. 결과 보기

```bash
bash experiments/scripts/report.sh [3000]     # 비교 epoch (기본: run들의 마지막 공통 epoch)
```

| 파일 | 내용 |
|---|---|
| `experiments/results/report/REPORT.md` | 컵/책 vs T09 주전자: 성공, 과제 성공, 진행률, **첫 실패 frame vs 잡기/들기 frame**, 실패 원인, 학습 곡선 |
| `experiments/results/report/fig_curves.png`, `curves.csv` | 성공/진행률/첫 실패 frame vs epoch |
| `experiments/results/audit/AUDIT.md`, `clips.csv` | 482개 구간의 운동학 지표 |
| `experiments/results/eval/<mode>/<run>/` | T09 평가기 형식 json/npz |
| `experiments/runs/<run>/` (git 제외) | 체크포인트, `train.log`, `gpu.csv`, `run_meta.json` |

## 6. 개별 명령

```bash
bash experiments/scripts/train_wm.sh cup_s110 3000          # 학습 1회 (scene, epochs, [seed])
bash experiments/scripts/eval_wm.sh wm_cup_s110_default_s0  # det + stoch 평가 (모든 체크포인트)
# 원본 ParaHome 시퀀스 렌더링 (open3d 필요, 업스트림 visualize/)
```

## 7. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `T09 환경 없음` | `../t09-wristmimic`에서 `uv sync` (`../../isaacgym-env/README.md`) |
| OOM / `PxgCudaDeviceMemoryAllocator` | 다른 GPU 작업을 끈다. 1024 env는 약 8.2 GB |
| `configs/env가 variants.yaml과 다릅니다` | T09 저장소의 설정을 건드렸다. `cd ../t09-wristmimic && .venv/bin/python experiments/tools/make_configs.py` |
