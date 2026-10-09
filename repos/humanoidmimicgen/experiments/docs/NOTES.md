# HumanoidMimicGen 노트 (2026-10-09)

업스트림: `github.com/NVlabs/humanoidmimicgen@d82844d` (2026-09-23, 공개 commit 1개). 데이터: `linkenv/humanoidmimicgen-g1-benchmark@acf6b53`.

## 1. 원본 대비 바뀐 것

| 파일 | 수정 | 이유 / 영향 |
|---|---|---|
| `scripts/train_policy_example.py` | `RELEASE_META_FILES`에서 `stats.json` 제거, snapshot 완전성 검사가 `meta/stats.json` 대신 `meta/episodes_stats.jsonl`을 확인 | 고정된 데이터 revision에는 `meta/stats.json`이 **없다** (HF API로 확인: episodes.jsonl, episodes_stats.jsonl, info.json, modality.json, tasks.jsonl만 있음). 원본 코드는 빈 캐시에서 항상 `HTTP 404`로 멈춘다. v2.1→v3 변환기가 `episodes_stats.jsonl`로 `stats.json`을 만들고 이후 단계는 그 파일을 쓰므로 학습 데이터는 같다 |
| 설치 | pip 대신 `uv pip` (같은 버전, 같은 순서). 리포트용 matplotlib 추가 | |
| 추가 파일 | `experiments/` 전체 (스크립트, `tools/prepare_data.py`, `tools/eval_baselines.py`, `tools/report.py`) | 업스트림 코드는 그대로 호출한다 |

`humanoidmimicgen/wbc/external_dependencies/.../policy/{stand,walk}.onnx`는 `download_wbc_policies`가 받은 파일이다 (NVIDIA Open Model License, git 제외).

## 2. 환경

| 항목 | 값 |
|---|---|
| Python | 3.10 (uv) |
| torch | 2.10.0+cu128 (공식 휠, sm_120 지원. 직접 빌드 불필요) |
| LeRobot | 0.4.4 @ `8fff0fd` (학습/평가 스크립트가 commit을 검사한다) |
| MuJoCo / RoboSuite | 3.2.6 / 1.5.1 (CPU) |
| onnxruntime | 1.22.1 CPU (WBC 하체 정책) |

- README 순서대로 설치하면 `[wbc-replay]`의 `lerobot==0.3.3`이 학습용 0.4.4로 덮인다. 그 결과 `playback_dataset.py --action-source recorded`
  (공개 1K replay 데이터, LeRobot v2.1 형식)는 `BackwardCompatibilityError`로 실패한다. 업스트림 설치 순서의 문제이고 우리 실험(학습/평가,
  사람 시범의 `wbc-goal` 재생)에는 영향이 없다. 필요하면 별도 venv에 0.3.3을 설치한다.

## 3. 점검에서 확인한 것 (2026-10-09)

| 항목 | 결과 |
|---|---|
| 무작위 행동 300 step | 동작 (영상 `experiments/results/_check/random_action.mp4`) |
| 사람 시범 1개를 WBC 목표로 재생 (`wbc-goal`) | **task 성공** (563 step 중 145 step 눌림, 첫 성공 201 step) → 이 컴퓨터의 MuJoCo + WBC 경로가 정상 |
| 학습 속도 | 약 5.5 update/s (batch 16, GPU 사용률 낮음: 영상 디코딩이 병목, `NUM_WORKERS` 4) → 20K 약 1시간 |
| 최대 GPU 메모리 | 약 6.1 GB (데스크톱 포함) |
| 시뮬레이션 속도 | 정책 없이 약 2.8 ms/step (CPU) |
| 체크포인트 크기 | 3.0 GB / 개 |

## 4. 성공 판정의 바닥값 (결과 해석에 중요)

`IsJointQposInRange(control_panel, 0, -1, -0.01)`: 버튼 관절이 1 cm 이상 들어가면 그 순간 성공 (1250 step 안에 한 번이라도).

| 정책 (점검, 소수 episode) | 성공 |
|---|---|
| 200 update DP (loss 0.97) | 2/2 |
| mean (데이터 평균 행동 반복) | 1/5 |
| hold (가만히) | 0/5 |
| replay (시범 하나의 행동을 열린 루프로) | 0/5 |

→ Phase 1은 기준 정책 4개를 100 episode씩 같이 돌린다. 학습 정책의 성공률은 이 바닥값과 비교해서 읽는다.
