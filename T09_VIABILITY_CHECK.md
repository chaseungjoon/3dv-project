# T09 viability 점검: 하드웨어 한계인가, 과제 한계인가

작성: 2026-10-09 (실험 준비 완료, 결과 전). 결과가 나오면 이 문서 아래에 보고서를 붙인다.

## 0. 질문

T09 baseline (WristMimic, 주전자 옮기기)은 RTX 5070 12GB에서 **성공 0%**였다. 병목은 "쥐고 들어 올리는 순간"(frame 45~60)
(`BASELINE_T09.md`). 원인 후보는 둘이다.

- **하드웨어**: 12GB 때문에 env 수(2048 → 1024), PhysX 버퍼, 학습량이 줄었다. 그렇다면 이런 종류의 과제는 이 GPU에서 계속 막힌다.
- **과제/방법**: 주전자 장면, 또는 WristMimic의 학습 구조(항상 frame 0 시작, 엄격한 종료 조건)가 문제다. 그렇다면 하드웨어와 무관하다.

세 저장소에서 변수를 하나씩 바꿔서 가른다.

| 저장소 | 바꾸는 것 | 고정하는 것 | 가르는 질문 |
|---|---|---|---|
| `repos/skillmimic-v2` | **방법** (SkillMimic-V2, SIGGRAPH 2025) | GPU, 시뮬레이터 (Isaac Gym), 데이터 (ParaHome) | 같은 GPU에서 다른 방법은 ParaHome의 잡고 들기를 배우는가? (주전자 포함) |
| `repos/parahome` | **물체/장면** (컵, 책) | GPU, 방법, 설정, 평가기 (T09 그대로) | T09의 0%는 주전자 탓인가? |
| `repos/humanoidmimicgen` | **패러다임** (시범 1,000개 모방학습, MuJoCo, G1 로봇) | GPU | 최신 휴머노이드 조작 파이프라인이 이 GPU에서 논문 규모로 도는가? |

## 1. Task별 적용 범위

| | Place Book | Drink Cup | Push Button |
|---|---|---|---|
| SkillMimic-V2 | ✅ `place_book` (논문 clip) | ✅ `drink_cup` (논문 clip) | ❌ 데이터 없음 (ParaHome에 버튼 없음) |
| ParaHome (+ T09 WristMimic) | ✅ `book_s11` (+ `book_s149`) | ✅ `cup_s110` (T09와 같은 사람) (+ `cup_s89`) | ❌ ParaHome에 버튼 없음. 가장 가까운 "가스레인지 손잡이"는 audit에만 |
| HumanoidMimicGen | ❌ 벤치마크에 없음 | ❌ 벤치마크에 없음 | ✅ `02_push_button` (1,012 demo) |

추가로 SkillMimic-V2의 `place_kettle`을 넣었다. T09와 같은 물체라서 "방법만 바꾸면 주전자도 되는가"를 직접 본다.

## 2. 준비 단계에서 이미 확인된 사실 (2026-10-09, pipeline check)

| 항목 | 결과 | 의미 |
|---|---|---|
| SkillMimic-V2 **2048 env** (논문과 같음) | 들어감: 최대 9.2~11.2 GB, 학습 중 증가 없음 | T09는 2048에서 OOM. **메모리는 방법에 따라 한계가 아닐 수 있다** |
| SkillMimic-V2 속도 | 5,500~10,500 samples/s (4090 논문 대비 약 2.4배 느림) | 논문 1.0B 샘플은 clip당 약 45시간 → T09와 같은 1.97억 샘플로 비교 |
| WristMimic, 컵/책 장면 (1024 env) | 6.0~6.4 s/epoch, 8.0 GB | T09 주전자와 같은 비용 |
| HumanoidMimicGen | 학습 5.5 update/s (20K = 약 1시간), 6 GB. 사람 시범 WBC 재생 성공 | MuJoCo는 CPU, 공식 torch 2.10이 sm_120 지원. **하드웨어 문제 없음** |
| HMG Push Button 판정 | 200 update만 학습한 정책도 2/2 성공, 평균 행동 반복도 1/5 | 성공률만으로는 약하다 → 학습 없는 기준 정책과 같이 본다 |
| 업스트림 문제 | SkillMimic-V2: ParaHome 명령/모델/history encoder 미공개, 가구 위치를 저장소에 없는 원본에서 읽음. WristMimic: 책장 mesh 누락. HMG: 데이터 준비가 없는 `stats.json`을 요구 (404) | 모두 고침/재구성, 각 저장소 NOTES에 기록 |

## 3. 실행 순서 (사용자가 직접, tmux 권장, GPU 작업은 한 번에 하나)

| 순서 | 명령 | 시간 | 우선순위 |
|---|---|---|---|
| 1 | `cd ~/code/3dv-project/repos/humanoidmimicgen && bash experiments/scripts/phase1_push_button.sh` | 약 2.5~3시간 | 필수 (Push Button) |
| 2 | `cd ~/code/3dv-project/repos/parahome && SCENES="cup_s110" bash experiments/scripts/phase1_wristmimic.sh` | 약 5.5시간 | 필수 (Drink Cup, T09 방법) |
| 3 | `cd ~/code/3dv-project/repos/skillmimic-v2 && CLIPS="drink_cup" bash experiments/scripts/phase1_main.sh` | 약 9시간 | 필수 (Drink Cup, V2) |
| 4 | `cd ~/code/3dv-project/repos/parahome && SCENES="book_s11" bash experiments/scripts/phase1_wristmimic.sh` | 약 5.5시간 | 필수 (Place Book, T09 방법) |
| 5 | `cd ~/code/3dv-project/repos/skillmimic-v2 && CLIPS="place_book" bash experiments/scripts/phase1_main.sh` | 약 9시간 | 필수 (Place Book, V2) |
| 6 | `cd ~/code/3dv-project/repos/skillmimic-v2 && CLIPS="place_kettle" bash experiments/scripts/phase1_main.sh` | 약 10시간 | 강력 권장 (주전자, T09와 직접 비교) |
| 7 | `cd ~/code/3dv-project/repos/skillmimic-v2 && bash experiments/scripts/phase2_baseline.sh` | 약 12시간 | 권장 (SM v1 baseline) |
| 8 | `cd ~/code/3dv-project/repos/skillmimic-v2 && bash experiments/scripts/phase3_envs.sh` | 약 10시간 | 선택 (env 1024 vs 2048) |
| 9 | `cd ~/code/3dv-project/repos/parahome && bash experiments/scripts/phase2_diag.sh` | 약 4.5시간 | 선택 (랜덤 시작) |

필수 1~5 약 32시간, 6까지 약 42시간, 전부 약 68시간. 각 스크립트는 끊겨도 같은 명령으로 이어진다.
각 저장소의 `QUICKSTART.md`에 개별 명령과 문제 해결이 있다. 결과는 각 저장소의 `experiments/results/report/REPORT.md`.

## 4. 판정 표 (결과를 보기 전에 고정, 자세한 기준은 각 저장소 `experiments/docs/PROTOCOL.md`)

| SkillMimic-V2 (같은 GPU, 다른 방법) | ParaHome × WristMimic (같은 방법, 다른 물체) | 결론 |
|---|---|---|
| 과제/들어올림 성공 ≥ 0.5 | 컵/책도 0 | **하드웨어는 근본 장애가 아니다.** 실패는 WristMimic의 학습 구조/설정 쪽. T09는 "방법 쪽 원인(시작 상태, 종료 조건, 손목 설정)"을 연구 질문으로 살릴 수 있다 |
| 과제/들어올림 성공 ≥ 0.5 | 컵/책 > 0 (주전자만 0) | 하드웨어 문제 아님 + 주전자 장면 특이적. **T09는 주 장면을 컵/책으로 바꾸면 원래 계획(손목 설정 vs 성공률)대로 가능** |
| 0이고 학습 곡선이 계속 오름 | 0 | 메모리는 아니고 **속도(예산)** 문제일 가능성. 4090 대비 약 2.4배 느림. 예산을 늘리거나(이어 학습) 범위를 좁혀야 함 |
| 0이고 곡선이 평평 | 0 | 이 GPU 설정에서 이 종류의 과제가 막힌다는 증거. **T09 같은 손-물체 물리 모방은 이 하드웨어에서 큰 장애** → 다른 과제 검토 |

HumanoidMimicGen은 "로봇 + 모방학습" 쪽 대안의 실행 가능성을 본다: 학습 정책이 기준 정책 바닥값보다 확실히 높으면
(95% CI가 겹치지 않으면) 이 GPU에서 휴머노이드 조작 연구를 할 수 있는 다른 경로가 있다는 근거가 된다.

## 5. 보고서 (결과 후 작성)

(실험 후 각 `REPORT.md`를 모아 여기에 작성)
