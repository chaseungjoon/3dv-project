# T10 실험 프로토콜

과제 T10 Cross-embodiment robot manipulation. Baseline은 과제 가이드(`3dv-project/05_robotic_manipulation.pdf` T10)가 지정한
**CrossFormer** (공식 130M 체크포인트, frozen)이다. 이 문서는 **본 결과를 보기 전에** 정해 둔 규칙이다.
결과를 본 뒤 규칙을 바꾸면 그 사실과 이유를 끝의 "변경 기록"에 적는다.

작성일: 2026-10-07. 기준 코드: `repos/t10-crossformer` (원본 `rail-berkeley/crossformer@4a56b64` + `CODE_NOTES.md` 1절의 로컬 수정).

---

## 1. 무엇을 고정하는가

| 항목 | 값 | 근거 |
|---|---|---|
| 정책 | `hf://rail-berkeley/crossformer` step 300000, 129.7M 파라미터, **가중치 변경 없음** | T10 범위: inference/test-time만 |
| head | `single_arm` (L1 회귀, 4-step chunk, 7차원) | 두 로봇 모두 이 head로 학습됨 |
| 관측 | `image_primary` 1개 (Bridge `image_0`, fractal `image`), 224×224 Lanczos3, history 5 | 체크포인트 config의 학습 설정. single-arm 데이터는 proprio/손목 카메라를 쓰지 않음 |
| embodiment | WidowX (`bridge_dataset`), Google Robot (`fractal20220817_data`) | 가이드 "두 arm setup". 학습 가중치 최대(각 0.17)이고 SimplerEnv가 있는 유일한 두 로봇 |
| 오프라인 평가 데이터 | bridge `val` 앞 4 shard (215 episode), fractal `train[95%:]` 마지막 4 shard (342 episode) | 둘 다 학습에 쓰이지 않음 (2절). 언어 지시가 빈 episode는 모든 variant에서 제외 → 모든 variant가 같은 episode 집합 |
| action 통계 | 체크포인트의 `dataset_statistics.json` (데이터셋 이름별) | 공식 notebook과 SimplerEnv wrapper가 쓰는 값 |
| closed-loop | SimplerEnv@eddb569 visual matching, 공식 스크립트의 episode grid (README 2절 표) | 공식 평가와 비교 가능 |
| batch | 32 | 12 GB에서 안전 (128 OOM) |

### 1.1 Held-out 근거

CrossFormer 데이터 로더(`crossformer/data/dataset.py:457`)는 `val` split이 있으면 학습에 `train`을, 없으면 `train[:95%]`를 쓴다.
- bridge_dataset: `val` split 존재 → `val`은 학습에 안 쓰임.
- fractal: `val` 없음 → `train[95%:]` (episode 82851번 이후)는 학습에 안 쓰임. 받은 shard는 episode 86870~87211 (`download_subset.py`가 assert).
- taco_play (선택): `test` split은 로더가 읽지 않음.

각 subset의 출처는 `experiments/data/<name>/<version>/SUBSET.json`에 기록된다.

## 2. 오프라인 지표 (Phase 1~3)

모든 시점 t (episode 길이 T)에서 한 번 추론. **실행되는 action = chunk의 step 0**. 별도 표시가 없으면 이것으로 계산한다.

| 지표 | 정의 | 왜 |
|---|---|---|
| `norm_l1` | 위치/회전 6차원의 \|pred − GT\| / σ_own 평균 (σ = 그 로봇 데이터의 action std) | 모델이 학습한 공간. 로봇 간 비교 가능 |
| `trans_err_mm` | ‖A(pred_xyz − GT_xyz)‖, A = 그 로봇의 action→m 변환 (`conventions.py`), mm/step | 단위를 m로 맞춘 이동 오차 |
| `trans_err_mm_s` | `trans_err_mm` × 제어 주기 (Hz) | 제어 주기가 다른 로봇끼리 비교 가능 (mm/s) |
| `dir_cos` | 이동 방향 cosine (GT 이동 ≥ 2 mm인 step만) | 방향은 맞는데 크기만 틀린 경우를 구분 |
| `mag_ratio` | ‖pred 이동‖ / ‖GT 이동‖ (GT ≥ 2 mm), window 중앙값 | "action scale이 2배 틀림" 같은 실패를 직접 측정 |
| `grip_acc` | gripper를 0.5에서 이진화했을 때 GT와 일치 비율 | |
| `chunk_norm_l1_by_h` | chunk step h = 0..3별 `norm_l1` (episode 끝 너머 step 제외) | chunk 실행 길이 선택의 근거 |
| `raw_mse` | 데이터셋 원래 단위의 MSE | **로봇 간 합산 금지**. 왜 안 되는지 보여주기 위해서만 보고 (REPORT "단위가 다른 raw MSE") |

### 2.1 실행 제약 위반 (execution-constraint violations)

예측 action에 적용하고, **같은 검사를 GT action에도 적용한 비율을 기준선으로 함께 보고한다** (GT도 위반이면 검사가 너무 엄격한 것).
한계값은 항상 **실제 로봇(=그 데이터)** 의 것이다.

| 검사 | 정의 |
|---|---|
| `v_p01_p99` | 6차원 중 하나라도 학습 데이터의 [p01, p99] 밖 (약한 검사, GT도 일부 위반) |
| `v_range` | 6차원 중 하나라도 학습 데이터의 [min, max] 밖 (그 로봇이 학습에서 한 번도 안 한 명령) |
| `v_speed` | 한 step 이동(m)이 GT 한 step 이동 p99의 1.5배 초과 |
| `v_workspace_step0`, `v_workspace_chunk` | 현재 EEF 위치 + 누적 이동이 작업공간 상자 밖 (상자 = 평가 데이터 EEF 위치의 min/max ± 5 cm; chunk는 4 step 중 하나라도) |
| `v_gripper` | gripper 출력이 [−0.05, 1.05] 밖 (head에 clip이 없음) |

관절 한계와 특이점(singularity)은 baseline에서 재지 않는다 (IK와 URDF 필요). 연구 단계 후보로 남긴다 (5절).

### 2.2 action 단위 probe (Phase 3, 추가 추론 없음)

`baseline_*` 예측을 (a) 자기 통계 (b) 통계 없이(정규화 출력 그대로) (c) 다른 로봇 통계로 unnormalize해서 같은 지표를 계산한다.
(b), (c)는 "변환을 틀린 배포"를 흉내 낸 것이다.

## 3. 통계 처리와 "차이 있음" 판정

- **독립 단위는 episode다.** window 지표는 episode 안에서 먼저 평균하고, episode 평균의 평균을 보고한다. CI는 episode bootstrap 95% (2000회).
- variant 비교는 **같은 episode끼리의 paired 차이** (variant − `baseline_goal`)로 한다. paired 차이의 bootstrap 95% CI가 0을 포함하지 않으면 "차이 있음" (REPORT의 `*`).
- 오프라인 추론은 결정적이다 (같은 입력 두 번, 최대 차이 1.7e-6). seed 반복은 하지 않는다.
- closed-loop 성공률은 task별로 모든 URDF/자세 변형을 합쳐(pooled) 보고하고, Wilson 95% CI를 붙인다.
  closed-loop은 거의 결정적이지만 GPU 부동소수 차이로 gripper가 드물게 다르게 나올 수 있다 (점검: 60 step 중 1번). 한 번만 돌린다.
- 여러 variant를 비교하므로 `*`는 탐색적 표시로 읽는다. 발표에서 주장할 차이는 효과 크기(얼마나)와 함께 말한다.

## 4. 결과를 보기 전에 적어 둔 예상 (가설)

개발 중 30 episode로 일부를 이미 봤다 (괄호에 표시). 본 결과로 확인한다.

| # | 예상 | 확인할 표 |
|---|---|---|
| H1 | Google Robot에서 goal-image 조건이 언어 조건보다 오차가 작다. 언어 조건은 "조건 없음"과 비슷하다 (개발 중 관찰: norm L1 0.49 vs 0.57 vs 0.56) | Phase 1, 2 paired |
| H2 | 두 로봇 모두 예측 이동 크기가 GT보다 작다 (`mag_ratio` < 1, L1 회귀의 수축). 언어 조건의 Google Robot에서 가장 작다 | Phase 1 |
| H3 | 올바른 변환에서는 제약 위반이 GT 기준선 수준이고, 통계를 틀리면(없음/다른 로봇) 위반이 크게 늘어난다 | 위반 표, 단위 probe |
| H4 | 제어 주기를 1/2, 1/3로 낮추면 mm/s 오차가 커지고 `mag_ratio`가 더 작아진다 (모델은 학습 주기의 step 크기를 낸다) | Phase 3 paired |
| H5 | 좌우 반전은 이동 방향(`dir_cos`)을 크게 망가뜨린다. center crop은 Google Robot에서만 영향이 있다 | Phase 3 |
| H6 | closed-loop 성공률은 WidowX보다 Google Robot에서 낮다 (오프라인에서 Google Robot의 `mag_ratio`가 작고, 점검 episode에서 gripper를 일찍 닫음) | Phase 4 |

## 5. Baseline과 연구의 경계

baseline = **공식 사용법 그대로** (Phase 1의 두 variant, Phase 4의 `sim_baseline`). Phase 2/3과 `sim_notask`는 baseline의 성질을 재는 진단이며,
그 결과로 연구 질문의 크기를 정한다. T10 연구 단계에서 바꿀 수 있는 것 (가중치는 고정):

- 출력 변환: action scale 보정 (`mag_ratio` < 1 대응), gripper 규칙, chunk 실행/ensemble 방식, 통계 보정
- 실행 제약: 작업공간/속도 clip, IK 기반 관절 한계 투영
- 입력 변환: 카메라 crop/aspect, 제어 주기에 맞춘 frame 간격

closed-loop의 `action_scale`, `rotation`, `stats`, `sticky_repeat` 등은 이미 `sim_variants.yaml`의 knob으로 열어 두었다.

## 6. 변경 기록

| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-10-07 | 최초 작성 | |
